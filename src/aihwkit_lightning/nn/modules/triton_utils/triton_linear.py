# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Functions for fast linear in triton."""
from typing import List
import triton  # type: ignore
import triton.language as tl  # type: ignore
from torch.autograd import Function
from torch.autograd.function import FunctionCtx
from torch import Tensor, empty, tensor
from aihwkit_lightning.nn.modules.triton_utils.abs_max import sliced_fast_abs_max 
from aihwkit_lightning.simulator.configs import (
    TorchInferenceRPUConfig,
    WeightClipType,
    WeightModifierType,
)


# @triton.autotune(
#     configs=[triton.Config(
#             {"BLOCK_SIZE_INP": 128, "BLOCK_SIZE_HIDDEN": 32, "BLOCK_SIZE_OUT": 128, "GROUP_SIZE_INP": 1},
#             num_warps=8, num_stages=0)],
#     key=["inp_size", "hidden_size", "out_size"],
# )
@triton.jit
def matmul_kernel(
    # pointers to tensors
    inp_ptr, # 2D [inp_size, hidden_size]
    weights_ptr, # 2D [hidden_size, out_size]
    out_ptr, # 2D [inp_size, out_size]
    assumed_wmax_ptr, # 2D [num_slices, out_size]
    input_range_ptr, # 1D [num_slices]
    upper_end_of_slices_ptr, # 1D [num_slices]
    # sizes
    inp_size,
    hidden_size,
    out_size,
    num_slices,
    # strides
    stride_inp_inp_size,
    stride_inp_hidden_size,
    stride_weights_hidden_size,
    stride_weights_out_size,
    stride_out_inp_size,
    stride_out_out_size,
    stride_assumed_wmax_num_slices,
    stride_assumed_wmax_out_size,
    # block sizes
    BLOCK_SIZE_INP: tl.constexpr,
    BLOCK_SIZE_HIDDEN: tl.constexpr,
    BLOCK_SIZE_OUT: tl.constexpr,
    GROUP_SIZE_INP: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, GROUP_SIZE_INP)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * GROUP_SIZE_INP + tl.arange(0, GROUP_SIZE_INP)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((GROUP_SIZE_INP, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        # We accumulate along the K dimension.
        accumulator = tl.dot(a, b, accumulator)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    # You can fuse arbitrary activation functions here
    # while the accumulator is still in FP32!
    if ACTIVATION == "leaky_relu":
        accumulator = leaky_relu(accumulator)
    c = accumulator.to(tl.float16)

    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * GROUP_SIZE_INP + tl.arange(0, GROUP_SIZE_INP)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


class TritonLinear(Function):
    @staticmethod
    def forward(
        ctx: FunctionCtx,
        inp: Tensor,
        weights: Tensor,
        input_range: Tensor,
        in_sizes: List[int],
        rpu_config: TorchInferenceRPUConfig,
        training: bool,
        apply_weight_modifier: bool
    ):
        assert input_range.is_contiguous(), "input_range not contiguous"
        assert weights.is_contiguous(), "weights not contiguous"
        assert inp.is_contiguous(), "inp not contiguous"

        assumed_wmax = sliced_fast_abs_max(weights=weights, split_sizes=in_sizes)
        assert assumed_wmax.is_contiguous(), "assumed_wmax not contiguous"

        upper_end_of_slices = tensor(in_sizes, device=inp.device, dtype=inp.dtype).cumsum(dim=0).contiguous()

        out_shape = (*inp.shape[:-1], weights.size(0))
        inp = inp.flatten(end_dim=-2)
        num_slices = len(input_range)

        out_size, hidden_size = weights.shape
        inp_size = inp.size(1)

        # invoke kernel
        grid = lambda META: (
            triton.cdiv(inp_size, META["BLOCK_SIZE_INP"]) * triton.cdiv(hidden_size, META["BLOCK_SIZE_HIDDEN"])\
                    * triton.cdiv(out_size, META["BLOCK_SIZE_OUT"]),
        )

        out = empty((inp_size, out_size), device=inp.device, dtype=inp.dtype)

        matmul_kernel[grid](
            inp, # 2D [inp_size, hidden_size]
            weights.T, # 2D [hidden_size, out_size]
            out, # 2D [inp_size, out_size]
            assumed_wmax, # 2D [num_slices, out_size]
            input_range, # 1D [num_slices]
            upper_end_of_slices, # 1D [num_slices]
            # sizes
            inp_size,
            hidden_size,
            out_size,
            num_slices,
            # strides
            inp.stride(0),
            inp.stride(1),
            weights.stride(1), # flipped because of transpose
            weights.stride(0),
            out.stride(0),
            out.stride(1),
            assumed_wmax.stride(0),
            assumed_wmax.stride(1),
            # for the other ptrs, we assume stride 1
            # block sizes
            128,
            32,
            128,
            1,
        )

        return out.view(out_shape)


    @staticmethod
    def backward(ctx: FunctionCtx, grad_output: Tensor):
        pass