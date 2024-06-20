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
from torch import Tensor, empty, tensor, float32
from aihwkit_lightning.nn.modules.triton_utils.abs_max import sliced_fast_abs_max
from aihwkit_lightning.simulator.configs import (
    TorchInferenceRPUConfig,
    WeightClipType,
    WeightModifierType,
)


# @triton.autotune(
#     configs=[triton.Config(
#     {"BLOCK_SIZE_INP": 128, "BLOCK_SIZE_HIDDEN": 32, "BLOCK_SIZE_OUT": 128, "GROUP_SIZE_INP": 1},
#     num_warps=8, num_stages=0)],
#     key=["inp_size", "hidden_size", "out_size"],
# )
@triton.jit
def matmul_kernel(
    # pointers to tensors
    inp_ptr,  # 2D [inp_size, hidden_size]
    weights_ptr,  # 2D [hidden_size, out_size]
    out_ptr,  # 2D [inp_size, out_size]
    assumed_wmax_ptr,  # 2D [num_slices, out_size]
    input_range_ptr,  # 1D [num_slices]
    upper_end_of_slices_ptr,  # 1D [num_slices]
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
    # miscellaneous
    inp_res: tl.constexpr,
    is_fp: tl.constexpr,
    is_float32: tl.constexpr,
    allow_tf32: tl.constexpr,
    # block sizes
    BLOCK_SIZE_INP: tl.constexpr,
    BLOCK_SIZE_HIDDEN: tl.constexpr,
    BLOCK_SIZE_OUT: tl.constexpr,
    GROUP_SIZE_INP: tl.constexpr,
):
    # cldiv = lambda a, b : (a + b - 1) // b # How often does b fit into a (rounded up)

    # Example GROUP_SIZE_INP: 8, inp_size: 256, hidden_size: 64 out_size: 256
    # BLOCK_SIZE_INP: 32 BLOCK_SIZE_HIDDEN: 16 BLOCK_SIZE_OUT: 32
    # Grid: 8 * 8 -> PIDs 0 1 2 3 ... 63

    pid = tl.program_id(axis=0) # 16
    num_pid_m = tl.cdiv(inp_size, BLOCK_SIZE_INP) # 8
    num_pid_n = tl.cdiv(out_size, BLOCK_SIZE_OUT) # 8
    num_pid_in_group = GROUP_SIZE_INP * num_pid_n # 2 * 8 = 16
    group_id = pid // num_pid_in_group # 0 .. 15 belong to 0 and 16 .. 31 to 1 -> 1
    first_pid_m = group_id * GROUP_SIZE_INP # 0
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_INP) # 2
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m) # 1
    pid_n = (pid % num_pid_in_group) // group_size_m # 0

    # [pid_m, pid_n] determines "x,y" coordinates of the result we compute

    # offs_am = (pid_m * BLOCK_SIZE_INP + tl.arange(0, BLOCK_SIZE_INP)) % inp_size
    # offs_bn = (pid_n * BLOCK_SIZE_OUT + tl.arange(0, BLOCK_SIZE_OUT)) % out_size
    # offs_k = tl.arange(0, BLOCK_SIZE_HIDDEN)
    # a_ptrs = inp_ptr + (offs_am[:, None] * stride_inp_inp_size + offs_k[None, :] * stride_inp_hidden_size)
    # b_ptrs = weights_ptr + (offs_k[:, None] * stride_weights_hidden_size + offs_bn[None, :] * stride_weights_out_size)

    # accumulator = tl.zeros((BLOCK_SIZE_INP, BLOCK_SIZE_OUT), dtype=tl.float32)
    # for k in range(0, tl.cdiv(hidden_size, BLOCK_SIZE_HIDDEN)):
    #     # Load the next block of A and B, generate a mask by checking the K dimension.
    #     # If it is out of bounds, set it to 0.

    #     a = tl.load(a_ptrs, mask=offs_k[None, :] < hidden_size - k * BLOCK_SIZE_HIDDEN, other=0.0)
    #     b = tl.load(b_ptrs, mask=offs_k[:, None] < hidden_size - k * BLOCK_SIZE_HIDDEN, other=0.0)


    #     # We accumulate along the K dimension.
    #     accumulator = tl.dot(a, b, accumulator)
    #     # Advance the ptrs to the next K block.
    #     a_ptrs += BLOCK_SIZE_HIDDEN * stride_inp_hidden_size
    #     b_ptrs += BLOCK_SIZE_HIDDEN * stride_weights_hidden_size



    accumulator = tl.zeros((BLOCK_SIZE_INP, BLOCK_SIZE_OUT), dtype=tl.float32 if is_float32 else tl.float16)

    ir_range_lower = 0
    for slice_idx in range(0, num_slices):

        ir_range_upper = tl.load(upper_end_of_slices_ptr + slice_idx)
        current_lower = ir_range_lower

        # Generate the pointers
        offs_am = (pid_m * BLOCK_SIZE_INP + tl.arange(0, BLOCK_SIZE_INP)) % inp_size
        offs_bn = (pid_n * BLOCK_SIZE_OUT + tl.arange(0, BLOCK_SIZE_OUT)) % out_size

        for k in range(0, tl.cdiv(ir_range_upper-ir_range_lower, BLOCK_SIZE_HIDDEN)):
            current_upper = min(ir_range_upper, current_lower + (k+1) * BLOCK_SIZE_HIDDEN, hidden_size)

            offs_k = current_lower + tl.arange(0, BLOCK_SIZE_HIDDEN)
            a_ptrs = inp_ptr + (offs_am[:, None] * stride_inp_inp_size + offs_k[None, :] * stride_inp_hidden_size)
            b_ptrs = weights_ptr + (offs_k[:, None] * stride_weights_hidden_size + offs_bn[None, :] * stride_weights_out_size)
            
            a = tl.load(a_ptrs, mask=offs_k[None, :] < current_upper, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < current_upper, other=0.0)

            # load the correct input range
            input_range = tl.load(input_range_ptr + slice_idx)

            # clip between the input ranges
            over_ir_mask = (a > input_range)
            under_ir_mask = (a < -input_range)

            a = tl.where(over_ir_mask, input_range, a)
            a = tl.where(under_ir_mask, -input_range, a)

            # scale with input ranges
            a = a / input_range

            if not is_fp:
                # DEBUG needs to be removed
                tl.device_assert(tl.max(tl.abs(a)) <= 1.0, "max abs is bigger 1.0")
                a = a / inp_res
                a = tl.extra.cuda.libdevice.rint(a)
                a = a * inp_res
            
            if not is_float32:
                a = a.to(tl.float16)
                b = b.to(tl.float16)

            dot_prod = tl.dot(a, b, allow_tf32=allow_tf32)
            # scale back with the input range
            dot_prod = input_range * dot_prod

            # accumulate
            if not is_float32:
                dot_prod = dot_prod.to(tl.float16)
            accumulator = accumulator + dot_prod

            current_lower = current_upper

        ir_range_lower = ir_range_upper


    # You can fuse arbitrary activation functions here
    # while the accumulator is still in FP32!
    c = accumulator
    if not is_float32:
        c = c.to(tl.float16)

    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_INP + tl.arange(0, BLOCK_SIZE_INP)
    offs_cn = pid_n * BLOCK_SIZE_OUT + tl.arange(0, BLOCK_SIZE_OUT)
    c_ptrs = out_ptr + stride_out_inp_size * offs_cm[:, None] + stride_out_out_size * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < inp_size) & (offs_cn[None, :] < out_size)
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
        apply_weight_modifier: bool,
    ):
        assert input_range.is_contiguous(), "input_range not contiguous"
        assert weights.is_contiguous(), "weights not contiguous"
        assert inp.is_contiguous(), "inp not contiguous"

        assumed_wmax = sliced_fast_abs_max(weights=weights, split_sizes=in_sizes)
        assert assumed_wmax.is_contiguous(), "assumed_wmax not contiguous"

        upper_end_of_slices = (
            tensor(in_sizes, device=inp.device, dtype=inp.dtype).cumsum(dim=0).contiguous().int()
        )

        out_shape = (*inp.shape[:-1], weights.size(0))
        inp = inp.flatten(end_dim=-2)
        num_slices = len(input_range)

        out_size, hidden_size = weights.shape
        inp_size = inp.size(0)
        assert hidden_size == inp.size(1), f"Input hidden size is {inp.size(1)} but weight hidden size is {hidden_size}"

        # invoke kernel
        def grid(meta):
            return (
                triton.cdiv(inp_size, meta["BLOCK_SIZE_INP"])
                * triton.cdiv(out_size, meta["BLOCK_SIZE_OUT"]),
            )

        out = empty((inp_size, out_size), device=inp.device, dtype=inp.dtype)

        # bring the input resolution into a state that we can interpret
        inp_res = rpu_config.forward.inp_res
        if inp_res > 0:
            inp_res = 2 / inp_res if inp_res > 1.0 else 2 * inp_res
            assert inp_res > 0, "resolution is <= 0"

        matmul_kernel[grid](
            inp,  # 2D [inp_size, hidden_size]
            weights.T,  # 2D [hidden_size, out_size]
            out,  # 2D [inp_size, out_size]
            assumed_wmax,  # 2D [num_slices, out_size]
            input_range,  # 1D [num_slices]
            upper_end_of_slices,  # 1D [num_slices]
            # sizes
            inp_size,
            hidden_size,
            out_size,
            num_slices,
            # strides
            inp.stride(0),
            inp.stride(1),
            weights.stride(1),  # flipped because of transpose
            weights.stride(0),
            out.stride(0),
            out.stride(1),
            assumed_wmax.stride(0),
            assumed_wmax.stride(1),
            # for the other ptrs, we assume stride 1
            # miscellaneous
            inp_res, # inp_res
            inp_res == -1, # is_fp
            inp.dtype == float32,
            False, # allow_tf32 DEBUG.. Change to True in the end
            # block sizes
            32,
            16,
            32,
            2,
        )
        out = out.view(out_shape)
        
        # import torch
        # ideal = inp @ weights.T
        # assert torch.allclose(out, ideal, atol=1e-3)

        return out

    @staticmethod
    def backward(ctx: FunctionCtx, grad_output: Tensor):  # type: ignore
        pass
