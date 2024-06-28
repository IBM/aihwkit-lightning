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
import os
from typing import List
import triton  # type: ignore
import triton.language as tl  # type: ignore
from torch.autograd import Function
from torch.autograd.function import FunctionCtx
from torch import Tensor, empty, tensor, float32, randint, zeros_like, clamp
from aihwkit_lightning.nn.modules.triton_utils.triton_abs_max import sliced_fast_abs_max
from aihwkit_lightning.nn.modules.triton_utils.triton_std import sliced_fast_std
from aihwkit_lightning.simulator.configs import TorchInferenceRPUConfig


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_OUT": 256, "BLOCK_SIZE_HIDDEN": 64}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_SIZE_OUT": 256, "BLOCK_SIZE_HIDDEN": 32}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_SIZE_OUT": 128, "BLOCK_SIZE_HIDDEN": 32}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_SIZE_OUT": 64, "BLOCK_SIZE_HIDDEN": 32}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_SIZE_OUT": 128, "BLOCK_SIZE_HIDDEN": 32}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_SIZE_OUT": 32, "BLOCK_SIZE_HIDDEN": 32}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_SIZE_OUT": 32, "BLOCK_SIZE_HIDDEN": 32}, num_stages=5, num_warps=2),
        triton.Config({"BLOCK_SIZE_OUT": 64, "BLOCK_SIZE_HIDDEN": 32}, num_stages=5, num_warps=2),
    ],
    key=["hidden_size", "out_size"],
)
@triton.jit
def modifier_kernel(
    # pointers to tensors
    weights_ptr,  # 2D [hidden_size, out_size]
    assumed_wmax_ptr,  # 2D [num_slices, out_size]
    reduced_assumed_wmax_ptr,  # 2D [num_slices, 1]
    upper_end_of_slices_ptr,  # 1D [num_slices]
    # sizes
    hidden_size,
    out_size,
    num_slices,
    # strides
    stride_weights_hidden_size,
    stride_weights_out_size,
    stride_assumed_wmax_num_slices,
    stride_assumed_wmax_out_size,
    # miscellaneous
    modifier_type: tl.constexpr,  # str
    modifier_weight_res: tl.constexpr,  # float
    modifier_seed: tl.constexpr,  # int
    modifier_std: tl.constexpr,  # float
    # block sizes
    BLOCK_SIZE_HIDDEN: tl.constexpr,
    BLOCK_SIZE_OUT: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs_bn = (pid * BLOCK_SIZE_OUT + tl.arange(0, BLOCK_SIZE_OUT)) % out_size
    offs_assumed_wmax = pid * BLOCK_SIZE_OUT + tl.arange(0, BLOCK_SIZE_OUT)

    # for random number generation of output
    increase_weight_offsets_by = BLOCK_SIZE_HIDDEN * BLOCK_SIZE_OUT
    weight_random_offsets = tl.arange(0, BLOCK_SIZE_HIDDEN * BLOCK_SIZE_OUT).reshape(
        (BLOCK_SIZE_HIDDEN, BLOCK_SIZE_OUT), can_reorder=True
    )

    ir_range_lower = 0
    for slice_idx in range(0, num_slices):

        # load the abs-max we need
        abs_max_slice_ptrs = (
            assumed_wmax_ptr
            + slice_idx * stride_assumed_wmax_num_slices
            + offs_bn * stride_assumed_wmax_out_size
        )
        if modifier_type == "AddNormal" or (
            modifier_type == "Discretize" or modifier_type == "DiscretizeAddNormal"
        ):
            assumed_wmax_per_slice = tl.load(reduced_assumed_wmax_ptr + slice_idx)
        else:
            assumed_wmax_per_slice = tl.load(
                abs_max_slice_ptrs, mask=offs_assumed_wmax < out_size, other=float("-inf")
            )
            assumed_wmax_per_slice = assumed_wmax_per_slice[None, :]

        ir_range_upper = tl.load(upper_end_of_slices_ptr + slice_idx)
        current_lower = ir_range_lower

        num_k = tl.cdiv(ir_range_upper - ir_range_lower, BLOCK_SIZE_HIDDEN)
        for k in range(0, num_k):
            current_upper = min(
                ir_range_upper, current_lower + (k + 1) * BLOCK_SIZE_HIDDEN, hidden_size
            )

            offs_k = current_lower + tl.arange(0, BLOCK_SIZE_HIDDEN)

            b_ptrs = weights_ptr + (
                offs_k[:, None] * stride_weights_hidden_size
                + offs_bn[None, :] * stride_weights_out_size
            )

            b = tl.load(b_ptrs, mask=offs_k[:, None] < current_upper, other=0.0)

            if (modifier_type == "Discretize" or modifier_type == "DiscretizeAddNormal") or (
                modifier_type == "DiscretizePerChannel"
                or modifier_type == "DiscretizeAddNormalPerChannel"
            ):
                if modifier_weight_res > 0:
                    n_states = max(modifier_weight_res, 1 / modifier_weight_res)
                    res = 2 * assumed_wmax_per_slice / n_states
                    b = b / res
                    b = tl.extra.cuda.libdevice.rint(b)
                    b = b * res

            if (modifier_type == "AddNormal" or modifier_type == "AddNormalPerChannel") or (
                modifier_type == "DiscretizeAddNormal"
                or modifier_type == "DiscretizeAddNormalPerChannel"
            ):
                randn_block = tl.randn(modifier_seed + pid, weight_random_offsets)
                weight_random_offsets += increase_weight_offsets_by
                randn_block = assumed_wmax_per_slice * modifier_std * randn_block
                b += randn_block

            # store b back to DRAM...
            tl.store(
                b_ptrs,
                b,
                mask=(offs_k[:, None] < current_upper) & (offs_assumed_wmax[None, :] < out_size),
            )

            current_lower = current_upper

        ir_range_lower = ir_range_upper


@triton.autotune(
    configs=[
        triton.Config(
            {
                "BLOCK_SIZE_INP": 128,
                "BLOCK_SIZE_OUT": 256,
                "BLOCK_SIZE_HIDDEN": 64,
                "GROUP_SIZE_INP": 8,
            },
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_INP": 64,
                "BLOCK_SIZE_OUT": 256,
                "BLOCK_SIZE_HIDDEN": 32,
                "GROUP_SIZE_INP": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_INP": 128,
                "BLOCK_SIZE_OUT": 128,
                "BLOCK_SIZE_HIDDEN": 32,
                "GROUP_SIZE_INP": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_INP": 128,
                "BLOCK_SIZE_OUT": 64,
                "BLOCK_SIZE_HIDDEN": 32,
                "GROUP_SIZE_INP": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_INP": 64,
                "BLOCK_SIZE_OUT": 128,
                "BLOCK_SIZE_HIDDEN": 32,
                "GROUP_SIZE_INP": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_INP": 128,
                "BLOCK_SIZE_OUT": 32,
                "BLOCK_SIZE_HIDDEN": 32,
                "GROUP_SIZE_INP": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_INP": 64,
                "BLOCK_SIZE_OUT": 32,
                "BLOCK_SIZE_HIDDEN": 32,
                "GROUP_SIZE_INP": 8,
            },
            num_stages=5,
            num_warps=2,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_INP": 32,
                "BLOCK_SIZE_OUT": 64,
                "BLOCK_SIZE_HIDDEN": 32,
                "GROUP_SIZE_INP": 8,
            },
            num_stages=5,
            num_warps=2,
        ),
    ],
    key=["inp_size", "hidden_size", "out_size"],
)
@triton.jit
def matmul_kernel(
    # pointers to tensors
    inp_ptr,  # 2D [inp_size, hidden_size]
    weights_ptr,  # 2D [hidden_size, out_size]
    out_ptr,  # 2D [inp_size, out_size]
    ir_vector_ptr,  # 1D [hidden_size]
    assumed_wmax_ptr,  # 2D [num_slices, out_size]
    reduced_assumed_wmax_ptr,  # 2D [num_slices, 1]
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
    out_noise: tl.constexpr,
    out_noise_seed: tl.constexpr,
    out_noise_std: tl.constexpr,
    out_noise_per_channel: tl.constexpr,
    ir_vector_is_none: tl.constexpr,
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

    pid = tl.program_id(axis=0)  # 16
    num_pid_m = tl.cdiv(inp_size, BLOCK_SIZE_INP)  # 8
    num_pid_n = tl.cdiv(out_size, BLOCK_SIZE_OUT)  # 8
    num_pid_in_group = GROUP_SIZE_INP * num_pid_n  # 2 * 8 = 16
    group_id = pid // num_pid_in_group  # 0 .. 15 belong to 0 and 16 .. 31 to 1 -> 1
    first_pid_m = group_id * GROUP_SIZE_INP  # 0
    GROUP_SIZE_INP = min(num_pid_m - first_pid_m, GROUP_SIZE_INP)  # 2
    pid_m = first_pid_m + ((pid % num_pid_in_group) % GROUP_SIZE_INP)  # 1
    pid_n = (pid % num_pid_in_group) // GROUP_SIZE_INP  # 0

    accumulator = tl.zeros(
        (BLOCK_SIZE_INP, BLOCK_SIZE_OUT), dtype=tl.float32 if is_float32 else tl.float16
    )

    # for random number generation of output
    increase_out_offsets_by = BLOCK_SIZE_INP * BLOCK_SIZE_OUT
    output_random_offsets = tl.arange(0, BLOCK_SIZE_INP * BLOCK_SIZE_OUT).reshape(
        (BLOCK_SIZE_INP, BLOCK_SIZE_OUT), can_reorder=True
    )

    # Generate the pointers
    offs_am = pid_m * BLOCK_SIZE_INP + tl.arange(0, BLOCK_SIZE_INP)
    offs_bn = pid_n * BLOCK_SIZE_OUT + tl.arange(0, BLOCK_SIZE_OUT)
    offs_assumed_wmax = pid_n * BLOCK_SIZE_OUT + tl.arange(0, BLOCK_SIZE_OUT)

    ir_range_lower = 0
    for slice_idx in range(0, num_slices):

        # load the abs-max we need
        abs_max_slice_ptrs = (
            assumed_wmax_ptr
            + slice_idx * stride_assumed_wmax_num_slices
            + offs_bn * stride_assumed_wmax_out_size
        )

        if out_noise and out_noise_per_channel:
            assumed_wmax_per_slice = tl.load(
                abs_max_slice_ptrs, mask=offs_assumed_wmax < out_size, other=float("-inf")
            )
            assumed_wmax_per_slice = assumed_wmax_per_slice[None, :]
        else:
            assumed_wmax_per_slice = tl.load(reduced_assumed_wmax_ptr + slice_idx)

        ir_range_upper = tl.load(upper_end_of_slices_ptr + slice_idx)
        current_lower = ir_range_lower

        # load the correct input range
        input_range = tl.load(input_range_ptr + slice_idx)

        num_k = tl.cdiv(ir_range_upper - ir_range_lower, BLOCK_SIZE_HIDDEN)
        for k in range(0, num_k):
            current_upper = min(
                ir_range_upper, ir_range_lower + (k + 1) * BLOCK_SIZE_HIDDEN, hidden_size
            )

            offs_k = current_lower + tl.arange(0, BLOCK_SIZE_HIDDEN)
            a_ptrs = inp_ptr + (
                offs_am[:, None] * stride_inp_inp_size + offs_k[None, :] * stride_inp_hidden_size
            )
            b_ptrs = weights_ptr + (
                offs_k[:, None] * stride_weights_hidden_size
                + offs_bn[None, :] * stride_weights_out_size
            )

            a = tl.load(
                a_ptrs,
                mask=(offs_am[:, None] < inp_size) & (offs_k[None, :] < current_upper),
                other=0.0,
            )
            b = tl.load(
                b_ptrs,
                mask=(offs_k[:, None] < current_upper) & (offs_bn[None, :] < out_size),
                other=0.0,
            )

            if not ir_vector_is_none:
                # save the correct IR per dimension in the hidden
                tl.store(ir_vector_ptr + offs_k, input_range, mask=offs_k < current_upper)

            # clip between the input ranges
            over_ir_mask = a > input_range
            under_ir_mask = a < -input_range

            a = tl.where(over_ir_mask, input_range, a)
            a = tl.where(under_ir_mask, -input_range, a)

            # scale with input ranges
            a = a / input_range

            if not is_fp:
                a = a / inp_res
                a = tl.extra.cuda.libdevice.rint(a)
                a = a * inp_res

            if not is_float32:
                a = a.to(tl.float16)
                b = b.to(tl.float16)

            # do the MVM
            dot_prod = tl.dot(a, b, allow_tf32=allow_tf32)

            if out_noise:
                randn_block = tl.randn(out_noise_seed + pid, output_random_offsets)
                # we add a N(0,1)*std/sqrt(N_slices * N_K)
                randn_block = (
                    assumed_wmax_per_slice
                    * out_noise_std
                    / tl.sqrt(num_slices * num_k.to(tl.float32))
                    * randn_block
                )
                # add the noise
                dot_prod += randn_block
                # advance the output_random_offsets
                output_random_offsets += increase_out_offsets_by

            # scale back with the input range
            dot_prod = input_range * dot_prod

            # accumulate
            if not is_float32:
                dot_prod = dot_prod.to(tl.float16)
            accumulator = accumulator + dot_prod

            current_lower = current_upper

        ir_range_lower = ir_range_upper

    c = accumulator
    if not is_float32:
        c = c.to(tl.float16)

    offs_cm = pid_m * BLOCK_SIZE_INP + tl.arange(0, BLOCK_SIZE_INP)
    offs_cn = pid_n * BLOCK_SIZE_OUT + tl.arange(0, BLOCK_SIZE_OUT)
    c_ptrs = (
        out_ptr + stride_out_inp_size * offs_cm[:, None] + stride_out_out_size * offs_cn[None, :]
    )
    c_mask = (offs_cm[:, None] < inp_size) & (offs_cn[None, :] < out_size)
    tl.store(c_ptrs, c, mask=c_mask)


class TritonLinear(Function):
    @staticmethod
    def forward(
        ctx: FunctionCtx,
        inp: Tensor,
        weights: Tensor,
        input_range: Tensor,
        input_range_update_idx: Tensor,
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

        reduced_assumed_wmax = assumed_wmax.amax(dim=1, keepdim=True)
        assert reduced_assumed_wmax.is_contiguous(), "reduced_assumed_wmax is not contiguous"

        upper_end_of_slices = (
            tensor(in_sizes, device=inp.device, dtype=inp.dtype).cumsum(dim=0).contiguous().int()
        )

        out_shape = (*inp.shape[:-1], weights.size(0))
        if inp.ndim == 1:
            inp = inp.view(1, -1)
        inp = inp.flatten(end_dim=-2)
        num_slices = len(input_range)

        # update the input ranges if necessary
        if training:
            ir_params = rpu_config.pre_post.input_range
            if (input_range_update_idx < ir_params.init_from_data).any():
                stds = sliced_fast_std(inp, upper_end_of_slices)
                for slice_idx in range(num_slices):
                    idx = input_range_update_idx[slice_idx]
                    if idx < ir_params.init_from_data:
                        if stds[slice_idx] > 0.0:
                            input_range.data[slice_idx] = (
                                input_range.data[slice_idx] * idx
                                + ir_params.init_std_alpha * stds[slice_idx]
                            ) / (idx + 1)
                            input_range_update_idx[slice_idx] += 1
                        input_range.data[slice_idx] = input_range.data[slice_idx].abs()

        out_size, hidden_size = weights.shape
        inp_size = inp.size(0)
        assert hidden_size == inp.size(
            1
        ), f"Input hidden size is {inp.size(1)} but weight hidden size is {hidden_size}"

        def weight_modifier_grid(meta):
            return (triton.cdiv(out_size, meta["BLOCK_SIZE_OUT"]),)

        if apply_weight_modifier:
            modifier_std = rpu_config.modifier.std_dev
            modifier_seed = randint(2**31, (1,)).item()
            # bring the weight resolution into a state that we can interpret
            modifier_weight_res = rpu_config.modifier.res

            modifier_kernel[weight_modifier_grid](
                # pointers to tensors
                weights.T,  # 2D [hidden_size, out_size]
                assumed_wmax,  # 2D [num_slices, out_size]
                reduced_assumed_wmax,  # 2D [num_slices, 1]
                upper_end_of_slices,  # 1D [num_slices]
                # sizes
                hidden_size,
                out_size,
                num_slices,
                # strides
                weights.stride(1),  # flipped because of transpose
                weights.stride(0),
                assumed_wmax.stride(0),
                assumed_wmax.stride(1),
                # miscellaneous
                rpu_config.modifier.type.value,
                modifier_weight_res,
                modifier_seed,
                modifier_std,
                # block sizes
                # 16,  # for debugging
                # 32,
            )

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
            inp_res = 2.0 / inp_res if inp_res > 1.0 else 2.0 * inp_res
            assert inp_res > 0, "resolution is <= 0"

        # bring output noise variables into shape
        out_noise = training and (not rpu_config.forward.out_noise == 0.0)
        out_noise_std = rpu_config.forward.out_noise
        out_noise_per_channel = rpu_config.forward.out_noise_per_channel
        out_noise_seed = randint(2**31, (1,)).item()

        # we fill this vector with the element wise IR
        ir_vector = empty((1, hidden_size), device=inp.device, dtype=inp.dtype)
        ir_vector = ir_vector.contiguous()

        # for some tests, we skip rounding
        skip_rounding = os.environ.get("_AIHWKIT_NO_ROUNDING", False)

        matmul_kernel[grid](
            inp,  # 2D [inp_size, hidden_size]
            weights.T,  # 2D [hidden_size, out_size]
            out,  # 2D [inp_size, out_size]
            ir_vector,  # 1D [hidden_size]
            assumed_wmax,  # 2D [num_slices, out_size]
            reduced_assumed_wmax,  # 2D [num_slices, 1]
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
            inp_res,  # inp_res
            inp_res == -1 or skip_rounding,  # is_fp
            inp.dtype == float32,
            True,
            out_noise,
            out_noise_seed,
            out_noise_std,
            out_noise_per_channel,
            False,  # ir_vector is None
            # block sizes
            # 16,  # This is for debugging
            # 256,
            # 16,
            # 2,
        )

        # save some stuff for backwards
        ctx.rpu_config = rpu_config
        ctx.save_for_backward(inp, weights, input_range, ir_vector)

        out = out.view(out_shape)
        return out

    @staticmethod
    def backward(ctx: FunctionCtx, grad_output: Tensor):  # type: ignore
        # # DEBUG
        # import pydevd
        # pydevd.settrace(suspend=False, trace_only_current_thread=True)

        rpu_config: TorchInferenceRPUConfig
        rpu_config = ctx.rpu_config
        inp, weights, input_range, ir_vector = ctx.saved_tensors

        weights: Tensor
        grad_inp_shape = (*grad_output.shape[:-1], weights.size(1))
        if grad_output.ndim == 1:
            grad_output = grad_output.view(1, -1)
        grad_output = grad_output.flatten(end_dim=-2)

        # bring the input resolution into a state that we can interpret
        inp_res = rpu_config.forward.inp_res
        if inp_res > 0:
            inp_res = 2.0 / inp_res if inp_res > 1.0 else 2.0 * inp_res
            assert inp_res > 0, "resolution is <= 0"

        # for some tests, we skip rounding
        skip_rounding = os.environ.get("_AIHWKIT_NO_ROUNDING", False)

        # input gradient
        inp: Tensor
        ir_vector: Tensor
        inp_rounded = inp.clamp(-ir_vector, ir_vector)
        if inp_res > 0 and not skip_rounding:
            scale = ir_vector * inp_res
            inp_rounded = (inp_rounded / scale).round() * scale
        grad_w = grad_output.T @ inp_rounded
        grad_inp = grad_output @ weights
        grad_inp = grad_inp.view(grad_inp_shape)

        # ir gradient
        decay = rpu_config.pre_post.input_range.decay  # type: ignore[attr-defined]
        input_min_percentage = rpu_config.pre_post.input_range.input_min_percentage  # type: ignore[attr-defined]

        upper_thresh = (inp > ir_vector).view_as(grad_inp)
        lower_thresh = (inp < -ir_vector).view_as(grad_inp)
        ir_grad = zeros_like(input_range)
        ir_grad += clamp(grad_inp[upper_thresh], min=None, max=0.0).sum()
        ir_grad -= clamp(grad_inp[lower_thresh], min=0.0, max=None).sum()
        ir_grad *= input_range
        did_clamp_mask = upper_thresh | lower_thresh
        if decay > 0:
            # - We shrink the input range if less than X% of the inputs are clipping.
            # where X is 1-ir_params.input_min_percentage
            percentage = 1.0 - (did_clamp_mask).sum().div(upper_thresh.numel())
            ir_grad += decay * input_range * (percentage > input_min_percentage)

        grad_inp[did_clamp_mask] = 0.0
        return grad_inp, grad_w, ir_grad, None, None, None, None, None
