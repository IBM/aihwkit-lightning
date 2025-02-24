# -*- coding: utf-8 -*-

# (C) Copyright 2024 IBM. All Rights Reserved.
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
from typing import Tuple
import triton  # type: ignore
import triton.language as tl  # type: ignore
from torch.autograd import Function
from torch.autograd.function import FunctionCtx
from torch import Tensor, empty, float32, randint, zeros_like, clamp
from aihwkit_lightning.nn.modules.triton_utils.triton_abs_max import sliced_fast_abs_max
from aihwkit_lightning.nn.modules.triton_utils.triton_std import sliced_fast_std
from aihwkit_lightning.simulator.configs import TorchInferenceRPUConfig, WeightClipType


FLOAT32_TINY: tl.constexpr = 1.1754943508222875e-38


# fmt: off
@triton.autotune(
        # pylint: disable=line-too-long
    configs=[
        triton.Config({"BLOCK_SIZE_OUT": 256, "BLOCK_SIZE_HIDDEN": 64}, num_stages=3, num_warps=8),  # noqa: E501
        triton.Config({"BLOCK_SIZE_OUT": 256, "BLOCK_SIZE_HIDDEN": 32}, num_stages=4, num_warps=4),  # noqa: E501
        triton.Config({"BLOCK_SIZE_OUT": 128, "BLOCK_SIZE_HIDDEN": 32}, num_stages=4, num_warps=4),  # noqa: E501
        triton.Config({"BLOCK_SIZE_OUT": 64, "BLOCK_SIZE_HIDDEN": 32}, num_stages=4, num_warps=4),  # noqa: E501
        triton.Config({"BLOCK_SIZE_OUT": 128, "BLOCK_SIZE_HIDDEN": 32}, num_stages=4, num_warps=4),  # noqa: E501
        triton.Config({"BLOCK_SIZE_OUT": 32, "BLOCK_SIZE_HIDDEN": 32}, num_stages=4, num_warps=4),  # noqa: E501
        triton.Config({"BLOCK_SIZE_OUT": 32, "BLOCK_SIZE_HIDDEN": 32}, num_stages=5, num_warps=2),  # noqa: E501
        triton.Config({"BLOCK_SIZE_OUT": 64, "BLOCK_SIZE_HIDDEN": 32}, num_stages=5, num_warps=2),  # noqa: E501
    ],
    key=["hidden_size", "out_size"],
    restore_value=["weights_ptr"]
)
@triton.jit
def modifier_kernel(  # pylint: disable=too-many-arguments
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
    modifier_seed,  # int
    modifier_std: tl.constexpr,  # float
    # block sizes
    BLOCK_SIZE_HIDDEN: tl.constexpr,  # pylint: disable=invalid-name
    BLOCK_SIZE_OUT: tl.constexpr,  # pylint: disable=invalid-name
):
    """
    Modifier kernel for the weights.
    """
    # pylint: disable=too-many-locals

    pid = tl.program_id(axis=0)
    offs_bn = (pid * BLOCK_SIZE_OUT + tl.arange(0, BLOCK_SIZE_OUT)) % out_size
    offs_assumed_wmax = pid * BLOCK_SIZE_OUT + tl.arange(0, BLOCK_SIZE_OUT)

    # for random number generation of output
    increase_weight_offsets_by = BLOCK_SIZE_HIDDEN * BLOCK_SIZE_OUT
    # pylint: disable=assignment-from-no-return, unexpected-keyword-arg
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
        # pylint: disable=consider-using-in
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
                ir_range_upper, ir_range_lower + (k + 1) * BLOCK_SIZE_HIDDEN, hidden_size
            )
            offs_k = current_lower + tl.arange(0, BLOCK_SIZE_HIDDEN)
            b_ptrs = weights_ptr + (
                offs_k[:, None] * stride_weights_hidden_size
                + offs_bn[None, :] * stride_weights_out_size
            )
            weight_block = tl.load(b_ptrs, mask=offs_k[:, None] < current_upper, other=0.0)

            # pylint: disable=consider-using-in
            if (modifier_type == "Discretize" or modifier_type == "DiscretizeAddNormal") or (
                modifier_type == "DiscretizePerChannel"
                or modifier_type == "DiscretizeAddNormalPerChannel"
            ):
                if modifier_weight_res > 0:
                    n_states = max(modifier_weight_res, 1 / modifier_weight_res)
                    res = 2 * assumed_wmax_per_slice / n_states
                    weight_block = weight_block / res
                    weight_block = tl.extra.cuda.libdevice.rint(weight_block)
                    weight_block = weight_block * res

            # pylint: disable=consider-using-in
            if (modifier_type == "AddNormal" or modifier_type == "AddNormalPerChannel") or (
                modifier_type == "DiscretizeAddNormal"
                or modifier_type == "DiscretizeAddNormalPerChannel"
            ):
                randn_block = tl.randn(modifier_seed + pid, weight_random_offsets)
                weight_random_offsets += increase_weight_offsets_by
                randn_block = assumed_wmax_per_slice * modifier_std * randn_block
                weight_block += randn_block

            # store weight_block back to DRAM...
            tl.store(
                b_ptrs,
                weight_block,
                mask=(offs_k[:, None] < current_upper) & (offs_assumed_wmax[None, :] < out_size),
            )
            current_lower = current_upper
        ir_range_lower = ir_range_upper


@triton.autotune(
    # pylint: disable=line-too-long
    configs=[
        triton.Config({"BLOCK_SIZE_INP": 128, "BLOCK_SIZE_OUT": 256, "BLOCK_SIZE_HIDDEN": 64, "GROUP_SIZE_INP": 8}, num_stages=3, num_warps=8),  # noqa: E501
        triton.Config({"BLOCK_SIZE_INP": 64, "BLOCK_SIZE_OUT": 256, "BLOCK_SIZE_HIDDEN": 32, "GROUP_SIZE_INP": 8}, num_stages=4, num_warps=4),  # noqa: E501
        triton.Config({"BLOCK_SIZE_INP": 128, "BLOCK_SIZE_OUT": 128, "BLOCK_SIZE_HIDDEN": 32, "GROUP_SIZE_INP": 8}, num_stages=4, num_warps=4),  # noqa: E501
        triton.Config({"BLOCK_SIZE_INP": 128, "BLOCK_SIZE_OUT": 64, "BLOCK_SIZE_HIDDEN": 32, "GROUP_SIZE_INP": 8}, num_stages=4, num_warps=4),  # noqa: E501
        triton.Config({"BLOCK_SIZE_INP": 64, "BLOCK_SIZE_OUT": 128, "BLOCK_SIZE_HIDDEN": 32, "GROUP_SIZE_INP": 8}, num_stages=4, num_warps=4),  # noqa: E501
        triton.Config({"BLOCK_SIZE_INP": 128, "BLOCK_SIZE_OUT": 32, "BLOCK_SIZE_HIDDEN": 32, "GROUP_SIZE_INP": 8}, num_stages=4, num_warps=4),  # noqa: E501
        triton.Config({"BLOCK_SIZE_INP": 64, "BLOCK_SIZE_OUT": 32, "BLOCK_SIZE_HIDDEN": 32, "GROUP_SIZE_INP": 8}, num_stages=5, num_warps=2),  # noqa: E501
        triton.Config({"BLOCK_SIZE_INP": 32, "BLOCK_SIZE_OUT": 64, "BLOCK_SIZE_HIDDEN": 32, "GROUP_SIZE_INP": 8}, num_stages=5, num_warps=2),  # noqa: E501
    ],
    key=["inp_size", "hidden_size", "out_size"],
)
@triton.jit
def matmul_kernel(
    # pointers to tensors
    inp_ptr,  # 2D [inp_size, hidden_size]
    weights_ptr,  # 2D [hidden_size, out_size]
    out_ptr,  # 2D [inp_size, out_size]
    ir_vector_ptr,  # 2D [inp_size, hidden_size]
    assumed_wmax_ptr,  # 2D [num_slices, out_size]
    reduced_assumed_wmax_ptr,  # 2D [num_slices, 1]
    input_range_ptr,  # 2D [num_slices, inp_size]
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
    stride_input_range_slice,
    stride_ir_vector_inp,
    stride_ir_vector_hidden,
    # miscellaneous
    out_noise_seed,
    inp_res: tl.constexpr,
    is_fp: tl.constexpr,
    out_res: tl.constexpr,
    out_quant: tl.constexpr,
    out_bound: tl.constexpr,
    bound_per_channel: tl.constexpr,
    out_noise: tl.constexpr,
    out_noise_std: tl.constexpr,
    out_noise_per_channel: tl.constexpr,
    ir_vector_is_none: tl.constexpr,
    dtype: tl.constexpr,
    precision: tl.constexpr,
    # block sizes
    BLOCK_SIZE_INP: tl.constexpr,  # pylint: disable=invalid-name
    BLOCK_SIZE_HIDDEN: tl.constexpr,  # pylint: disable=invalid-name
    BLOCK_SIZE_OUT: tl.constexpr,  # pylint: disable=invalid-name
    GROUP_SIZE_INP: tl.constexpr,  # pylint: disable=invalid-name
):
    """
    Computes the block-wise matmul.
    Applies input range to the input and quantizes it. Converts
    back to the original range before accumulating the dot products.
    Can handle different input ranges per slice in the input dimension.
    Stores the MVM result inp_ptr @ weights_ptr in out_ptr.
    """

    # pylint: disable=too-many-arguments, too-many-locals, too-many-statements, too-many-branches

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(inp_size, BLOCK_SIZE_INP)
    num_pid_n = tl.cdiv(out_size, BLOCK_SIZE_OUT)
    num_pid_in_group = GROUP_SIZE_INP * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_INP
    GROUP_SIZE_INP = min(num_pid_m - first_pid_m, GROUP_SIZE_INP)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % GROUP_SIZE_INP)
    pid_n = (pid % num_pid_in_group) // GROUP_SIZE_INP

    accumulator = tl.zeros((BLOCK_SIZE_INP, BLOCK_SIZE_OUT), dtype=tl.float32)
    # for every slice, this gets reset to zero. at the end of every slice,
    # this gets added to the accumulator
    per_slice_accumulator = tl.zeros((BLOCK_SIZE_INP, BLOCK_SIZE_OUT), dtype=tl.float32)

    # for random number generation of output
    increase_out_offsets_by = BLOCK_SIZE_INP * BLOCK_SIZE_OUT
    # pylint: disable=assignment-from-no-return, unexpected-keyword-arg
    output_random_offsets = tl.arange(0, BLOCK_SIZE_INP * BLOCK_SIZE_OUT).reshape(
        (BLOCK_SIZE_INP, BLOCK_SIZE_OUT), can_reorder=True
    )

    # Generate the pointers
    offs_am = pid_m * BLOCK_SIZE_INP + tl.arange(0, BLOCK_SIZE_INP)
    offs_bn = pid_n * BLOCK_SIZE_OUT + tl.arange(0, BLOCK_SIZE_OUT)

    offs_am = tl.max_contiguous(tl.multiple_of(offs_am, BLOCK_SIZE_INP), BLOCK_SIZE_INP)
    offs_bn = tl.max_contiguous(tl.multiple_of(offs_bn, BLOCK_SIZE_OUT), BLOCK_SIZE_OUT)

    offs_assumed_wmax = pid_n * BLOCK_SIZE_OUT + tl.arange(0, BLOCK_SIZE_OUT)

    ir_range_lower = 0
    for slice_idx in range(0, num_slices):
        if slice_idx > 0:
            # don't reset in first round
            # need to reset the per-slice acc. to zero
            per_slice_accumulator = tl.zeros((BLOCK_SIZE_INP, BLOCK_SIZE_OUT), dtype=tl.float32)

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

        if bound_per_channel and not (out_noise and out_noise_per_channel):
            bound_scale = tl.load(
                abs_max_slice_ptrs, mask=offs_assumed_wmax < out_size, other=float("-inf")
            )
            bound_scale = bound_scale[None, :]
        else:
            bound_scale = assumed_wmax_per_slice

        # this gives a speedup for large input dimension
        if num_slices == 1:
            ir_range_upper = hidden_size
        else:
            ir_range_upper = tl.load(upper_end_of_slices_ptr + slice_idx)

        current_lower = ir_range_lower

        # load the correct input range
        input_range_ptrs = input_range_ptr + slice_idx * stride_input_range_slice + offs_am[:, None]
        input_range = tl.load(input_range_ptrs, mask=offs_am[:, None] < inp_size, other=1.0)

        offs_k = current_lower + tl.arange(0, BLOCK_SIZE_HIDDEN)
        a_ptrs = inp_ptr + (
            offs_am[:, None] * stride_inp_inp_size + offs_k[None, :] * stride_inp_hidden_size
        )
        b_ptrs = weights_ptr + (
            offs_k[:, None] * stride_weights_hidden_size
            + offs_bn[None, :] * stride_weights_out_size
        )

        num_k = tl.cdiv(ir_range_upper - ir_range_lower, BLOCK_SIZE_HIDDEN)
        for k in range(0, num_k):
            current_upper = min(
                ir_range_upper, ir_range_lower + (k + 1) * BLOCK_SIZE_HIDDEN, hidden_size
            )

            inp_block = tl.load(
                a_ptrs,
                mask=(offs_am[:, None] < inp_size) & (offs_k[None, :] < current_upper),
                other=0.0,
            )
            weight_block = tl.load(
                b_ptrs,
                mask=(offs_k[:, None] < current_upper) & (offs_bn[None, :] < out_size),
                other=0.0,
            )

            input_range = input_range.to(dtype)

            if not ir_vector_is_none:
                # save the correct IR per dimension in the hidden
                ir_vector_ptrs = (
                    ir_vector_ptr
                    + offs_am[:, None] * stride_ir_vector_inp
                    + offs_k[None, :] * stride_ir_vector_hidden
                )
                tl.store(
                    ir_vector_ptrs,
                    input_range,
                    mask=(offs_am[:, None] < inp_size) & (offs_k[None, :] < current_upper)
                )

            # clip between the input ranges
            over_ir_mask = inp_block > input_range
            under_ir_mask = inp_block < -input_range

            inp_block = tl.where(over_ir_mask, input_range, inp_block)
            inp_block = tl.where(under_ir_mask, -input_range, inp_block)

            # scale with input ranges
            inp_block = inp_block / input_range  # -> float32

            if not is_fp:
                inp_block = inp_block / inp_res
                inp_block = tl.extra.cuda.libdevice.rint(inp_block)
                # here inp_block is float32
                inp_block = inp_block * inp_res

            inp_block = inp_block.to(dtype)

            # do the MVM
            dot_prod = tl.dot(inp_block, weight_block, input_precision=precision)

            # scale back with the input range
            dot_prod = input_range.to(tl.float32) * dot_prod
            per_slice_accumulator = per_slice_accumulator + dot_prod

            # increment the pointers
            offs_k += current_upper - current_lower
            a_ptrs += (current_upper - current_lower) * stride_inp_hidden_size
            b_ptrs += (current_upper - current_lower) * stride_weights_hidden_size

            current_lower = current_upper

        # here, the MVM for the slice was completed. We can apply the
        # out_noise and ADC
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
            per_slice_accumulator += randn_block
            # advance the output_random_offsets
            output_random_offsets += increase_out_offsets_by

        if out_quant or out_bound > 0:
            # compute the bound
            # we just scale with abs-max of weight
            bound = bound_scale * out_bound * input_range.to(tl.float32)
            if out_quant:
                alpha = (bound.to(tl.float32) * out_res)
                per_slice_accumulator = per_slice_accumulator / tl.where(
                    alpha == 0.0, FLOAT32_TINY,
                    alpha
                )
                per_slice_accumulator = tl.extra.cuda.libdevice.rint(per_slice_accumulator)
                per_slice_accumulator = per_slice_accumulator * alpha

            if out_bound > 0:
                # clip between the input ranges
                over_out_bound_mask = per_slice_accumulator > bound
                under_out_bound_mask = per_slice_accumulator < -bound
                per_slice_accumulator = tl.where(
                    over_out_bound_mask,
                    bound,
                    per_slice_accumulator
                )
                per_slice_accumulator = tl.where(
                    under_out_bound_mask,
                    -bound,
                    per_slice_accumulator
                )

        accumulator = accumulator + per_slice_accumulator

        ir_range_lower = ir_range_upper

    out_block = accumulator.to(dtype)

    offs_cm = pid_m * BLOCK_SIZE_INP + tl.arange(0, BLOCK_SIZE_INP)
    offs_cn = pid_n * BLOCK_SIZE_OUT + tl.arange(0, BLOCK_SIZE_OUT)
    c_ptrs = (
        out_ptr + stride_out_inp_size * offs_cm[:, None] + stride_out_out_size * offs_cn[None, :]
    )
    c_mask = (offs_cm[:, None] < inp_size) & (offs_cn[None, :] < out_size)
    tl.store(c_ptrs, out_block, mask=c_mask)
# fmt: on


# pylint: disable=abstract-method, arguments-differ
class TritonLinear(Function):
    """autograd.Function for triton-based linear layer."""

    @staticmethod
    # type: ignore[override]
    def forward(  # pylint: disable=too-many-locals, too-many-statements, too-many-branches
        ctx: FunctionCtx,
        inp: Tensor,
        weights: Tensor,
        input_range: Tensor | None,
        input_range_update_idx: Tensor | None,
        upper_end_of_slices: Tensor,
        rpu_config: TorchInferenceRPUConfig,
        training: bool,
        apply_weight_modifier: bool,
    ) -> Tensor:
        """
        Forward of the triton-based linear layer

        Args:
            ctx: Context.
            inp: Input matrix
            weights: Weight matrix. inp @ weights.T is being performed
            input_range: Tensor of input range(s)
            input_range_update_idx: How often did every input range
                get updated by data already?
            upper_end_of_slices: [128, 256] if a 256-sized layer is
                split by 128-sized chunks
            rpu_config: The configuration for HW-aware training
            training: Are we in training or eval mode
            apply_weight_modifier: Do we need to call the weight modifier kernel or not

        Returns:
            Gradients w.r.t. inputs, weights and input ranges.
        """
        ir_dynamic = rpu_config.pre_post.input_range.dynamic
        assert ir_dynamic == (input_range is None), "Received input range when dynamic IR"
        assert ir_dynamic == (input_range_update_idx is None), "Received IR counter when dynamic IR"

        if input_range is not None:
            assert input_range.is_contiguous(), "input_range not contiguous"

        assert weights.is_contiguous(), "weights not contiguous"
        assert inp.is_contiguous(), "inp not contiguous"

        assumed_wmax = sliced_fast_abs_max(weights=weights, upper_end_of_slices=upper_end_of_slices)
        assert assumed_wmax.is_contiguous(), "assumed_wmax not contiguous"

        reduced_assumed_wmax = assumed_wmax.amax(dim=1, keepdim=True)
        assert reduced_assumed_wmax.is_contiguous(), "reduced_assumed_wmax is not contiguous"

        out_shape = (*inp.shape[:-1], weights.size(0))
        if inp.ndim == 1:
            inp = inp.view(1, -1)
        inp = inp.flatten(end_dim=-2)
        num_slices = len(upper_end_of_slices)

        # update the input ranges if necessary
        if training and input_range_update_idx is not None and input_range is not None:
            ir_params = rpu_config.pre_post.input_range
            if input_range_update_idx[0] < ir_params.init_from_data:
                # Avoiding the for loop yields a speed-up.
                stds = sliced_fast_std(inp, upper_end_of_slices)
                # if (stds > 0.0).all():
                #     # stds = naive_per_slice_std(inp, upper_end_of_slices)
                #     input_range.data = (
                #                 input_range * input_range_update_idx
                #                 + ir_params.init_std_alpha * stds
                #             ) / (input_range_update_idx + 1)
                #     input_range_update_idx += 1
                for slice_idx in range(num_slices):
                    idx = input_range_update_idx[slice_idx]
                    if idx < ir_params.init_from_data:
                        std = stds[slice_idx]
                        if std > 0.0:
                            input_range.data[slice_idx] = (
                                input_range.data[slice_idx] * idx + ir_params.init_std_alpha * std
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
                # 32,  # for debugging
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

        # do the same for the out_res
        out_bound = rpu_config.forward.out_bound
        out_res = rpu_config.forward.out_res
        if out_res > 0:
            out_res = 2.0 / out_res if out_res > 1.0 else 2.0 * out_res
            assert out_res > 0, "resolution is <= 0"
        # do we need wmax per column?
        bound_per_channel = (
            out_bound > 0
        ) and rpu_config.clip.type == WeightClipType.LAYER_GAUSSIAN_PER_CHANNEL

        # bring output noise variables into shape
        out_noise = training and (not rpu_config.forward.out_noise == 0.0)
        out_noise_std = rpu_config.forward.out_noise
        out_noise_per_channel = rpu_config.forward.out_noise_per_channel
        out_noise_seed = randint(2**31, (1,)).item()

        # create an input range vector that has shape [-1, hidden_size]
        ir_vector = empty((inp.size(0), hidden_size), device=inp.device, dtype=inp.dtype)
        ir_vector = ir_vector.contiguous()

        # preprocess the input range here to have shape [num_slices, -1]
        # i.e. for every token, we have num_slices many IRs
        if input_range is None:
            expanded_input_range = sliced_fast_abs_max(inp, upper_end_of_slices=upper_end_of_slices)
        else:
            expanded_input_range = input_range.unsqueeze(1).repeat(num_slices, inp.size(0))

        # for some tests, we skip rounding
        skip_rounding = os.environ.get("_AIHWKIT_NO_ROUNDING", False)
        skip_rounding = skip_rounding == "1"

        dtype = tl.float32 if inp.dtype == float32 else tl.float16

        # disable tf32 when testing is on
        precision = None
        if os.environ.get("AIHWKIT_TESTING", None) == "1":
            precision = "ieee"

        matmul_kernel[grid](
            inp,  # 2D [inp_size, hidden_size]
            weights.T,  # 2D [hidden_size, out_size]
            out,  # 2D [inp_size, out_size]
            ir_vector,  # 2D [inp_size, hidden_size]
            assumed_wmax,  # 2D [num_slices, out_size]
            reduced_assumed_wmax,  # 2D [num_slices, 1]
            expanded_input_range,  # 2D [num_slices, inp_size]
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
            expanded_input_range.stride(0),
            ir_vector.stride(0),
            ir_vector.stride(1),
            # for the other ptrs, we assume stride 1
            # miscellaneous
            out_noise_seed,
            inp_res,  # inp_res
            inp_res <= 0 or skip_rounding,  # is_fp
            out_res,  # out_res
            not skip_rounding and out_res > 0,  # out_quant?
            out_bound,
            bound_per_channel,
            out_noise,
            out_noise_std,
            out_noise_per_channel,
            False,  # ir_vector is None
            dtype,
            precision,
            # block sizes
            # 64,  # This is for debugging
            # 32,
            # 128,
            # 8,
        )

        # save some stuff for backwards
        ctx.rpu_config = rpu_config  # type: ignore[attr-defined]
        ctx.save_for_backward(
            inp, weights, None if ir_dynamic else input_range, ir_vector  # type: ignore[arg-type]
        )

        out = out.view(out_shape)
        return out

    @staticmethod
    # type: ignore[override]
    # pylint: disable=too-many-locals
    def backward(
        ctx: FunctionCtx, grad_output: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor | None, None, None, None, None, None]:  # type: ignore
        """Straight-through estimator for linear layer

        Args:
            ctx: Context.
            grad_output: Backward flowing gradient w.r.t. outputs.

        Returns:
            Gradients w.r.t. inputs, weights and input ranges.
        """
        # # DEBUG
        # import pydevd
        # pydevd.settrace(suspend=False, trace_only_current_thread=True)

        rpu_config: TorchInferenceRPUConfig
        rpu_config = ctx.rpu_config  # type: ignore[attr-defined]
        inp, weights, input_range, ir_vector = ctx.saved_tensors  # type: ignore[attr-defined]

        weights: Tensor  # type: ignore[no-redef]
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
        inp: Tensor  # type: ignore[no-redef]
        ir_vector: Tensor  # type: ignore[no-redef]
        inp_rounded = inp.clamp(-ir_vector, ir_vector)
        if inp_res > 0 and not skip_rounding:
            scale = ir_vector * inp_res
            inp_rounded = (inp_rounded / scale).round() * scale

        grad_w = grad_output.T @ inp_rounded
        grad_inp = grad_output @ weights
        grad_inp = grad_inp.view(grad_inp_shape)

        if rpu_config.pre_post.input_range.dynamic:
            ir_grad = None
        else:
            # ir gradient
            decay = rpu_config.pre_post.input_range.decay  # type: ignore[attr-defined]
            # type: ignore[attr-defined]
            input_min_percentage = rpu_config.pre_post.input_range.input_min_percentage

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
