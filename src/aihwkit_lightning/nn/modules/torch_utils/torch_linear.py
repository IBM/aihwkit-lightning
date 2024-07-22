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

"""Functions for normal linear in PyTorch."""
import os
from typing import List, Tuple, Union
from math import sqrt
from torch import Tensor, randn_like, clamp, zeros_like
from torch.autograd import no_grad, Function
from torch.autograd.function import FunctionCtx
from aihwkit_lightning.simulator.configs import TorchInferenceRPUConfig, WeightModifierType
from aihwkit_lightning.exceptions import ConfigError


class UniformQuantize(Function):
    """Quantization function."""

    # pylint: disable=abstract-method, redefined-builtin, arguments-differ

    @staticmethod
    def forward(ctx: FunctionCtx, inp: Tensor, res: float, inplace: bool) -> Tensor:
        """Quantizes the input tensor and performs straight-through estimation.

        Args:
            ctx (FunctionCtx): Context.
            inp (torch.Tensor): Input to be discretized.
            res (float): Resolution (number of states).
            inplace (bool): Clone the input?

        Returns:
            torch.Tensor: Quantized input.
        """
        # - Compute 1 / states if the number of states are provided
        if isinstance(res, Tensor):
            if res.ndim > 1:
                # pylint: disable=line-too-long
                err_string = f"res is tensor but first dim {res.size(0)} mismatches with first dim of input: {inp.size(0)}"  # noqa: E501
                assert res.size(0) == inp.size(0), err_string
            res = 2 / res if (res > 1.0).all() else 2 * res
            assert (res > 0).all(), "resolution is <= 0"
        else:
            res = 2 / res if res > 1.0 else 2 * res
            assert res > 0, "resolution is <= 0"
        output = inp if inplace else inp.clone()
        output = output / res
        # - Perform explicit rounding
        skip_rounding = os.environ.get("_AIHWKIT_NO_ROUNDING", False)
        if not skip_rounding:
            output = output.round()
        else:
            is_testing = os.environ.get("AIHWKIT_TESTING", False)
            if not is_testing:
                del os.environ["_AIHWKIT_NO_ROUNDING"]
                assert is_testing, "_AIHWKIT_NO_ROUNDING was set but we are not in testing mode."
        # - Scale back down
        output *= res
        return output

    @staticmethod
    # type: ignore[override]
    def backward(ctx: FunctionCtx, grad_output: Tensor) -> Tuple[Tensor, None, None, None]:
        """Straight-through estimator.

        Args:
            ctx: Context.
            grad_output: Gradient w.r.t. the inputs.

        Returns:
            Gradients w.r.t. inputs to forward.
        """
        # - Straight-through estimator
        grad_input = grad_output
        return grad_input, None, None, None


class StraightThroughClamp(Function):
    """
    Straight-through clamp function.
    """

    # pylint: disable=abstract-method, redefined-builtin, arguments-differ

    @staticmethod
    def forward(
        ctx: FunctionCtx,
        x_input: Tensor,
        upper_thresh: Tensor,
        lower_thresh: Tensor,
        input_range: Tensor,
    ) -> Tensor:
        clamped_mask = upper_thresh | lower_thresh
        exactly_the_same = x_input.abs() == input_range
        ctx.save_for_backward(clamped_mask & ~exactly_the_same)
        return clamp(x_input, min=-input_range, max=input_range)

    @staticmethod
    # type: ignore[override]
    def backward(ctx: FunctionCtx, d_output: Tensor) -> Tuple[Tensor, None, None, None, None]:
        (clamped_mask,) = ctx.saved_tensors  # type: ignore[attr-defined]
        d_output[clamped_mask] = 0.0
        return d_output, None, None, None, None


class InputRangeForward(Function):
    """
    Enable custom input range gradient computation using torch's autograd.
    """

    # pylint: disable=abstract-method, redefined-builtin, arguments-differ

    @staticmethod
    def forward(
        ctx: FunctionCtx,
        x_input: Tensor,
        input_range: Tensor,
        upper_thresh: Tensor,
        lower_thresh: Tensor,
        decay: float,
        input_min_percentage: float,
    ) -> Tensor:
        ctx.save_for_backward(upper_thresh, lower_thresh, input_range)
        ctx.decay = decay  # type: ignore[attr-defined]
        ctx.input_min_percentage = input_min_percentage  # type: ignore[attr-defined]
        return x_input

    @staticmethod
    # type: ignore[override]
    def backward(
        ctx: FunctionCtx, d_output: Tensor
    ) -> Tuple[Tensor, Tensor, None, None, None, None]:

        # # DEBUG
        # import pydevd

        # pydevd.settrace(suspend=False, trace_only_current_thread=True)

        upper_thresh: Tensor
        lower_thresh: Tensor
        input_range: Tensor

        upper_thresh, lower_thresh, input_range = ctx.saved_tensors  # type: ignore[attr-defined]
        ir_grad = None

        if input_range is not None:
            decay = ctx.decay  # type: ignore[attr-defined]
            input_min_percentage = ctx.input_min_percentage  # type: ignore[attr-defined]

            # upper_thres = x_input >= input_range  # pylint: disable=invalid-unary-operand-type
            # lower_thres = x_input <= -input_range  # pylint: disable=invalid-unary-operand-type
            ir_grad = zeros_like(input_range)
            ir_grad += clamp(d_output[upper_thresh], min=None, max=0.0).sum()
            ir_grad -= clamp(d_output[lower_thresh], min=0.0, max=None).sum()
            ir_grad *= input_range
            if decay > 0:
                # - We shrink the input range if less than X% of the inputs are clipping.
                # where X is 1-ir_params.input_min_percentage
                percentage = 1.0 - (upper_thresh | lower_thresh).sum().div(upper_thresh.numel())
                ir_grad += decay * input_range * (percentage > input_min_percentage)

        return d_output, ir_grad, None, None, None, None


class TorchLinear:
    """Linear layer with RPU support."""

    # pylint: disable=too-many-arguments, too-many-locals, too-many-branches, too-many-statements
    @staticmethod
    def linear(
        inp: Tensor,
        weights: Tensor,
        bias: Union[None, Tensor],
        input_range: Union[None, Tensor],
        input_range_update_idx: Union[None, Tensor],
        x_min: Union[None, Tensor],
        x_max: Union[None, Tensor],
        in_sizes: List[int],
        training: bool,
        rpu_config: TorchInferenceRPUConfig,
        apply_weight_modifier: bool,
    ):
        """Forward function for the linear layer. Performs x @ W^T + b."""
        current_upper = 0
        out = 0.0
        for slice_idx, inp_size in enumerate(in_sizes):
            inp_slice = inp[..., current_upper : current_upper + inp_size]  # noqa: E203

            if rpu_config.pre_post.input_range.enable:
                assert input_range is not None, "Input range must be provided"
                assert (
                    input_range_update_idx is not None
                ), "Input range update index must be provided"

                if rpu_config.pre_post.input_range.fast_mode:
                    assert x_min is not None, "x_min must be provided"
                    assert x_max is not None, "x_max must be provided"
                    inp_slice = TorchLinear.apply_input_range_fast(
                        values=inp_slice,
                        slice_idx=slice_idx,
                        rpu_config=rpu_config,
                        input_range=input_range,
                        input_range_update_idx=input_range_update_idx,
                        x_min=x_min,
                        x_max=x_max,
                        update_from_data=training,
                    )
                    inp_slice = inp_slice / input_range[slice_idx]
                else:
                    inp_slice, upper_thresh, lower_thresh = TorchLinear.apply_input_range(
                        values=inp_slice,
                        slice_idx=slice_idx,
                        rpu_config=rpu_config,
                        input_range=input_range,
                        input_range_update_idx=input_range_update_idx,
                        update_from_data=training,
                    )

                    inp_slice = InputRangeForward.apply(
                        inp_slice,
                        input_range[slice_idx],
                        upper_thresh,
                        lower_thresh,
                        rpu_config.pre_post.input_range.decay,
                        rpu_config.pre_post.input_range.input_min_percentage,
                    )

                    inp_slice = inp_slice / input_range[slice_idx]

            # maybe do some quantization
            if rpu_config.forward.inp_res > 0:
                inp_slice = UniformQuantize.apply(inp_slice, rpu_config.forward.inp_res, False)

            # do we meed assumed_wmax per-column or per-tensor? or not at all?
            need_assumed_wmax = False
            need_assumed_wmax_per_channel = False
            reduce_assumed_wmax_for_weight_modifier = False
            if training and rpu_config.forward.out_noise > 0:
                need_assumed_wmax = True
                if rpu_config.forward.out_noise_per_channel:
                    need_assumed_wmax_per_channel = True
            if apply_weight_modifier and rpu_config.modifier.type in [
                WeightModifierType.DISCRETIZE,
                WeightModifierType.DISCRETIZE_ADD_NORMAL,
                WeightModifierType.ADD_NORMAL,
            ]:
                need_assumed_wmax = True
                reduce_assumed_wmax_for_weight_modifier = True
            elif apply_weight_modifier and rpu_config.modifier.type in [
                WeightModifierType.ADD_NORMAL_PER_CHANNEL,
                WeightModifierType.DISCRETIZE_PER_CHANNEL,
                WeightModifierType.DISCRETIZE_ADD_NORMAL_PER_CHANNEL,
            ]:
                need_assumed_wmax = True
                need_assumed_wmax_per_channel = True

            if need_assumed_wmax:
                if need_assumed_wmax_per_channel:
                    assumed_wmax = (
                        weights[:, current_upper : current_upper + inp_size]
                        .abs()
                        .amax(dim=1, keepdim=True)
                    )
                else:
                    assumed_wmax = weights[:, current_upper : current_upper + inp_size].abs().max()
            else:
                assumed_wmax = None

            if apply_weight_modifier and assumed_wmax is not None:
                modified_slice = TorchLinear.modify_weight(
                    weights[:, current_upper : current_upper + inp_size],
                    assumed_wmax=(
                        assumed_wmax.max()
                        if reduce_assumed_wmax_for_weight_modifier
                        else assumed_wmax
                    ),
                    rpu_config=rpu_config,
                )
            else:
                modified_slice = weights[:, current_upper : current_upper + inp_size]

            out_slice = inp_slice @ modified_slice.T

            if training and rpu_config.forward.out_noise > 0:
                assert assumed_wmax is not None, "Assumed wmax must be provided for out noise"
                with no_grad():
                    # note that assumed_wmax has the correct shape here
                    if out_slice.ndim == 1:
                        assumed_wmax = assumed_wmax.view(-1)
                    wmax = (
                        assumed_wmax
                        if assumed_wmax.numel() == 1 or out_slice.ndim == 1
                        else assumed_wmax.view(-1, out_slice.size(-1))
                    )
                    out_noise = (
                        wmax
                        * rpu_config.forward.out_noise
                        / sqrt(len(in_sizes))
                        * randn_like(out_slice)
                    )
                out_slice += out_noise

            if rpu_config.pre_post.input_range.enable:
                assert input_range is not None, "Input range must be provided"
                out_slice *= input_range[slice_idx]

            out += out_slice  # type: ignore[assignment]

            current_upper += inp_size

        out = out.to(dtype=weights.dtype)  # type: ignore[attr-defined]
        return out + bias if bias is not None else out

    @staticmethod
    def apply_input_range(
        values: Tensor,
        slice_idx: int,
        rpu_config: TorchInferenceRPUConfig,
        input_range: Tensor,
        input_range_update_idx: Tensor,
        update_from_data: bool = False,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Apply the input clipping.

        Args:
            values: tensor to clip
            slice_idx: index of the input slice
            rpu_config: RPU configuration
            input_range: input range tensor
            input_range_update_idx: input range update index tensor
            update_from_data: whether to update from data if applicable

        Returns:
            clipped output tensor
        """
        if update_from_data:
            ir_params = rpu_config.pre_post.input_range
            idx = input_range_update_idx[slice_idx]
            if idx < ir_params.init_from_data:
                std = values.std()
                if std > 0.0:
                    input_range.data[slice_idx] = (
                        input_range.data[slice_idx] * idx + ir_params.init_std_alpha * std
                    ) / (idx + 1)
                    input_range_update_idx[slice_idx] += 1
                input_range.data[slice_idx] = input_range.data[slice_idx].abs()

        input_range = input_range[slice_idx]
        upper_thresh = values >= input_range
        lower_thresh = values <= -input_range
        x_clamped = StraightThroughClamp.apply(values, upper_thresh, lower_thresh, input_range)
        return x_clamped, upper_thresh, lower_thresh

    @staticmethod
    def apply_input_range_fast(
        values: Tensor,
        slice_idx: int,
        rpu_config: TorchInferenceRPUConfig,
        input_range: Tensor,
        input_range_update_idx: Tensor,
        x_min: Tensor,
        x_max: Tensor,
        update_from_data: bool = False,
    ) -> Tensor:
        """Fast updating and clipping without saving additional data for the BW.

        Args:
            values: tensor to clip
            slice_idx: index of the input slice
            rpu_config: RPUConfig used
            input_range: Input ranges per slice
            input_range_update_idx: How many times each input range got updated already
            x_min: Minimum value observed so far.
            x_max: Maximum value observed so far.
            update_from_data: whether to update from data if applicable

        Returns:
            clipped output tensor
        """
        with no_grad():
            if update_from_data:
                ir_params = rpu_config.pre_post.input_range
                input_range_update_idx: Tensor  # type: ignore[no-redef]
                idx = input_range_update_idx[slice_idx]
                if idx < ir_params.init_from_data:
                    act_x_min = values.min()
                    act_x_max = values.max()
                    # initialization
                    if x_min[slice_idx].min() > -1.1e-5 and x_max[slice_idx].max() < 1.1e-5:
                        x_min[slice_idx] = x_min[slice_idx] + act_x_min
                        x_max[slice_idx] = x_max[slice_idx] + act_x_max
                    elif ir_params.act_range_momentum == -1:
                        x_min[slice_idx] = min(
                            x_min[slice_idx], act_x_min
                        )  # type: ignore[call-overload]
                        x_max[slice_idx] = max(
                            x_max[slice_idx], act_x_max
                        )  # type: ignore[call-overload]
                    else:
                        x_min[slice_idx] = x_min[
                            slice_idx
                        ] * ir_params.act_range_momentum + act_x_min * (
                            1 - ir_params.act_range_momentum
                        )
                        x_max[slice_idx] = x_max[
                            slice_idx
                        ] * ir_params.act_range_momentum + act_x_max * (
                            1 - ir_params.act_range_momentum
                        )

                    input_range_update_idx[slice_idx] += 1
                    input_range.data[slice_idx] = max(  # type: ignore[call-overload]
                        x_min[slice_idx].abs(), x_max[slice_idx].abs()
                    )

        x_clamped = clamp(values, min=-input_range[slice_idx], max=input_range[slice_idx])
        return x_clamped

    @staticmethod
    def modify_weight(
        inp_weight: Tensor, assumed_wmax: Union[Tensor, None], rpu_config: TorchInferenceRPUConfig
    ) -> Tensor:
        """Modifies weights in-place, so .clone() before passing the weights here.

        Args:
            inp_weight: Input weights.
            assumed_wmax: Assumed maximum weight value.
            rpu_config: RPUConfig used.

        Raises:
            ConfigError: Unsupported/unknown weight modifier type.

        Returns:
            Weights with noise injected.
        """
        modifier = rpu_config.modifier
        if modifier.type == WeightModifierType.NONE:
            return inp_weight

        assert assumed_wmax is not None, "Assumed wmax must be provided for weight modifier"
        if modifier.type in [
            WeightModifierType.DISCRETIZE,
            WeightModifierType.DISCRETIZE_PER_CHANNEL,
            WeightModifierType.DISCRETIZE_ADD_NORMAL,
            WeightModifierType.DISCRETIZE_ADD_NORMAL_PER_CHANNEL,
        ]:
            res = modifier.res
            n_states = max(res, 1 / res)
            # assumed_wamax.item() would result in fp16 imprecision
            res = assumed_wmax / n_states  # type: ignore[assignment]

        if modifier.type in [
            WeightModifierType.DISCRETIZE,
            WeightModifierType.DISCRETIZE_PER_CHANNEL,
        ]:
            # - Discretize the weights on the fly and backprob through them
            inp_weight = UniformQuantize.apply(inp_weight, res, True)
        elif modifier.type in [
            WeightModifierType.ADD_NORMAL,
            WeightModifierType.ADD_NORMAL_PER_CHANNEL,
        ]:
            with no_grad():
                noise = modifier.std_dev * assumed_wmax * randn_like(inp_weight)
            inp_weight = inp_weight + noise
        elif modifier.type in [
            WeightModifierType.DISCRETIZE_ADD_NORMAL,
            WeightModifierType.DISCRETIZE_ADD_NORMAL_PER_CHANNEL,
        ]:
            inp_weight = UniformQuantize.apply(inp_weight, res, True)
            with no_grad():
                noise = modifier.std_dev * assumed_wmax * randn_like(inp_weight)
            inp_weight = inp_weight + noise
        else:
            raise ConfigError(f"Weight modifier {modifier} not supported")
        return inp_weight
