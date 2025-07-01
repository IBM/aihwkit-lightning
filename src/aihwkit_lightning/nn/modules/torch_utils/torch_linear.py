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

"""Functions for normal linear in PyTorch."""
from typing import List, Tuple, Union
from math import sqrt
from torch import Tensor, randn_like, clamp, zeros_like
from torch.autograd import no_grad, Function
from torch.autograd.function import FunctionCtx
from aihwkit_lightning.simulator.configs import (
    TorchInferenceRPUConfig,
    WeightNoiseInjectionType,
    WeightQuantizationType,
    WeightClipType,
)
from aihwkit_lightning.exceptions import ConfigError
from .quant_utils import UniformQuantize, clip_and_quantize


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

        # doesn't this blow up memory?
        d_output_zeroed_out = d_output.clone()
        d_output_zeroed_out[clamped_mask] = 0.0

        # # what works, but not when I use torch.compile
        # d_output[clamped_mask] = 0.0

        return d_output_zeroed_out, None, None, None, None


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
                # We shrink the input range if less than X% of the inputs are clipping.
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
        input_range_delta: Union[None, Tensor],
        x_min: Union[None, Tensor],
        x_max: Union[None, Tensor],
        learnable_weight_clip: Union[None, Tensor],
        in_sizes: List[int],
        training: bool,
        rpu_config: TorchInferenceRPUConfig,
        apply_out_quantization: bool,
    ):
        """Forward function for the linear layer. Performs x @ W^T + b."""
        current_upper = 0
        out = 0.0
        for slice_idx, inp_size in enumerate(in_sizes):
            inp_slice = inp[..., current_upper : current_upper + inp_size]  # noqa: E203

            if rpu_config.pre_post.input_range.enable:
                is_dynamic = rpu_config.pre_post.input_range.dynamic
                if rpu_config.pre_post.input_range.fast_mode:
                    assert x_min is not None, "x_min must be provided"
                    assert x_max is not None, "x_max must be provided"
                    assert input_range is not None, "Input range must be provided"
                    assert (
                        input_range_update_idx is not None
                    ), "Input range update index must be provided"
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
                    slice_input_range = input_range[slice_idx]
                elif is_dynamic:
                    slice_input_range = inp_slice.abs().max(dim=-1, keepdim=True)[0]
                else:
                    assert input_range is not None, "Input range must be provided"
                    assert (
                        input_range_update_idx is not None
                    ), "Input range update index must be provided"

                    # slice_input_range might get changed in diff'able manner here
                    # originally from input_range which is the learnable parameter
                    inp_slice, upper_thresh, lower_thresh, slice_input_range = (
                        TorchLinear.apply_input_range(
                            values=inp_slice,
                            slice_idx=slice_idx,
                            rpu_config=rpu_config,
                            input_range=input_range,
                            input_range_update_idx=input_range_update_idx,
                            update_from_data=training,
                        )
                    )

                    if input_range_delta is not None:
                        input_range_delta[slice_idx] = input_range[slice_idx] - slice_input_range

                    inp_slice = InputRangeForward.apply(
                        inp_slice,
                        slice_input_range,
                        upper_thresh,
                        lower_thresh,
                        rpu_config.pre_post.input_range.decay,
                        rpu_config.pre_post.input_range.input_min_percentage,
                    )

            # maybe do some quantization
            if rpu_config.forward.inp_res > 0:
                assert rpu_config.pre_post.input_range.enable, "Input range must be enabled."
                assert slice_input_range is not None, "Input range must be provided"
                inp_slice = UniformQuantize.apply(
                    inp_slice,
                    rpu_config.forward.inp_res,
                    slice_input_range,
                    True,
                    False,
                    None,
                    False,
                )

            # do we meed assumed_wmax per-column or per-tensor? or not at all?
            need_assumed_wmax = False
            if rpu_config.clip.type == WeightClipType.LEARNABLE_PER_CHANNEL:
                need_assumed_wmax = True

            if apply_out_quantization or rpu_config.forward.out_bound > 0:
                need_assumed_wmax = True

            if training and rpu_config.forward.out_noise > 0:
                need_assumed_wmax = True

            if training and rpu_config.modifier.noise_type in [
                WeightNoiseInjectionType.ADD_NORMAL,
                WeightNoiseInjectionType.ADD_NORMAL_PER_CHANNEL,
            ]:
                need_assumed_wmax = True

            if rpu_config.modifier.quantization_type in [
                WeightQuantizationType.DISCRETIZE,
                WeightQuantizationType.DISCRETIZE_PER_CHANNEL,
            ]:
                need_assumed_wmax = True

            modified_slice = weights[:, current_upper : current_upper + inp_size]

            if need_assumed_wmax:
                assumed_wmax = modified_slice.abs().amax(dim=-1, keepdim=True)
            else:
                assumed_wmax = None

            modified_slice = clip_and_quantize(
                inp_weight=modified_slice,
                assumed_wmax=assumed_wmax,
                learnable_weight_clip=(
                    None
                    if learnable_weight_clip is None
                    else learnable_weight_clip[slice_idx].unsqueeze(-1)
                ),
                rpu_config=rpu_config,
            )
            if rpu_config.clip.type == WeightClipType.LEARNABLE_PER_CHANNEL:
                assert rpu_config.modifier.quantization_type in [
                    WeightQuantizationType.NONE,
                    WeightQuantizationType.DISCRETIZE_PER_CHANNEL,
                ], "You can't learn weight ranges per column but quantize per tensor"
                # we just clipped. the assumed_wmax is therefore tighter.
                assert assumed_wmax is not None and isinstance(
                    assumed_wmax, Tensor
                ), "assumed_wmax here must be tensor"
                assert learnable_weight_clip is not None and isinstance(
                    learnable_weight_clip, Tensor
                ), "learnable_weight_clip here must be tensor"
                assumed_wmax = learnable_weight_clip[slice_idx].view_as(assumed_wmax)

            if training:
                modified_slice = TorchLinear.modify_weight(
                    modified_slice, assumed_wmax=assumed_wmax, rpu_config=rpu_config
                )

            out_slice = inp_slice @ modified_slice.T

            if training and rpu_config.forward.out_noise > 0:
                assert assumed_wmax is not None, "Assumed wmax must be provided for out noise"
                assert rpu_config.pre_post.input_range.enable, "Input range must be enabled"
                assert input_range is not None, "Input range must be provided"
                with no_grad():
                    # note that assumed_wmax has the correct shape here
                    if out_slice.ndim == 1:
                        assumed_wmax = assumed_wmax.view(-1)

                    if rpu_config.forward.out_noise_per_channel:
                        maybe_reduced_assumed_wmax = assumed_wmax
                    else:
                        maybe_reduced_assumed_wmax = assumed_wmax.max()

                    maybe_reduced_assumed_wmax = (
                        maybe_reduced_assumed_wmax
                        if maybe_reduced_assumed_wmax.numel() == 1 or out_slice.ndim == 1
                        else maybe_reduced_assumed_wmax.view(-1, out_slice.size(-1))
                    )
                    out_noise = (
                        maybe_reduced_assumed_wmax
                        * (
                            rpu_config.forward.out_noise
                            / sqrt(len(in_sizes))
                            * input_range[slice_idx]
                        )
                        * randn_like(out_slice)
                    )
                out_slice += out_noise

            if rpu_config.forward.out_bound > 0 or apply_out_quantization:
                assert (
                    slice_input_range is not None
                ), "Input range must be provided when using an out_bound or out quantization"
                assert assumed_wmax is not None, "Assumed wmax must be provided for out bound"
                if rpu_config.clip.type in [
                    WeightClipType.LAYER_GAUSSIAN_PER_CHANNEL,
                    WeightClipType.LEARNABLE_PER_CHANNEL,
                ]:
                    maybe_reduced_assumed_wmax = assumed_wmax
                else:
                    maybe_reduced_assumed_wmax = assumed_wmax.max()

                with no_grad():
                    bound = (
                        slice_input_range * rpu_config.forward.out_bound
                    ) * maybe_reduced_assumed_wmax.view(
                        1, -1
                    )  # type: ignore[union-attr]
                if apply_out_quantization:
                    out_slice = UniformQuantize.apply(
                        out_slice, rpu_config.forward.out_res, bound, True, False, None, False
                    )
                if rpu_config.forward.out_bound > 0:
                    out_slice = clamp(out_slice, min=-bound, max=bound)

            out += out_slice  # type: ignore[assignment]
            current_upper += inp_size

        return out + bias if bias is not None else out

    @staticmethod
    def apply_input_range(
        values: Tensor,
        slice_idx: int,
        rpu_config: TorchInferenceRPUConfig,
        input_range: Tensor,
        input_range_update_idx: Tensor,
        update_from_data: bool = False,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
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
            idx = input_range_update_idx[slice_idx].clone()
            if idx < ir_params.init_from_data:
                std = values.std()
                if std > 0.0:
                    # purposefully cut the gradient flow
                    # to input_range[slice_idx] because
                    # it's meaningless in this phase
                    with no_grad():
                        input_range_hat = (
                            (
                                (
                                    input_range[slice_idx].float() * idx
                                    + ir_params.init_std_alpha * std.float()
                                )
                                / (idx + 1)
                            ).abs()
                        ).to(dtype=input_range.dtype)
                        input_range_update_idx[slice_idx] += 1
                else:
                    with no_grad():
                        input_range_hat = input_range[slice_idx].clone()
            else:
                input_range_hat = input_range[slice_idx].clone()
        else:
            input_range_hat = input_range[slice_idx].clone()

        upper_thresh = values >= input_range_hat
        lower_thresh = values <= -input_range_hat
        x_clamped = StraightThroughClamp.apply(values, upper_thresh, lower_thresh, input_range_hat)
        return x_clamped, upper_thresh, lower_thresh, input_range_hat

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

        if modifier.noise_type == WeightNoiseInjectionType.NONE:
            return inp_weight

        if assumed_wmax is None:
            assumed_wmax = inp_weight.abs().amax(dim=1, keepdim=True)

        modified_weight = inp_weight.clone()
        if modifier.noise_type == WeightNoiseInjectionType.ADD_NORMAL:
            with no_grad():
                noise = modifier.std_dev * assumed_wmax.max() * randn_like(inp_weight)
        elif modifier.noise_type == WeightNoiseInjectionType.ADD_NORMAL_PER_CHANNEL:
            with no_grad():
                noise = modifier.std_dev * assumed_wmax * randn_like(inp_weight)
        else:
            raise ConfigError(f"Weight modifier {modifier} not supported")

        modified_weight = modified_weight + noise

        return modified_weight
