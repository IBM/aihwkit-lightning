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

"""Quantization utility functions."""
import os
from typing import Tuple, Union
from math import sqrt
from torch import Tensor, finfo, clamp
from torch import sum as torch_sum
from torch.autograd import Function
from torch.autograd.function import FunctionCtx
from aihwkit_lightning.simulator.configs import (
    TorchInferenceRPUConfig,
    WeightQuantizationType,
    WeightClipType,
)
from aihwkit_lightning.exceptions import ConfigError


class UniformQuantize(Function):
    """Quantization function."""

    # pylint: disable=abstract-method, redefined-builtin, arguments-differ, unused-argument

    @staticmethod
    def forward(
        ctx: FunctionCtx,
        inp: Tensor,
        res: Union[Tensor, float, int],
        bound: Union[float, Tensor],
        inplace: bool,
        learn_res: bool,
        max_quant_state: Union[None, float],
        skip_rounding: bool,
    ) -> Tensor:
        """Quantizes the input tensor and performs straight-through estimation.

        Args:
            ctx (FunctionCtx): Context.
            inp (torch.Tensor): Input to be discretized.
            res (Tensor, float, int): Resolution (number of states).
            bound (Tensor, float, int): Output bound.
            inplace (bool): Clone the input?
            learn_res (bool): Do we want to learn the ranges.
            max_quant_state (Union[None, float]): 2**(n-bits -1)-1
            skip_rounding (bool): Skip the rounding in quantization.

        Returns:
            torch.Tensor: Quantized input.
        """
        skip_rounding = skip_rounding or bool(os.environ.get("_AIHWKIT_NO_ROUNDING", False))
        ctx.learn_res = learn_res  # type: ignore[attr-defined]
        # Compute 1 / states if the number of states are provided
        alpha = 2 * bound
        if isinstance(res, Tensor) or isinstance(alpha, Tensor):
            if isinstance(res, (int, float)):
                res = alpha / res if res > 1.0 else alpha * res
            else:
                if res.ndim > 1:
                    # pylint: disable=line-too-long
                    err_string = f"res is tensor but first dim {res.size(0)} mismatches with first dim of input: {inp.size(0)}"  # noqa: E501
                    assert res.size(0) == inp.size(0), err_string
                res = alpha / res if (res > 1.0).all() else alpha * res
                res[res == 0.0] = 1.0  # avoid division by zero
                assert (res > 0).all(), "resolution is <= 0"
        else:
            res = alpha / res if res > 1.0 else alpha * res
            assert res > 0, "resolution is <= 0"
        output = inp if inplace else inp.clone()
        clamped_res = (
            max(finfo(inp.dtype).tiny, res)
            if isinstance(res, (float, int))
            else res.clamp_min(finfo(inp.dtype).tiny)
        )
        output = output / clamped_res  # avoid zero res entries
        if max_quant_state is not None:
            output = clamp(output, min=-max_quant_state, max=max_quant_state)
        # Perform explicit rounding
        if not skip_rounding:
            output = output.round()
        else:
            is_testing = os.environ.get("AIHWKIT_TESTING", False)
            if not is_testing:
                if os.environ.get("_AIHWKIT_NO_ROUNDING", False):
                    del os.environ["_AIHWKIT_NO_ROUNDING"]
                    assert (
                        is_testing
                    ), "_AIHWKIT_NO_ROUNDING was set but we are not in testing mode."
        # Scale back down
        output *= res
        if learn_res:
            assert max_quant_state is not None, "max_quant_state should be float"
            assert isinstance(clamped_res, Tensor), "clamped_res must be Tensor to be learnable"
            # grad_scale = 1.0 / sqrt(inp.numel() * max_quant_state)
            grad_scale = (
                1 / sqrt(inp.numel()) if skip_rounding else 2 * sqrt(max_quant_state / inp.numel())
            )
            ctx.save_for_backward(inp, clamped_res)
            ctx.other = grad_scale, max_quant_state, skip_rounding  # type: ignore[attr-defined]
        return output

    @staticmethod
    # type: ignore[override]
    def backward(
        ctx: FunctionCtx, grad_output: Tensor
    ) -> Tuple[Tensor, Union[None, Tensor], None, None, None, None, None]:
        """Straight-through estimator.

        Args:
            ctx: Context.
            grad_output: Gradient w.r.t. the inputs.

        Returns:
            Gradients w.r.t. inputs to forward.
        """
        if ctx.learn_res:  # type: ignore[attr-defined]
            inp, res = ctx.saved_tensors  # type: ignore[attr-defined]
            grad_scale, quant_max, skip_rounding = ctx.other  # type: ignore[attr-defined]
            q_w = inp / res
            indicate_small = (q_w < -quant_max).float()
            indicate_big = (q_w > quant_max).float()
            indicate_middle = (
                1.0 - indicate_small - indicate_big
            )  # this is more cpu-friendly than torch.ones(input_.shape)
            if skip_rounding:
                grad_res = (
                    (indicate_small * (-quant_max) + indicate_big * quant_max)
                    * grad_output
                    * grad_scale
                )
            else:
                grad_res = (
                    (
                        indicate_small * (-quant_max)
                        + indicate_big * quant_max
                        + indicate_middle * (-q_w + q_w.round())
                    )
                    * grad_output
                    * grad_scale
                )
            grad_res = torch_sum(grad_res, dim=-1, keepdim=True)
            grad_input = indicate_middle * grad_output
        else:
            grad_input = grad_output
            grad_res = None

        return grad_input, grad_res, None, None, None, None, None


def clip_and_quantize(
    inp_weight: Tensor,
    assumed_wmax: Union[Tensor, None],
    learnable_weight_clip: Union[Tensor, None],
    rpu_config: TorchInferenceRPUConfig,
) -> Tensor:
    """
    Applies learned weight clipping ranges. This is only done if the weight modifier
    is not one of the quantization weight modifiers.

    Args:
        inp_weight: Input weights.
        assumed_wmax: Assumed maximum weight value.
        learnable_weight_clip: The learnable per-column clipping values.
        rpu_config: RPUConfig used.

    Raises:
        ConfigError: Unsupported/unknown weight modifier type.

    Returns:
        Clamped weights if clamping was applicable (not in place).
    """
    modifier = rpu_config.modifier

    # 1, maybe clip
    learnable_weight_clipping = rpu_config.clip.type == WeightClipType.LEARNABLE_PER_CHANNEL
    if learnable_weight_clipping:
        assert learnable_weight_clip is not None, "learnable_weight_clip should not be None"
        assert learnable_weight_clip.shape == (
            inp_weight.size(0),
            1,
        ), f"learnable_weight_clip.shape must be ({inp_weight.size(0)}, 1)"
        if modifier.quantization_type == WeightQuantizationType.NONE:
            # we pass the learned learnable_weight_clip
            # we want to clip at -1 1 after "normalization"
            # we want to skip rounding in this case
            # we can do in place since we don't modify the weight
            inp_weight = UniformQuantize.apply(
                inp_weight, learnable_weight_clip, 0.5, False, learnable_weight_clipping, 1.0, True
            )

    # after we might have done clipping, return if we don't want to quantize
    if modifier.quantization_type == WeightQuantizationType.NONE:
        return inp_weight

    # 2, if we're here, we definitely quantize
    if assumed_wmax is None:
        assumed_wmax = inp_weight.abs().amax(dim=1, keepdim=True)

    if modifier.quantization_type in [
        WeightQuantizationType.DISCRETIZE,
        WeightQuantizationType.DISCRETIZE_PER_CHANNEL,
    ]:
        res = modifier.res
        n_states = max(res, 1 / res)
        max_quant_state = n_states / 2
        if learnable_weight_clipping:
            assert learnable_weight_clip is not None, "learnable_weight_clip should not be None"
            scaling_factors = learnable_weight_clip / n_states  # type: ignore[assignment]
        else:
            if modifier.quantization_type == WeightQuantizationType.DISCRETIZE:
                scaling_factors = assumed_wmax.max() / n_states
            else:
                scaling_factors = assumed_wmax / n_states  # type: ignore[assignment]

        # Discretize the weights on the fly and backprob through them
        inp_weight = UniformQuantize.apply(
            inp_weight,
            scaling_factors,
            1.0,
            False,
            learnable_weight_clipping,
            max_quant_state,
            False,
        )
    else:
        raise ConfigError(f"modifier.quantization_type {modifier.quantization_type} not supported")

    return inp_weight
