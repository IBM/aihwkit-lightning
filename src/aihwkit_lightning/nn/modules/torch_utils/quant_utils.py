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
