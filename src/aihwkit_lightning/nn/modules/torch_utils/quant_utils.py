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
from torch import Tensor, finfo
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
    ) -> Tensor:
        """Quantizes the input tensor and performs straight-through estimation.

        Args:
            ctx (FunctionCtx): Context.
            inp (torch.Tensor): Input to be discretized.
            res (Tensor, float, int): Resolution (number of states).
            bound (Tensor, float, int): Output bound.
            inplace (bool): Clone the input?

        Returns:
            torch.Tensor: Quantized input.
        """
        # - Compute 1 / states if the number of states are provided
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
    def backward(ctx: FunctionCtx, grad_output: Tensor) -> Tuple[Tensor, None, None, None, None]:
        """Straight-through estimator.

        Args:
            ctx: Context.
            grad_output: Gradient w.r.t. the inputs.

        Returns:
            Gradients w.r.t. inputs to forward.
        """
        # - Straight-through estimator
        grad_input = grad_output
        return grad_input, None, None, None, None
