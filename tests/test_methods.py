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

"""Test methods that got merged into AIHWKIT-Lightning."""

# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Tuple

from torch import tensor, where, sum, randn, float16, float32
from torch import dtype as torch_dtype
from torch import device as torch_device
from torch.autograd import Function
from torch.nn import Linear

from aihwkit_lightning.simulator.configs import TorchInferenceRPUConfig
from aihwkit_lightning.simulator.configs import WeightClipType


class LsqBinaryTernaryExtension(Function):
    """
    Modified from Learned Step-size Quantization.
    https://arxiv.org/abs/1902.08153
    """

    @staticmethod
    def forward(ctx, input, alpha, num_bits, layerwise):
        """
        :param input: input to be quantized
        :param alpha: the step size
        :param num_bits: quantization bits
        :param layerwise: rowwise quant
        :return: quantized output
        """
        ctx.num_bits = num_bits
        if num_bits >= 16:
            return input

        # we consider case where negative regime
        # has equal num. states as positive
        Qn = -(2 ** (num_bits - 1) - 1)
        Qp = 2 ** (num_bits - 1) - 1

        eps = torch.tensor(0.00001, device=alpha.device).float()

        alpha = torch.where(alpha > eps, alpha, eps)

        grad_scale = (
            1.0 / math.sqrt(input.numel())
            if not Qp
            else 1.0 / math.sqrt(input.numel() * Qp)
        )
        ctx.save_for_backward(input, alpha)
        ctx.other = grad_scale, Qn, Qp, layerwise
        q_w = (input / alpha).round().clamp(Qn, Qp)
        w_q = q_w * alpha
        return w_q

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.num_bits >= 16:
            return grad_output, None, None, None

        input_, alpha = ctx.saved_tensors
        grad_scale, Qn, Qp, layerwise = ctx.other
        q_w = input_ / alpha
        indicate_small = (q_w < Qn).float()
        indicate_big = (q_w > Qp).float()
        indicate_middle = (
            1.0 - indicate_small - indicate_big
        )  # this is more cpu-friendly than torch.ones(input_.shape)
        if ctx.num_bits == 1:
            if layerwise:
                grad_alpha = (
                    ((input_.sign()) * grad_output * grad_scale).sum().unsqueeze(dim=0)
                )
            else:
                grad_alpha = (input_.sign()) * grad_output * grad_scale
                grad_alpha = torch.sum(grad_alpha, dim=-1, keepdim=True)
        else:
            if layerwise:
                grad_alpha = (
                    (
                        (
                            indicate_small * Qn
                            + indicate_big * Qp
                            + indicate_middle * (-q_w + q_w.round())
                        )
                        * grad_output
                        * grad_scale
                    )
                    .sum()
                    .unsqueeze(dim=0)
                )
            else:
                grad_alpha = (
                    (
                        indicate_small * Qn
                        + indicate_big * Qp
                        + indicate_middle * (-q_w + q_w.round())
                    )
                    * grad_output
                    * grad_scale
                )
                grad_alpha = torch.sum(grad_alpha, dim=-1, keepdim=True)

        grad_input = indicate_middle * grad_output
        return grad_input, grad_alpha, None, None


class ParetoQQuantizeLinear(Linear):
    def __init__(
        self,
        *kargs,
        symmetric=True,
        bias=False,
        w_bits=16,
        weight_layerwise=False,
    ):
        super(QuantizeLinear, self).__init__(*kargs, bias=False)
        self.w_bits = w_bits
        self.weight_layerwise = weight_layerwise
        # params for weight quant
        if self.w_bits < 16:
            self.weight_clip_val = nn.Parameter(tensor(self.weight.shape[0], 1))

    def forward(self, input_):
        # quantize weight
        assert len(self.weight.size()) == 2
        real_weights = self.weight

        if self.w_bits >= 16:
            weight = self.weight
        elif self.w_bits <= 4:
            weight = LsqBinaryTernaryExtension.apply(
                real_weights,
                self.weight_clip_val,
                self.w_bits,
                self.weight_layerwise,
            ).to(input_.dtype)
        else:
            raise NotImplementedError

        out = nn.functional.linear(input_, weight)
        if self.bias is not None:
            out += self.bias.view(1, -1).expand_as(out)

        return out


def test_pareto_q_correctness(
    inp_size: int,
    out_size: int,
    bsz: int,
    num_inp_dims: int,
    device: torch_device,
    dtype: torch_dtype,
    bias: bool = False,
    symmetric: bool = True,
    w_bits: int = 4,
    weight_layerwise: bool = False
):
    orig_linear = ParetoQQuantizeLinear(
        in_features=inp_size,
        out_features=out_size,
        bias=bias,
        symmetric=symmetric,
        w_bits=w_bits,
        weight_layerwise=weight_layerwise
    )

    if num_inp_dims == 1:
        inp = randn(inp_size, device=device, dtype=dtype)
    elif num_inp_dims == 2:
        inp = randn(bsz, inp_size, device=device, dtype=dtype)
    else:
        inp = randn(bsz, inp_size, inp_size, device=device, dtype=dtype)


def fixture_rpus(
    modifier_res: int
) -> TorchInferenceRPUConfig:
    """Fixture for initializing rpus globally for all tests that need them"""
    rpu_config = TorchInferenceRPUConfig()
    rpu_config.modifier.res = modifier_res
    rpu_config.clip.type = WeightClipType.ParetoQ
    return rpu_config


if __name__ == "__main__":
    rpu_config = fixture_rpus(
        modifier_res=2**4 - 2  # 4 bits
    )
    test_pareto_q_correctness(
        inp_size=10,
        out_size=20,
        bsz=1,
        num_inp_dims=1,
        device=torch_device("cpu"),
        dtype=float16
    )