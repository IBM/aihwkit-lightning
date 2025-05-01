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

import os
import math
from unittest import SkipTest
from pytest import mark

from torch import Tensor, tensor, allclose, where, randn, bfloat16, float16, float32, manual_seed
from torch.autograd.function import FunctionCtx
from torch import sum as torch_sum
from torch import dtype as torch_dtype
from torch import device as torch_device
from torch.autograd import Function
from torch.nn import Linear, Parameter
import torch.nn.functional as F
import torch.cuda as torch_cuda

from aihwkit_lightning.nn import AnalogLinear
from aihwkit_lightning.simulator.configs import TorchInferenceRPUConfig
from aihwkit_lightning.simulator.configs import WeightClipType, WeightModifierType


SKIP_CUDA_TESTS = os.getenv("SKIP_CUDA_TESTS") or not torch_cuda.is_available()


# Code adapted from
# https://github.com/facebookresearch/ParetoQ/blob/main/models/utils_quant.py
class LsqBinaryTernaryExtension(Function):
    """
    Modified from Learned Step-size Quantization.
    https://arxiv.org/abs/1902.08153
    """

    # pylint: disable=abstract-method, redefined-builtin, arguments-differ, unused-argument

    @staticmethod
    def forward(ctx: FunctionCtx, inp: Tensor, alpha: Tensor, num_bits: int):
        """Forward of learnable quant."""

        ctx.num_bits = num_bits
        if num_bits >= 16:
            return inp

        # we consider case where negative regime
        # has equal num. states as positive
        q_n = -(2 ** (num_bits - 1) - 1)
        q_p = 2 ** (num_bits - 1) - 1

        eps = tensor(0.00001, device=alpha.device).float()

        alpha = where(alpha > eps, alpha, eps)

        grad_scale = (
            1.0 / math.sqrt(inp.numel()) if not q_p else 1.0 / math.sqrt(inp.numel() * q_p)
        )
        ctx.save_for_backward(inp, alpha)
        ctx.other = grad_scale, q_n, q_p
        q_w = (inp / alpha).round().clamp(q_n, q_p)
        w_q = q_w * alpha
        return w_q

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.num_bits >= 16:
            return grad_output, None, None

        inp, alpha = ctx.saved_tensors
        grad_scale, q_n, q_p = ctx.other
        q_w = inp / alpha
        indicate_small = (q_w < q_n).float()
        indicate_big = (q_w > q_p).float()
        indicate_middle = (
            1.0 - indicate_small - indicate_big
        )  # this is more cpu-friendly than torch.ones(inp.shape)
        grad_alpha = (
            (indicate_small * q_n + indicate_big * q_p + indicate_middle * (-q_w + q_w.round()))
            * grad_output
            * grad_scale
        )
        grad_alpha = torch_sum(grad_alpha, dim=-1, keepdim=True)

        grad_input = indicate_middle * grad_output
        return grad_input, grad_alpha, None


class ParetoQQuantizeLinear(Linear):
    """
    Adapted from https://github.com/facebookresearch/ParetoQ/blob/main/models/utils_quant.py
    """

    def __init__(self, *args, w_bits=16, **kwargs):
        super().__init__(*args, **kwargs)
        self.w_bits = w_bits
        # params for weight quant
        if self.w_bits < 16:
            self.weight_clip_val = Parameter(
                self.weight.abs().amax(dim=-1, keepdim=True) / (2 ** (w_bits - 1) - 1)
            )

    def forward(self, inp):  # pylint: disable=arguments-renamed
        # quantize weight
        assert len(self.weight.size()) == 2
        real_weights = self.weight

        if self.w_bits >= 16:
            weight = self.weight
        weight = LsqBinaryTernaryExtension.apply(
            real_weights, self.weight_clip_val, self.w_bits
        ).to(inp.dtype)

        out = F.linear(inp, weight)
        if self.bias is not None:
            out += self.bias.view(1, -1).expand_as(out)

        return out


@mark.parametrize("n_bits", [4, 5, 6, 7, 8])
@mark.parametrize("inp_size", [10, 100])
@mark.parametrize("out_size", [20, 400])
@mark.parametrize("bsz,num_inp_dims", [(1, 1), (2, 2)])
@mark.parametrize("device", [torch_device("cpu"), torch_device("cuda")])
@mark.parametrize("dtype", [float32])
def test_pareto_q_correctness(
    n_bits: int,
    inp_size: int,
    out_size: int,
    bsz: int,
    num_inp_dims: int,
    device: torch_device,
    dtype: torch_dtype,
):
    """
    Test correctness against Meta implementation.
    """

    # pylint: disable=too-many-locals

    manual_seed(0)

    rpu_config = get_rpu_config(modifier_res=2**n_bits - 2)  # 4 bits

    if device == torch_device("cuda") and SKIP_CUDA_TESTS:
        raise SkipTest("CUDA tests are disabled/ can't be performed")

    orig_linear = ParetoQQuantizeLinear(
        in_features=inp_size,
        out_features=out_size,
        bias=False,
        w_bits=n_bits,
        weight_layerwise=False,
        device=device,
        dtype=dtype,
    )

    if num_inp_dims == 1:
        inp = randn(inp_size, device=device, dtype=dtype)
    elif num_inp_dims == 2:
        inp = randn(bsz, inp_size, device=device, dtype=dtype)
    else:
        inp = randn(bsz, inp_size, inp_size, device=device, dtype=dtype)

    orig_out = orig_linear(inp)
    orig_loss = orig_out.sum()
    orig_loss.backward()

    aihwkit_linear = AnalogLinear(
        in_features=inp_size,
        out_features=out_size,
        bias=False,
        device=device,
        dtype=dtype,
        rpu_config=rpu_config,
    )
    aihwkit_linear.set_weights(orig_linear.weight)
    aihwkit_out = aihwkit_linear(inp)
    aihwkit_loss = aihwkit_out.sum()
    aihwkit_loss.backward()

    atol = 1e-4 if dtype in [float16, bfloat16] else 1e-5
    assert allclose(orig_loss, aihwkit_loss, atol=atol), "loss does not match"
    aihwkit_grad = aihwkit_linear.learnable_weight_clip.grad.flatten()
    orig_grad = orig_linear.weight_clip_val.grad.flatten()
    assert allclose(aihwkit_grad, orig_grad, atol=atol), "Gradients don't match"


def get_rpu_config(modifier_res: int) -> TorchInferenceRPUConfig:
    """Fixture for initializing rpus globally for all tests that need them"""
    rpu_config = TorchInferenceRPUConfig()
    rpu_config.modifier.res = modifier_res
    rpu_config.modifier.type = WeightModifierType.DISCRETIZE_PER_CHANNEL
    rpu_config.clip.type = WeightClipType.LEARNABLE_PER_CHANNEL
    return rpu_config


if __name__ == "__main__":
    test_pareto_q_correctness(
        n_bits=4,
        inp_size=10,
        out_size=20,
        bsz=1,
        num_inp_dims=1,
        device=torch_device("cpu"),
        dtype=float32,
    )
