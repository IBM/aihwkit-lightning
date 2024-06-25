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

# pylint: disable=too-many-locals, too-many-public-methods, no-member
"""Test the forward/backward correctness of purely torch vs triton."""

import os
from typing import Union
from unittest import SkipTest
from pytest import mark

from torch import dtype as torch_dtype
from torch import device as torch_device
from torch import cuda as torch_cuda
from torch import (
    allclose,
    randn,
    randn_like,
    empty_like,
    float32,
    float16,
    bfloat16,
    Tensor,
    matmul,
    manual_seed,
    arange,
)
from torch.optim import AdamW
from aihwkit_lightning.nn import AnalogLinear
from aihwkit_lightning.simulator.configs import TorchInferenceRPUConfig as RPUConfig
from aihwkit_lightning.optim import AnalogOptimizer
from aihwkit_lightning.simulator.configs import WeightClipType, WeightModifierType


SKIP_CUDA_TESTS = os.getenv("SKIP_CUDA_TESTS") or not torch_cuda.is_available()


@mark.parametrize("bsz", [1, 10])
@mark.parametrize("num_inp_dims", [1, 2, 3])
@mark.parametrize("inp_size", [10, 32])
@mark.parametrize("out_size", [10, 32])
@mark.parametrize("bias", [True])
@mark.parametrize("inp_res", [2**8 - 2, 1 / (2**8 - 2)])
@mark.parametrize("max_inp_size", [20])
@mark.parametrize("ir_enable", [True, False])
@mark.parametrize("ir_learn_input_range", [True, False])
@mark.parametrize("ir_init_value", [3.0])
@mark.parametrize("ir_init_std_alpha", [2.0])
@mark.parametrize("out_noise", [True, False])
@mark.parametrize("out_noise_per_channel", [True, False])
@mark.parametrize(
    "weight_modifier",
    [
        WeightModifierType.DISCRETIZE,
        WeightModifierType.DISCRETIZE_PER_CHANNEL,
        WeightModifierType.NONE,
    ],
)
@mark.parametrize("weight_modifier_res", [2**8 - 2, 1 / (2**8 - 2)])
@mark.parametrize("device", ["cuda"])  # cpu not supported for triton
@mark.parametrize("dtype", [float32, float16, bfloat16])
def test_linear_forward(  # pylint: disable=too-many-arguments
    bsz: int,
    num_inp_dims: int,
    inp_size: int,
    out_size: int,
    bias: bool,
    inp_res: float,
    max_inp_size: int,
    ir_enable: bool,
    ir_learn_input_range: bool,
    ir_init_value: float,
    ir_init_std_alpha: float,
    out_noise: bool,
    out_noise_per_channel: bool,
    weight_modifier: WeightModifierType,
    weight_modifier_res: int,
    device: torch_device,
    dtype: torch_dtype,
):
    """Test the forward pass."""

    manual_seed(0)

    if device == "cuda" and SKIP_CUDA_TESTS:
        raise SkipTest("CUDA tests are disabled/ can't be performed")

    if not ir_enable and inp_res > 0:
        raise SkipTest("IR not enabled but inp_res > 0")

    if num_inp_dims == 1 and bsz > 1:
        raise SkipTest("1D input but bsz > 1")

    if dtype == bfloat16:
        raise SkipTest("Bfloat16 currently not supported for triton")

    if out_noise and weight_modifier != WeightModifierType.NONE:
        raise SkipTest("Output noise and non-None weight modifier skipped")

    def populate_rpu(rpu_config: RPUConfig):
        rpu_config.modifier.type = weight_modifier
        rpu_config.modifier.std_dev = 0.03
        rpu_config.modifier.res = weight_modifier_res
        rpu_config.forward.inp_res = inp_res
        rpu_config.forward.out_noise = 0.03 if out_noise else 0.0
        rpu_config.forward.out_noise_per_channel = out_noise_per_channel
        rpu_config.mapping.max_input_size = max_inp_size
        rpu_config.pre_post.input_range.enable = ir_enable
        rpu_config.pre_post.input_range.learn_input_range = ir_learn_input_range
        rpu_config.pre_post.input_range.init_value = ir_init_value
        rpu_config.pre_post.input_range.init_std_alpha = ir_init_std_alpha
        rpu_config.pre_post.input_range.init_from_data = 0  # we force init from data zero here
        return rpu_config

    rpu = populate_rpu(RPUConfig())
    linear = AnalogLinear(in_features=inp_size, out_features=out_size, bias=bias, rpu_config=rpu)
    # if out noise is to be tested, we don't want to be in eval mode
    if out_noise:
        assert (
            weight_modifier == WeightModifierType.NONE
        ), "Found out_noise and non-None weight modifier"
    elif weight_modifier != WeightModifierType.NONE:
        assert weight_modifier in [
            WeightModifierType.ADD_NORMAL,
            WeightModifierType.ADD_NORMAL_PER_CHANNEL,
            WeightModifierType.DISCRETIZE,
            WeightModifierType.DISCRETIZE_PER_CHANNEL,
            WeightModifierType.DISCRETIZE_ADD_NORMAL,
            WeightModifierType.DISCRETIZE_ADD_NORMAL_PER_CHANNEL,
        ], "Unkown weight modifier"
    else:
        linear = linear.eval()

    linear.input_range.data = 1.0 + arange(len(linear.in_sizes))
    linear = linear.to(device=device, dtype=dtype)

    if num_inp_dims == 1:
        inp = randn(inp_size, device=device, dtype=dtype)
    if num_inp_dims == 2:
        inp = randn(bsz, inp_size, device=device, dtype=dtype)
    else:
        inp = randn(bsz, inp_size, inp_size, device=device, dtype=dtype)

    # pure pytorch
    out = linear(inp)  # pylint: disable=not-callable

    # with triton
    os.environ["AIHWKIT_USE_TRITON"] = "1"
    out_triton = linear(inp)  # pylint: disable=not-callable
    del os.environ["AIHWKIT_USE_TRITON"]

    if out_noise:
        linear.rpu_config.forward.out_noise = 0.0
        out_noise_free = linear(inp)  # pylint: disable=not-callable
        delta_triton = out_triton - out_noise_free
        delta_torch = out - out_noise_free
        assert allclose(delta_torch.std(), delta_triton.std(), atol=1e-2)
    elif not weight_modifier in [
        WeightModifierType.NONE,
        WeightModifierType.DISCRETIZE,
        WeightModifierType.DISCRETIZE_PER_CHANNEL,
    ]:
        linear.rpu_config.modifier.std_dev = 0.0
        out_noise_free = linear(inp)  # pylint: disable=not-callable
        linear.rpu_config.modifier.std_dev = 0.03
        sum_mean_triton = 0
        sum_mean_torch = 0
        for _ in range(5):
            os.environ["AIHWKIT_USE_TRITON"] = "1"
            out_triton = linear(inp)  # pylint: disable=not-callable
            delta_triton = out_triton - out_noise_free
            del os.environ["AIHWKIT_USE_TRITON"]
            out = linear(inp)  # pylint: disable=not-callable
            delta_torch = out - out_noise_free
            mean_l2_triton = delta_triton.norm(dim=1).mean()
            mean_l2_torch = delta_torch.norm(dim=1).mean()
            sum_mean_torch += mean_l2_torch
            sum_mean_triton += mean_l2_triton
            print(f"Norm triton {mean_l2_triton:.4f} norm torch {mean_l2_torch:.4f}")
        overall_mean_triton = sum_mean_triton / 5
        overall_mean_torch = sum_mean_torch / 5
        assert allclose(overall_mean_torch, overall_mean_triton, atol=1e-2)
    else:
        atol = 1e-5
        if dtype == float16:
            atol = 1e-2  # accumulation is slightly different in triton vs torch
        assert allclose(out_triton, out, atol=atol)


@mark.parametrize("bsz", [1, 10])
@mark.parametrize("num_inp_dims", [1, 2, 3])
@mark.parametrize("inp_size", [10, 255, 513])
@mark.parametrize("out_size", [10, 255, 513])
@mark.parametrize("bias", [True, False])
@mark.parametrize("inp_res", [2**8 - 2, 1 / (2**8 - 2)])
@mark.parametrize("max_inp_size", [256, 512])
@mark.parametrize("ir_init_value", [2.0, 3.0])
@mark.parametrize("ir_init_from_data", [-1, 0, 10])
@mark.parametrize("ir_init_std_alpha", [2.0, 3.0])
@mark.parametrize("device", ["cpu", "cuda"])
@mark.parametrize("dtype", [float32])
def test_input_range_backward(  # pylint: disable=too-many-arguments
    bsz: int,
    num_inp_dims: int,
    inp_size: int,
    out_size: int,
    bias: bool,
    inp_res: float,
    max_inp_size: int,
    ir_init_value: float,
    ir_init_from_data: int,
    ir_init_std_alpha: float,
    device: str,
    dtype: torch_dtype,
):
    """Test the input range backward pass."""

    if device == "cuda" and SKIP_CUDA_TESTS:
        raise SkipTest("CUDA tests are disabled/ can't be performed")

    if num_inp_dims == 1 and bsz > 1:
        raise SkipTest("1D input but bsz > 1")

    manual_seed(0)

    def populate_rpu(rpu_config: RPUConfig):
        rpu_config.forward.inp_res = inp_res
        rpu_config.forward.out_noise = 0.0
        rpu_config.mapping.max_input_size = max_inp_size
        rpu_config.pre_post.input_range.enable = True
        rpu_config.pre_post.input_range.learn_input_range = True
        rpu_config.pre_post.input_range.init_value = ir_init_value
        rpu_config.pre_post.input_range.init_from_data = ir_init_from_data
        rpu_config.pre_post.input_range.init_std_alpha = ir_init_std_alpha
        return rpu_config

    rpu = populate_rpu(RPUConfig())
    linear = AnalogLinear(in_features=inp_size, out_features=out_size, bias=bias, rpu_config=rpu)
    linear = linear.to(device=device, dtype=dtype)

    linear_triton = AnalogLinear(
        in_features=inp_size, out_features=out_size, bias=bias, rpu_config=rpu
    )
    linear_triton.load_state_dict(linear.state_dict())
    linear_triton = linear_triton.to(device=device, dtype=dtype)

    if num_inp_dims == 1:
        inp = randn(inp_size, device=device, dtype=dtype, requires_grad=True)
    if num_inp_dims == 2:
        inp = randn(bsz, inp_size, device=device, dtype=dtype, requires_grad=True)
    else:
        inp = randn(bsz, inp_size, inp_size, device=device, dtype=dtype, requires_grad=True)
    inp_triton = empty_like(inp, requires_grad=True)
    inp_triton.data = inp.data.clone()

    out: Tensor
    out = linear(inp)  # pylint: disable=not-callable

    os.environ["AIHWKIT_USE_TRITON"] = "1"
    out_triton: Tensor
    out_triton = linear_triton(inp_triton)  # pylint: disable=not-callable
    del os.environ["AIHWKIT_USE_TRITON"]

    atol = 1e-5 if dtype == float32 else 1e-2
    assert allclose(out, out_triton, atol=atol)

    out.sum().backward()
    out_triton.sum().backward()

    # compare the weight gradient
    assert allclose(linear.weight.grad, linear_triton.weight.grad, atol=atol)

    # compare the inp gradient
    assert allclose(inp.grad, inp_triton.grad, atol=atol)

    # compare the input range
    # # This doesn't work for fundamental reasons: we don't have access to d L / d
    # slice-x since we sum up the slices in the triton kernel.
    # assert allclose(linear.input_range.grad, linear_triton.input_range.grad, atol=atol)


if __name__ == "__main__":
    test_input_range_backward(
        bsz=1,
        num_inp_dims=1,
        inp_size=10,
        out_size=5,
        bias=False,
        inp_res=254,
        max_inp_size=5,
        ir_init_value=1.0,
        ir_init_from_data=10,
        ir_init_std_alpha=3.0,
        device="cuda",
        dtype=float32,
    )