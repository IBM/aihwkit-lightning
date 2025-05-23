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

# pylint: disable=too-many-locals, too-many-public-methods, no-member
"""Test the forward/backward correctness of purely torch vs triton."""

import os
from typing import Tuple
from unittest import SkipTest
from pytest import mark

from torch import dtype as torch_dtype
from torch import device as torch_device
from torch import cuda as torch_cuda
from torch import (
    allclose,
    randn,
    empty_like,
    float32,
    float16,
    bfloat16,
    Tensor,
    manual_seed,
    arange,
)
from aihwkit_lightning.nn import AnalogLinear
from aihwkit_lightning.simulator.configs import TorchInferenceRPUConfig as RPUConfig
from aihwkit_lightning.simulator.configs import WeightModifierType, WeightClipType


SKIP_CUDA_TESTS = os.getenv("SKIP_CUDA_TESTS") or not torch_cuda.is_available()


@mark.parametrize("bsz", [1, 10])
@mark.parametrize("num_inp_dims", [1, 2, 3])
@mark.parametrize("inp_size", [10, 32])
@mark.parametrize("out_size", [10, 32])
@mark.parametrize("bias", [True])
@mark.parametrize("inp_res", [2**8 - 2, 1 / (2**8 - 2)])
@mark.parametrize("max_inp_size", [20])
@mark.parametrize("ir_enable", [True, False])
@mark.parametrize("ir_dynamic", [True, False])
@mark.parametrize("ir_learn_input_range", [True, False])
@mark.parametrize("ir_init_value", [3.0])
@mark.parametrize("ir_init_std_alpha", [2.0])
@mark.parametrize("adc_config", [(10, 2**8 - 2), (10, 1 / (2**8 - 2))])
@mark.parametrize("out_noise", [False])
@mark.parametrize("out_noise_per_channel", [False])
@mark.parametrize("weight_modifier", [WeightModifierType.NONE])
@mark.parametrize("weight_modifier_res", [2**8 - 2])
@mark.parametrize("clip_type", [WeightClipType.LAYER_GAUSSIAN_PER_CHANNEL, WeightClipType.NONE])
@mark.parametrize("device", ["cpu"] if SKIP_CUDA_TESTS else ["cuda"])
@mark.parametrize("dtype", [float32])
# pylint: disable=too-many-arguments, too-many-branches, too-many-statements
def test_linear_forward(
    bsz: int,
    num_inp_dims: int,
    inp_size: int,
    out_size: int,
    bias: bool,
    inp_res: float,
    max_inp_size: int,
    ir_enable: bool,
    ir_dynamic: bool,
    ir_learn_input_range: bool,
    ir_init_value: float,
    ir_init_std_alpha: float,
    adc_config: Tuple[float, float],
    out_noise: bool,
    out_noise_per_channel: bool,
    weight_modifier: WeightModifierType,
    weight_modifier_res: int,
    clip_type: WeightClipType,
    device: torch_device,
    dtype: torch_dtype,
):
    """Test the forward pass."""

    out_bound, out_res = adc_config

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

    if not ir_enable and not adc_config == (-1, -1):
        raise SkipTest("No ADC when no inp. quantization.")

    if out_noise and not adc_config == (-1, -1):
        # we ensure that the noise vector instance is the same for
        # triton and torch. As a result, the outputs might end up
        # in different quantization bins, causing stronger changes
        raise SkipTest("No out_noise when ADC used.")

    # turn off rounding as this can lead to large errors as a result
    # of tiny fp errors
    os.environ["_AIHWKIT_NO_ROUNDING"] = "1"
    os.environ["AIHWKIT_TESTING"] = "1"
    manual_seed(0)

    def populate_rpu(rpu_config: RPUConfig):
        rpu_config.modifier.type = weight_modifier
        rpu_config.modifier.std_dev = 0.03
        rpu_config.modifier.res = weight_modifier_res
        rpu_config.forward.inp_res = inp_res
        rpu_config.forward.out_noise = 0.03 if out_noise else 0.0
        rpu_config.forward.out_noise_per_channel = out_noise_per_channel
        rpu_config.forward.out_bound = out_bound
        rpu_config.forward.out_res = out_res
        rpu_config.mapping.max_input_size = max_inp_size
        rpu_config.pre_post.input_range.enable = ir_enable
        rpu_config.pre_post.input_range.dynamic = ir_dynamic
        rpu_config.pre_post.input_range.learn_input_range = ir_learn_input_range
        rpu_config.pre_post.input_range.init_value = ir_init_value
        rpu_config.pre_post.input_range.init_std_alpha = ir_init_std_alpha
        rpu_config.pre_post.input_range.init_from_data = 0  # we force init from data zero here
        rpu_config.clip.type = clip_type
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

    if ir_enable and not ir_dynamic:
        linear.input_range.data = 1.0 + arange(len(linear.in_sizes))
    linear = linear.to(device=device, dtype=dtype)

    if num_inp_dims == 1:
        inp = randn(inp_size, device=device, dtype=dtype)
    elif num_inp_dims == 2:
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
    elif weight_modifier not in [
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

    del os.environ["_AIHWKIT_NO_ROUNDING"]
    del os.environ["AIHWKIT_TESTING"]


@mark.parametrize("bsz", [1, 10])
@mark.parametrize("num_inp_dims", [1, 2, 3])
@mark.parametrize("inp_size", [10, 255, 513])
@mark.parametrize("out_size", [10, 255, 513])
@mark.parametrize("bias", [True, False])
@mark.parametrize("inp_res", [2**8 - 2, 1 / (2**8 - 2)])
@mark.parametrize("ir_dynamic", [True, False])
@mark.parametrize("max_inp_size", [256, 512])
@mark.parametrize("ir_init_value", [2.0, 3.0])
@mark.parametrize("ir_init_from_data", [0, 10])
@mark.parametrize("ir_init_std_alpha", [2.0, 3.0])
@mark.parametrize("device", ["cpu"] if SKIP_CUDA_TESTS else ["cuda"])
@mark.parametrize("dtype", [float32])
def test_input_range_backward(  # pylint: disable=too-many-arguments
    bsz: int,
    num_inp_dims: int,
    inp_size: int,
    out_size: int,
    bias: bool,
    inp_res: float,
    ir_dynamic: bool,
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

    if dtype == bfloat16:
        raise SkipTest("Bfloat16 currently not supported for triton")

    os.environ["_AIHWKIT_NO_ROUNDING"] = "1"
    os.environ["AIHWKIT_TESTING"] = "1"
    manual_seed(0)

    def populate_rpu(rpu_config: RPUConfig):
        rpu_config.forward.inp_res = inp_res
        rpu_config.forward.out_noise = 0.0
        rpu_config.mapping.max_input_size = max_inp_size
        rpu_config.pre_post.input_range.enable = True
        rpu_config.pre_post.input_range.dynamic = ir_dynamic
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
    elif num_inp_dims == 2:
        inp = randn(bsz, inp_size, device=device, dtype=dtype, requires_grad=True)
    else:
        inp = randn(bsz, inp_size, inp_size, device=device, dtype=dtype, requires_grad=True)
    inp_triton = empty_like(inp, requires_grad=True)
    inp_triton.data = inp.data.clone()

    os.environ["AIHWKIT_USE_TRITON"] = "1"
    out_triton: Tensor
    out_triton = linear_triton(inp_triton)  # pylint: disable=not-callable
    del os.environ["AIHWKIT_USE_TRITON"]

    out: Tensor
    out = linear(inp)  # pylint: disable=not-callable

    atol = 1e-5 if dtype == float32 else 1e-2
    assert allclose(out, out_triton, atol=atol)

    out.mean().backward()
    out_triton.mean().backward()

    # compare the weight gradient
    assert allclose(linear.weight.grad, linear_triton.weight.grad, atol=atol)

    # compare the inp gradient
    assert allclose(inp.grad, inp_triton.grad, atol=atol)

    del os.environ["_AIHWKIT_NO_ROUNDING"]
    del os.environ["AIHWKIT_TESTING"]


if __name__ == "__main__":
    # test_input_range_backward(
    #     bsz=10,
    #     num_inp_dims=2,
    #     inp_size=513,
    #     out_size=10,
    #     bias=False,
    #     inp_res=254,
    #     ir_dynamic=False,
    #     max_inp_size=512,
    #     ir_init_value=3.0,
    #     ir_init_from_data=10,
    #     ir_init_std_alpha=2.0,
    #     device="cpu",
    #     dtype=float16,
    # )
    test_linear_forward(
        bsz=10,
        num_inp_dims=2,
        inp_size=10,
        out_size=20,
        bias=True,
        inp_res=254,
        max_inp_size=20,
        ir_enable=True,
        ir_dynamic=True,
        ir_learn_input_range=True,
        ir_init_value=3.0,
        ir_init_std_alpha=2.0,
        adc_config=(10, 2**8 - 2),
        out_noise=False,
        out_noise_per_channel=False,
        weight_modifier=WeightModifierType.NONE,
        weight_modifier_res=254,
        clip_type=WeightClipType.LAYER_GAUSSIAN_PER_CHANNEL,
        device="cuda",
        dtype=float32,
    )
