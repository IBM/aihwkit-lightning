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
"""Test the forward/backward correctness of our CPU/CUDA version to AIHWKIT."""

import os
from typing import Union, List
from unittest import SkipTest
from pytest import mark

from torch import dtype as torch_dtype
from torch import device as torch_device
from torch import cuda as torch_cuda
from torch import (
    allclose,
    randn,
    randn_like,
    float32,
    float16,
    bfloat16,
    Tensor,
    matmul,
    manual_seed,
)
from torch.optim import AdamW
from torch.nn import Conv2d

from aihwkit.nn import AnalogLinear as AIHWKITAnalogLinear
from aihwkit.nn import AnalogConv2d as AIWHKITAnalogConv2d
from aihwkit.simulator.configs import TorchInferenceRPUConfig as AIHWKITRPUConfig
from aihwkit.simulator.configs import NoiseManagementType, BoundManagementType
from aihwkit.simulator.configs import WeightModifierType as AIHWKITWeightModifierType

from aihwkit_lightning.nn import AnalogLinear, AnalogConv2d
from aihwkit_lightning.simulator.configs import TorchInferenceRPUConfig as RPUConfig
from aihwkit_lightning.optim import AnalogOptimizer
from aihwkit_lightning.simulator.configs import WeightClipType, WeightModifierType


SKIP_CUDA_TESTS = os.getenv("SKIP_CUDA_TESTS") or not torch_cuda.is_available()


@mark.parametrize("bsz", [1, 10])
@mark.parametrize("num_inp_dims", [1, 2, 3])
@mark.parametrize("inp_size", [10, 255, 513])
@mark.parametrize("out_size", [10, 255, 513])
@mark.parametrize("bias", [True, False])
@mark.parametrize("inp_res", [2**8 - 2, 1 / (2**8 - 2)])
@mark.parametrize("max_inp_size", [256, 512])
@mark.parametrize("ir_enable", [True, False])
@mark.parametrize("ir_learn_input_range", [True, False])
@mark.parametrize("ir_init_value", [2.0, 3.0])
@mark.parametrize("ir_init_from_data", [-1, 0, 10])
@mark.parametrize("ir_init_std_alpha", [2.0, 3.0])
@mark.parametrize("device", ["cpu", "cuda"])
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
    ir_init_from_data: int,
    ir_init_std_alpha: float,
    device: torch_device,
    dtype: torch_dtype,
):
    """Test the forward pass."""

    if device == "cuda" and SKIP_CUDA_TESTS:
        raise SkipTest("CUDA tests are disabled/ can't be performed")

    if not ir_enable and inp_res > 0:
        raise SkipTest("IR not enabled but inp_res > 0")

    if num_inp_dims == 1 and bsz > 1:
        raise SkipTest("1D input but bsz > 1")

    def populate_rpu(rpu_config: Union[AIHWKITRPUConfig, RPUConfig]):
        # AIHWKIT-specific configurations
        if isinstance(rpu_config, AIHWKITRPUConfig):
            rpu_config.mapping.max_output_size = -1
            rpu_config.forward.noise_management = (
                NoiseManagementType.ABS_MAX if not ir_enable else NoiseManagementType.NONE
            )
            rpu_config.forward.bound_management = BoundManagementType.NONE

        rpu_config.forward.inp_res = inp_res
        rpu_config.forward.out_res = -1
        rpu_config.forward.out_bound = -1
        rpu_config.forward.out_noise = 0.0
        rpu_config.mapping.max_input_size = max_inp_size
        rpu_config.pre_post.input_range.enable = ir_enable
        rpu_config.pre_post.input_range.learn_input_range = ir_learn_input_range
        rpu_config.pre_post.input_range.init_value = ir_init_value
        rpu_config.pre_post.input_range.init_from_data = ir_init_from_data
        rpu_config.pre_post.input_range.init_std_alpha = ir_init_std_alpha
        return rpu_config

    aihwkit_rpu = populate_rpu(AIHWKITRPUConfig())
    rpu = populate_rpu(RPUConfig())

    aihwkit_linear = AIHWKITAnalogLinear(
        in_features=inp_size, out_features=out_size, bias=bias, rpu_config=aihwkit_rpu
    )
    linear = AnalogLinear(in_features=inp_size, out_features=out_size, bias=bias, rpu_config=rpu)

    aihwkit_linear = aihwkit_linear.eval()
    linear = linear.eval()

    aihwkit_linear = aihwkit_linear.to(device=device, dtype=dtype)

    aihwkit_weight, aihwkit_bias = aihwkit_linear.get_weights()
    linear.set_weights_and_biases(aihwkit_weight, aihwkit_bias)
    linear = linear.to(device=device, dtype=dtype)

    if num_inp_dims == 1:
        inp = randn(inp_size, device=device, dtype=dtype)
    if num_inp_dims == 2:
        inp = randn(bsz, inp_size, device=device, dtype=dtype)
    else:
        inp = randn(bsz, inp_size, inp_size, device=device, dtype=dtype)

    out_aihwkit = aihwkit_linear(inp)  # pylint: disable=not-callable
    out = linear(inp)  # pylint: disable=not-callable
    assert allclose(out_aihwkit, out, atol=1e-5)


@mark.parametrize("bsz", [1, 10])
@mark.parametrize("num_inp_dims", [1, 2])
@mark.parametrize("height", [10, 513])
@mark.parametrize("width", [10, 513])
@mark.parametrize("in_channels", [3, 10])
@mark.parametrize("out_channels", [3, 10])
@mark.parametrize("kernel_size", [[3, 3], [3, 4]])
@mark.parametrize("stride", [[1, 1]])
@mark.parametrize("padding", [[1, 1]])
@mark.parametrize("dilation", [[1, 1]])
@mark.parametrize("groups", [1])
@mark.parametrize("bias", [True])
@mark.parametrize("inp_res", [2**8 - 2])
@mark.parametrize("max_inp_size", [256, 512])
@mark.parametrize("ir_enable", [True, False])
@mark.parametrize("ir_learn_input_range", [True, False])
@mark.parametrize("ir_init_value", [3.0])
@mark.parametrize("ir_init_from_data", [10])
@mark.parametrize("ir_init_std_alpha", [3.0])
@mark.parametrize("device", ["cpu", "cuda"])
@mark.parametrize("dtype", [float32, float16, bfloat16])
def test_conv2d_forward(  # pylint: disable=too-many-arguments
    bsz: int,
    num_inp_dims: int,
    height: int,
    width: int,
    in_channels: int,
    out_channels: int,
    kernel_size: List[int],
    stride: List[int],
    padding: List[int],
    dilation: List[int],
    groups: int,
    bias: bool,
    inp_res: float,
    max_inp_size: int,
    ir_enable: bool,
    ir_learn_input_range: bool,
    ir_init_value: float,
    ir_init_from_data: int,
    ir_init_std_alpha: float,
    device: torch_device,
    dtype: torch_dtype,
):
    """Test the Con2D forward pass."""

    if device == "cuda" and SKIP_CUDA_TESTS:
        raise SkipTest("CUDA tests are disabled/ can't be performed")

    if not ir_enable and inp_res > 0:
        raise SkipTest("IR not enabled but inp_res > 0")

    if groups > 1:
        raise SkipTest("AIHWKIT currently does not support groups > 1")

    def populate_rpu(rpu_config: Union[AIHWKITRPUConfig, RPUConfig]):
        # AIHWKIT-specific configurations
        if isinstance(rpu_config, AIHWKITRPUConfig):
            rpu_config.mapping.max_output_size = -1
            rpu_config.forward.noise_management = (
                NoiseManagementType.ABS_MAX if not ir_enable else NoiseManagementType.NONE
            )
            rpu_config.forward.bound_management = BoundManagementType.NONE

        rpu_config.forward.inp_res = inp_res
        rpu_config.forward.out_res = -1
        rpu_config.forward.out_bound = -1
        rpu_config.forward.out_noise = 0.0
        rpu_config.mapping.max_input_size = max_inp_size
        rpu_config.pre_post.input_range.enable = ir_enable
        rpu_config.pre_post.input_range.learn_input_range = ir_learn_input_range
        rpu_config.pre_post.input_range.init_value = ir_init_value
        rpu_config.pre_post.input_range.init_from_data = ir_init_from_data
        rpu_config.pre_post.input_range.init_std_alpha = ir_init_std_alpha
        return rpu_config

    aihwkit_rpu = populate_rpu(AIHWKITRPUConfig())
    rpu = populate_rpu(RPUConfig())

    aihwkit_analog_conv2d = AIWHKITAnalogConv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias=bias,
        rpu_config=aihwkit_rpu,
    )

    analog_conv2d = AnalogConv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias=bias,
        rpu_config=rpu,
        device=device,
        dtype=dtype,
    )

    aihwkit_analog_conv2d = aihwkit_analog_conv2d.eval()
    analog_conv2d = analog_conv2d.eval()

    aihwkit_analog_conv2d = aihwkit_analog_conv2d.to(device=device, dtype=dtype)

    if num_inp_dims == 1:
        inp = randn(in_channels, height, width, device=device, dtype=dtype)
    else:
        assert num_inp_dims == 2, "Only batched or non-batched inputs are supported"
        inp = randn(bsz, in_channels, height, width, device=device, dtype=dtype)

    digital_aihwkit_conv2d = AIWHKITAnalogConv2d.to_digital(aihwkit_analog_conv2d)
    conv_weight = digital_aihwkit_conv2d.weight.to(device=device, dtype=dtype)
    conv_bias = digital_aihwkit_conv2d.bias
    conv_bias = conv_bias.to(device=device, dtype=dtype) if conv_bias is not None else None
    analog_conv2d.set_weights_and_biases(conv_weight, conv_bias)
    analog_conv2d = analog_conv2d.to(device=device, dtype=dtype)

    out_aihwkit = aihwkit_analog_conv2d(inp)  # pylint: disable=not-callable
    out = analog_conv2d(inp)  # pylint: disable=not-callable
    assert allclose(out_aihwkit, out, atol=1e-5)

    if inp_res == -1 and not ir_enable:
        conv2d = Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            device=device,
            dtype=dtype,
        )

        conv2d.weight.data = conv_weight.clone()
        conv2d.bias.data = conv_bias.clone()
        out_digital = conv2d(inp)
        assert allclose(out, out_digital, atol=1e-5)


@mark.parametrize("height", [10, 513])
@mark.parametrize("width", [10, 513])
@mark.parametrize("in_channels", [3, 10])
@mark.parametrize("out_channels", [3, 10])
@mark.parametrize("kernel_size", [[3, 3]])
@mark.parametrize("stride", [[1, 1]])
@mark.parametrize("padding", [[1, 1]])
@mark.parametrize("dilation", [[1, 1]])
@mark.parametrize("groups", [1])
@mark.parametrize("bias", [True])
@mark.parametrize("device", ["cpu", "cuda"])
@mark.parametrize("dtype", [float32, float16, bfloat16])
def test_conv2d_to_and_from_digital(  # pylint: disable=too-many-arguments
    height: int,
    width: int,
    in_channels: int,
    out_channels: int,
    kernel_size: List[int],
    stride: List[int],
    padding: List[int],
    dilation: List[int],
    groups: int,
    bias: bool,
    device: torch_device,
    dtype: torch_dtype,
):
    """Test the Con2D forward pass."""

    if device == "cuda" and SKIP_CUDA_TESTS:
        raise SkipTest("CUDA tests are disabled/ can't be performed")

    if groups > 1:
        raise SkipTest("AIHWKIT currently does not support groups > 1")

    rpu = RPUConfig()
    analog_conv2d = AnalogConv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias=bias,
        rpu_config=rpu,
        device=device,
        dtype=dtype,
    )

    digital_conv2d = AnalogConv2d.to_digital(analog_conv2d)
    re_analog_conv2d = AnalogConv2d.from_digital(digital_conv2d, rpu_config=rpu)
    inp = randn(in_channels, height, width, device=device, dtype=dtype)
    out_orig = analog_conv2d(inp)
    out_re_analog = re_analog_conv2d(inp)  # pylint: disable=not-callable
    assert allclose(out_orig, out_re_analog, atol=1e-5)


@mark.parametrize(
    "clip_type",
    [WeightClipType.NONE, WeightClipType.LAYER_GAUSSIAN, WeightClipType.LAYER_GAUSSIAN_PER_CHANNEL],
)
@mark.parametrize("clip_sigma", [2.0, 3.0])
@mark.parametrize("device", ["cpu", "cuda"])
@mark.parametrize("dtype", [float32, float16, bfloat16])
def test_clipping(
    clip_type: WeightClipType, clip_sigma: float, device: torch_device, dtype: torch_dtype
):
    """Test the clipping."""

    device = torch_device("cpu") if device == "cpu" or SKIP_CUDA_TESTS else torch_device(device)

    rpu_config = RPUConfig()
    rpu_config.clip.type = clip_type
    rpu_config.clip.sigma = clip_sigma
    model = AnalogLinear(in_features=10, out_features=20, rpu_config=rpu_config, bias=False)
    model = model.to(device=device, dtype=dtype)
    weights = randn_like(model.weight, device=device, dtype=dtype)
    model.set_weights(weights)  # note that this performs a clone internally
    optim = AnalogOptimizer(AdamW, model.analog_layers(), model.parameters(), lr=0.0)
    loss: Tensor
    loss = model(randn(10, 10, device=device, dtype=dtype)).sum()  # pylint: disable=not-callable
    loss.backward()
    # actually not doing an update, but just clipping
    optim.step()
    if clip_type == WeightClipType.NONE:
        assert allclose(weights, model.weight)
    elif clip_type == WeightClipType.LAYER_GAUSSIAN:
        bound = weights.std() * clip_sigma
        assert allclose(weights.clamp(-bound, bound), model.weight)
    elif clip_type == WeightClipType.LAYER_GAUSSIAN_PER_CHANNEL:
        bound = weights.std(1, keepdim=True) * clip_sigma
        assert allclose(weights.clamp(-bound, bound), model.weight)
    else:
        raise ValueError(f"Unknown clip type {clip_type}")


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

    def populate_rpu(rpu_config: Union[AIHWKITRPUConfig, RPUConfig]):
        # AIHWKIT-specific configurations
        if isinstance(rpu_config, AIHWKITRPUConfig):
            rpu_config.mapping.max_output_size = -1
            rpu_config.forward.noise_management = NoiseManagementType.NONE
            rpu_config.forward.bound_management = BoundManagementType.NONE

        rpu_config.forward.inp_res = inp_res
        rpu_config.forward.out_res = -1
        rpu_config.forward.out_bound = -1
        rpu_config.forward.out_noise = 0.0
        rpu_config.mapping.max_input_size = max_inp_size
        rpu_config.pre_post.input_range.enable = True
        rpu_config.pre_post.input_range.learn_input_range = True
        rpu_config.pre_post.input_range.init_value = ir_init_value
        rpu_config.pre_post.input_range.init_from_data = ir_init_from_data
        rpu_config.pre_post.input_range.init_std_alpha = ir_init_std_alpha
        return rpu_config

    aihwkit_rpu = populate_rpu(AIHWKITRPUConfig())
    rpu = populate_rpu(RPUConfig())

    aihwkit_linear = AIHWKITAnalogLinear(
        in_features=inp_size, out_features=out_size, bias=bias, rpu_config=aihwkit_rpu
    )
    linear = AnalogLinear(in_features=inp_size, out_features=out_size, bias=bias, rpu_config=rpu)

    aihwkit_linear = aihwkit_linear.to(device=device, dtype=dtype)

    aihwkit_weights, aihwkit_biases = aihwkit_linear.get_weights()
    linear.set_weights_and_biases(aihwkit_weights, aihwkit_biases)
    linear = linear.to(device=device, dtype=dtype)

    if num_inp_dims == 1:
        inp = randn(inp_size, device=device, dtype=dtype, requires_grad=True)
    if num_inp_dims == 2:
        inp = randn(bsz, inp_size, device=device, dtype=dtype, requires_grad=True)
    else:
        inp = randn(bsz, inp_size, inp_size, device=device, dtype=dtype, requires_grad=True)

    inp_aihwkit = randn_like(inp, device=device, dtype=dtype, requires_grad=True)
    inp_aihwkit.data = inp.data

    out_aihwkit: Tensor
    out_aihwkit = aihwkit_linear(inp_aihwkit)  # pylint: disable=not-callable

    out: Tensor
    out = linear(inp)  # pylint: disable=not-callable

    out_aihwkit.sum().backward()
    out.sum().backward()

    atol = 1e-4
    assert allclose(inp_aihwkit.grad, inp.grad, atol=atol), "grad w.r.t. the input not matching"
    assert allclose(out_aihwkit, out, atol=atol)

    for tile_idx, tiles in enumerate(aihwkit_linear.analog_module.array):
        assert len(tiles) == 1, "Output size must be inf"
        tile = tiles[0]
        input_range = tile.input_range
        assert allclose(
            input_range.grad, linear.input_range.grad[tile_idx], atol=atol
        ), f"AIHWKIT: {input_range.grad} lightning: {linear.input_range.grad[tile_idx]}"


@mark.parametrize(
    "modifier_type",
    [
        WeightModifierType.DISCRETIZE,
        WeightModifierType.NONE,
        WeightModifierType.ADD_NORMAL,
        WeightModifierType.ADD_NORMAL_PER_CHANNEL,
        WeightModifierType.DISCRETIZE_ADD_NORMAL,
    ],
)
@mark.parametrize("res", [2**5 - 2, 1 / (2**5 - 2)])
@mark.parametrize("device", ["cpu", "cuda"])
@mark.parametrize("dtype", [float32, float16, bfloat16])
def test_weight_modifier(
    modifier_type: WeightModifierType, res: float, device: str, dtype: torch_dtype
):
    """Test the weight modifier."""

    if device == "cuda" and SKIP_CUDA_TESTS:
        raise SkipTest("CUDA tests are disabled/ can't be performed")

    if res > 0 and modifier_type not in [
        WeightModifierType.DISCRETIZE,
        WeightModifierType.DISCRETIZE_ADD_NORMAL,
    ]:
        raise SkipTest("res but modifier is not discretize")

    manual_seed(0)
    in_size = 10
    rpu_config = RPUConfig()
    rpu_config.modifier.type = modifier_type
    rpu_config.modifier.res = res
    rpu_config.modifier.std_dev = 0.05

    model = AnalogLinear(in_features=in_size, out_features=1, rpu_config=rpu_config, bias=False)
    model = model.to(device=device, dtype=dtype)

    weights = randn_like(model.weight, device=device)
    model.set_weights(weights)  # note that this performs a clone internally
    inp = randn(10, in_size, device=device, dtype=dtype)

    manual_seed(0)
    out = model(inp)  # pylint: disable=not-callable

    assumed_wmax = (
        weights.abs().amax(dim=1, keepdim=True)
        if modifier_type == WeightModifierType.ADD_NORMAL_PER_CHANNEL
        else weights.abs().max()
    )
    if modifier_type in [WeightModifierType.DISCRETIZE, WeightModifierType.DISCRETIZE_ADD_NORMAL]:
        res = rpu_config.modifier.res
        n_states = res / 2 if res > 1.0 else 1 / (2 * res)
        res = (1 / n_states) * assumed_wmax
        quantized_weights = (weights / res).round()
        quantized_weights *= res
    else:
        quantized_weights = weights

    manual_seed(0)
    if modifier_type in [
        WeightModifierType.ADD_NORMAL,
        WeightModifierType.ADD_NORMAL_PER_CHANNEL,
        WeightModifierType.DISCRETIZE_ADD_NORMAL,
    ]:
        noise = (
            rpu_config.modifier.std_dev
            * assumed_wmax
            * randn_like(quantized_weights, device=device, dtype=dtype)
        )
        quantized_weights += noise
    out_expected = matmul(inp, quantized_weights.T)

    assert allclose(out, out_expected, atol=1e-5)


@mark.parametrize("is_test", [True, False])
@mark.parametrize("enable_during_test", [False])
@mark.parametrize("device", ["cpu", "cuda"])
@mark.parametrize("dtype", [float32, float16, bfloat16])
def test_weight_modifier_gradient(
    is_test: bool, enable_during_test: bool, device: str, dtype: torch_dtype
):
    """Test the weight modifier backward behavior."""

    if device == "cuda" and SKIP_CUDA_TESTS:
        raise SkipTest("CUDA tests are disabled/ can't be performed")

    manual_seed(0)
    in_size = 10
    out_size = 20

    def populate_rpu(rpu_config: Union[AIHWKITRPUConfig, RPUConfig]):
        # AIHWKIT-specific configurations
        if isinstance(rpu_config, AIHWKITRPUConfig):
            rpu_config.mapping.max_output_size = -1
            rpu_config.forward.is_perfect = True
            rpu_config.modifier.type = AIHWKITWeightModifierType.ADD_NORMAL
        else:
            rpu_config.modifier.type = WeightModifierType.ADD_NORMAL

        rpu_config.modifier.std_dev = 0.05
        rpu_config.modifier.enable_during_test = enable_during_test
        rpu_config.forward.inp_res = -1
        rpu_config.forward.out_res = -1
        rpu_config.forward.out_bound = -1
        rpu_config.forward.out_noise = 0.0
        rpu_config.mapping.max_input_size = 256
        rpu_config.pre_post.input_range.enable = False
        return rpu_config

    aihwkit_rpu = populate_rpu(AIHWKITRPUConfig())
    rpu = populate_rpu(RPUConfig())

    aihwkit_linear = AIHWKITAnalogLinear(
        in_features=in_size, out_features=out_size, rpu_config=aihwkit_rpu
    )
    linear = AnalogLinear(in_features=in_size, out_features=out_size, rpu_config=rpu)

    aihwkit_linear = aihwkit_linear.to(device=device, dtype=dtype)

    aihwkit_weights, aihwkit_biases = aihwkit_linear.get_weights()
    linear.set_weights_and_biases(aihwkit_weights, aihwkit_biases)
    linear = linear.to(device=device, dtype=dtype)

    if is_test:
        aihwkit_linear = aihwkit_linear.eval()
        linear = linear.eval()

    inp = randn(in_size, device=device, dtype=dtype)

    manual_seed(0)
    out_aihwkit: Tensor
    out_aihwkit = aihwkit_linear(inp)  # pylint: disable=not-callable

    manual_seed(0)
    out: Tensor
    out = linear(inp)  # pylint: disable=not-callable

    assert allclose(out_aihwkit, out, atol=1e-5)


@mark.parametrize("is_test", [True, False])
@mark.parametrize("out_noise_per_channel", [True, False])
@mark.parametrize("device", ["cpu", "cuda"])
@mark.parametrize("dtype", [float32])  # bug in AIHWKIT for fp16 and bfloat16
def test_output_noise(is_test: bool, out_noise_per_channel: bool, device: str, dtype: torch_dtype):
    """Test the weight modifier backward behavior."""

    if device == "cuda" and SKIP_CUDA_TESTS:
        raise SkipTest("CUDA tests are disabled/ can't be performed")

    manual_seed(0)
    in_size = 10
    out_size = 20

    def populate_rpu(rpu_config: Union[AIHWKITRPUConfig, RPUConfig]):
        # AIHWKIT-specific configurations
        if isinstance(rpu_config, AIHWKITRPUConfig):
            rpu_config.mapping.max_output_size = -1
            rpu_config.forward.noise_management = NoiseManagementType.NONE
            rpu_config.forward.bound_management = BoundManagementType.NONE
            rpu_config.mapping.weight_scaling_columnwise = out_noise_per_channel
            rpu_config.forward.out_noise = 0.0 if is_test else 0.05
        else:
            rpu_config.forward.out_noise_per_channel = out_noise_per_channel
            rpu_config.forward.out_noise = 0.05

        rpu_config.forward.inp_res = 2**8 - 2
        rpu_config.forward.out_res = -1
        rpu_config.forward.out_bound = -1
        rpu_config.mapping.max_input_size = 256
        rpu_config.pre_post.input_range.enable = True
        return rpu_config

    aihwkit_rpu = populate_rpu(AIHWKITRPUConfig())
    rpu = populate_rpu(RPUConfig())

    aihwkit_linear = AIHWKITAnalogLinear(
        in_features=in_size, out_features=out_size, rpu_config=aihwkit_rpu
    )
    linear = AnalogLinear(in_features=in_size, out_features=out_size, rpu_config=rpu)

    aihwkit_linear = aihwkit_linear.to(device=device, dtype=dtype)

    aihwkit_weights, aihwkit_biases = aihwkit_linear.get_weights()
    linear.set_weights_and_biases(aihwkit_weights, aihwkit_biases)
    linear = linear.to(device=device, dtype=dtype)

    if is_test:
        aihwkit_linear = aihwkit_linear.eval()
        linear = linear.eval()

    inp = randn(in_size, device=device, dtype=dtype)

    aihwkit_linear.remap_analog_weights()

    manual_seed(0)
    out_aihwkit: Tensor
    out_aihwkit = aihwkit_linear(inp)  # pylint: disable=not-callable

    manual_seed(0)
    out: Tensor
    out = linear(inp)  # pylint: disable=not-callable

    assert allclose(out_aihwkit, out, atol=1e-5)


if __name__ == "__main__":

    test_conv2d_forward(
        1,
        1,
        10,
        101,
        3,
        3,
        [3, 3],
        [1, 1],
        [1, 1],
        [1, 1],
        1,
        False,
        254,
        256,
        True,
        True,
        3.0,
        10,
        3.0,
        "cpu",
        float32,
    )
