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
# pylint: disable=too-many-arguments, too-many-branches, too-many-statements
"""Test the forward/backward correctness of our CPU/CUDA version to AIHWKIT."""

import os
from typing import Union, List, Tuple
from unittest import SkipTest
from itertools import product
from pytest import mark, fixture

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

from aihwkit.nn import AnalogLinear as AIHWKITAnalogLinear
from aihwkit.nn import AnalogConv2d as AIWHKITAnalogConv2d
from aihwkit.nn.modules.rnn.rnn import AnalogRNN as AIHWKITAnalogRNN
from aihwkit.nn.modules.rnn.cells import (
    AnalogVanillaRNNCell as AIHWKITAnalogVanillaRNNCell,
    AnalogLSTMCell as AIHWKITAnalogLSTMCell,
    AnalogLSTMCellCombinedWeight as AIHWKITAnalogLSTMCellCombinedWeight,
    AnalogGRUCell as AIHWKITAnalogGRUCell,
)
from aihwkit.simulator.configs import TorchInferenceRPUConfig as AIHWKITRPUConfig
from aihwkit.simulator.configs import NoiseManagementType, BoundManagementType
from aihwkit.simulator.configs import WeightModifierType as AIHWKITWeightModifierType

from aihwkit_lightning.nn import AnalogLinear, AnalogConv2d
from aihwkit_lightning.nn.modules.rnn.rnn import AnalogRNN
from aihwkit_lightning.nn.modules.rnn.cells import (
    AnalogVanillaRNNCell,
    AnalogLSTMCell,
    AnalogLSTMCellCombinedWeight,
    AnalogGRUCell,
)
from aihwkit_lightning.simulator.configs import TorchInferenceRPUConfig as RPUConfig
from aihwkit_lightning.optim import AnalogOptimizer
from aihwkit_lightning.simulator.configs import (
    WeightNoiseInjectionType,
    WeightQuantizationType,
    WeightClipType,
)


SKIP_CUDA_TESTS = os.getenv("SKIP_CUDA_TESTS") or not torch_cuda.is_available()


@fixture(scope="module", name="max_inp_size")
def fixture_max_inp_size(request) -> int:
    """Maximum input size parameter"""
    return request.param


@fixture(scope="module", name="ir_enable_inp_res")
def fixture_ir_enable_inp_res(request) -> Tuple[bool, float]:
    """Combination of ir_enable and inp_res parameters"""
    return request.param


@fixture(scope="module", name="ir_learn_input_range")
def fixture_ir_learn_input_range(request) -> bool:
    """Learn input range parameter"""
    return request.param


@fixture(scope="module", name="ir_dynamic")
def fixture_ir_dynamic(request) -> bool:
    """Dynamic input range"""
    return request.param


@fixture(scope="module", name="ir_init_value")
def fixture_ir_init_value(request) -> float:
    """IR initialization value parameter"""
    return request.param


@fixture(scope="module", name="ir_init_from_data")
def fixture_ir_init_from_data(request) -> int:
    """IR initialization from data parameter"""
    return request.param


@fixture(scope="module", name="ir_init_std_alpha")
def fixture_ir_init_std_alpha(request) -> float:
    """IR initialization alpha parameter"""
    return request.param


@fixture(scope="module", name="adc_config")
def fixture_adc_config(request) -> Tuple[float, float]:
    """Tuple of out_bound, out_res for ADC"""
    return request.param


@fixture(scope="module", name="rpus")
def fixture_rpus(
    max_inp_size: int,
    ir_enable_inp_res: Tuple[bool, float],
    ir_dynamic: bool,
    ir_learn_input_range: bool,
    ir_init_value: float,
    ir_init_from_data: int,
    ir_init_std_alpha: float,
    adc_config: Tuple[float, float],
) -> Tuple[AIHWKITRPUConfig, RPUConfig]:
    """Fixture for initializing rpus globally for all tests that need them"""
    ir_enable, inp_res = ir_enable_inp_res
    out_bound, out_res = adc_config
    aihwkit_rpu_config = AIHWKITRPUConfig()
    lightning_rpu_config = RPUConfig()
    for rpu_config in [aihwkit_rpu_config, lightning_rpu_config]:
        if isinstance(rpu_config, AIHWKITRPUConfig):
            rpu_config.mapping.max_output_size = -1
            if ir_dynamic:
                rpu_config.forward.noise_management = NoiseManagementType.ABS_MAX
                rpu_config.pre_post.input_range.enable = False
            else:
                rpu_config.forward.noise_management = NoiseManagementType.NONE
                rpu_config.pre_post.input_range.enable = ir_enable
            rpu_config.forward.bound_management = BoundManagementType.NONE
            if not ir_enable:
                rpu_config.forward.inp_bound = -1
        elif isinstance(rpu_config, RPUConfig):
            rpu_config.pre_post.input_range.dynamic = ir_dynamic
            rpu_config.pre_post.input_range.enable = ir_enable
        else:
            raise Exception(f"Unknown rpu config type {rpu_config}.")

        rpu_config.forward.inp_res = inp_res
        rpu_config.forward.out_res = out_res
        rpu_config.forward.out_bound = out_bound
        rpu_config.forward.out_noise = 0.0
        rpu_config.mapping.max_input_size = max_inp_size
        rpu_config.pre_post.input_range.learn_input_range = ir_learn_input_range
        rpu_config.pre_post.input_range.init_value = ir_init_value
        rpu_config.pre_post.input_range.init_from_data = ir_init_from_data
        rpu_config.pre_post.input_range.init_std_alpha = ir_init_std_alpha
    return aihwkit_rpu_config, lightning_rpu_config


def recurse_compare(aihwkit_out, lightning_out, atol):
    """Recurse compare to compare LSTM states the same between bidir and unidir layers"""
    if isinstance(aihwkit_out, (list, tuple)):
        # Recurse into lists of states
        return [recurse_compare(aa, bb, atol) for aa, bb in zip(aihwkit_out, lightning_out)]
    if isinstance(lightning_out, (list, tuple)):
        # Handle the fact that AIHWKIT doesn't return a tuple sometimes in AnalogRNN
        return allclose(aihwkit_out, lightning_out[0], atol=atol)
    return allclose(aihwkit_out, lightning_out, atol=atol)


def out_allclose(out_1, out_2, dtype):
    """Check that outs are close, and report if atol has to be less than 1e-5"""
    min_atol = 1e-4 if dtype in [float16, bfloat16] else 1e-5
    return allclose(out_1, out_2, atol=min_atol)


bsz_num_inp_dims_parameters = [
    (bsz, num_inp_dims)
    for bsz, num_inp_dims in list(product([1, 2], [1, 2, 3]))
    if not (num_inp_dims == 1 and bsz > 1)
]


@mark.parametrize("bsz, num_inp_dims", bsz_num_inp_dims_parameters)
@mark.parametrize("inp_size", [10])
@mark.parametrize("out_size", [10])
@mark.parametrize("bias", [True])
@mark.parametrize("max_inp_size", [9, 11], indirect=True)
@mark.parametrize(
    "ir_enable_inp_res", [(True, 2**8 - 2), (True, 1 / (2**8 - 2))], ids=str, indirect=True
)
@mark.parametrize("ir_dynamic", [True, False], indirect=True)
@mark.parametrize("ir_learn_input_range", [True], indirect=True)
@mark.parametrize("ir_init_value", [2.0], indirect=True)
@mark.parametrize("ir_init_from_data", [-1, 0, 10], indirect=True)
@mark.parametrize("ir_init_std_alpha", [2.0, 3.0], indirect=True)
@mark.parametrize(
    "adc_config", [(-1, -1), (10, 2**8 - 2), (10, 1 / (2**8 - 2))], ids=str, indirect=True
)
@mark.parametrize("device", ["cpu"] if SKIP_CUDA_TESTS else ["cpu", "cuda"])
@mark.parametrize("dtype", [float32], ids=str)
def test_linear_forward(
    bsz: int,
    num_inp_dims: int,
    inp_size: int,
    out_size: int,
    bias: bool,
    device: torch_device,
    dtype: torch_dtype,
    rpus: Tuple[AIHWKITRPUConfig, RPUConfig],
):
    """Test the forward pass."""

    aihwkit_rpu, rpu = rpus

    out_bound = rpu.forward.out_bound
    out_res = rpu.forward.out_res
    if not ((out_bound > 0 and out_res > 0) or (out_bound <= 0 and out_res <= 0)):
        raise SkipTest("Can't mix out_bound and out_res")

    aihwkit_linear = AIHWKITAnalogLinear(
        in_features=inp_size, out_features=out_size, bias=bias, rpu_config=aihwkit_rpu
    )
    if out_res > 0:
        # we need the weights to be normalized
        if dtype in [float16, bfloat16]:
            raise SkipTest(
                "ADC tests are done in FP32 only. "
                "This is because it requires re-mapping. "
                "Comparing an MVM with normalized vs."
                "un-normalized weights in FP16 is not feasible "
                "because the quantization error will be too high."
            )
        aihwkit_linear.remap_analog_weights()

    linear = AnalogLinear(in_features=inp_size, out_features=out_size, bias=bias, rpu_config=rpu)

    aihwkit_linear = aihwkit_linear.eval()
    linear = linear.eval()

    aihwkit_linear = aihwkit_linear.to(device=device, dtype=dtype)

    aihwkit_weight, aihwkit_bias = aihwkit_linear.get_weights()
    linear.set_weights_and_biases(aihwkit_weight, aihwkit_bias)
    linear = linear.to(device=device, dtype=dtype)

    if num_inp_dims == 1:
        inp = randn(inp_size, device=device, dtype=dtype)
    elif num_inp_dims == 2:
        inp = randn(bsz, inp_size, device=device, dtype=dtype)
    else:
        inp = randn(bsz, inp_size, inp_size, device=device, dtype=dtype)

    # Some tests might fail due to rounding issues.
    # This can be reproduced by commenting out
    # "aihwkit/simulator/tiles/utils.py, output = output.round()"
    # in AIHWKIT and setting the env variables _AIHWKIT_NO_ROUNDING
    # and AIHWKIT_TESTING.
    out_aihwkit = aihwkit_linear(inp)  # pylint: disable=not-callable
    out = linear(inp)  # pylint: disable=not-callable

    atol = 1e-4 if dtype in [float16, bfloat16] else 1e-5
    assert allclose(out_aihwkit, out, atol=atol)


@mark.parametrize("bsz", [1, 10])
@mark.parametrize("num_inp_dims", [1, 2, 3])
@mark.parametrize("height", [10])
@mark.parametrize("width", [10])
@mark.parametrize("in_channels", [3, 10])
@mark.parametrize("out_channels", [3, 10])
@mark.parametrize("kernel_size", [[3, 3], [3, 4]], ids=str)
@mark.parametrize("stride", [[1, 1]], ids=str)
@mark.parametrize("padding", [[1, 1]], ids=str)
@mark.parametrize("dilation", [[1, 1]], ids=str)
@mark.parametrize("groups", [1])
@mark.parametrize("bias", [True])
@mark.parametrize("max_inp_size", [256, 512], indirect=True)
@mark.parametrize("ir_enable_inp_res", [(True, 2**8 - 2)], ids=str, indirect=True)
@mark.parametrize("ir_dynamic", [True, False], indirect=True)
@mark.parametrize("ir_learn_input_range", [True, False], indirect=True)
@mark.parametrize("ir_init_value", [3.0], indirect=True)
@mark.parametrize("ir_init_from_data", [10], indirect=True)
@mark.parametrize("ir_init_std_alpha", [3.0], indirect=True)
@mark.parametrize("adc_config", [(-1, -1)], ids=str, indirect=True)
@mark.parametrize("device", ["cpu"] if SKIP_CUDA_TESTS else ["cpu", "cuda"])
@mark.parametrize("dtype", [float32], ids=str)
def test_conv2d_forward(
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
    device: torch_device,
    dtype: torch_dtype,
    rpus,
):
    """Test the Conv2D forward pass."""

    manual_seed(0)

    if groups > 1:
        raise SkipTest("AIHWKIT currently does not support groups > 1")

    if num_inp_dims == 1:
        raise SkipTest("AIHWKIT has a bug with 1D inputs in Conv layers")

    aihwkit_rpu, rpu = rpus

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
        inp = randn(bsz, in_channels, height, width, device=device, dtype=dtype)

    digital_aihwkit_conv2d = AIWHKITAnalogConv2d.to_digital(aihwkit_analog_conv2d)
    conv_weight = digital_aihwkit_conv2d.weight.to(device=device, dtype=dtype)
    conv_bias = digital_aihwkit_conv2d.bias
    conv_bias = conv_bias.to(device=device, dtype=dtype) if conv_bias is not None else None
    analog_conv2d.set_weights_and_biases(conv_weight, conv_bias)
    analog_conv2d = analog_conv2d.to(device=device, dtype=dtype)

    out_aihwkit = aihwkit_analog_conv2d(inp)  # pylint: disable=not-callable
    out = analog_conv2d(inp)  # pylint: disable=not-callable
    atol = 1e-4 if dtype in [float16, bfloat16] else 1e-5
    assert allclose(out_aihwkit, out, atol=atol)


@mark.parametrize("height", [10, 513])
@mark.parametrize("width", [10, 513])
@mark.parametrize("in_channels", [3, 10])
@mark.parametrize("out_channels", [3, 10])
@mark.parametrize("kernel_size", [[3, 3]], ids=str)
@mark.parametrize("stride", [[1, 1]], ids=str)
@mark.parametrize("padding", [[1, 1]], ids=str)
@mark.parametrize("dilation", [[1, 1]], ids=str)
@mark.parametrize("groups", [1])
@mark.parametrize("bias", [True])
@mark.parametrize("device", ["cpu"] if SKIP_CUDA_TESTS else ["cpu", "cuda"])
@mark.parametrize("dtype", [float32, float16, bfloat16], ids=str)
def test_conv2d_to_and_from_digital(
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

    if groups > 1:
        raise SkipTest("AIHWKIT currently does not support groups > 1")

    if device == "cpu" and dtype != float32:
        raise SkipTest("Skipping non-float32 tests for CPU")

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
    atol = 1e-4 if dtype in [float16, bfloat16] else 1e-5
    assert allclose(out_orig, out_re_analog, atol=atol)


# Don't run tests where dropout > 0.0 but num_layers == 1
layers_dropout_parameters = [
    (num_layers, dropout)
    for num_layers, dropout in list(product([1, 3], [0.0, 0.1]))
    if (num_layers > 1) or (num_layers == 1 and dropout == 0.0)
]


@mark.parametrize("bsz", [0, 1, 2])
@mark.parametrize(
    "cell_type", [AnalogVanillaRNNCell, AnalogLSTMCell, AnalogLSTMCellCombinedWeight, AnalogGRUCell]
)
@mark.parametrize("sequence_length", [1, 2])
@mark.parametrize("input_size", [9, 11])
@mark.parametrize("hidden_size", [9, 11])
@mark.parametrize("num_layers, dropout", layers_dropout_parameters)
@mark.parametrize("bias", [True])
@mark.parametrize("batch_first", [False])
@mark.parametrize("bidir", [True, False])
@mark.parametrize("proj_size", [0])
@mark.parametrize("realistic_read_write", [True])
@mark.parametrize("max_inp_size", [10], indirect=True)
@mark.parametrize("ir_enable_inp_res", [(True, 2**8 - 2)], ids=str, indirect=True)
@mark.parametrize("ir_dynamic", [True, False], indirect=True)
@mark.parametrize("ir_learn_input_range", [True, False], indirect=True)
@mark.parametrize("ir_init_value", [3.0], indirect=True)
@mark.parametrize("ir_init_from_data", [10], indirect=True)
@mark.parametrize("ir_init_std_alpha", [3.0], indirect=True)
@mark.parametrize("adc_config", [(-1, -1)], ids=str, indirect=True)
@mark.parametrize("device", ["cpu"] if SKIP_CUDA_TESTS else ["cpu", "cuda"])
@mark.parametrize("dtype", [float32], ids=str)
def test_lstm_forward(
    bsz: int,
    cell_type,
    sequence_length: int,
    input_size: int,
    hidden_size: int,
    num_layers: int,
    bias: bool,
    batch_first: bool,
    dropout: float,
    bidir: bool,
    proj_size: int,
    realistic_read_write: bool,
    device: torch_device,
    dtype: torch_dtype,
    rpus,
):
    """Test the lstm forward pass."""
    if cell_type == AnalogVanillaRNNCell:
        aihwkit_cell = AIHWKITAnalogVanillaRNNCell
    elif cell_type == AnalogLSTMCell:
        aihwkit_cell = AIHWKITAnalogLSTMCell
    elif cell_type == AnalogLSTMCellCombinedWeight:
        aihwkit_cell = AIHWKITAnalogLSTMCellCombinedWeight
    elif cell_type == AnalogGRUCell:
        aihwkit_cell = AIHWKITAnalogGRUCell
    else:
        raise RuntimeError("Unrecognized cell type in test_lstm_forward")

    aihwkit_rpu, rpu = rpus

    aihwkit_lstm = AIHWKITAnalogRNN(
        cell=aihwkit_cell,
        input_size=input_size,
        hidden_size=hidden_size,
        bias=bias,
        rpu_config=aihwkit_rpu,
        xavier=realistic_read_write,
        num_layers=num_layers,
        bidir=bidir,
        dropout=dropout,
    )
    lstm = AnalogRNN(
        cell=cell_type,
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        bias=bias,
        batch_first=batch_first,
        bidir=bidir,
        dropout=dropout,
        proj_size=proj_size,
        xavier=realistic_read_write,
        device=device,
        dtype=dtype,
        rpu_config=rpu,
    )
    aihwkit_lstm = aihwkit_lstm.eval()
    lstm = lstm.eval()

    # AIHWKIT does not properly convert the AnalogContext of TorchSimulatorTile to (b)float16,
    # resulting in a failure
    aihwkit_lstm = aihwkit_lstm.to(device=device, dtype=dtype)
    for param in aihwkit_lstm.parameters():
        param.to(dtype)
    for aihwkit_layer, layer in zip(aihwkit_lstm.rnn.layers, lstm.rnn.layers):
        if bidir:
            for aihwkit_dir, l_dir in zip(aihwkit_layer.directions, layer.directions):
                if cell_type == AnalogLSTMCellCombinedWeight:
                    aihwkit_weight, aihwkit_bias = aihwkit_dir.cell.weight.get_weights()
                    l_dir.cell.weight.set_weights_and_biases(aihwkit_weight, aihwkit_bias)
                else:
                    aihwkit_weight, aihwkit_bias = aihwkit_dir.cell.weight_ih.get_weights()
                    l_dir.cell.weight_ih.set_weights_and_biases(aihwkit_weight, aihwkit_bias)
                    aihwkit_weight, aihwkit_bias = aihwkit_dir.cell.weight_hh.get_weights()
                    l_dir.cell.weight_hh.set_weights_and_biases(aihwkit_weight, aihwkit_bias)
        else:
            if cell_type == AnalogLSTMCellCombinedWeight:
                aihwkit_weight, aihwkit_bias = aihwkit_layer.cell.weight.get_weights()
                layer.cell.weight.set_weights_and_biases(aihwkit_weight, aihwkit_bias)
            else:
                aihwkit_weight, aihwkit_bias = aihwkit_layer.cell.weight_ih.get_weights()
                layer.cell.weight_ih.set_weights_and_biases(aihwkit_weight, aihwkit_bias)
                aihwkit_weight, aihwkit_bias = aihwkit_layer.cell.weight_hh.get_weights()
                layer.cell.weight_hh.set_weights_and_biases(aihwkit_weight, aihwkit_bias)

    lstm = lstm.to(device=device, dtype=dtype)

    if bsz == 0:
        inp = randn(sequence_length, input_size, device=device, dtype=dtype)
        # aihwkit can't do unbatched inputs
        aihwkit_inp = inp.unsqueeze(1)
    else:
        inp = randn(sequence_length, bsz, input_size, device=device, dtype=dtype)
        aihwkit_inp = inp
    out, out_hidden = lstm(inp)  # pylint: disable=not-callable
    out_aihwkit, out_hidden_aihwkit = aihwkit_lstm(aihwkit_inp)  # pylint: disable=not-callable

    atol = 1e-3 if dtype in [float16, bfloat16] else 1e-4
    assert allclose(out_aihwkit, out, atol=atol)
    assert all(recurse_compare(out_hidden_aihwkit, out_hidden, atol))


@mark.parametrize(
    "clip_type",
    [WeightClipType.NONE, WeightClipType.LAYER_GAUSSIAN, WeightClipType.LAYER_GAUSSIAN_PER_CHANNEL],
)
@mark.parametrize("clip_sigma", [2.0, 3.0])
@mark.parametrize("device", ["cpu"] if SKIP_CUDA_TESTS else ["cpu", "cuda"])
@mark.parametrize("dtype", [float32, float16, bfloat16], ids=str)
def test_clipping(
    clip_type: WeightClipType, clip_sigma: float, device: torch_device, dtype: torch_dtype
):
    """Test the clipping."""

    if device == "cpu" and dtype != float32:
        raise SkipTest("Skipping non-float32 tests for CPU")

    device = torch_device(device)

    rpu_config = RPUConfig()
    rpu_config.clip.type = clip_type
    rpu_config.clip.sigma = clip_sigma
    model = AnalogLinear(in_features=10, out_features=20, rpu_config=rpu_config, bias=False)
    model = model.to(device=device, dtype=dtype)
    weights = randn_like(model.weight, device=device, dtype=dtype)
    model.set_weights(weights)  # note that this performs a clone internally
    optim = AnalogOptimizer(AdamW, model.analog_layers, model.parameters(), lr=0.0)
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


@mark.parametrize("bsz, num_inp_dims", bsz_num_inp_dims_parameters)
@mark.parametrize("inp_size", [10, 32])
@mark.parametrize("out_size", [10, 32])
@mark.parametrize("bias", [True, False])
@mark.parametrize(
    "ir_enable_inp_res", [(True, 2**8 - 2), (True, 1 / (2**8 - 2))], ids=str, indirect=True
)
@mark.parametrize("ir_dynamic", [True, False], indirect=True)
@mark.parametrize("ir_learn_input_range", [True], indirect=True)
@mark.parametrize("max_inp_size", [32], indirect=True)
@mark.parametrize("ir_init_value", [2.0], indirect=True)
@mark.parametrize("ir_init_from_data", [-1, 0, 10], indirect=True)
@mark.parametrize("ir_init_std_alpha", [2.0], indirect=True)
@mark.parametrize("adc_config", [(-1, -1), (10, 254)], ids=str, indirect=True)
@mark.parametrize("device", ["cpu"] if SKIP_CUDA_TESTS else ["cpu", "cuda"])
@mark.parametrize("dtype", [float32], ids=str)
def test_backward(
    bsz: int,
    num_inp_dims: int,
    inp_size: int,
    out_size: int,
    bias: bool,
    device: str,
    dtype: torch_dtype,
    rpus: Tuple[AIHWKITRPUConfig, RPUConfig],
):
    """Test the input range backward pass."""

    manual_seed(0)
    aihwkit_rpu, rpu = rpus
    aihwkit_linear = AIHWKITAnalogLinear(
        in_features=inp_size, out_features=out_size, bias=bias, rpu_config=aihwkit_rpu
    )
    if rpu.forward.out_res > 0:
        # we need the weights to be normalized
        if dtype in [float16, bfloat16]:
            raise SkipTest(
                "ADC tests are done in FP32 only. "
                "This is because it requires re-mapping. "
                "Comparing an MVM with normalized vs."
                "un-normalized weights in FP16 is not feasible "
                "because the quantization error will be too high."
            )
        aihwkit_linear.remap_analog_weights()

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

    # aihwkit simulator/tiles/analog_mvm.py l. 306 and l. 115, if these
    # calls are wrapped in no_grad (i.e. ignore the scaling and re-scaling)
    # the gradients would be exactly the same. we don't apply scaling and
    # re-scaling of the inputs and outputs
    no_check = inp.abs() == inp.abs().amax(-1, keepdim=True)
    assert allclose(
        inp_aihwkit.grad[~no_check], inp.grad[~no_check], atol=atol
    ), "grad w.r.t. the input not matching"
    assert allclose(out_aihwkit, out, atol=atol)


@mark.parametrize(
    "modifier_type",
    [
        WeightNoiseInjectionType.NONE,
        WeightNoiseInjectionType.ADD_NORMAL,
        WeightNoiseInjectionType.ADD_NORMAL_PER_CHANNEL,
    ],
)
@mark.parametrize(
    "quantization_type",
    [
        WeightQuantizationType.NONE,
        WeightQuantizationType.DISCRETIZE,
        WeightQuantizationType.DISCRETIZE_PER_CHANNEL,
    ],
)
@mark.parametrize("res", [2**5 - 2, 1 / (2**5 - 2)])
@mark.parametrize("device", ["cpu"] if SKIP_CUDA_TESTS else ["cpu", "cuda"])
@mark.parametrize("dtype", [float32, float16, bfloat16], ids=str)
def test_weight_modifier(
    modifier_type: WeightNoiseInjectionType,
    quantization_type: WeightQuantizationType,
    res: float,
    device: str,
    dtype: torch_dtype,
):
    """Test the weight modifier."""

    if res > 0 and quantization_type not in [
        WeightQuantizationType.DISCRETIZE,
        WeightQuantizationType.DISCRETIZE_PER_CHANNEL,
    ]:
        raise SkipTest("res but quantizer is not discretize")

    if device == "cpu" and dtype != float32:
        raise SkipTest("Skipping non-float32 tests for CPU")

    manual_seed(0)
    in_size = 10
    rpu_config = RPUConfig()
    rpu_config.modifier.noise_type = modifier_type
    rpu_config.modifier.quantization_type = quantization_type
    rpu_config.modifier.res = res
    rpu_config.modifier.std_dev = 0.05

    model = AnalogLinear(in_features=in_size, out_features=1, rpu_config=rpu_config, bias=False)
    model = model.to(device=device, dtype=dtype)

    weights = randn_like(model.weight, device=device)
    model.set_weights(weights)  # note that this performs a clone internally
    inp = randn(10, in_size, device=device, dtype=dtype)

    manual_seed(0)
    out = model(inp)  # pylint: disable=not-callable

    if quantization_type in [
        WeightQuantizationType.DISCRETIZE,
        WeightQuantizationType.DISCRETIZE_PER_CHANNEL,
    ]:
        if quantization_type == WeightQuantizationType.DISCRETIZE_PER_CHANNEL:
            assumed_wmax = weights.abs().amax(dim=1, keepdim=True)
        else:
            assumed_wmax = weights.abs().max()
        res = rpu_config.modifier.res
        n_states = res / 2 if res > 1.0 else 1 / (2 * res)
        res = (1 / n_states) * assumed_wmax
        quantized_weights = (weights / res).round()
        quantized_weights *= res
    else:
        quantized_weights = weights

    manual_seed(0)
    if modifier_type in [
        WeightNoiseInjectionType.ADD_NORMAL,
        WeightNoiseInjectionType.ADD_NORMAL_PER_CHANNEL,
    ]:
        if modifier_type == WeightNoiseInjectionType.ADD_NORMAL_PER_CHANNEL:
            assumed_wmax = weights.abs().amax(dim=1, keepdim=True)
        else:
            assumed_wmax = weights.abs().max()
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
@mark.parametrize("device", ["cpu"] if SKIP_CUDA_TESTS else ["cpu", "cuda"])
@mark.parametrize("dtype", [float32, float16, bfloat16], ids=str)
def test_weight_modifier_gradient(
    is_test: bool, enable_during_test: bool, device: str, dtype: torch_dtype
):
    """Test the weight modifier backward behavior."""

    if device == "cpu" and dtype != float32:
        raise SkipTest("Skipping non-float32 tests for CPU")

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
            rpu_config.modifier.noise_type = WeightNoiseInjectionType.ADD_NORMAL

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
@mark.parametrize("device", ["cpu"] if SKIP_CUDA_TESTS else ["cpu", "cuda"])
@mark.parametrize("dtype", [float32], ids=str)  # bug in AIHWKIT for fp16 and bfloat16
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

        rpu_config.pre_post.input_range.enable = True
        rpu_config.pre_post.input_range.init_from_data = 0
        rpu_config.pre_post.input_range.init_value = 0.2
        rpu_config.forward.inp_res = 2**8 - 2
        rpu_config.forward.out_res = -1
        rpu_config.forward.out_bound = -1
        rpu_config.mapping.max_input_size = 256
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
    test_rpus = fixture_rpus(
        max_inp_size=9,
        ir_enable_inp_res=(True, 254),
        ir_dynamic=True,
        ir_learn_input_range=True,
        ir_init_value=2.0,
        ir_init_from_data=-1,
        ir_init_std_alpha=2.0,
        adc_config=(10, 254),
    )
    # test_conv2d_forward(
    #     bsz=1,
    #     num_inp_dims=2,
    #     height=10,
    #     width=10,
    #     in_channels=3,
    #     out_channels=3,
    #     kernel_size=(3, 3),
    #     stride=(1, 1),
    #     padding=(1, 1),
    #     dilation=(1, 1),
    #     groups=1,
    #     bias=True,
    #     device=torch_device("cpu"),
    #     dtype=float32,
    #     rpus=test_rpus,
    # )
    # test_backward(
    #     bsz=1,
    #     num_inp_dims=2,
    #     inp_size=32,
    #     out_size=32,
    #     bias=False,
    #     device=torch_device("cpu"),
    #     dtype=float32,
    #     rpus=test_rpus,
    # )
    # test_linear_forward(
    #     bsz=3,
    #     num_inp_dims=2,
    #     inp_size=10,
    #     out_size=10,
    #     bias=True,
    #     device=torch_device("cpu"),
    #     dtype=float32,
    #     rpus=test_rpus,
    # )
    test_output_noise(
        is_test=False, out_noise_per_channel=False, device=torch_device("cpu"), dtype=float32
    )
