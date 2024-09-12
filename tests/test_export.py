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
# pylint: disable=too-many-arguments, too-many-branches, too-many-statements\
# mypy: disable-error-code="arg-type"
"""Test the export method to AIHWKIT."""

import os
from typing import Tuple
from unittest import SkipTest
from itertools import product
from pytest import mark, fixture

from torch import dtype as torch_dtype
from torch import device as torch_device
from torch import cuda as torch_cuda
from torch import arange, allclose, randn, manual_seed, float32, float16, bfloat16, Tensor
from torch.nn import Linear, Conv2d, Module, Sequential

from aihwkit_lightning.nn.conversion import convert_to_analog
from aihwkit_lightning.simulator.configs import WeightClipType, WeightModifierType
from aihwkit_lightning.simulator.configs import TorchInferenceRPUConfig as RPUConfig
from aihwkit_lightning.nn.export import export_to_aihwkit
from aihwkit_lightning.nn import AnalogLinear

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


@fixture(scope="module", name="clip_type")
def fixture_clip_type(request) -> WeightClipType:
    """Weight clip type parameter"""
    return request.param


@fixture(scope="module", name="weight_modifier_type")
def fixture_weight_modifier_type(request) -> WeightModifierType:
    """Weight modifier type parameter"""
    return request.param


bsz_num_inp_dims_parameters = [
    (bsz, num_inp_dims)
    for bsz, num_inp_dims in list(product([1, 2], [1, 2, 3]))
    if not (num_inp_dims == 1 and bsz > 1)
]


# @fixture(scope="module", name="rpu")
def fixture_rpu(
    max_inp_size: int,
    ir_enable_inp_res: Tuple[bool, float],
    ir_learn_input_range: bool,
    ir_init_value: float,
    ir_init_from_data: int,
    ir_init_std_alpha: float,
    adc_config: Tuple[float, float],
    clip_type: WeightClipType,
    weight_modifier_type: WeightModifierType,
) -> RPUConfig:
    """Fixture for initializing rpu globally for all tests that need them"""
    ir_enable, inp_res = ir_enable_inp_res
    out_bound, out_res = adc_config
    rpu_config = RPUConfig()

    rpu_config.clip.type = clip_type
    rpu_config.clip.sigma = 2.0

    rpu_config.modifier.type = weight_modifier_type
    rpu_config.modifier.std_dev = 0.0

    rpu_config.forward.inp_res = inp_res

    rpu_config.forward.out_res = out_res
    rpu_config.forward.out_bound = out_bound
    rpu_config.forward.out_noise = 0.0

    rpu_config.mapping.max_input_size = max_inp_size

    rpu_config.pre_post.input_range.enable = ir_enable
    rpu_config.pre_post.input_range.learn_input_range = ir_learn_input_range
    rpu_config.pre_post.input_range.init_value = ir_init_value
    rpu_config.pre_post.input_range.init_from_data = ir_init_from_data
    rpu_config.pre_post.input_range.init_std_alpha = ir_init_std_alpha
    return rpu_config


@mark.parametrize("bsz, num_inp_dims", bsz_num_inp_dims_parameters)
@mark.parametrize("inp_size", [10])
@mark.parametrize("max_inp_size", [9, 11], indirect=True)
@mark.parametrize(
    "ir_enable_inp_res", [(True, 2**8 - 2), (True, 1 / (2**8 - 2))], ids=str, indirect=True
)
@mark.parametrize("ir_learn_input_range", [True, False], indirect=True)
@mark.parametrize("ir_init_value", [2.0, 3.0], indirect=True)
@mark.parametrize("ir_init_from_data", [-1, 0, 10], indirect=True)
@mark.parametrize("ir_init_std_alpha", [2.0, 3.0], indirect=True)
@mark.parametrize(
    "adc_config", [(-1, -1), (10, 2**8 - 2), (10, 1 / (2**8 - 2))], ids=str, indirect=True
)
@mark.parametrize(
    "clip_type",
    [WeightClipType.NONE, WeightClipType.LAYER_GAUSSIAN, WeightClipType.LAYER_GAUSSIAN_PER_CHANNEL],
    ids=str,
    indirect=True,
)
@mark.parametrize("weight_modifier_type", [WeightModifierType.NONE], ids=str, indirect=True)
@mark.parametrize("device", ["cpu"] if SKIP_CUDA_TESTS else ["cpu", "cuda"])
@mark.parametrize("dtype", [float32], ids=str)
def test_linear_forward(
    bsz: int,
    num_inp_dims: int,
    inp_size: int,
    device: torch_device,
    dtype: torch_dtype,
    rpu: RPUConfig,
):
    """Test the forward pass."""

    if num_inp_dims == 1:
        raise SkipTest("AIHWKIT has a bug with 1D inputs in Conv layers")

    # Set the seed for debugging
    manual_seed(0)

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

    class SimpleNet(Module):
        """Simple network for testing."""

        def __init__(self):
            super().__init__()
            self.conv = Conv2d(in_channels=inp_size, out_channels=1, kernel_size=3)
            self.fc_seq = Sequential(
                Linear(
                    4 * 4, 10
                ),  # Adjust based on input size, assuming input size after conv is 6x6
                Linear(10, 10),
            )
            self.fc_final = Linear(10, 5)  # Assuming 10 output classes

        def forward(self, inp: Tensor) -> Tensor:
            """Forward pass."""
            x = self.conv(inp)
            x = x.view(x.size(0), -1)
            x = self.fc_seq(x)
            x = self.fc_final(x)
            return x

    # Convert to analog using aihwkit-lightning
    model = convert_to_analog(SimpleNet(), rpu_config=rpu)

    # Make the input range a bit harder
    analog_layer: AnalogLinear
    for analog_layer in model.analog_layers():
        if rpu.pre_post.input_range.enable:
            input_range_tensor = analog_layer.input_range
            analog_layer.input_range.data = 0.1 + arange(input_range_tensor.numel()).float()

    model = model.eval().to(device=device, dtype=dtype)
    aihwkit_model = export_to_aihwkit(model=model)

    if num_inp_dims == 1:
        inp = randn(inp_size, 6, 6, device=device, dtype=dtype)
    else:
        inp = randn(bsz, inp_size, 6, 6, device=device, dtype=dtype)

    out = model(inp)  # pylint: disable=not-callable
    out_aihwkit = aihwkit_model(inp)  # pylint: disable=not-callable

    atol = 1e-4 if dtype in [float16, bfloat16] else 1e-5
    assert allclose(out_aihwkit, out, atol=atol)


if __name__ == "__main__":
    test_rpu = fixture_rpu(
        max_inp_size=5,
        ir_enable_inp_res=(True, 254),
        ir_learn_input_range=True,
        ir_init_value=3.0,
        ir_init_from_data=10,
        ir_init_std_alpha=3.0,
        adc_config=(-1, -1),
        clip_type=WeightClipType.LAYER_GAUSSIAN_PER_CHANNEL,
        weight_modifier_type=WeightModifierType.NONE,
    )

    test_linear_forward(
        bsz=3, num_inp_dims=2, inp_size=10, device="cuda", dtype=float32, rpu=test_rpu
    )
