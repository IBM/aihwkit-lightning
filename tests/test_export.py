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
# pylint: disable=too-many-arguments, too-many-branches, too-many-statements
"""Test the export method to AIHWKIT."""

import os
from typing import Union, List, Tuple
from unittest import SkipTest
import logging
from itertools import product
from pytest import mark, fixture

from torch import dtype as torch_dtype
from torch import device as torch_device
from torch import cuda as torch_cuda
from torch import allclose, randn, float32, float16
from torch.nn import Conv2d, Linear, Module
from aihwkit.simulator.configs import TorchInferenceRPUConfig as AIHWKITRPUConfig
from aihwkit.simulator.configs import NoiseManagementType, BoundManagementType
from aihwkit_lightning.nn.conversion import convert_to_analog
from aihwkit_lightning.simulator.configs import TorchInferenceRPUConfig as RPUConfig
from aihwkit_lightning.nn.export import export_to_aihwkit

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


bsz_num_inp_dims_parameters = [
    (bsz, num_inp_dims)
    for bsz, num_inp_dims in list(product([1, 10], [1, 2, 3]))
    if not (num_inp_dims == 1 and bsz > 1)
]


def out_allclose(out_1, out_2, dtype, caplog):
    """Check that outs are close"""
    atol = 1e-4 if dtype == float16 else 1e-5
    return allclose(out_1, out_2, atol=atol)


@fixture(scope="module", name="rpu")
def fixture_rpus(
    max_inp_size,
    ir_enable_inp_res,
    ir_learn_input_range,
    ir_init_value,
    ir_init_from_data,
    ir_init_std_alpha,
) -> Tuple[AIHWKITRPUConfig, RPUConfig]:
    """Fixture for initializing rpus globally for all tests that need them"""
    ir_enable = ir_enable_inp_res[0]
    inp_res = ir_enable_inp_res[1]
    rpu_config = RPUConfig()
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


@mark.parametrize("bsz, num_inp_dims", bsz_num_inp_dims_parameters)
@mark.parametrize("inp_size", [10, 265])
@mark.parametrize("out_size", [10, 265])
@mark.parametrize("bias", [True, False])
@mark.parametrize("max_inp_size", [256], indirect=True)
@mark.parametrize(
    "ir_enable_inp_res", [(True, 2**8 - 2), (True, 1 / (2**8 - 2))], ids=str, indirect=True
)
@mark.parametrize("ir_learn_input_range", [True, False], indirect=True)
@mark.parametrize("ir_init_value", [2.0, 3.0], indirect=True)
@mark.parametrize("ir_init_from_data", [-1, 0, 10], indirect=True)
@mark.parametrize("ir_init_std_alpha", [2.0, 3.0], indirect=True)
@mark.parametrize("device", ["cpu"] if SKIP_CUDA_TESTS else ["cpu", "cuda"])
@mark.parametrize("dtype", [float32, float16], ids=str)  # bfloat is
def test_linear_forward(
    bsz: int,
    num_inp_dims: int,
    inp_size: int,
    out_size: int,
    bias: bool,
    device: torch_device,
    dtype: torch_dtype,
    rpu,
    caplog,
):
    """Test the forward pass."""
    rpu_config = rpu

    class Net(Module):
        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = Linear(in_features=inp_size, out_features=out_size, bias=bias)

        def forward(self, x):
            return self.fc1(x)

    linear = convert_to_analog(Net(), rpu_config=rpu_config)
    linear = linear.eval().to(device=device, dtype=dtype)
    aihwkit_linear = export_to_aihwkit(linear)
    if num_inp_dims == 1:
        inp = randn(inp_size, device=device, dtype=dtype)
    if num_inp_dims == 2:
        inp = randn(bsz, inp_size, device=device, dtype=dtype)
    else:
        inp = randn(bsz, inp_size, inp_size, device=device, dtype=dtype)

    out = linear(inp)  # pylint: disable=not-callable
    out_aihwkit = aihwkit_linear(inp)  # pylint: disable=not-callable
    assert out_allclose(out_aihwkit, out, dtype, caplog)


@mark.parametrize("bsz", [1, 10])
@mark.parametrize("num_inp_dims", [1, 2])
@mark.parametrize("height", [10, 265])
@mark.parametrize("width", [10, 265])
@mark.parametrize("in_channels", [3, 10])
@mark.parametrize("out_channels", [3, 10])
@mark.parametrize("kernel_size", [[3, 3], [3, 4]], ids=str)
@mark.parametrize("stride", [[1, 1]], ids=str)
@mark.parametrize("padding", [[1, 1]], ids=str)
@mark.parametrize("dilation", [[1, 1]], ids=str)
@mark.parametrize("groups", [1])
@mark.parametrize("bias", [True])
@mark.parametrize("max_inp_size", [256], indirect=True)
@mark.parametrize("ir_enable_inp_res", [(True, 2**8 - 2)], ids=str, indirect=True)
@mark.parametrize("ir_learn_input_range", [True, False], indirect=True)
@mark.parametrize("ir_init_value", [3.0], indirect=True)
@mark.parametrize("ir_init_from_data", [10], indirect=True)
@mark.parametrize("ir_init_std_alpha", [3.0], indirect=True)
@mark.parametrize("device", ["cpu"] if SKIP_CUDA_TESTS else ["cpu", "cuda"])
@mark.parametrize("dtype", [float32, float16], ids=str)
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
    rpu,
    caplog,
):
    """Test the Conv2D forward pass."""
    if groups > 1:
        raise SkipTest("AIHWKIT currently does not support groups > 1")

    rpu_config = rpu

    class Net(Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
            )

        def forward(self, x):
            return self.conv1(x)

    conv = convert_to_analog(Net(), rpu_config=rpu_config)
    conv = conv.eval().to(device=device, dtype=dtype)
    aihwkit_conv = export_to_aihwkit(conv)
    if num_inp_dims == 1:
        inp = randn(in_channels, height, width, device=device, dtype=dtype)
    else:
        assert num_inp_dims == 2, "Only batched or non-batched inputs are supported"
        inp = randn(bsz, in_channels, height, width, device=device, dtype=dtype)

    out = conv(inp)  # pylint: disable=not-callable
    out_aihwkit = aihwkit_conv(inp)  # pylint: disable=not-callable
    assert out_allclose(out_aihwkit, out, dtype, caplog)
