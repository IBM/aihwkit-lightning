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
"""Test the speed."""

import os
from unittest import SkipTest
from pytest import mark

from torch import dtype as torch_dtype
from torch import device as torch_device
from torch import cuda as torch_cuda
from torch import randn, float32, float16, bfloat16, Tensor
from torch import compile as torch_compile

from aihwkit_lightning.nn import AnalogLinear
from aihwkit_lightning.simulator.configs import TorchInferenceRPUConfig as RPUConfig


SKIP_CUDA_TESTS = os.getenv("SKIP_CUDA_TESTS") or not torch_cuda.is_available()


@mark.parametrize("is_test", [True, False])
@mark.parametrize("inp_size", [10])
@mark.parametrize("out_size", [10])
@mark.parametrize("bias", [True])
@mark.parametrize("inp_res", [2**8 - 2])
@mark.parametrize("max_inp_size", [256])
@mark.parametrize("ir_enable", [True, False])
@mark.parametrize("ir_learn_input_range", [True, False])
@mark.parametrize("ir_init_value", [2.0])
@mark.parametrize("ir_init_from_data", [-1, 0, 10])
@mark.parametrize("ir_init_std_alpha", [2.0])
@mark.parametrize("device", ["cpu", "cuda"])
@mark.parametrize("dtype", [float32, float16, bfloat16])
def test_torch_compile(  # pylint: disable=too-many-arguments
    is_test: bool,
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
    """Test the speed of the forward pass."""

    if device == "cuda" and SKIP_CUDA_TESTS:
        raise SkipTest("CUDA tests are disabled/ can't be performed")

    if not ir_enable and inp_res > 0:
        raise SkipTest("IR not enabled but inp_res > 0")

    if ir_enable:
        raise SkipTest("Compile doesn't work with IR learning. We're working on that.")

    def populate_rpu(rpu_config: RPUConfig):
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

    rpu = populate_rpu(RPUConfig())
    linear = AnalogLinear(in_features=inp_size, out_features=out_size, bias=bias, rpu_config=rpu)
    if is_test:
        linear = linear.eval()
    linear = linear.to(device=device, dtype=dtype)
    compiled_linear = torch_compile(linear)
    inp = randn(inp_size, device=device, dtype=dtype)
    linear(inp)  # pylint: disable=not-callable
    compiled_linear(inp)

    @torch_compile
    def forward_backward(model: AnalogLinear, inp: Tensor):
        out = model(inp)
        out.sum().backward()
        return out

    forward_backward(linear, inp)
