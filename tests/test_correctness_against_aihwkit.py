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
from typing import Union
from pytest import mark

from torch import dtype as torch_dtype
from torch import device as torch_device
from torch import cuda as torch_cuda
from torch import allclose, randn, randn_like, float32, float16, bfloat16, Tensor
from torch.optim import AdamW

from aihwkit.nn import AnalogLinear as AIHWKITAnalogLinear
from aihwkit.simulator.configs import TorchInferenceRPUConfig as AIHWKITRPUConfig
from aihwkit.simulator.configs import NoiseManagementType

from aihwkit_lightning.nn import AnalogLinear as AnalogLinear
from aihwkit_lightning.simulator.configs import TorchInferenceRPUConfig as RPUConfig
from aihwkit_lightning.optim import AnalogOptimizer
from aihwkit_lightning.simulator.configs import WeightClipType


SKIP_CUDA_TESTS = os.getenv("SKIP_CUDA_TESTS") or not torch_cuda.is_available()


@mark.parametrize("bsz", [1, 10])
@mark.parametrize("inp_size", [10, 255, 512, 513, 1024])
@mark.parametrize("out_size", [10, 255, 512, 513, 1024])
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
def test_linear_forward(
    bsz: int,
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

    device = (
        torch_device("cpu")
        if device == "cpu" or SKIP_CUDA_TESTS
        else torch_device(device)
    )

    def populate_rpu(rpu_config: Union[AIHWKITRPUConfig, RPUConfig]):
        # AIHWKIT-specific configurations
        if isinstance(rpu_config, AIHWKITRPUConfig):
            rpu_config.mapping.max_output_size = -1
            rpu_config.forward.noise_management = (
                NoiseManagementType.ABS_MAX
                if not ir_enable
                else NoiseManagementType.NONE
            )

        rpu_config.forward.inp_res = inp_res
        rpu_config.forward.out_noise = 0.0
        rpu_config.mapping.max_input_size = max_inp_size
        rpu_config.pre_post.input_range.enable = ir_enable
        rpu_config.pre_post.input_range.learn_input_range = ir_learn_input_range
        rpu_config.pre_post.input_range.init_value = ir_init_value
        rpu_config.pre_post.input_range.init_from_data = ir_init_from_data
        rpu_config.pre_post.input_range.init_std_alpha = ir_init_std_alpha

    aihwkit_rpu = populate_rpu(AIHWKITRPUConfig())
    rpu = populate_rpu(RPUConfig())

    aihwkit_linear = AIHWKITAnalogLinear(
        in_features=inp_size, out_features=out_size, bias=bias, rpu_config=aihwkit_rpu
    )
    linear = AnalogLinear(
        in_features=inp_size, out_features=out_size, bias=bias, rpu_config=rpu
    )

    aihwkit_linear = aihwkit_linear.eval()
    linear = linear.eval()

    aihwkit_linear = aihwkit_linear.to(device=device, dtype=dtype)
    linear = linear.to(device=device, dtype=dtype)

    inp = randn(bsz, inp_size, device=device, dtype=dtype)
    out_aihwkit = aihwkit_linear(inp)
    out = linear(inp)
    assert allclose(out_aihwkit, out, atol=1e-5)


@mark.parametrize(
    "clip_type",
    [
        WeightClipType.NONE,
        WeightClipType.LAYER_GAUSSIAN,
        WeightClipType.LAYER_GAUSSIAN_PER_CHANNEL,
    ],
)
@mark.parametrize("clip_sigma", [2.0, 3.0])
@mark.parametrize("device", ["cpu", "cuda"])
@mark.parametrize("dtype", [float32, float16, bfloat16])
def test_clipping(
    clip_type: WeightClipType,
    clip_sigma: float,
    device: torch_device,
    dtype: torch_dtype,
):
    device = (
        torch_device("cpu")
        if device == "cpu" or SKIP_CUDA_TESTS
        else torch_device(device)
    )

    rpu_config = RPUConfig()
    rpu_config.clip.type = clip_type
    rpu_config.clip.sigma = clip_sigma
    model = AnalogLinear(in_features=10, out_features=20, rpu_config=rpu_config, bias=False)
    model = model.to(device=device, dtype=dtype)
    weights = randn_like(model.weight, device=device, dtype=dtype)
    model.set_weights(weights)  # note that this performs a clone internally
    optim = AnalogOptimizer(AdamW, model.analog_layers(), model.parameters(), lr=0.0)
    loss: Tensor
    loss = model(randn(10, 10, device=device, dtype=dtype)).sum()
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


def test_input_range_backward():
    pass
