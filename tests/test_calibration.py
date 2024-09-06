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

# pylint: disable=too-many-locals, too-many-public-methods
# pylint: disable=no-member, too-many-arguments, too-many-branches

"""Test the calibration of the input ranges."""

from typing import Union
import os
from unittest import SkipTest
from pytest import mark

from torch import dtype as torch_dtype
from torch import device as torch_device
from torch import cuda as torch_cuda
from torch import allclose, randn, float32, float16, bfloat16
from torch.nn.functional import unfold
from aihwkit_lightning.nn import AnalogLinear, AnalogConv2d
from aihwkit_lightning.simulator.configs import TorchInferenceRPUConfig
from aihwkit_lightning.inference.calibration import (
    calibrate_input_ranges,
    InputRangeCalibrationType,
)
from aihwkit_lightning.exceptions import ConfigError


SKIP_CUDA_TESTS = os.getenv("SKIP_CUDA_TESTS") or not torch_cuda.is_available()


@mark.parametrize("module", [AnalogLinear, AnalogConv2d])
@mark.parametrize(
    "cal_type",
    [
        InputRangeCalibrationType.NONE,
        InputRangeCalibrationType.MOVING_STD,
        InputRangeCalibrationType.CACHE_QUANTILE,
        InputRangeCalibrationType.MOVING_QUANTILE,
        InputRangeCalibrationType.MAX,
    ],
)
@mark.parametrize("total_num_samples", [100, 5000])
@mark.parametrize("in_size", [10, 255, 257])
@mark.parametrize("out_size", [10])
@mark.parametrize("inp_res", [-1, 2**8 - 2, 1 / (2**8 - 2)])
@mark.parametrize("max_inp_size", [256])
@mark.parametrize("ir_enable", [False])
@mark.parametrize("is_test", [True])
@mark.parametrize("device", ["cpu", "cuda"])
@mark.parametrize("dtype", [float32, float16, bfloat16])
def test_calibration(
    module: Union[AnalogLinear, AnalogConv2d],
    cal_type: InputRangeCalibrationType,
    total_num_samples: int,
    in_size: int,
    out_size: int,
    inp_res: float,
    max_inp_size: int,
    ir_enable: bool,
    is_test: bool,
    device: str,
    dtype: torch_dtype,
):
    """Test the calibration."""

    if device == "cuda" and SKIP_CUDA_TESTS:
        raise SkipTest("CUDA tests are disabled/ can't be performed")

    if in_size > 10 and module == AnalogConv2d:
        raise SkipTest("Skipping large input size for Conv2d")

    if device == "cpu" and dtype != float32:
        raise SkipTest("Skipping non-float32 tests for CPU")

    def populate_rpu(rpu_config: TorchInferenceRPUConfig):
        rpu_config.forward.inp_res = inp_res
        rpu_config.forward.out_noise = 0.0
        rpu_config.mapping.max_input_size = max_inp_size
        rpu_config.pre_post.input_range.enable = ir_enable
        rpu_config.pre_post.input_range.learn_input_range = True
        return rpu_config

    rpu = populate_rpu(TorchInferenceRPUConfig())
    if module == AnalogConv2d:
        linear_or_conv = module(
            in_channels=in_size, out_channels=out_size, kernel_size=3, rpu_config=rpu
        )
    else:
        linear_or_conv = module(in_features=in_size, out_features=out_size, rpu_config=rpu)

    linear_or_conv.to(dtype=dtype, device=torch_device(device))

    if is_test:
        linear_or_conv.eval()

    if isinstance(linear_or_conv, AnalogConv2d):
        all_inputs = randn(
            (total_num_samples, in_size, 10, 10), dtype=dtype, device=torch_device(device)
        )
    else:
        all_inputs = randn((total_num_samples, in_size), dtype=dtype, device=torch_device(device))

    class Sampler:
        """Example of a sampler used for calibration."""

        def __init__(self):
            self.idx = 0

        def __iter__(self):
            return self

        def __next__(self):
            if self.idx < total_num_samples:
                x = all_inputs[self.idx]
            else:
                raise StopIteration
            self.idx += 1
            if isinstance(linear_or_conv, AnalogConv2d):
                return (x,), {}

            return (), {"inp": x}

    try:
        calibrate_input_ranges(linear_or_conv, calibration_type=cal_type, dataloader=Sampler())
    except ConfigError as exc:
        raise SkipTest("Calibration not supported for this configuration") from exc

    if cal_type in [InputRangeCalibrationType.CACHE_QUANTILE, InputRangeCalibrationType.MAX]:
        current_upper = 0
        for slice_idx, inp_size in enumerate(linear_or_conv.in_sizes):
            if isinstance(linear_or_conv, AnalogConv2d):
                all_inputs_unfolded = unfold(
                    all_inputs,
                    kernel_size=linear_or_conv.kernel_size,
                    dilation=linear_or_conv.dilation,
                    padding=linear_or_conv.padding,
                    stride=linear_or_conv.stride,
                ).transpose(-1, -2)
                inp_slice = all_inputs_unfolded[
                    ..., current_upper : current_upper + inp_size
                ]  # noqa: E203
                num_samples = all_inputs_unfolded.shape[:-1].numel()
            else:
                inp_slice = all_inputs[..., current_upper : current_upper + inp_size]  # noqa: E203
                num_samples = all_inputs.shape[0]
            if cal_type == InputRangeCalibrationType.MAX:
                assert allclose(
                    linear_or_conv.input_range.data[slice_idx], inp_slice.abs().max(), atol=1e-5
                )
            else:
                if num_samples <= 1000:
                    assert allclose(
                        linear_or_conv.input_range.data[slice_idx],
                        inp_slice.float().flatten().quantile(0.99995).to(dtype=dtype),
                        atol=1e-5,
                    )
            current_upper += inp_size
