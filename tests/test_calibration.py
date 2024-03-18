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
"""Test the calibration of the input ranges."""

import os
from unittest import SkipTest
from pytest import mark

from torch import dtype as torch_dtype
from torch import device as torch_device
from torch import cuda as torch_cuda
from torch import allclose, randn, float32, float16, bfloat16
from aihwkit_lightning.nn import AnalogLinear
from aihwkit_lightning.simulator.configs import TorchInferenceRPUConfig
from aihwkit_lightning.inference.calibration import (
    calibrate_input_ranges,
    InputRangeCalibrationType,
)
from aihwkit_lightning.exceptions import ConfigError


SKIP_CUDA_TESTS = os.getenv("SKIP_CUDA_TESTS") or not torch_cuda.is_available()


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
@mark.parametrize("ir_enable", [False, True])
@mark.parametrize("is_test", [False, True])
@mark.parametrize("device", ["cpu", "cuda"])
@mark.parametrize("dtype", [float32, float16, bfloat16])
def test_calibration(
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

    def populate_rpu(rpu_config: TorchInferenceRPUConfig):
        rpu_config.forward.inp_res = inp_res
        rpu_config.forward.out_noise = 0.0
        rpu_config.mapping.max_input_size = max_inp_size
        rpu_config.pre_post.input_range.enable = ir_enable
        rpu_config.pre_post.input_range.learn_input_range = True
        return rpu_config

    rpu = populate_rpu(TorchInferenceRPUConfig())
    linear = AnalogLinear(in_features=in_size, out_features=out_size, rpu_config=rpu)

    linear.to(dtype=dtype, device=torch_device(device))

    if is_test:
        linear.eval()

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
            return (), {"inp": x}

    try:
        calibrate_input_ranges(linear, calibration_type=cal_type, dataloader=Sampler())
    except ConfigError as exc:
        raise SkipTest("Calibration not supported for this configuration") from exc

    if cal_type in [InputRangeCalibrationType.CACHE_QUANTILE, InputRangeCalibrationType.MAX]:
        current_upper = 0
        for slice_idx, inp_size in enumerate(linear.in_sizes):
            inp_slice = all_inputs[..., current_upper : current_upper + inp_size]  # noqa: E203
            if cal_type == InputRangeCalibrationType.MAX:
                assert allclose(
                    linear.input_range.data[slice_idx], inp_slice.abs().max(), atol=1e-5
                )
            else:
                if total_num_samples <= 1000:
                    assert allclose(
                        linear.input_range.data[slice_idx],
                        inp_slice.float().flatten().quantile(0.99995).to(dtype=dtype),
                        atol=1e-5,
                    )
            current_upper += inp_size
