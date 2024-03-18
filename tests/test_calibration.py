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
    float32,
    float16,
    bfloat16,
    Tensor,
    matmul,
    manual_seed,
)
from torch.optim import AdamW

from aihwkit_lightning.nn import AnalogLinear
from aihwkit_lightning.simulator.configs import TorchInferenceRPUConfig
from aihwkit_lightning.optim import AnalogOptimizer
from aihwkit_lightning.inference.calibration import calibrate_input_ranges, InputRangeCalibrationType


SKIP_CUDA_TESTS = os.getenv("SKIP_CUDA_TESTS") or not torch_cuda.is_available()


def test_calibration(
    inp_size: int,
    out_size: int,
    inp_res: float,
    max_inp_size: int,
    ir_enable: bool,
    is_test: bool,
    device: str,
    dtype: torch_dtype
):
    """Test the calibration."""

    if device == "cuda" and SKIP_CUDA_TESTS:
        raise SkipTest("CUDA tests are disabled/ can't be performed")

    def populate_rpu(rpu_config: TorchInferenceRPUConfig):
        rpu_config.forward.inp_res = inp_res
        rpu_config.forward.out_noise = 0.0
        rpu_config.mapping.max_input_size = max_inp_size
        rpu_config.pre_post.input_range.enable = True
        rpu_config.pre_post.input_range.learn_input_range = ir_enable
        return rpu_config

    rpu = populate_rpu(TorchInferenceRPUConfig())
    linear = AnalogLinear(in_features=inp_size, out_features=out_size, rpu_config=rpu)

    linear.to(dtype=dtype, device=torch_device(device))

    if is_test:
        linear.eval()

    calibrate_input_ranges(linear, calibration_type=InputRangeCalibrationType.MAX, dataloader=sampler)


if __name__ == "__main__":
    test_calibration(257, 10, 2**8-2, 256, True, True, "cpu", float32)

    