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

import os
import argparse
from functools import reduce
import torch
import torch.nn as nn
from torch.nn import Parameter, Module
import torch.nn.functional as F

from aihwkit_lightning.nn.modules.container import AnalogWrapper
from aihwkit_lightning.nn.conversion import convert_to_digital
from aihwkit.simulator.parameters.enums import RPUDataType
from aihwkit.simulator.configs import TorchInferenceRPUConfig as AIHWKITRPUConfig
from aihwkit.nn.conversion import convert_to_analog as AIHWKITconvert_to_analog
from aihwkit.simulator.configs import (
    NoiseManagementType,
    BoundManagementType,
    WeightClipType,
    WeightModifierType,
    WeightRemapType,
)

DTYPE_MAP = {torch.float32: RPUDataType.FLOAT, torch.float16: RPUDataType.HALF}


def base_aihwkit_rpu_config() -> AIHWKITRPUConfig:
    """Construct a 'generic' base AIHWKIT rpu configuration."""
    rpu_config = AIHWKITRPUConfig()
    rpu_config.clip.type = WeightClipType.NONE
    rpu_config.forward.noise_management = NoiseManagementType.NONE
    rpu_config.forward.bound_management = BoundManagementType.NONE
    rpu_config.forward.inp_res = -1
    rpu_config.forward.out_res = -1
    rpu_config.forward.inp_bound = -1
    rpu_config.forward.out_bound = -1
    rpu_config.forward.out_noise = 0.0
    rpu_config.mapping.max_input_size = -1
    rpu_config.mapping.max_output_size = -1
    rpu_config.mapping.digital_bias = True
    rpu_config.mapping.out_scaling_columnwise = False
    rpu_config.mapping.weight_scaling_columnwise = True
    rpu_config.mapping.weight_scaling_lr_compensation = False
    return rpu_config


def get_module_by_name(module: Module, access_string: str) -> Module:
    """Get a torch module by name from a base module."""
    if access_string == "":
        return module
    else:
        names = access_string.split(sep=".")
        return reduce(getattr, names, module)


def export_to_aihwkit(module: Module):
    """Export a AIHWKITLighting model to an AIHWKIT model."""
    rpu_config = next(module.analog_layers(), None)
    assert rpu_config is not None
    rpu_config = rpu_config.rpu_config
    dtype = next(module.parameters()).dtype
    training = module.training
    aihwkit_rpu_config = base_aihwkit_rpu_config()
    if dtype not in DTYPE_MAP:
        raise NotImplementedError("dtype %s not implemented." % dtype)

    aihwkit_rpu_config.runtime.data_type = DTYPE_MAP[dtype]
    aihwkit_rpu_config.clip.type = WeightClipType[rpu_config.clip.type.name]
    aihwkit_rpu_config.clip.sigma = rpu_config.clip.sigma
    aihwkit_rpu_config.forward.inp_res = rpu_config.forward.inp_res
    aihwkit_rpu_config.forward.out_noise = rpu_config.forward.out_noise
    if rpu_config.forward.out_noise_per_channel:
        raise NotImplementedError

    aihwkit_rpu_config.remap.type = WeightRemapType.CHANNELWISE_SYMMETRIC
    aihwkit_rpu_config.mapping.max_input_size = rpu_config.mapping.max_input_size
    aihwkit_rpu_config.mapping.weight_scaling_columnwise = True
    aihwkit_rpu_config.mapping.weight_scaling_omega = 1.0
    aihwkit_rpu_config.modifier.assumed_wmax = 1.0
    aihwkit_rpu_config.modifier.enable_during_test = rpu_config.modifier.enable_during_test
    aihwkit_rpu_config.modifier.res = rpu_config.modifier.res
    aihwkit_rpu_config.modifier.std_dev = rpu_config.modifier.std_dev
    aihwkit_rpu_config.modifier.type = WeightModifierType[aihwkit_rpu_config.modifier.type.name]
    aihwkit_rpu_config.pre_post.input_range.decay = rpu_config.pre_post.input_range.decay
    aihwkit_rpu_config.pre_post.input_range.enable = rpu_config.pre_post.input_range.enable
    aihwkit_rpu_config.pre_post.input_range.init_from_data = (
        rpu_config.pre_post.input_range.init_from_data
    )
    aihwkit_rpu_config.pre_post.input_range.init_std_alpha = (
        rpu_config.pre_post.input_range.init_std_alpha
    )
    aihwkit_rpu_config.pre_post.input_range.init_value = rpu_config.pre_post.input_range.init_value
    aihwkit_rpu_config.pre_post.input_range.input_min_percentage = (
        rpu_config.pre_post.input_range.input_min_percentage
    )
    aihwkit_rpu_config.pre_post.input_range.learn_input_range = (
        rpu_config.pre_post.input_range.learn_input_range
    )
    module.clip_weights()
    module.eval()
    input_ranges = {}
    for name, module_ in module.named_modules():
        if hasattr(module_, "input_range"):
            input_ranges[name] = module_.input_range

    module = AIHWKITconvert_to_analog(convert_to_digital(module), aihwkit_rpu_config)
    module.to(dtype=dtype)
    module.remap_analog_weights()
    for name, input_range in input_ranges.items():
        module_ = get_module_by_name(module, name)
        assert hasattr(module_, "analog_module")
        for tile_id, tile in enumerate(module_.analog_tiles()):
            tile.input_range = Parameter(
                input_range[tile_id].to(dtype=tile.input_range.dtype), requires_grad=False
            )

    if training:
        module.train()
    else:
        module.eval()

    return module
