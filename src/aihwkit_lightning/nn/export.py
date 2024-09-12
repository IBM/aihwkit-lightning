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
# mypy: disable-error-code="arg-type, attr-defined"

"""
Export utility for converting AIHWKIT Lightning models to AIHWKIT models.
"""

import warnings
import re
from torch import no_grad, float32, float16
from torch import dtype as torch_dtype
from torch import device as torch_device

from aihwkit.simulator.parameters.enums import RPUDataType
from aihwkit.simulator.configs import TorchInferenceRPUConfig as AIHWKITRPUConfig
from aihwkit.nn.conversion import convert_to_analog as aihwkit_convert_to_analog
from aihwkit.nn import AnalogWrapper as AIHWKITAnalogWrapper
from aihwkit.simulator.tiles.inference import TileWithPeriphery
from aihwkit.simulator.configs import (
    NoiseManagementType as AIHWKITNoiseManagementType,
    BoundManagementType as AIHWKITBoundManagementType,
    WeightClipType as AIHWKITWeightClipType,
    WeightModifierType as AIHWKITWeightModifierType,
    WeightRemapType as AIHWKITWeightRemapType,
)

from aihwkit_lightning.nn.conversion import convert_to_digital
from aihwkit_lightning.nn import AnalogLayerBase
from aihwkit_lightning.nn.modules.container import AnalogWrapper
from aihwkit_lightning.simulator.configs import WeightClipType, WeightModifierType


def base_aihwkit_rpu_config() -> AIHWKITRPUConfig:
    """Construct a 'generic' base AIHWKIT rpu configuration."""
    rpu_config = AIHWKITRPUConfig()
    rpu_config.clip.type = AIHWKITWeightClipType.NONE
    rpu_config.forward.noise_management = AIHWKITNoiseManagementType.NONE
    rpu_config.forward.bound_management = AIHWKITBoundManagementType.NONE
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


@no_grad()
def export_to_aihwkit(model: AnalogWrapper, max_output_size: int = -1) -> AIHWKITAnalogWrapper:
    """Export a AIHWKITLighting model to an AIHWKIT model."""
    modifier_name_conversion = {
        "DiscretizePerChannel": "Discretize",
        "AddNormalPerChannel": "AddNormal",
        "DiscretizeAddNormalPerChannel": "DiscretizeAddNormal",
    }

    dtype_map = {float32: RPUDataType.FLOAT, float16: RPUDataType.HALF}

    analog_layer: AnalogLayerBase
    assert hasattr(model, "analog_layers"), "Model does not have analog layers."
    analog_layer = next(model.analog_layers())  # pylint: disable=not-callable
    assert analog_layer is not None, "No RPUConfig found in the model."
    rpu_config = analog_layer.rpu_config

    # Get dtype
    dtype: torch_dtype
    dtype = next(model.parameters()).dtype
    if dtype not in dtype_map:
        raise NotImplementedError(f"dtype {dtype} not implemented.")

    # Get device
    device: torch_device
    device = next(model.parameters()).device

    training = analog_layer.training
    aihwkit_rpu_config = base_aihwkit_rpu_config()

    # Assign correct dtype
    aihwkit_rpu_config.runtime.data_type = dtype_map[dtype]

    # Will decide if we remap the weights per channel or not
    remap_per_channel = False

    # Clipping
    clip_name = rpu_config.clip.type.name
    if rpu_config.clip.type == WeightClipType.LAYER_GAUSSIAN_PER_CHANNEL:
        remap_per_channel = True
        clip_name = WeightClipType.LAYER_GAUSSIAN.name
    aihwkit_rpu_config.clip.type = AIHWKITWeightClipType[clip_name]
    aihwkit_rpu_config.clip.sigma = rpu_config.clip.sigma

    # Weight modifier
    modifier_name = rpu_config.modifier.type.name
    if rpu_config.modifier.type.name in modifier_name_conversion:
        remap_per_channel = True
        modifier_name = modifier_name_conversion[rpu_config.modifier.type.name]
    aihwkit_rpu_config.modifier.type = AIHWKITWeightModifierType[modifier_name]
    if not rpu_config.modifier.type == WeightModifierType.NONE:
        warnings.warn(f"The weight modifier is active and set to {modifier_name}")
    aihwkit_rpu_config.modifier.std_dev = rpu_config.modifier.std_dev
    aihwkit_rpu_config.modifier.res = rpu_config.modifier.res
    aihwkit_rpu_config.modifier.enable_during_test = rpu_config.modifier.enable_during_test
    aihwkit_rpu_config.modifier.assumed_wmax = 1.0

    # Output noise
    if rpu_config.forward.out_noise > 0.0:
        warnings.warn("The output_noise is active")
    aihwkit_rpu_config.forward.out_noise = rpu_config.forward.out_noise
    if rpu_config.forward.out_noise_per_channel:
        remap_per_channel = True

    # Mapping
    if remap_per_channel:
        aihwkit_rpu_config.remap.type = AIHWKITWeightRemapType.CHANNELWISE_SYMMETRIC
    else:
        aihwkit_rpu_config.remap.type = AIHWKITWeightRemapType.LAYERWISE_SYMMETRIC
    aihwkit_rpu_config.mapping.max_input_size = rpu_config.mapping.max_input_size
    aihwkit_rpu_config.mapping.max_output_size = max_output_size
    aihwkit_rpu_config.mapping.weight_scaling_columnwise = remap_per_channel
    aihwkit_rpu_config.mapping.weight_scaling_omega = 1.0

    # Forward - Input quantization
    aihwkit_rpu_config.forward.inp_res = rpu_config.forward.inp_res
    if rpu_config.forward.inp_res > 0:
        assert (
            rpu_config.pre_post.input_range.enable
        ), "Input range must be enabled for input resolution"
    else:
        if not rpu_config.pre_post.input_range.enable:
            warnings.warn("Input resolution is not set and input range is not enabled")
            aihwkit_rpu_config.forward.noise_management = AIHWKITNoiseManagementType.ABS_MAX
        else:
            warnings.warn("Input resolution is not set, but still input range")

    # Forward - Output quantization
    aihwkit_rpu_config.forward.out_res = rpu_config.forward.out_res
    aihwkit_rpu_config.forward.out_bound = rpu_config.forward.out_bound
    if rpu_config.forward.out_res > 0:
        assert rpu_config.forward.out_bound > 0, "Output bound must be set for output resolution"

    # Input range learning
    aihwkit_rpu_config.pre_post.input_range.enable = rpu_config.pre_post.input_range.enable
    aihwkit_rpu_config.pre_post.input_range.decay = rpu_config.pre_post.input_range.decay
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

    # Loading model to CPU...
    model = model.to(device=torch_device("cpu"))

    # Extract input ranges if any
    input_ranges = {}  # string: Tensor (1D or num_tiles)
    if rpu_config.pre_post.input_range.enable:
        for name, analog_layer in model.named_analog_layers():
            input_ranges[name] = analog_layer.input_range

    # Convert to AIHWKIT model
    aihwkit_model = aihwkit_convert_to_analog(
        convert_to_digital(model), rpu_config=aihwkit_rpu_config
    )

    model = model.to(device=device, dtype=dtype)
    aihwkit_model = aihwkit_model.to(device=device, dtype=dtype)

    # Set remap weights
    aihwkit_model.remap_analog_weights()  # pylint: disable=not-callable

    # Set input ranges
    aihwkit_model: AIHWKITAnalogWrapper  # type: ignore[no-redef]
    analog_tile: TileWithPeriphery
    for analog_tile_name, analog_tile in aihwkit_model.named_analog_tiles():
        module_pattern = r"analog_module\.array\.(\d+)"
        analog_name_pattern = r"^(.*?)\.analog_module"

        module_match = re.search(module_pattern, analog_tile_name)
        analog_name_match = re.search(analog_name_pattern, analog_tile_name)

        assert module_match is not None, f"Could not find tile id in {analog_tile_name}"
        assert analog_name_match is not None, f"Could not find module name in {analog_tile_name}"

        tile_id_along_input_dim = int(module_match.group(1))
        module_name = analog_name_match.group(1)

        input_range = input_ranges[module_name][tile_id_along_input_dim]
        analog_tile.input_range.data = input_range.clone().view_as(analog_tile.input_range)

    # Set back to training if needed
    aihwkit_model = aihwkit_model.train() if training else aihwkit_model.eval()
    return aihwkit_model
