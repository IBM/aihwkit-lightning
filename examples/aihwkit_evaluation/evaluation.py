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

"""Example on how to evaluate an AIHWKIT-Lightning model with AIHWKIT."""

import torch
from aihwkit.inference.noise.pcm import PCMLikeNoiseModel
from aihwkit.inference.compensation.drift import GlobalDriftCompensation
from model import resnet32
from aihwkit_lightning.simulator.configs import TorchInferenceRPUConfig
from aihwkit_lightning.nn.conversion import convert_to_analog
from aihwkit_lightning.nn.export import export_to_aihwkit

if __name__ == "__main__":
    model = resnet32()
    rpu_config = TorchInferenceRPUConfig()
    model = convert_to_analog(model, rpu_config)
    aihwkit_model = export_to_aihwkit(model=model, max_output_size=-1)
    aihwkit_model.to(torch.float32)
    for analog_tile in aihwkit_model.analog_tiles():
        new_rpu_config = analog_tile.rpu_config
        break

    new_rpu_config.noise_model = PCMLikeNoiseModel(g_max=25.0)
    new_rpu_config.drift_compensation = GlobalDriftCompensation()
    aihwkit_model.replace_rpu_config(new_rpu_config)
    aihwkit_model.eval()
    aihwkit_model.drift_analog_weights(0.0)
