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

"""Configurations for resistive processing units."""

from aihwkit_lightning.simulator.parameters import (
    IOParameters,
    WeightModifierParameter,
    WeightClipParameter,
    MappingParameter,
    InputRangeParameter,
    PrePostProcessingParameter,
)
from aihwkit_lightning.simulator.parameters.enums import (
    WeightModifierType,
    WeightNoiseInjectionType,
    WeightQuantizationType,
    WeightClipType,
)
from .configs import TorchInferenceRPUConfig
