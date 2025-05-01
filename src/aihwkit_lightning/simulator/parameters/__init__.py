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

"""RPU simulator bindings."""

# This import is required in order to load the `torch` shared libraries, which
# the simulator shared library is linked against.

from .enums import WeightNoiseInjectionType, WeightQuantizationType, WeightClipType

from .io import IOParameters

from .mapping import MappingParameter

from .pre_post import InputRangeParameter, PrePostProcessingParameter

from .inference import WeightModifierParameter, WeightClipParameter
