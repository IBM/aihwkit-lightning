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

# pylint: disable=too-many-instance-attributes

"""Utility enumerators for resistive processing units."""

from enum import Enum


class WeightModifierType(Enum):
    """Weight modifier type."""

    NONE = "None"
    """No weight modifier. Nothing happens to the weight. """

    DISCRETIZE = "Discretize"
    """Quantize the weights."""

    ADD_NORMAL = "AddNormal"
    """Additive Gaussian noise."""

    ADD_NORMAL_PER_CHANNEL = "AddNormalPerChannel"
    """Additive Gaussian noise per channel."""

    DISCRETIZE_ADD_NORMAL = "DiscretizeAddNormal"
    """First discretize and then additive Gaussian noise."""


class WeightClipType(Enum):
    """Weight clipper type."""

    NONE = "None"
    """None."""

    LAYER_GAUSSIAN = "LayerGaussian"
    """Calculates the second moment of the whole weight matrix and clips
    at ``sigma`` times the result symmetrically around zero."""

    LAYER_GAUSSIAN_PER_CHANNEL = "LayerGaussianPerChannel"
    """Calculates the second moment of the whole weight matrix per output column and clips
    at ``sigma`` times the result symmetrically around zero."""
