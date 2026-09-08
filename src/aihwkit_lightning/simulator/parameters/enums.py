# -*- coding: utf-8 -*-

# (C) Copyright 2026 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

# pylint: disable=too-many-instance-attributes

"""Utility enumerators for resistive processing units."""

from enum import Enum
from typing_extensions import deprecated


@deprecated("You are using AIHWKIT-Lightning version <2. Consider upgrading to v2.")
class WeightModifierType(Enum):
    """Weight modifier type."""

    NONE = "None"
    """No weight modifier. Nothing happens to the weight. """

    DISCRETIZE = "Discretize"
    """Quantize the weights."""

    DISCRETIZE_PER_CHANNEL = "DiscretizePerChannel"
    """Quantize the weights per channel."""

    ADD_NORMAL = "AddNormal"
    """Additive Gaussian noise."""

    ADD_NORMAL_PER_CHANNEL = "AddNormalPerChannel"
    """Additive Gaussian noise per channel."""

    DISCRETIZE_ADD_NORMAL = "DiscretizeAddNormal"
    """First discretize and then additive Gaussian noise."""

    DISCRETIZE_ADD_NORMAL_PER_CHANNEL = "DiscretizeAddNormalPerChannel"
    """First discretize per channel and then additive Gaussian noise per channel."""


class WeightNoiseInjectionType(Enum):
    """Weight modifier type."""

    NONE = "None"
    """No weight modifier. Nothing happens to the weight. """

    ADD_NORMAL = "AddNormal"
    """Additive Gaussian noise."""

    ADD_NORMAL_PER_CHANNEL = "AddNormalPerChannel"
    """Additive Gaussian noise per channel."""


class WeightQuantizationType(Enum):
    """Weight quantization type."""

    NONE = "None"
    """No weight quantizer. Nothing happens to the weight. """

    DISCRETIZE = "Discretize"
    """Quantize the weights."""

    DISCRETIZE_PER_CHANNEL = "DiscretizePerChannel"
    """Quantize the weights per channel."""


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

    LEARNABLE_PER_CHANNEL = "LearnablePerChannel"
    """Implements ParetoQ-like learnable weight clipping (originally LSQ)
    see eq. 3 from https://arxiv.org/pdf/1902.08153. Implements this on a
    per-channel basis."""
