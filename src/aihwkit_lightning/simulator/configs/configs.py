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

"""Configurations for resistive processing units."""

# pylint: disable=too-few-public-methods

from dataclasses import dataclass, field
from aihwkit_lightning.simulator.parameters import (
    IOParameters,
    WeightClipParameter,
    WeightModifierParameter,
)


@dataclass
class TorchInferenceRPUConfig:
    """Configuration for an analog tile that is used only for inference.

    Training is done in *hardware-aware* manner, thus using only the
    non-idealities of the forward-pass, but backward and update passes
    are ideal.

    During inference, statistical models of programming, drift
    and read noise can be used.
    """

    # pylint: disable=too-many-instance-attributes

    forward: IOParameters = field(
        default_factory=IOParameters, metadata=dict(bindings_include=True)
    )
    """Input-output parameter setting for the forward direction.

    This parameters govern the hardware definitions specifying analog
    MVM non-idealities.

    Note:

        This forward pass is applied equally in training and
        inference. In addition, materials effects such as drift and
        programming noise can be enabled during inference by
        specifying the ``noise_model``
    """

    clip: WeightClipParameter = field(default_factory=WeightClipParameter)
    """Parameter for weight clip.

    If a clipping type is set, the weights are clipped according to
    the type specified.

    Caution:

        The clipping type is set to ``None`` by default, setting
        parameters of the clipping will not be taken into account, if
        the clipping type is not specified.
    """

    modifier: WeightModifierParameter = field(default_factory=WeightModifierParameter)

    """Parameter for weight modifier.

    If a modifier type is set, it is called once per mini-match in the
    ``post_update_step`` and modifies the weight in forward and
    backward direction for the next mini-batch during training, but
    updates hidden reference weights. In eval mode, the reference
    weights are used instead for forward.

    The modifier is used to do hardware-aware training, so that the
    model becomes more noise robust during inference (e.g. when the
    ``noise_model`` is employed).
    """
