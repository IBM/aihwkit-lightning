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
# pylint: disable=too-many-lines

"""Inference related parameters for resistive processing units."""

from dataclasses import dataclass, field

from aihwkit_lightning.simulator.parameters.helpers import _PrintableMixin
from aihwkit_lightning.simulator.parameters.enums import WeightModifierType, WeightClipType


@dataclass
class WeightModifierParameter(_PrintableMixin):
    """Parameter that modify the forward/backward weights during hardware-aware training."""

    std_dev: float = 0.0
    """Standard deviation of the added noise to the weight matrix.

    This parameter affects the modifier types ``AddNormal``, ``MultNormal`` and
    ``DiscretizeAddNormal``.

    Note:
        If the parameter ``rel_to_actual_wmax`` is set then the ``std_dev`` is
        computed in relative terms to the abs max of the given weight matrix,
        otherwise it in relative terms to the assumed max, which is set by
        ``assumed_wmax``.
    """

    res: float = 0.0
    r"""Resolution of the discretization.

    For example, for 8 bits specify as 2**8-2 or the inverse.

    ``res`` is only used in the modifier types ``Discretize`` and
    ``DiscretizeAddNormal``.
    """

    enable_during_test: bool = False
    """Whether to use the last modified weight matrix during testing.

    Caution:
        This will **not** remove drop connect or any other noise
        during evaluation, and thus should only used with care.
    """

    type: WeightModifierType = field(
        default_factory=lambda: WeightModifierType.NONE, metadata={"always_show": True}
    )
    """Type of the weight modification."""


@dataclass
class WeightClipParameter(_PrintableMixin):
    """Parameter that clip the weights during hardware-aware training.

    Important:
        A clipping ``type`` has to be set before any of the parameter
        changes take any effect.

    """

    sigma: float = 2.5
    """Sigma value for clipping for the ``LayerGaussian`` type."""

    type: WeightClipType = field(
        default_factory=lambda: WeightClipType.NONE, metadata={"always_show": True}
    )
    """Type of clipping."""
