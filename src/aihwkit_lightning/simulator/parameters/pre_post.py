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

# pylint: disable=too-many-instance-attributes

"""Pre-post processing related parameters for resistive processing units."""

from dataclasses import dataclass, field

from .helpers import _PrintableMixin


@dataclass
class InputRangeParameter(_PrintableMixin):
    """Parameter related to input range learning"""

    fast_mode: bool = False
    """Whether to use fast mode for input range learning."""

    act_range_momentum: float = 0.95
    """Momentum for the activation range."""

    enable: bool = field(default_factory=lambda: False, metadata={"always_show": True})
    """Whether to enable to learn the input range. Note that if enable is
    ``False`` then no clip is applied.

    Note:

        The input bound (``forward.inp_bound``) is assumed to be 1 if
        enabled as the input range already scales the input into to the
        range :math:`(-1, 1)` by dividing the input to the type by
        itself and multiplying the output accordingly.

        Typically, noise and bound management should be set to `NONE`
        for the input range learning as it replaces the dynamic
        managements with a static but learned input bound. However, in
        some exceptional experimental cases one might want to enable
        the management techniques on top of the input range learning,
        so that no error is raised if they are not set to `NONE`.
    """

    learn_input_range: bool = True
    """Whether to learn the input range when enabled.

    Note:

       If not learned, the input range should in general be set
       with some calibration method before training the DNN.

    """

    init_value: float = 3.0
    """Initial setting of the input range in case of input range learning."""

    init_from_data: int = 100
    """Number of batches to use for initialization from data. Set 0 to turn off."""

    init_std_alpha: float = 3.0
    """Standard deviation multiplier for initialization from data."""

    decay: float = 0.001
    """Decay rate for input range learning."""

    input_min_percentage: float = 0.95
    """Decay is only applied if percentage of non-clipped values is above this value.

    Note:

        The added gradient is (in case of non-clipped input
        percentage ``percentage > input_min_percentage``)::

            grad += decay * input_range
    """


@dataclass
class PrePostProcessingParameter(_PrintableMixin):
    """Parameter related to digital input and output processing, such as input clip
    learning.
    """

    input_range: InputRangeParameter = field(default_factory=InputRangeParameter)


@dataclass
class PrePostProcessingRPU(_PrintableMixin):
    """Defines the pre-post parameters and utility factories"""

    pre_post: PrePostProcessingParameter = field(default_factory=PrePostProcessingParameter)
    """Parameter related digital pre and post processing."""
