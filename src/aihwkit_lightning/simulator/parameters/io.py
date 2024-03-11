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

"""Forward related parameters for resistive processing units."""

from dataclasses import dataclass

from .helpers import _PrintableMixin


@dataclass
class IOParameters(_PrintableMixin):
    """Parameter that define the analog-matvec (forward / backward) and
    peripheral digital input-output behavior.

    Here one can enable analog-digital conversion, dynamic input
    scaling, and define the properties of the analog-matvec
    computations, such as noise and non-idealities (e.g. IR-drop).
    """

    inp_res: float = 1 / (2**8 - 2)
    r"""Number of discretization steps for DAC (:math:`\le0` means infinite steps)
    or resolution (1/steps)."""

    out_noise: float = 0.0
    r"""Output noise strength at each output of a tile.

    This sets the std-deviation of the Gaussian output noise
    (:math:`\sigma_\text{out}`) at each output, i.e. noisiness of
    device summation at the output.
    """
