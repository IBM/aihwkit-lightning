# -*- coding: utf-8 -*-

# (C) Copyright 2026 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

# pylint: disable=too-many-instance-attributes

"""Mapping parameters for resistive processing units."""

from dataclasses import dataclass, field
from .helpers import _PrintableMixin


@dataclass
class MappingParameter(_PrintableMixin):
    """Parameter related to hardware design and the mapping of logical
    weight matrices to physical tiles.

    Caution:

        Some of these parameters have only an effect for modules that
        support tile mappings.
    """

    max_input_size: int = -1
    """Maximal input size of the weight matrix
    that is handled on a single analog tile.

    If the logical weight matrix size exceeds this size it will be
    split and mapped onto multiple analog tiles. -1 means no maximum input size.
    """


@dataclass
class MappableRPU(_PrintableMixin):
    """Defines the mapping parameters and utility factories"""

    mapping: MappingParameter = field(default_factory=MappingParameter)
    """Parameter related to mapping weights to tiles for supporting modules."""
