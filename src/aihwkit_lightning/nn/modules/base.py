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

"""Base class for adding functionality to analog layers."""
from typing import Tuple, Generator, Callable


class AnalogLayerBase:
    """Mixin that adds functionality on the layer level.

    In general, the defined methods will be looped for all analog tile
    modules and delegate the function.
    """

    IS_CONTAINER: bool = False
    """Class constant indicating whether sub-layers exist or whether
    this layer is a leave node (that is only having tile modules)"""

    # pylint: disable=no-member

    def apply_to_analog_layers(self, fn: Callable) -> "AnalogLayerBase":
        """Apply a function to all the analog layers.

        Note:
            Here analog layers are all sub modules of the current
            module that derive from ``AnalogLayerBase`` (such as
            ``AnalogLinear``) _except_ ``AnalogSequential``.

        Args:
            fn: function to be applied.

        Returns:
            This module after the function has been applied.

        """
        for _, module in self.named_analog_layers():
            fn(module)

        return self

    def analog_layers(self) -> Generator["AnalogLayerBase", None, None]:
        """Generator over analog layers only.

        Note:
            Here analog layers are all sub modules of the current module that
            derive from ``AnalogLayerBase`` (such as ``AnalogLinear``)
            _except_ ``AnalogSequential``.
        """
        for _, layer in self.named_analog_layers():  # type: ignore
            yield layer

    def named_analog_layers(self) -> Generator[Tuple[str, "AnalogLayerBase"], None, None]:
        """Generator over analog layers only.

        Note:
            Here analog layers are all sub-modules of the current
            module that derive from ``AnalogLayerBase`` (such as
            ``AnalogLinear``) _except_ those that are containers
            (`IS_CONTAINER=True`) such as ``AnalogSequential``.

        """
        for name, layer in self.named_modules():  # type: ignore
            if isinstance(layer, AnalogLayerBase) and not layer.IS_CONTAINER:
                yield name, layer

    @classmethod
    def move_to_meta(cls, _):
        """Move the module to the meta class.

        This is used to move the module to the meta class. This is
        useful for the conversion of the module to analog.
        """
        return

    def clip_weights(self) -> None:
        """Clip the weights of the analog layers."""
        return
