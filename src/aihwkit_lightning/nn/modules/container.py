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

"""Analog Modules that contain children Modules."""

# pylint: disable=unused-argument, arguments-differ
from types import new_class
from typing import Any, Optional, Dict, Type
from collections import OrderedDict

from torch.nn import Sequential, Module

from aihwkit_lightning.nn.modules.base import AnalogLayerBase


class AnalogContainerBase(AnalogLayerBase):
    """Base class for analog containers."""

    IS_CONTAINER: bool = True


class AnalogSequential(AnalogContainerBase, Sequential):
    """An analog-aware sequential container.

    Specialization of torch ``nn.Sequential`` with extra functionality for
    handling analog layers:

    * apply analog-specific functions to all its children (drift and program
      weights).

    Note:
        This class is recommended to be used in place of ``nn.Sequential`` in
        order to correctly propagate the actions to all the children analog
        layers. If using regular containers, please be aware that operations
        need to be applied manually to the children analog layers when needed.
    """

    @classmethod
    def from_digital(cls, module: Sequential, *args: Any, **kwargs: Any) -> "AnalogSequential":
        """Construct AnalogSequential in-place from Sequential."""
        return cls(OrderedDict(mod for mod in module.named_children()))

    @classmethod
    def to_digital(cls, module: "AnalogSequential", *args: Any, **kwargs: Any) -> Sequential:
        """Construct Sequential in-place from AnalogSequential."""
        return Sequential(OrderedDict(mod for mod in module.named_children()))


class AnalogWrapper(AnalogContainerBase):
    """Generic wrapper over an given Module.

    Will add the AnalogLayerBase functionality to the given Module
    (as an added subclass).

    Note:
        Here the state dictionary of the give module will be simply
        copied by reference. The original model therefore should not
        be used any more as the underlying tensor data is shared.

    Args:
         model: model to wrap with the analog wrapper.
    """

    SUBCLASSES = {}  # type: Dict[str, Type]
    """Registry of the created subclasses."""

    def __new__(cls, module: Optional[Module] = None, **__: Any) -> "AnalogWrapper":
        if module is None:
            # for deepcopy and the like
            return super().__new__(cls)

        module_cls = module.__class__
        subclass_name = "{}{}".format(cls.__name__, module_cls.__name__)

        # Retrieve or create a new subclass, that inherits both from
        # `AnalogModuleBase` and for the specific torch module
        # (`module_cls`).
        if subclass_name not in cls.SUBCLASSES:
            cls.SUBCLASSES[subclass_name] = new_class(subclass_name, (cls, module_cls), {})

        return super().__new__(cls.SUBCLASSES[subclass_name])

    def __init__(self, module: Module):
        self.__dict__.update(module.__dict__)

    @classmethod
    def from_digital(cls, module: Module, *args: Any, **kwargs: Any) -> "AnalogWrapper":
        """Construct AnalogSequential in-place from any module."""
        return cls(module)

    @classmethod
    def to_digital(cls, module: "AnalogWrapper", *args: Any, **kwargs: Any) -> Module:
        """Construct Sequential in-place from AnalogSequential."""
        digital_class = module.__class__.__bases__[1]
        new_module = digital_class.__new__(digital_class)  # type: ignore
        new_module.__dict__ = module.__dict__
        return new_module

    def quantize_weights(self):
        """Iterate through analog layers in model and quantize weights in-place."""
        for layer in self.analog_layers():
            layer.quantize_weights()
