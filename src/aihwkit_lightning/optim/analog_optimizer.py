# -*- coding: utf-8 -*-

# (C) Copyright 2026 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""Analog-aware inference optimizer."""

from types import new_class
from typing import Any, Dict, Type, Generator, Callable

from torch.optim import Optimizer

from aihwkit_lightning.nn import AnalogLayerBase


class AnalogOptimizer(Optimizer):
    """Generic optimizer that wraps an existing ``Optimizer`` for analog inference.

    This class wraps an existing ``Optimizer``, customizing the optimization
    step for triggering the analog update needed for analog tiles. All other
    (digital) parameters are governed by the given torch optimizer. In case of
    hardware-aware training (``InferenceTile``) the tile weight update is also
    governed by the given optimizer, otherwise it is using the internal analog
    update as defined in the ``rpu_config``.

    The ``AnalogOptimizer`` constructor expects the wrapped optimizer class as
    the first parameter, followed by any arguments required by the wrapped
    optimizer.

    Note:
        The instances returned are of a *new* type that is a subclass of:

        * the wrapped ``Optimizer`` (allowing access to all their methods and
          attributes).
        * this ``AnalogOptimizer``.

    Example:
        The following block illustrate how to create an optimizer that wraps
        standard SGD:

        >>> from torch.optim import SGD
        >>> from torch.nn import Linear
        >>> from aihwkit.simulator.configs.configs import InferenceRPUConfig
        >>> from aihwkit.optim import AnalogOptimizer
        >>> model = AnalogLinear(3, 4, rpu_config=InferenceRPUConfig)
        >>> optimizer = AnalogOptimizer(SGD, model.parameters(), lr=0.02)
    """

    SUBCLASSES = {}  # type: Dict[str, Type]
    """Registry of the created subclasses."""

    def __new__(cls, optimizer_cls: Type, *_: Any, **__: Any) -> "AnalogOptimizer":
        subclass_name = "{}{}".format(cls.__name__, optimizer_cls.__name__)

        # Retrieve or create a new subclass, that inherits both from
        # `AnalogOptimizer` and for the specific torch optimizer
        # (`optimizer_cls`).
        if subclass_name not in cls.SUBCLASSES:
            cls.SUBCLASSES[subclass_name] = new_class(subclass_name, (cls, optimizer_cls), {})

        return super().__new__(cls.SUBCLASSES[subclass_name])

    # pylint: disable=unused-argument
    def __init__(self, _: Type, analog_layers: Callable[[], Generator], *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

        def hook(*_: Any, **__: Any):
            for analog_layer in analog_layers():
                analog_layer: AnalogLayerBase  # type: ignore[no-redef]
                analog_layer.clip_weights()

        self.register_step_post_hook(hook)
