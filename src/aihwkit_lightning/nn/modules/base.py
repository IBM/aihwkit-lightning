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
from typing import Tuple, Generator, Callable, List, Any
from torch import dtype as torch_dtype
from torch import device as torch_device
from torch.nn import Parameter
from torch import Tensor, empty, zeros
from ...simulator.configs import TorchInferenceRPUConfig
from ...simulator.parameters import WeightClipType


class AnalogLayerBase:
    """Mixin that adds functionality on the layer level.

    In general, the defined methods will be looped for all analog tile
    modules and delegate the function.
    """

    rpu_config: TorchInferenceRPUConfig
    in_sizes: List[int]
    register_buffer: Any

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

    def init_learnable_weight_ranges(
        self, init_value: Tensor, device: torch_device, dtype: torch_dtype
    ) -> None:
        """Initializes parameters for learnable weight ranges."""

        # pylint: disable=attribute-defined-outside-init

        clip_config = self.rpu_config.clip
        if clip_config.type == WeightClipType.LEARNABLE_PER_CHANNEL:
            # do the init. take the weight sharding into consideration as well
            self.learnable_weight_clip = Parameter(
                data=init_value.to(dtype=dtype, device=device), requires_grad=True
            )
        else:
            self.learnable_weight_clip = None  # type: ignore[assignment]

    def init_ir(
        self,
        device: torch_device,
        dtype: torch_dtype,
        init_value_ir: float,
        init_value_counter: int = 0,
    ) -> None:
        """Initialize input range parameters."""

        # pylint: disable=attribute-defined-outside-init

        ir_config = self.rpu_config.pre_post.input_range
        if ir_config.enable and not ir_config.dynamic:
            # for every vertical tile, we have an input range
            self.input_range = Parameter(
                data=empty((len(self.in_sizes),), dtype=dtype, device=device).fill_(init_value_ir),
                requires_grad=self.rpu_config.pre_post.input_range.learn_input_range,
            )
            self.register_buffer(  # type: ignore[call-arg]
                "input_range_update_idx",
                tensor=empty((len(self.in_sizes),), dtype=dtype, device=device).fill_(
                    init_value_counter
                ),
            )
            # needed for the fast mode
            self.register_buffer(  # type: ignore[call-arg]
                "x_min", tensor=zeros((len(self.in_sizes),), dtype=dtype, device=device)
            )
            self.register_buffer(  # type: ignore[call-arg]
                "x_max", tensor=zeros((len(self.in_sizes),), dtype=dtype, device=device)
            )
            self.x_min: Tensor
            self.x_min -= 1e-5
            self.x_max: Tensor
            self.x_max += 1e-5
        else:
            self.input_range = None  # type: ignore
            self.input_range_update_idx = None
            self.x_min = None  # type: ignore
            self.x_max = None  # type: ignore
