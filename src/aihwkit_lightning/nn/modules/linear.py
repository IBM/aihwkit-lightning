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

"""Module class for analog linear layer."""
from typing import Optional, Tuple
from torch import Tensor
from torch.nn import Linear
from aihwkit_lightning.simulator.configs import TorchInferenceRPUConfig, WeightClipType
from aihwkit_lightning.nn.modules.base import AnalogLayerBase


class AnalogLinear(Linear, AnalogLayerBase):
    """Analog linear layer."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
        rpu_config: Optional[TorchInferenceRPUConfig] = None,
    ) -> None:
        if rpu_config is None:
            raise ValueError("rpu_config must be provided. Try TorchInferenceRPUConfig()")
        Linear.__init__(self, in_features, out_features, bias, device, dtype)
        self.rpu_config = rpu_config

    @classmethod
    def from_digital(cls, module: Linear, rpu_config: TorchInferenceRPUConfig) -> "AnalogLinear":
        """Return an AnalogLinear layer from a torch Linear layer.

        Args:
            module: The torch module to convert. All layers that are
                defined in the ``conversion_map``.
            rpu_config: RPU config to apply to all converted tiles.
                Applied to all converted tiles.

        Returns:
            an AnalogLinear layer based on the digital Linear ``module``.
        """
        analog_layer = cls(
            module.in_features,
            module.out_features,
            module.bias is not None,
            device=module.weight.device,
            dtype=module.weight.dtype,
            rpu_config=rpu_config,
        )

        analog_layer.set_weights_and_biases(
            module.weight.data, None if module.bias is None else module.bias.data
        )
        return analog_layer

    @classmethod
    def move_to_meta(cls, module: Linear):
        """Move the module to the meta class.

        This is used to move the module to the meta class. This is
        useful for the conversion of the module to analog.

        Args:
            module: The module to move to the meta class.

        """
        module.weight.data = module.weight.data.to(device="meta")
        if module.bias is not None:
            module.bias.data = module.bias.data.to(device="meta")

    def set_weights(self, weight: Tensor) -> None:
        """Set the weight tensor to the analog crossbar. Creates a copy of the tensors.

        Args:
            weight: the weight tensor
        """
        assert (
            self.weight.shape == weight.shape
        ), f"weight shape mismatch. Got {weight.shape}, expected {self.weight.shape}"
        self.weight.data = weight.detach().clone()

    def set_weights_and_biases(self, weight: Tensor, bias: Optional[Tensor] = None) -> None:
        """Set the weight (and bias) tensors to the analog crossbar. Creates a copy of the tensors.

        Args:
            weight: the weight tensor
            bias: the bias tensor is available
        """
        self.set_weights(weight)
        if bias is not None:
            assert (
                self.bias.shape == bias.shape
            ), f"bias shape mismatch. Got {bias.shape}, expected {self.bias.shape}"
            self.bias.data = bias.detach().clone()

    def get_weights_and_biases(self) -> Tuple[Tensor, Optional[Tensor]]:
        """Get the weight (and bias) tensors from the analog crossbar.

        Returns:
            tuple: weight matrix, bias vector
        """
        return (self.weight, self.bias)

    def clip_weights(self) -> None:
        """Clip the weights."""
        clip_type = self.rpu_config.clip.type
        clip_sigma = self.rpu_config.clip.sigma

        if clip_type == WeightClipType.NONE:
            return
        assert clip_sigma > 0, "Clip sigma must be greater than 0"
        sigma_std = clip_sigma * self.weight.std(
            None if clip_type == WeightClipType.LAYER_GAUSSIAN else 1, keepdim=True
        )
        if clip_type in [WeightClipType.LAYER_GAUSSIAN, WeightClipType.LAYER_GAUSSIAN_PER_CHANNEL]:
            self.weight.data = self.weight.data.clamp(-sigma_std, sigma_std)
        else:
            raise ValueError(f"Unknown clip type {clip_type}")
