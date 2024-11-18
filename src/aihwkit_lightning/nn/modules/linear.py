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

"""Module class for analog linear layer."""
from typing import Optional, Tuple, List
import os
from logging import warning
from torch import Tensor, cuda, empty, zeros, tensor, int32
from torch.nn import Linear, Parameter
from aihwkit_lightning.simulator.configs import (
    TorchInferenceRPUConfig,
    WeightClipType,
    WeightModifierType,
)
from aihwkit_lightning.nn.modules.base import AnalogLayerBase
from aihwkit_lightning.nn.modules.torch_utils.torch_linear import TorchLinear


def is_at_least_volta_gpu():
    """Check if the GPU is at least Volta."""
    if cuda.is_available():
        gpu_properties = cuda.get_device_properties(0)
        if gpu_properties.major >= 7:
            return True
    return False


TRITON_AVAIL = False
try:
    from aihwkit_lightning.nn.modules.triton_utils.triton_linear import TritonLinear

    if not is_at_least_volta_gpu():
        raise ImportError("GPU must at least be Volta")
    TRITON_AVAIL = True
except ImportError:
    print("Could not import triton_utils.triton_linear. Using PyTorch variant.")


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

        max_input_size = rpu_config.mapping.max_input_size
        self.in_sizes = self.get_split_sizes(in_features, max_input_size)
        self.upper_end_of_slices = (
            tensor(self.in_sizes, device=device, dtype=dtype)
            .cumsum(dim=0, dtype=int32)
            .contiguous()
        )

        if rpu_config.pre_post.input_range.enable:
            # for every vertical tile, we have an input range
            self.input_range = Parameter(
                data=empty((len(self.in_sizes),), dtype=dtype, device=device).fill_(
                    rpu_config.pre_post.input_range.init_value
                ),
                requires_grad=rpu_config.pre_post.input_range.learn_input_range,
            )
            self.register_buffer(
                "input_range_update_idx",
                tensor=zeros((len(self.in_sizes),), dtype=dtype, device=device),
            )
            # needed for the fast mode
            self.register_buffer(
                "x_min", tensor=zeros((len(self.in_sizes),), dtype=dtype, device=device)
            )
            self.register_buffer(
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

    def forward(self, inp: Tensor) -> Tensor:  # pylint: disable=arguments-renamed
        """Forward function."""

        # pylint: disable=too-many-branches, too-many-statements, too-many-locals

        modified_weights = self.weight
        apply_weight_modifier = (
            self.training or self.rpu_config.modifier.enable_during_test
        ) and self.rpu_config.modifier.type != WeightModifierType.NONE
        if apply_weight_modifier:
            modified_weights = self.weight.clone()

        apply_out_quantization = self.rpu_config.forward.out_res > 0
        if apply_out_quantization:
            assert self.rpu_config.forward.out_bound > 0, "Out quant. without a bound."
            assert self.rpu_config.pre_post.input_range.enable, "Out quant. without IR."
        # apply_out_quantization entails out_bound > 0

        triton_enabled = os.environ.get("AIHWKIT_USE_TRITON", False)
        if triton_enabled and not TRITON_AVAIL:
            warning("AIHWKIT_USE_TRITON is set, but triton is not installed")
        if TRITON_AVAIL and triton_enabled:
            self.upper_end_of_slices = self.upper_end_of_slices.to(device=modified_weights.device)
            out = TritonLinear.apply(
                inp,
                modified_weights,
                self.input_range,
                self.input_range_update_idx,
                self.upper_end_of_slices,
                self.rpu_config,
                self.training,
                apply_weight_modifier,
            )
            return out + self.bias if self.bias is not None else out

        self.input_range_update_idx: Tensor  # type: ignore
        out = TorchLinear.linear(
            inp,
            modified_weights,
            self.bias,
            self.input_range,
            self.input_range_update_idx,
            self.x_min,
            self.x_max,
            self.in_sizes,
            self.training,
            self.rpu_config,
            apply_weight_modifier,
            apply_out_quantization,
        )
        return out

    def get_split_sizes(self, size: int, split_max_size: int) -> List[int]:
        """Computed the split sizes.

        Args:
            size: number of elements of the layer in one dimension
            split_max_size: max size of the split

        Returns:
            List of split sizes
        """
        if split_max_size <= 0:
            return [size]

        n_splits = (size + split_max_size - 1) // split_max_size
        base, extra = divmod(size, n_splits)
        return [base + (i < extra) for i in range(n_splits)]

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
    def to_digital(cls, module: "AnalogLinear") -> Linear:
        """Return an nn.Linear layer from an AnalogLinear layer.

        Args:
            module: The analog module to convert.

        Returns:
            a torch Linear layer with the same dimension and weights
            as the analog version.
        """
        digital_layer = Linear(module.in_features, module.out_features, module.bias is not None)
        digital_layer.weight.data = module.weight.data.detach().clone()
        if module.bias is not None:
            digital_layer.bias.data = module.bias.data.detach().clone()
        return digital_layer.to(device=module.weight.device, dtype=module.weight.dtype)

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
            self.weight.data.clamp_(-sigma_std, sigma_std)
        else:
            raise ValueError(f"Unknown clip type {clip_type}")

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super()._save_to_state_dict(destination, prefix, keep_vars)
        # pylint: disable=protected-access
        destination._metadata[prefix.split(".")[0]]["rpu_config"] = self.rpu_config

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )
        if "rpu_config" in local_metadata:
            self.rpu_config = local_metadata["rpu_config"]
