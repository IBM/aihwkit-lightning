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
from copy import deepcopy
from logging import warning
from torch import Tensor, cuda, no_grad, tensor, int32
from torch.nn import Linear

from .base import AnalogLayerBase
from .torch_utils.quant_utils import clip_and_quantize
from .torch_utils.torch_abs_max import sliced_abs_max
from .torch_utils.torch_linear import TorchLinear
from ...simulator.configs import TorchInferenceRPUConfig, WeightClipType


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

    if not os.environ.get("TRITON_INTERPRET", None) == "1":
        # we are not in interpret mode
        if not is_at_least_volta_gpu():
            raise ImportError("GPU must at least be Volta")
    TRITON_AVAIL = True
except ImportError:
    print("Could not import triton_utils.triton_linear. Using PyTorch variant.")
except RuntimeError as e:
    if str(e) != "0 active drivers ([]). There should only be one.":
        raise RuntimeError(e) from e


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

        if str(device) == "meta":
            # init these small values on cpu
            device = "cpu"

        max_input_size = rpu_config.mapping.max_input_size
        self.in_sizes = self.get_split_sizes(in_features, max_input_size)
        self.upper_end_of_slices = (
            tensor(self.in_sizes, device=device, dtype=dtype)
            .cumsum(dim=0, dtype=int32)
            .contiguous()
        )

        self.init_ir(
            init_value_ir=self.rpu_config.pre_post.input_range.init_value,
            init_value_counter=0,
            device=device,
            dtype=dtype,
        )

        self.init_learnable_weight_ranges(
            init_value=sliced_abs_max(
                upper_end_of_slices=self.upper_end_of_slices, weights=self.weight
            ),
            device=device,
            dtype=dtype,
        )

        self.deprecation_adjustment()

    def forward(self, inp: Tensor) -> Tensor:  # pylint: disable=arguments-renamed
        """Forward function."""

        # pylint: disable=too-many-branches, too-many-statements, too-many-locals

        modified_weights = self.weight
        apply_out_quantization = self.rpu_config.forward.out_res > 0
        if apply_out_quantization:
            assert self.rpu_config.forward.out_bound > 0, "Out quant. without a bound."
            assert self.rpu_config.pre_post.input_range.enable, "Out quant. without IR."

        if self.rpu_config.clip.type == WeightClipType.LEARNABLE_PER_CHANNEL:
            assert (
                self.learnable_weight_clip is not None
            ), "Learnable weight clipping tensor not initialized."
            assert self.learnable_weight_clip.size(0) == len(self.upper_end_of_slices), (
                "Learnable weight clipping tensor must have the same number of rows as slices"
                " you have for the weight matrix."
            )
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
                self.input_range_delta,
                self.learnable_weight_clip,
                self.upper_end_of_slices,
                self.rpu_config,
                self.training,
            )
            return out + self.bias if self.bias is not None else out

        # pylint: disable=attribute-defined-outside-init
        self.input_range_update_idx: Tensor  # type: ignore
        out = TorchLinear.linear(
            inp,
            modified_weights,
            self.bias,
            self.input_range,
            self.input_range_update_idx,
            self.input_range_delta,
            self.x_min,
            self.x_max,
            self.learnable_weight_clip,
            self.in_sizes,
            self.training,
            self.rpu_config,
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

    @classmethod
    def move_to_meta(cls, module: Linear):
        """Move the module to the meta class.

        This is used to move the module to the meta class. This is
        useful for the conversion of the module to analog.

        Args:
            module: The module to move to the meta class.

        """
        module.to(device="meta")

    def set_weights(self, weight: Tensor) -> None:
        """Set the weight tensor to the analog crossbar. Creates a copy of the tensors.

        Args:
            weight: the weight tensor
        """
        assert (
            self.weight.shape == weight.shape
        ), f"weight shape mismatch. Got {weight.shape}, expected {self.weight.shape}"
        self.weight.data = weight.detach().clone()
        self.init_learnable_weight_ranges(
            init_value=sliced_abs_max(
                upper_end_of_slices=self.upper_end_of_slices, weights=self.weight
            ),
            device=weight.device,
            dtype=weight.dtype,
        )

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
        """
        Clip the weights.

        > Note: For DeepSpeed, we clamp both the fp16 and fp32 base weights.
            This is important when mixed precision (satge 1-3) is used.
        """
        clip_type = self.rpu_config.clip.type
        clip_sigma = self.rpu_config.clip.sigma

        if clip_type in [WeightClipType.NONE, WeightClipType.LEARNABLE_PER_CHANNEL]:
            return

        deepspeed = False
        if hasattr(self.weight, "ds_id") or hasattr(self.weight, "_hp_mapping"):
            from deepspeed.utils import safe_get_full_fp32_param, safe_set_full_fp32_param
            deepspeed = True

        assert clip_sigma > 0, "Clip sigma must be greater than 0"
        sigma_std = clip_sigma * self.weight.std(
            None if clip_type == WeightClipType.LAYER_GAUSSIAN else 1, keepdim=True
        )
        if deepspeed:
            hp_weight = safe_get_full_fp32_param(self.weight)
            hp_sigma = clip_sigma * hp_weight.std(
                None if clip_type == WeightClipType.LAYER_GAUSSIAN else 1, keepdim=True
            )
        if clip_type in [WeightClipType.LAYER_GAUSSIAN, WeightClipType.LAYER_GAUSSIAN_PER_CHANNEL]:
            self.weight.data.clamp_(-sigma_std, sigma_std)
            if deepspeed:
                hp_weight.data.clamp_(-hp_sigma, hp_sigma)
        else:
            raise ValueError(f"Unknown clip type {clip_type}")
        
        if deepspeed:
            safe_set_full_fp32_param(self.weight, hp_weight)

    @no_grad()
    def quantize_weights(self) -> None:
        """Quantize the weights."""
        current_upper = 0
        for slice_idx, inp_size in enumerate(self.in_sizes):
            modified_slice = self.weight[:, current_upper : current_upper + inp_size]
            modified_slice = clip_and_quantize(
                inp_weight=modified_slice,
                assumed_wmax=None,
                learnable_weight_clip=(
                    None
                    if self.learnable_weight_clip is None
                    else self.learnable_weight_clip[slice_idx].unsqueeze(-1)
                ),
                rpu_config=self.rpu_config,
            )
            self.weight[:, current_upper : current_upper + inp_size] = modified_slice
            current_upper += inp_size
        super().quantize_weights()

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super()._save_to_state_dict(destination, prefix, keep_vars)
        # pylint: disable=protected-access
        destination._metadata[prefix.split(".")[0]]["rpu_config"] = deepcopy(self.rpu_config)

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )
        if "rpu_config" in local_metadata:
            self.rpu_config = deepcopy(local_metadata["rpu_config"])
