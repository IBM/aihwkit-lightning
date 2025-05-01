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

"""Convolution layers."""

# pylint: disable=too-many-arguments, too-many-locals, too-many-instance-attributes

from typing import Optional, Tuple, Union, List
import os
from functools import reduce
from torch import Tensor, cuda, tensor, int32
from torch.nn.functional import unfold
from torch.nn.modules.conv import _ConvNd, Conv1d, Conv2d
from torch.nn.modules.utils import _single, _pair

from aihwkit_lightning.nn.modules.base import AnalogLayerBase
from aihwkit_lightning.nn.modules.torch_utils.torch_linear import TorchLinear
from aihwkit_lightning.simulator.configs import TorchInferenceRPUConfig, WeightClipType
from aihwkit_lightning.nn.modules.torch_utils.torch_abs_max import sliced_abs_max


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


class _AnalogConvNd(AnalogLayerBase, _ConvNd):
    """Base class for convolution layers."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, ...],
        stride: Tuple[int, ...],
        padding: Tuple[int, ...],
        dilation: Tuple[int, ...],
        transposed: bool,
        output_padding: Tuple[int, ...],
        groups: int,
        bias: bool,
        padding_mode: str,
        device=None,
        dtype=None,
        rpu_config: Optional[TorchInferenceRPUConfig] = None,
    ):
        if groups != 1:
            raise ValueError("Only one group is supported")
        if padding_mode != "zeros":
            raise ValueError('Only "zeros" padding mode is supported')
        assert rpu_config is not None, "RPU config must be provided"

        _ConvNd.__init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            transposed,
            output_padding,
            groups,
            bias,
            padding_mode,
            device,
            dtype,
        )

        if str(device) == "meta":
            # init these small values on cpu
            device = "cpu"

        self.in_features = self.get_tile_size(in_channels, groups, kernel_size)
        self.out_features = out_channels
        self.rpu_config = rpu_config

        self.in_sizes = self.get_split_sizes(self.in_features, rpu_config.mapping.max_input_size)
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
                upper_end_of_slices=self.upper_end_of_slices,
                weights=self.weight.view(self.out_channels, -1),
            ),
            device=device,
            dtype=dtype,
        )

    def get_tile_size(self, in_channels: int, groups: int, kernel_size: Tuple[int, ...]) -> int:
        """Calculate the tile size."""
        return (in_channels // groups) * reduce(lambda x, y: x * y, kernel_size)

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
    def move_to_meta(cls, module: _ConvNd):
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

    def set_weights_and_biases(self, weight: Tensor, bias: Optional[Tensor] = None) -> None:
        """Set the weight (and bias) tensors to the analog crossbar. Creates a copy of the tensors.

        Args:
            weight: the weight tensor
            bias: the bias tensor is available
        """
        self.set_weights(weight)
        if bias is not None:
            assert (
                self.bias is not None and self.bias.shape == bias.shape
            ), f"bias shape mismatch. Got {bias.shape}, expected \
                {None if self.bias is None else self.bias.shape}"
            self.bias.data = bias.detach().clone()

    def get_weights_and_biases(self) -> Tuple[Tensor, Optional[Tensor]]:
        """Get the weight (and bias) tensors from the analog crossbar.

        Returns:
            tuple: weight matrix, bias vector
        """
        return (self.weight, self.bias)

    def forward(self, x_input: Tensor) -> Tensor:
        """Compute the forward pass."""

        modified_weights = self.weight

        apply_out_quantization = self.rpu_config.forward.out_res > 0
        if apply_out_quantization:
            assert self.rpu_config.forward.out_bound > 0, "Out quant. without a bound."
            assert self.rpu_config.pre_post.input_range.enable, "Out quant. without IR."
        # apply_out_quantization entails out_bound > 0

        if self.rpu_config.clip.type == WeightClipType.LEARNABLE_PER_CHANNEL:
            assert (
                self.learnable_weight_clip is not None
            ), "Learnable weight clipping tensor not initialized."
            assert self.learnable_weight_clip.size(0) == len(self.upper_end_of_slices), (
                "Learnable weight clipping tensor must have the same number of rows as slices"
                " you have for the weight matrix."
            )

        im_shape = x_input.shape
        assert isinstance(self.padding, tuple), "Padding must be a tuple"
        x_input_ = (
            unfold(
                x_input,
                kernel_size=self.kernel_size,
                dilation=self.dilation,
                padding=self.padding,
                stride=self.stride,
            )
            .transpose(-1, -2)
            .contiguous()
        )

        modified_weights = modified_weights.view(self.out_channels, -1)
        triton_enabled = os.environ.get("AIHWKIT_USE_TRITON", False)
        if TRITON_AVAIL and triton_enabled:
            self.upper_end_of_slices = self.upper_end_of_slices.to(device=modified_weights.device)
            out = TritonLinear.apply(
                x_input_,
                modified_weights,
                self.input_range,
                self.input_range_update_idx,
                self.upper_end_of_slices,
                self.rpu_config,
                self.training,
            )
            out = out + self.bias if self.bias is not None else out
        else:
            out = TorchLinear.linear(
                inp=x_input_,
                weights=modified_weights,
                bias=self.bias,
                input_range=self.input_range,
                input_range_update_idx=self.input_range_update_idx,
                x_min=self.x_min,
                x_max=self.x_max,
                learnable_weight_clip=self.learnable_weight_clip,
                in_sizes=self.in_sizes,
                training=self.training,
                rpu_config=self.rpu_config,
                apply_out_quantization=apply_out_quantization,
            )

        out = out.transpose(-1, -2).contiguous()
        out_size = (
            im_shape[-2] + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1
        ) // self.stride[0] + 1
        if len(im_shape) == 3:
            return out.view(self.out_channels, out_size, -1)
        return out.view(im_shape[0], self.out_channels, out_size, -1)


class AnalogConv1d(_AnalogConvNd):
    """1D convolution layer that uses an analog tile.

    Applies a 1D convolution over an input signal composed of several input
    planes, using an analog tile for its forward, backward and update passes.

    Note:
        The tensor parameters of this layer (``.weight`` and ``.bias``) are not
        guaranteed to contain the same values as the internal weights and biases
        stored in the analog tile. Please use ``set_weights`` and
        ``get_weights`` when attempting to read or modify the weight/bias. This
        read/write process can simulate the (noisy and inexact) analog writing
        and reading of the resistive elements.

    Args:
        in_channels: number of channels in the input image.
        out_channels: number of channels produced by the convolution.
        kernel_size: size of the convolving kernel.
        stride: stride of the convolution.
        padding: zero-padding added to both sides of the input.
        dilation: spacing between kernel elements.
        groups: number of blocked connections from input channels to output
            channels.
        bias: whether to use a bias row on the analog tile or not.
        padding_mode: padding strategy. Only ``'zeros'`` is supported.
        rpu_config: resistive processing unit configuration.
        tile_module_class: Class for the tile module (default
            will be specified from the ``RPUConfig``).

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int]],
        stride: Union[int, Tuple[int]] = 1,
        padding: Union[int, Tuple[int]] = 0,
        dilation: Union[int, Tuple[int]] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
        rpu_config: Optional[TorchInferenceRPUConfig] = None,
    ):
        # pylint: disable=too-many-arguments
        kernel_size = _single(kernel_size)
        stride = _single(stride)
        padding = _single(padding)
        dilation = _single(dilation)

        if dilation != _single(1):
            raise ValueError("Only dilation = 1 is supported")

        super().__init__(
            in_channels,
            out_channels,
            (1, kernel_size[0]),  # type: ignore
            (1, stride[0]),  # type: ignore
            (0, padding[0]),  # type: ignore
            (1, dilation[0]),  # type: ignore
            False,
            _pair(0),
            groups,
            bias,
            padding_mode,
            device,
            dtype,
            rpu_config,
        )
        self._register_load_state_dict_pre_hook(self.update_state_dict)
        self._register_state_dict_hook(self.return_pytorch_state_dict)
        self.tensor_view = (-1, 1)

    def forward(self, x_input: Tensor) -> Tensor:
        x_input_ = x_input.unsqueeze(-2)
        y = super().forward(x_input_)
        return y.squeeze(-2)

    @classmethod
    def from_digital(cls, module: Conv1d, rpu_config: TorchInferenceRPUConfig) -> "AnalogConv1d":
        """Return an AnalogConv1d layer from a torch Conv1d layer.

        Args:
            module: The torch module to convert. All layers that are
                defined in the ``conversion_map``.
            rpu_config: RPU config to use.

        Returns:
            an AnalogConv1d layer based on the digital Conv1d ``module``.
        """
        analog_layer = cls(
            module.in_channels,
            module.out_channels,
            module.kernel_size,  # type: ignore
            module.stride,  # type: ignore
            module.padding,  # type: ignore
            module.dilation,  # type: ignore
            module.groups,
            module.bias is not None,
            module.padding_mode,
            None,
            None,
            rpu_config,
        )

        analog_layer.set_weights_and_biases(module.weight.unsqueeze(-2), module.bias)
        return analog_layer.to(device=module.weight.device, dtype=module.weight.dtype)

    @classmethod
    def to_digital(cls, module: "AnalogConv1d") -> Conv1d:
        """Return an nn.Conv1d layer from an AnalogConv1d layer.

        Args:
            module: The analog module to convert.

        Returns:
            an torch Linear layer with the same dimension and weights
            as the analog linear layer.
        """
        digital_layer = Conv1d(
            module.in_channels,
            module.out_channels,
            module.kernel_size[1],
            module.stride[1],
            module.padding[1],
            module.dilation[1],
            module.groups,
            module.bias is not None,
            module.padding_mode,
        )
        digital_layer.weight.data = module.weight.data.detach().clone().squeeze(-2)
        if module.bias is not None:
            assert digital_layer.bias is not None, "Bias must be present"
            digital_layer.bias.data = module.bias.data.detach().clone()
        return digital_layer.to(device=module.weight.device, dtype=module.weight.dtype)

    def update_state_dict(self, state_dict, *args, **kwargs):  # pylint: disable=unused-argument
        """Update the state dict weight shape to cast Conv1d as AnalogConv2d"""
        for name, item in state_dict.items():
            if "weight" in name:
                state_dict[name] = item.unsqueeze(-2)

    def return_pytorch_state_dict(self, module, state_dict, prefix, local_metadata):
        # pylint: disable=unused-argument
        """Return the cast AnalogConv2d weight shape to standard 1d shape"""
        keys = [key for key in state_dict.keys() if prefix in key]
        for name, item in [(key, state_dict[key]) for key in keys]:
            if "weight" in name:
                state_dict[name] = item.squeeze(-2)


class AnalogConv2d(_AnalogConvNd):
    """2D convolution layer that uses an analog tile.

    Applies a 2D convolution over an input signal composed of several input
    planes, using an analog tile for its forward, backward and update passes.

    Note:
        The tensor parameters of this layer (``.weight`` and ``.bias``) are not
        guaranteed to contain the same values as the internal weights and biases
        stored in the analog tile. Please use ``set_weights`` and
        ``get_weights`` when attempting to read or modify the weight/bias. This
        read/write process can simulate the (noisy and inexact) analog writing
        and reading of the resistive elements.

    Args:
        in_channels: number of channels in the input image.
        out_channels: number of channels produced by the convolution.
        kernel_size: size of the convolving kernel.
        stride: stride of the convolution.
        padding: zero-padding added to both sides of the input.
        dilation: spacing between kernel elements.
        groups: number of blocked connections from input channels to output
            channels.
        bias: whether to use a bias row on the analog tile or not.
        padding_mode: padding strategy. Only ``'zeros'`` is supported.
        rpu_config: resistive processing unit configuration.
        tile_module_class: Class for the tile module (default
            will be specified from the ``RPUConfig``).
        use_indexed: Whether to use explicit unfolding or implicit indexing. If
            None (default), it will use implicit indexing for CUDA and
            explicit unfolding for CPU
    """

    # pylint: disable=abstract-method

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple],
        stride: Union[int, Tuple] = 1,
        padding: Union[int, Tuple] = 0,
        dilation: Union[int, Tuple] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
        rpu_config: Optional[TorchInferenceRPUConfig] = None,
    ):
        # pylint: disable=too-many-arguments
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)

        super().__init__(
            in_channels,
            out_channels,
            kernel_size,  # type: ignore
            stride,  # type: ignore
            padding,  # type: ignore
            dilation,  # type: ignore
            False,
            _pair(0),
            groups,
            bias,
            padding_mode,
            device,
            dtype,
            rpu_config,
        )

    @classmethod
    def from_digital(cls, module: Conv2d, rpu_config: TorchInferenceRPUConfig) -> "AnalogConv2d":
        """Return an AnalogConv2d layer from a torch Conv2d layer.

        Args:
            module: The torch module to convert. All layers that are
                defined in the ``conversion_map``.
            rpu_config: RPU config to use.

        Returns:
            an AnalogConv2d layer based on the digital Conv2d ``module``.
        """
        analog_layer = cls(
            module.in_channels,
            module.out_channels,
            module.kernel_size,
            module.stride,  # type: ignore
            module.padding,  # type: ignore
            module.dilation,  # type: ignore
            module.groups,  # type: ignore
            module.bias is not None,
            module.padding_mode,
            None,
            None,
            rpu_config,
        )

        analog_layer.set_weights_and_biases(module.weight, module.bias)
        return analog_layer.to(device=module.weight.device, dtype=module.weight.dtype)

    @classmethod
    def to_digital(cls, module: "AnalogConv2d") -> Conv2d:
        """Return an nn.Conv2d layer from an AnalogConv2d layer.

        Args:
            module: The analog module to convert.

        Returns:
            a torch Conv2d layer with the same dimension and weights
            as the analog version.
        """
        digital_layer = Conv2d(
            module.in_channels,
            module.out_channels,
            module.kernel_size,  # type: ignore
            module.stride,  # type: ignore
            module.padding,  # type: ignore
            module.dilation,  # type: ignore
            module.groups,
            module.bias is not None,
            module.padding_mode,
        )
        digital_layer.weight.data = module.weight.data.detach().clone()
        if module.bias is not None:
            assert digital_layer.bias is not None, "Bias must be present"
            digital_layer.bias.data = module.bias.data.detach().clone()
        return digital_layer.to(device=module.weight.device, dtype=module.weight.dtype)

    def clip_weights(self) -> None:
        """Clip the weights."""
        two_dim_weights = self.weight.view(self.out_channels, -1)
        clip_type = self.rpu_config.clip.type
        clip_sigma = self.rpu_config.clip.sigma

        if clip_type in [WeightClipType.NONE, WeightClipType.LEARNABLE_PER_CHANNEL]:
            return
        assert clip_sigma > 0, "Clip sigma must be greater than 0"
        sigma_std = clip_sigma * two_dim_weights.std(
            None if clip_type == WeightClipType.LAYER_GAUSSIAN else 1, keepdim=True
        )
        if clip_type in [WeightClipType.LAYER_GAUSSIAN, WeightClipType.LAYER_GAUSSIAN_PER_CHANNEL]:
            two_dim_weights.data.clamp_(-sigma_std, sigma_std)
        else:
            raise ValueError(f"Unknown clip type {clip_type}")
        self.weight.data = two_dim_weights.view_as(self.weight)

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
