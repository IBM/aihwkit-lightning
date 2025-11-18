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

from typing import Optional, Tuple, Union, List, Literal
import os
from copy import deepcopy
from functools import reduce
from torch import Tensor, cuda, no_grad, tensor, int32
from torch.nn.functional import unfold, pad as torch_pad
from torch.nn.modules.conv import _ConvNd, Conv1d, Conv2d, Conv3d
from torch.nn.modules.utils import _single, _pair, _triple

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


def unfold3d(
    inp: Tensor,
    kernel_size: Tuple[int, int, int],
    padding: Tuple[int, int, int],
    stride: Tuple[int, int, int],
) -> Tensor:
    """Unfold a 5D tensor for 3D convolution using sequential unfold operations.

    This implements the approach from:
    https://discuss.pytorch.org/t/manual-implementation-of-unrolled-3d-convolutions/91021/4

    Args:
        inp: Input tensor of shape (B, C, D, H, W) or (C, D, H, W)
        kernel_size: 3D kernel size (kD, kH, kW)
        padding: 3D padding (pD, pH, pW)
        stride: 3D stride (sD, sH, sW)

    Returns:
        Unfolded tensor of shape (B, C * kD * kH * kW, num_patches) or
        (C * kD * kH * kW, num_patches) for unbatched input
    """
    # Handle unbatched input by adding batch dimension
    if inp.dim() == 4:
        inp = inp.unsqueeze(0)
        unbatched = True
    else:
        unbatched = False

    batch_size, in_channels, _, _, _ = inp.shape

    # Apply padding: F.pad expects (left, right, top, bottom, front, back)
    # padding is (pD, pH, pW), so we need (pW, pW, pH, pH, pD, pD)
    inp_padded = torch_pad(
        inp, (padding[2], padding[2], padding[1], padding[1], padding[0], padding[0])
    )

    # Unfold across all three spatial dimensions sequentially
    # Start with depth (dimension 2)
    unfolded = inp_padded.unfold(2, size=kernel_size[0], step=stride[0])
    # Then height (dimension 3, after first unfold)
    unfolded = unfolded.unfold(3, size=kernel_size[1], step=stride[1])
    # Finally width (dimension 4, after second unfold)
    unfolded = unfolded.unfold(4, size=kernel_size[2], step=stride[2])

    # After unfolding, shape is: (B, C, nD, nH, nW, kD, kH, kW)
    # where n* are the output spatial dimensions

    # Permute to get: (B, nD, nH, nW, C, kD, kH, kW)
    unfolded = unfolded.permute(0, 2, 3, 4, 1, 5, 6, 7)

    # Reshape to: (B, nD * nH * nW, C * kD * kH * kW)
    num_patches = unfolded.size(1) * unfolded.size(2) * unfolded.size(3)
    kernel_numel = in_channels * kernel_size[0] * kernel_size[1] * kernel_size[2]
    unfolded = unfolded.reshape(batch_size, num_patches, kernel_numel)

    # Transpose to: (B, C * kD * kH * kW, nD * nH * nW)
    unfolded = unfolded.transpose(1, 2).contiguous()

    # Remove batch dimension if input was unbatched
    if unbatched:
        unfolded = unfolded.squeeze(0)

    return unfolded


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
        padding_mode: Literal["zeros", "reflect", "replicate", "circular"],
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

        self.deprecation_adjustment()

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

    @no_grad()
    def quantize_weights(self) -> None:
        """Quantize the weights."""
        weight_2d = self.weight.view(self.out_channels, -1)
        current_upper = 0
        for slice_idx, inp_size in enumerate(self.in_sizes):
            modified_slice = weight_2d[:, current_upper : current_upper + inp_size]
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
            weight_2d[:, current_upper : current_upper + inp_size] = modified_slice
            current_upper += inp_size

        self.weight.data = weight_2d.view_as(self.weight)
        super().quantize_weights()

    def forward(self, inp: Tensor) -> Tensor:
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

        im_shape = inp.shape
        assert isinstance(self.padding, tuple), "Padding must be a tuple"
        inp_ = (
            unfold(
                inp,
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
                inp_,
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
                inp=inp_,
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
        padding_mode: Literal["zeros", "reflect", "replicate", "circular"] = "zeros",
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

    def forward(self, inp: Tensor) -> Tensor:
        inp_ = inp.unsqueeze(-2)
        y = super().forward(inp_)
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
            module.padding_mode,  # type: ignore[arg-type]
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
        padding_mode: Literal["zeros", "reflect", "replicate", "circular"] = "zeros",
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
            module.padding_mode,  # type: ignore[arg-type]
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
        destination._metadata[prefix.split(".")[0]]["rpu_config"] = deepcopy(self.rpu_config)

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )
        if "rpu_config" in local_metadata:
            self.rpu_config = deepcopy(local_metadata["rpu_config"])


class AnalogConv3d(_AnalogConvNd):
    """3D convolution layer that uses an analog tile.

    Applies a 3D convolution over an input signal composed of several input
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
        kernel_size: Union[int, Tuple],
        stride: Union[int, Tuple] = 1,
        padding: Union[int, Tuple] = 0,
        dilation: Union[int, Tuple] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: Literal["zeros", "reflect", "replicate", "circular"] = "zeros",
        device=None,
        dtype=None,
        rpu_config: Optional[TorchInferenceRPUConfig] = None,
    ):
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        dilation = _triple(dilation)

        assert dilation == (1, 1, 1), "Dilation must be 1."

        super().__init__(
            in_channels,
            out_channels,
            kernel_size,  # type: ignore
            stride,  # type: ignore
            padding,  # type: ignore
            dilation,  # type: ignore
            False,
            _triple(0),
            groups,
            bias,
            padding_mode,
            device,
            dtype,
            rpu_config,
        )

    def forward(self, inp: Tensor) -> Tensor:
        """Compute the forward pass for 3D convolution.

        This overrides the base class forward to use unfold3d instead of unfold.
        """
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

        im_shape = inp.shape
        assert isinstance(self.padding, tuple), "Padding must be a tuple"
        assert isinstance(self.kernel_size, tuple), "Kernel size must be a tuple"
        assert isinstance(self.stride, tuple), "Stride must be a tuple"
        assert isinstance(self.dilation, tuple), "Dilation must be a tuple"

        # Use unfold3d for 3D convolution
        inp_ = unfold3d(
            inp,
            kernel_size=self.kernel_size,  # type: ignore[arg-type]
            padding=self.padding,  # type: ignore[arg-type]
            stride=self.stride,  # type: ignore[arg-type]
        )

        # Handle unbatched input
        if inp_.dim() == 2:
            inp_ = inp_.unsqueeze(0)
            unbatched = True
        else:
            unbatched = False

        # Transpose to match expected format: (B, num_patches, kernel_size)
        inp_ = inp_.transpose(-1, -2).contiguous()

        modified_weights = modified_weights.view(self.out_channels, -1)
        triton_enabled = os.environ.get("AIHWKIT_USE_TRITON", False)
        if TRITON_AVAIL and triton_enabled:
            self.upper_end_of_slices = self.upper_end_of_slices.to(device=modified_weights.device)
            out = TritonLinear.apply(
                inp_,
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
                inp=inp_,
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

        # Calculate output spatial dimensions
        def calc_output_size(input_size, padding, dilation, kernel_size, stride):
            return (input_size + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1

        if len(im_shape) == 4:
            # Unbatched input: (C, D, H, W)
            depth, height, width = im_shape[1], im_shape[2], im_shape[3]
        else:
            # Batched input: (B, C, D, H, W)
            depth, height, width = im_shape[2], im_shape[3], im_shape[4]

        out_depth = calc_output_size(
            depth, self.padding[0], self.dilation[0], self.kernel_size[0], self.stride[0]
        )
        out_height = calc_output_size(
            height, self.padding[1], self.dilation[1], self.kernel_size[1], self.stride[1]
        )
        out_width = calc_output_size(
            width, self.padding[2], self.dilation[2], self.kernel_size[2], self.stride[2]
        )

        if unbatched or len(im_shape) == 4:
            # Unbatched output: (C_out, D_out, H_out, W_out)
            return out.view(self.out_channels, out_depth, out_height, out_width)
        # Batched output: (B, C_out, D_out, H_out, W_out)
        return out.view(im_shape[0], self.out_channels, out_depth, out_height, out_width)

    @classmethod
    def from_digital(cls, module: Conv3d, rpu_config: TorchInferenceRPUConfig) -> "AnalogConv3d":
        """Return an AnalogConv3d layer from a torch Conv3d layer.

        Args:
            module: The torch module to convert. All layers that are
                defined in the ``conversion_map``.
            rpu_config: RPU config to use.

        Returns:
            an AnalogConv3d layer based on the digital Conv3d ``module``.
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
            module.padding_mode,  # type: ignore[arg-type]
            None,
            None,
            rpu_config,
        )

        analog_layer.set_weights_and_biases(module.weight, module.bias)
        return analog_layer.to(device=module.weight.device, dtype=module.weight.dtype)

    @classmethod
    def to_digital(cls, module: "AnalogConv3d") -> Conv3d:
        """Return an nn.Conv3d layer from an AnalogConv3d layer.

        Args:
            module: The analog module to convert.

        Returns:
            a torch Conv3d layer with the same dimension and weights
            as the analog version.
        """
        digital_layer = Conv3d(
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
        destination._metadata[prefix.split(".")[0]]["rpu_config"] = deepcopy(self.rpu_config)

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )
        if "rpu_config" in local_metadata:
            self.rpu_config = deepcopy(local_metadata["rpu_config"])
