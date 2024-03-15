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
from typing import Optional, Tuple, List
from torch.autograd import no_grad, Function
from torch.autograd.function import FunctionCtx
from torch import Tensor, empty, zeros, zeros_like, clamp, randn_like
from torch.nn import Linear, Parameter
from aihwkit_lightning.simulator.configs import TorchInferenceRPUConfig, WeightClipType, WeightModifierType
from aihwkit_lightning.nn.modules.base import AnalogLayerBase
from aihwkit_lightning.exceptions import ConfigError


class UniformQuantize(Function):
    """Quantization function."""

    # pylint: disable=abstract-method, redefined-builtin, arguments-differ

    @staticmethod
    def forward(ctx: FunctionCtx, inp: Tensor, res: float, inplace: bool) -> Tensor:
        """Quantizes the input tensor and performs straight-through estimation.

        Args:
            ctx (FunctionCtx): Context.
            inp (torch.Tensor): Input to be discretized.
            res (float): Resolution (number of states).
            inplace (bool): Clone the input?

        Returns:
            torch.Tensor: Quantized input.
        """
        # - Compute 1 / states if the number of states are provided
        res = 2 / res if res > 1.0 else 2 * res
        assert res > 0, "resolution is <= 0"
        output = inp if inplace else inp.clone()
        output = output / res
        # - Perform explicit rounding
        output = output.round()
        # - Scale back down
        output *= res
        return output

    @staticmethod
    # type: ignore[override]
    def backward(ctx: FunctionCtx, grad_output: Tensor) -> Tuple[Tensor, None, None, None]:
        """Straight-through estimator.

        Args:
            ctx: Context.
            grad_output: Gradient w.r.t. the inputs.

        Returns:
            Gradients w.r.t. inputs to forward.
        """
        # - Straight-through estimator
        grad_input = grad_output
        return grad_input, None, None, None


class InputRangeForward(Function):
    """
    Enable custom input range gradient computation using torch's autograd.
    """

    # pylint: disable=abstract-method, redefined-builtin, arguments-differ

    @staticmethod
    def forward(
        ctx: FunctionCtx,
        x_input: Tensor,
        input_range: Tensor,
        decay: float,
        input_min_percentage: float,
    ) -> Tensor:
        ctx.save_for_backward(x_input, input_range)
        ctx.decay = decay  # type: ignore[attr-defined]
        ctx.input_min_percentage = input_min_percentage  # type: ignore[attr-defined]
        return x_input

    @staticmethod
    # type: ignore[override]
    def backward(ctx: FunctionCtx, d_output: Tensor) -> Tuple[Tensor, Tensor, None, None]:

        x_input: Tensor
        input_range: Tensor

        x_input, input_range = ctx.saved_tensors  # type: ignore[attr-defined]
        ir_grad = None

        if input_range is not None:
            decay = ctx.decay  # type: ignore[attr-defined]
            input_min_percentage = ctx.input_min_percentage  # type: ignore[attr-defined]

            upper_thres = x_input >= input_range  # pylint: disable=invalid-unary-operand-type
            lower_thres = x_input <= -input_range  # pylint: disable=invalid-unary-operand-type
            ir_grad = zeros_like(input_range)
            ir_grad += clamp(upper_thres * d_output, min=None, max=0.0).sum()
            ir_grad -= clamp(lower_thres * d_output, min=0.0, max=None).sum()
            ir_grad *= input_range
            if decay > 0:
                # - We shrink the input range if less than X% of the inputs are clipping.
                # where X is 1-ir_params.input_min_percentage
                percentage = (x_input.abs() < input_range).float().mean()
                ir_grad += decay * input_range * (percentage > input_min_percentage)

        return d_output, ir_grad, None, None


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

        if rpu_config.pre_post.input_range.enable:
            # for every vertical tile, we have an input range
            self.input_range = Parameter(
                data=empty((len(self.in_sizes),), dtype=dtype, device=device).fill_(
                    rpu_config.pre_post.input_range.init_value
                ),
                requires_grad=rpu_config.pre_post.input_range.learn_input_range,
            )
            self.input_range_update_idx = Parameter(
                data=zeros((len(self.in_sizes),), dtype=dtype, device=device), requires_grad=False
            )

    def forward(self, inp: Tensor) -> Tensor:  # pylint: disable=arguments-renamed
        """Forward function."""

        modified_weights = self.weight
        if self.rpu_config.modifier.type != WeightModifierType.NONE:
            modified_weights = self.weight.clone()

        current_upper = 0
        out = 0.0
        for slice_idx, inp_size in enumerate(self.in_sizes):
            inp_slice = inp[..., current_upper : current_upper + inp_size]  # noqa: E203

            if self.rpu_config.pre_post.input_range.enable:
                with no_grad():
                    inp_slice = self.apply_input_range(
                        values=inp_slice, slice_idx=slice_idx, update_from_data=self.training
                    )

                inp_slice = InputRangeForward.apply(
                    inp_slice,
                    self.input_range[slice_idx],
                    self.rpu_config.pre_post.input_range.decay,
                    self.rpu_config.pre_post.input_range.input_min_percentage,
                )
                inp_slice = inp_slice / self.input_range[slice_idx]

            # maybe do some quantization
            if self.rpu_config.forward.inp_res > 0:
                inp_slice = UniformQuantize.apply(inp_slice, self.rpu_config.forward.inp_res, False)

            modified_slice = self.modify_weight(modified_weights[:, current_upper : current_upper + inp_size])
            out_slice = inp_slice @ modified_slice.T

            # maybe add some output noise
            # TODO

            if self.rpu_config.pre_post.input_range.enable:
                out_slice *= self.input_range[slice_idx]

            out += out_slice  # type: ignore[assignment]

            current_upper += inp_size

        return out + self.bias if self.bias is not None else out

    def apply_input_range(
        self, values: Tensor, slice_idx: int, update_from_data: bool = False
    ) -> Tensor:
        """Apply the input clipping.

        Args:
            values: tensor to clip
            slice_idx: index of the input slice
            update_from_data: whether to update from data if applicable

        Returns:
            clipped output tensor
        """

        if not self.rpu_config.pre_post.input_range.enable:
            return values

        if update_from_data:
            ir_params = self.rpu_config.pre_post.input_range
            idx = self.input_range_update_idx[slice_idx]
            if idx < ir_params.init_from_data:
                std = values.std()
                if std > 0.0:
                    self.input_range.data[slice_idx] = (
                        self.input_range.data[slice_idx] * idx + ir_params.init_std_alpha * std
                    ) / (idx + 1)
                    self.input_range_update_idx[slice_idx] += 1
                self.input_range.data[slice_idx] = self.input_range.data[slice_idx].abs()

        return clamp(values, min=-self.input_range[slice_idx], max=self.input_range[slice_idx])

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
    
    def modify_weight(self, inp_weight: Tensor) -> Tensor:
        """Modified weights in-place, so .clone() before if it's not NONE.

        Args:
            inp_weight: Input weights.
            modifier: Noise injection configuration.

        Raises:
            ConfigError: Unsupported/unknown weight modifier type.

        Returns:
            Weights with noise injected.
        """
        modifier = self.rpu_config.modifier
        if modifier.type == WeightModifierType.NONE:
            return inp_weight
        
        assumed_wmax = inp_weight.abs().max()
        res = modifier.res
        n_states = max(res, 1 / res)
        res = assumed_wmax / n_states

        if modifier.type == WeightModifierType.DISCRETIZE:
            # - Discretize the weights on the fly and backprob through them
            inp_weight = UniformQuantize.apply(inp_weight, res, True)
        elif modifier.type == WeightModifierType.ADD_NORMAL:
            with no_grad():
                noise = (
                    modifier.std_dev * assumed_wmax * randn_like(inp_weight)
                )
            inp_weight = inp_weight + noise
        elif modifier.type == WeightModifierType.DISCRETIZE_ADD_NORMAL:
            inp_weight = UniformQuantize.apply(inp_weight, res, True)
            with no_grad():
                noise = (
                    modifier.std_dev * assumed_wmax * randn_like(inp_weight)
                )
            inp_weight = inp_weight + noise
        else:
            raise ConfigError(f"Weight modifier {modifier} not supported")
        return inp_weight

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
