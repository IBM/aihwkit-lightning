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

"""Calibration for inference."""

from typing import Optional, Dict, Tuple, List, Union
from copy import deepcopy
from collections.abc import Iterator
from functools import partial
from enum import Enum

from tqdm import tqdm

from torch import tensor, Tensor, cat, randperm, no_grad, empty, zeros, full, int32
from torch.nn.functional import unfold
from torch.nn import Parameter
from torch.nn.modules.module import RemovableHandle

from aihwkit_lightning.exceptions import ConfigError
from aihwkit_lightning.simulator.configs import WeightModifierType
from aihwkit_lightning.nn import AnalogLinear, AnalogConv2d
from aihwkit_lightning.simulator.configs import TorchInferenceRPUConfig

# mypy: disable-error-code="attr-defined"


class InputRangeCalibrationType(Enum):
    """Input range post-training calibration type.

    Different styles of calibrating the DAC ranges post-training.
    """

    NONE = "None"
    """No Calibration."""

    MOVING_STD = "MovingStd"
    """Computes a moving average of x*standard deviation of the inputs."""

    MOVING_QUANTILE = "MovingQuantile"
    """Computes the moving average of the quantiles. Saves memory."""

    CACHE_QUANTILE = "CacheQuantile"
    """Caches inputs that are then used to compute the Xth quantile for the input range."""

    MAX = "Max"
    """Takes the abs().max() over the inputs."""


def _calibration_pre_forward(
    mod: Union[AnalogLinear, AnalogConv2d],
    input_args: Tuple,
    input_kwargs: Dict,
    calibration_type: InputRangeCalibrationType,
    cache_key: str,
    global_cache: Dict[str, Tensor],
    max_samples: int = 1000,
    ir_quantile: float = 0.99,
) -> None:
    """Caches inputs for calibrating the input ranges.

    Args:
        input_args: Forward inputs.
        calibration_type: type used for calibration
        cache_key: key of global cache
        max_samples: Maximal number of cache samples
    """
    # pylint: disable=too-many-locals

    # get rid of entries that are all-zeros
    x_input: Tensor
    x_input = input_args[0] if len(input_args) > 0 else input_kwargs["inp"]
    if isinstance(mod, AnalogConv2d):
        assert isinstance(mod.padding, tuple), "Padding must be a tuple"
        x_input = unfold(
            x_input,
            kernel_size=mod.kernel_size,
            dilation=mod.dilation,
            padding=mod.padding,
            stride=mod.stride,
        ).transpose(-1, -2)

    x_input = x_input.reshape(-1, x_input.size(-1))
    x_input = x_input[~(x_input == 0.0).all(-1)]

    ir_params = mod.rpu_config.pre_post.input_range  # type: ignore
    cache = global_cache[cache_key]
    if calibration_type in [
        InputRangeCalibrationType.CACHE_QUANTILE,
        InputRangeCalibrationType.MAX,
    ]:
        # We need to cache the inputs
        # Add new samples to the cache
        if calibration_type in [InputRangeCalibrationType.CACHE_QUANTILE]:
            cache = cat(
                [cache, x_input.float().reshape(-1, x_input.size(-1)).clone().detach().cpu()]
            )
            # Shuffle and limit the number
            cache = cache[randperm(cache.size(0))[:max_samples]]
        else:
            if cache.numel() == 0:
                cache = full(
                    (len(mod.in_sizes),),
                    fill_value=float("-Inf"),
                    dtype=x_input.dtype,
                    device="cpu",
                )

            current_upper = 0
            for slice_idx, inp_size in enumerate(mod.in_sizes):
                inp_slice = x_input[..., current_upper : current_upper + inp_size]  # noqa: E203
                cache[slice_idx] = max(
                    cache[slice_idx], inp_slice.abs().max().detach()
                )  # type: ignore[call-overload]
                current_upper += inp_size

    elif calibration_type in [
        InputRangeCalibrationType.MOVING_QUANTILE,
        InputRangeCalibrationType.MOVING_STD,
    ]:
        current_upper = 0
        for slice_idx, inp_size in enumerate(mod.in_sizes):
            inp_slice = x_input[..., current_upper : current_upper + inp_size]  # noqa: E203
            assert mod.input_range_update_idx is not None, "Input range update idx is None"
            idx = mod.input_range_update_idx[slice_idx]

            if calibration_type == InputRangeCalibrationType.MOVING_QUANTILE:
                val = (
                    inp_slice.abs().max()
                    if ir_quantile == 1.0
                    else inp_slice.float().flatten().quantile(ir_quantile)
                ).item()
            else:
                std = inp_slice.std().item()
                val = ir_params.init_std_alpha * std

            old_val = mod.input_range[slice_idx].item()
            new_val = (old_val * idx + val) / (idx + 1)
            mod.input_range.data[slice_idx] = new_val.type_as(mod.input_range)
            mod.input_range_update_idx[slice_idx] += 1

            current_upper += inp_size
    else:
        raise ConfigError(f"Unknown InputRangeCalibrationType {calibration_type}")

    global_cache[cache_key] = cache


@no_grad()
def calibrate_input_ranges(
    model: Union[AnalogLinear, AnalogConv2d],
    calibration_type: InputRangeCalibrationType,
    dataloader: Iterator,
    quantile: float = 0.99995,
    max_samples: int = 1000,
    std_alpha: Optional[float] = None,
    verbose: bool = True,
) -> None:
    """Calibrate the input ranges according to the defined strategy.

    Only tiles that support and have enabled input range learning will
    be calibrated. If noise management is turned on an error is
    raised.

    Note:
        This implementation transiently registers a new `forward_pre_hook`
        on the analog tile level. It assumes that the user has not defined
        any other forward prehooks.

    Args:
        model: The analog model for
            which to calibrate the input ranges.
        calibration_type: Strategy of the calibration. See :class:`~InputRangeCalibrationType`
        dataloader: Iterator that yields the next inputs. Is used like this
            ``x = next(dataloader); model(x)``
        quantile: Quantile used for hard-coded quantile setting.
            Defaults to 0.99995.
        max_samples: Max batch samples to cache in each tile.
            Defaults to 1000.
        std_alpha: Number of standard deviations for moving
            standard deviation strategy. Defaults to ``init_std_alpha`` from RPUConfig
        verbose: Whether to print verbose output.

    Raises:
        ConfigError: If RPUConfig does not support input range learning

    """
    # pylint: disable=too-many-statements, too-many-locals, too-many-branches

    sample_layer: Union[AnalogLinear, AnalogConv2d]
    sample_layer = next(model.analog_layers())  # type: ignore[assignment]
    is_training = sample_layer.training
    rpu_config = sample_layer.rpu_config

    if rpu_config.pre_post.input_range.enable:
        raise ConfigError(
            "You can only calibrate input ranges for models that don't have input ranges."
        )
    if rpu_config.forward.inp_res > 0:
        raise ConfigError(
            "When calibrating the input ranges, the input res must be infinite (-1 or 0)"
        )
    if is_training:
        raise ConfigError("Calibration can only be done in test mode.")

    cache: Dict[str, Tensor]
    cache = {}

    old_rpu_config: Dict[str, TorchInferenceRPUConfig]
    old_rpu_config = {}

    handles: List[RemovableHandle]
    handles = []

    for layer_name, layer in model.named_analog_layers():
        layer: Union[AnalogLinear, AnalogConv2d]  # type: ignore[no-redef]

        if calibration_type in [
            InputRangeCalibrationType.MOVING_QUANTILE,
            InputRangeCalibrationType.MOVING_STD,
        ]:
            # we actually first change the rpu_config to enable the input range
            layer.rpu_config.pre_post.input_range.enable = True
            layer.rpu_config.pre_post.input_range.learn_input_range = True
            layer.rpu_config.pre_post.input_range.init_value = 3.0
            if std_alpha is not None:
                layer.rpu_config.pre_post.input_range.init_std_alpha = std_alpha

            layer.input_range = Parameter(
                data=full(
                    (len(layer.in_sizes),),
                    fill_value=rpu_config.pre_post.input_range.init_value,
                    dtype=layer.weight.dtype,
                    device=layer.weight.device,
                ),
                requires_grad=rpu_config.pre_post.input_range.learn_input_range,
            )
            layer.input_range_update_idx = Parameter(
                data=zeros((len(layer.in_sizes),), dtype=int32, device=layer.weight.device),
                requires_grad=False,
            )

        # turn off output noise and turn off the weight modifier
        # generate hook
        old_rpu_config[layer_name] = deepcopy(layer.rpu_config)
        layer.rpu_config.forward.out_noise = 0.0
        layer.rpu_config.modifier.type = WeightModifierType.NONE

        cache[layer_name] = tensor([])
        hook = partial(
            _calibration_pre_forward,
            ir_quantile=quantile,
            calibration_type=calibration_type,
            cache_key=layer_name,
            global_cache=cache,
            max_samples=max_samples,
        )
        handles.append(layer.register_forward_pre_hook(hook, with_kwargs=True))

    # Pass through the samples
    progress_bar = tqdm if verbose else lambda x: x
    for args, kwargs in progress_bar(dataloader):  # type: ignore[operator]
        model(*args, **kwargs)

    # Remove hooks
    for handle in handles:
        handle.remove()

    # Re-assign the rpu-configs but also enable the IR range
    # and create the according Parameters
    for layer_name, layer in model.named_analog_layers():
        layer: Union[AnalogLinear, AnalogConv2d]  # type: ignore[no-redef]

        layer.input_range_update_idx = Parameter(
            data=full((len(layer.in_sizes),), fill_value=float("-Inf"), device=layer.weight.device),
            requires_grad=False,
        )

        rpu_config: TorchInferenceRPUConfig  # type: ignore[no-redef]
        rpu_config = old_rpu_config[layer_name]

        if calibration_type in [
            InputRangeCalibrationType.CACHE_QUANTILE,
            InputRangeCalibrationType.MAX,
        ]:
            rpu_config.pre_post.input_range.enable = True

            layer.input_range = Parameter(
                data=empty(
                    (len(layer.in_sizes),), dtype=layer.weight.dtype, device=layer.weight.device
                ).fill_(rpu_config.pre_post.input_range.init_value),
                requires_grad=rpu_config.pre_post.input_range.learn_input_range,
            )

            cached_inputs = cache[layer_name]
            if calibration_type == InputRangeCalibrationType.CACHE_QUANTILE:
                current_upper = 0
                for slice_idx, inp_size in enumerate(layer.in_sizes):
                    inp_slice = cached_inputs[
                        ..., current_upper : current_upper + inp_size
                    ]  # noqa: E203
                    layer.input_range.data[slice_idx] = (
                        inp_slice.flatten().quantile(quantile).item()
                    )
                    current_upper += inp_size
            elif calibration_type == InputRangeCalibrationType.MAX:
                layer.input_range.data = cached_inputs

        layer.rpu_config = rpu_config
