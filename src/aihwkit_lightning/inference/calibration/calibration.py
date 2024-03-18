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

from typing import Optional, Dict, Tuple, TYPE_CHECKING
from collections.abc import Iterator
from functools import partial
from enum import Enum

from tqdm import tqdm

from torch import tensor, Tensor, cat, randperm, no_grad
from torch.nn import Module
from torch.nn.modules.module import RemovableHandle

from aihwkit_lightning.exceptions import ConfigError, ArgumentError
from aihwkit_lightning.simulator.parameters.pre_post import PrePostProcessingRPU
from aihwkit_lightning.nn import AnalogLinear

if TYPE_CHECKING:
    from aihwkit_lightning.simulator.parameters import IOParameters


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
    mod: AnalogLinear,
    input_args: Tuple,
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

    raise NotImplementedError("This function is not yet implemented.")


@no_grad()
def calibrate_input_ranges(
    model: AnalogLinear,
    calibration_type: InputRangeCalibrationType,
    dataloader: Iterator,
    quantile: float = 0.99995,
    max_samples: int = 1000,
    verbose: bool = False,
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
        force_all_layers: Whether to force all layers to be
            (re)-calibrated (default). Otherwise only the layer having
            ``input_range.enable = True`` will be calibrated.
        verbose: Whether to print verbose output.

    Raises:
        ConfigError: If RPUConfig does not support input range learning
        ArgumentError: If non-analog model is given

    """
    # pylint: disable=too-many-statements, too-many-locals, too-many-branches

    raise NotImplementedError("This function is not yet implemented.")
