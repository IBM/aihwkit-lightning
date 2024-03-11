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

# pylint: disable=too-many-locals, too-many-public-methods, no-member
"""Test for conversion utility."""

from typing import Union
from pytest import mark
from torch import Tensor
from torch import device as torch_device
from torch.nn import Module, Linear
from aihwkit_lightning.nn import AnalogLinear
from aihwkit_lightning.nn.conversion import AnalogWrapper, convert_to_analog
from aihwkit_lightning.simulator.configs import TorchInferenceRPUConfig


class Model(Module):
    """Test model."""

    def __init__(self):
        super().__init__()
        self.fc1 = Linear(10, 10)
        self.fc2 = Linear(10, 10)

    def forward(self, x: Tensor):
        """
        Forward method

        Args:
            x (Tensor): Input.

        Returns:
            Tensor: Output.
        """
        x = self.fc1(x)
        x = self.fc2(x)
        return x


@mark.parametrize("conversion_map", [None, {}])
@mark.parametrize("ensure_analog_root", [True, False])
@mark.parametrize("exclude_modules", [None, "fc2"])
@mark.parametrize("inplace", [True, False])
def test_conversion(
    conversion_map: Union[None, dict],
    ensure_analog_root: bool,
    exclude_modules: Union[None, str],
    inplace: bool,
):
    """
    Test the correctness of the conversion to analog.
    """
    # Create a model
    model = Model()

    # Convert the model to analog
    analog_model: Model
    analog_model = convert_to_analog(
        model,
        rpu_config=TorchInferenceRPUConfig(),
        conversion_map=conversion_map,
        ensure_analog_root=ensure_analog_root,
        exclude_modules=exclude_modules,
        inplace=inplace,
    )
    if inplace and conversion_map != {}:
        assert isinstance(model.fc1, AnalogLinear)
    if conversion_map == {}:
        assert isinstance(analog_model.fc1, Linear)
        assert isinstance(analog_model.fc2, Linear)
    if not inplace:
        assert isinstance(model.fc1, Linear) and model.fc1.weight.device == torch_device("cpu")
        assert isinstance(model.fc2, Linear) and model.fc2.weight.device == torch_device("cpu")
    if ensure_analog_root:
        assert isinstance(analog_model, AnalogWrapper)
    else:
        assert not isinstance(analog_model, AnalogWrapper)
        assert isinstance(analog_model, Model)
    if exclude_modules == ["fc2"]:
        assert isinstance(analog_model.fc1, AnalogLinear)
        assert isinstance(analog_model.fc2, Linear)
        assert not isinstance(analog_model.fc2, AnalogLinear)
