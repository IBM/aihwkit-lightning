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

import tempfile
from typing import Union
from pytest import mark
from torch import Tensor, randn, allclose, save, load
from torch import device as torch_device
from torch.nn import Module, Linear
from torch.optim import AdamW
from aihwkit_lightning.nn import AnalogLinear
from aihwkit_lightning.nn.conversion import AnalogWrapper, convert_to_analog
from aihwkit_lightning.simulator.configs import TorchInferenceRPUConfig
from aihwkit_lightning.optim import AnalogOptimizer


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


def test_optimizer_state():
    """Test loading the optimizer state and compare to torch"""
    rpu_config = TorchInferenceRPUConfig()
    inp = randn(10)

    model = Model()
    model_sd = model.state_dict()

    # convert the model to analog and load the state dict (not even necessary
    # but nice to check)
    analog_model = convert_to_analog(model, rpu_config=rpu_config)
    analog_model.load_state_dict(model_sd)

    # create an analog optimizer
    optim = AnalogOptimizer(AdamW, analog_model.analog_layers(), analog_model.parameters(), lr=0.01)
    loss = analog_model(inp).sum()
    loss.backward()
    optim.step()

    # create a normal optimizer
    normal_optim = AdamW(model.parameters(), lr=0.01)
    normal_loss = model(inp).sum()
    normal_loss.backward()
    normal_optim.step()

    allclose(
        optim.state_dict()["state"][0]["exp_avg"],
        normal_optim.state_dict()["state"][0]["exp_avg"],
        atol=1e-5,
    )


def test_save_and_load_state():
    """Test saving and loading the state dict."""
    rpu_config = TorchInferenceRPUConfig()

    model = Model()
    model_sd = model.state_dict()

    analog_model = convert_to_analog(model, rpu_config=rpu_config)

    # FP -> Analog
    analog_model.load_state_dict(model_sd)

    # Analog -> Analog
    analog_sd = analog_model.state_dict()
    analog_model.load_state_dict(analog_sd)

    # Create a temporary file
    with tempfile.TemporaryFile() as tmp:
        # Save the model to the temporary file
        save(analog_model.state_dict(), tmp)

        # To read the data back, you need to seek back to the start of the file
        tmp.seek(0)

        # Load the model state dict from the temporary file
        loaded_state_dict = load(tmp)

    analog_model.load_state_dict(loaded_state_dict)

    # we can even load the analog state dict into the non-analog model
    model.load_state_dict(analog_model.state_dict())