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

# pylint: disable=too-many-locals, too-many-public-methods, no-member
"""Test for conversion utility."""

import tempfile
from typing import Union
from pytest import mark
from torch import Tensor, randn, allclose, save, load
from torch import device as torch_device
from torch.nn import Module, Linear, Conv1d, Conv2d, RNN, LSTM, GRU
from torch.optim import AdamW
from aihwkit_lightning.nn import AnalogLinear, AnalogConv1d, AnalogConv2d, AnalogRNN
from aihwkit_lightning.nn.modules.rnn.cells import (
    AnalogVanillaRNNCell,
    AnalogLSTMCell,
    AnalogGRUCell,
)
from aihwkit_lightning.nn.conversion import AnalogWrapper, convert_to_analog, convert_to_digital
from aihwkit_lightning.simulator.configs import TorchInferenceRPUConfig
from aihwkit_lightning.optim import AnalogOptimizer


class Model(Module):
    """Test model."""

    def __init__(self):
        super().__init__()
        self.conv1 = Conv1d(in_channels=3, out_channels=3, kernel_size=3, padding=1)
        self.conv2 = Conv2d(in_channels=3, out_channels=10, kernel_size=3)
        self.fc1 = Linear(10, 10)
        self.fc2 = Linear(10, 10)
        self.gru1 = GRU(10, 10, 1, bidirectional=False)
        self.lstm1 = LSTM(10, 20, 3, bidirectional=True)
        self.lstm2 = LSTM(40, 20, 3, bias=False, bidirectional=True)
        self.rnn1 = RNN(40, 20, 3, dropout=0.1, bidirectional=False)

    def forward(self, x: Tensor):
        """
        Forward method

        Args:
            x (Tensor): Input.

        Returns:
            Tensor: Output.
        """
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], 10, 10)
        x = self.conv2(x)
        x = x.reshape(x.shape[0], -1, 10)
        x = self.fc1(x)
        x = self.fc2(x)
        x = x.permute(1, 0, 2)
        x, _ = self.gru1(x)
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x, _ = self.rnn1(x)
        return x


@mark.parametrize("conversion_map", [None, {}])
@mark.parametrize("ensure_analog_root", [True, False])
@mark.parametrize("exclude_modules", [None, "fc2"])
@mark.parametrize("inplace", [True, False])
def test_conversion_to_analog(
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
        assert isinstance(model.conv1, AnalogConv1d)
        assert isinstance(model.conv2, AnalogConv2d)
        assert isinstance(model.gru1, AnalogRNN)
        assert isinstance(model.gru1.rnn.layers[0].cell, AnalogGRUCell)
        assert isinstance(model.lstm1, AnalogRNN)
        assert isinstance(model.lstm1.rnn.layers[0].directions[0].cell, AnalogLSTMCell)
        assert isinstance(model.lstm2, AnalogRNN)
        assert isinstance(model.lstm2.rnn.layers[0].directions[0].cell, AnalogLSTMCell)
        assert isinstance(model.rnn1, AnalogRNN)
        assert isinstance(model.rnn1.rnn.layers[0].cell, AnalogVanillaRNNCell)
    if conversion_map == {}:
        assert isinstance(analog_model.fc1, Linear)
        assert isinstance(analog_model.fc2, Linear)
        assert isinstance(analog_model.conv1, Conv1d)
        assert isinstance(analog_model.conv2, Conv2d)
        assert isinstance(analog_model.gru1, GRU)
        assert isinstance(analog_model.lstm1, LSTM)
        assert isinstance(analog_model.lstm2, LSTM)
        assert isinstance(analog_model.rnn1, RNN)
    if not inplace:
        assert isinstance(model.fc1, Linear) and model.fc1.weight.device == torch_device("cpu")
        assert isinstance(model.fc2, Linear) and model.fc2.weight.device == torch_device("cpu")
        assert isinstance(model.conv1, Conv1d) and model.conv1.weight.device == torch_device("cpu")
        assert isinstance(model.conv2, Conv2d) and model.conv2.weight.device == torch_device("cpu")
        assert isinstance(model.gru1, GRU) and model.gru1.weight_hh_l0.device == torch_device("cpu")
        assert isinstance(model.lstm1, LSTM) and model.lstm1.weight_hh_l0.device == torch_device(
            "cpu"
        )
        assert isinstance(model.lstm2, LSTM) and model.lstm2.weight_hh_l0.device == torch_device(
            "cpu"
        )
        assert isinstance(model.rnn1, RNN) and model.rnn1.weight_hh_l0.device == torch_device("cpu")
    if ensure_analog_root:
        assert isinstance(analog_model, AnalogWrapper)
    else:
        assert not isinstance(analog_model, AnalogWrapper)
        assert isinstance(analog_model, Model)
    if exclude_modules == ["fc2"]:
        assert isinstance(analog_model.conv1, AnalogConv1d)
        assert isinstance(analog_model.conv2, AnalogConv2d)
        assert isinstance(analog_model.fc1, AnalogLinear)
        assert isinstance(analog_model.fc2, Linear)
        assert not isinstance(analog_model.fc2, AnalogLinear)
        assert isinstance(analog_model.gru1, AnalogRNN)
        assert isinstance(analog_model.lstm1, AnalogRNN)
        assert isinstance(analog_model.lstm2, AnalogRNN)
        assert isinstance(analog_model.rnn1, AnalogRNN)


@mark.parametrize("conversion_map", [None, set()])
@mark.parametrize("inplace", [True, False])
def test_conversion_to_digital(conversion_map: Union[None, set], inplace: bool):
    """
    Test the correctness of the conversion to digital.
    """
    # Create analog model
    model = convert_to_analog(Model(), rpu_config=TorchInferenceRPUConfig(), inplace=False)
    # Convert the model to digital
    digital_model = convert_to_digital(model, conversion_map, inplace=inplace)

    if inplace and conversion_map != set():
        assert isinstance(model.fc1, Linear)
        assert isinstance(model.conv1, Conv1d)
        assert isinstance(model.conv2, Conv2d)
        assert isinstance(model.gru1, GRU)
        assert isinstance(model.lstm1, LSTM)
        assert isinstance(model.lstm2, LSTM)
        assert isinstance(model.rnn1, RNN)
    if conversion_map == set():
        assert isinstance(digital_model.fc1, AnalogLinear)
        assert isinstance(digital_model.fc2, AnalogLinear)
        assert isinstance(digital_model.conv1, AnalogConv1d)
        assert isinstance(digital_model.conv2, AnalogConv2d)
        assert isinstance(digital_model.gru1, AnalogRNN)
        assert isinstance(digital_model.gru1.rnn.layers[0].cell, AnalogGRUCell)
        assert isinstance(digital_model.lstm1, AnalogRNN)
        assert isinstance(digital_model.lstm1.rnn.layers[0].directions[0].cell, AnalogLSTMCell)
        assert isinstance(digital_model.lstm2, AnalogRNN)
        assert isinstance(digital_model.lstm2.rnn.layers[0].directions[0].cell, AnalogLSTMCell)
        assert isinstance(digital_model.rnn1, AnalogRNN)
        assert isinstance(digital_model.rnn1.rnn.layers[0].cell, AnalogVanillaRNNCell)
    if not inplace:
        assert isinstance(model.fc1, AnalogLinear)
        assert isinstance(model.fc2, AnalogLinear)
        assert isinstance(model.conv1, AnalogConv1d)
        assert isinstance(model.conv2, AnalogConv2d)
        assert isinstance(model.gru1, AnalogRNN)
        assert isinstance(model.lstm1, AnalogRNN)
        assert isinstance(model.lstm2, AnalogRNN)
        assert model.fc1.weight.device == torch_device("cpu")
        assert model.fc2.weight.device == torch_device("cpu")
        assert model.conv1.weight.device == torch_device("cpu")
        assert model.conv2.weight.device == torch_device("cpu")
        assert model.gru1.rnn.layers[0].cell.weight_hh.weight.device == torch_device("cpu")
        assert model.lstm1.rnn.layers[0].directions[0].cell.weight_hh.weight.device == torch_device(
            "cpu"
        )
        assert model.rnn1.rnn.layers[0].cell.weight_hh.weight.device == torch_device("cpu")


def test_optimizer_state():
    """Test loading the optimizer state and compare to torch"""
    rpu_config = TorchInferenceRPUConfig()
    inp = randn(2, 3, 100)

    model = Model()
    model_sd = model.state_dict()

    # convert the model to analog and load the state dict (not even necessary
    # but nice to check)
    analog_model = convert_to_analog(model, rpu_config=rpu_config)
    analog_model.load_state_dict(model_sd)

    # convert the model back to digitral and load the state dict
    digital_model = convert_to_digital(analog_model)

    # create a normal optimizer
    normal_optim = AdamW(model.parameters(), lr=0.01)
    normal_loss = model(inp)
    normal_loss.sum().backward()
    normal_optim.step()

    # create an analog optimizer
    optim = AnalogOptimizer(AdamW, analog_model.analog_layers, analog_model.parameters(), lr=0.01)
    loss = analog_model(inp)
    loss.sum().backward()
    optim.step()

    # infer from digital converted network
    digital_loss = digital_model(inp)

    allclose(loss, normal_loss, atol=1e-5)
    allclose(digital_loss, normal_loss, atol=1e-5)
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
        loaded_state_dict = load(tmp, weights_only=False)

    analog_model.load_state_dict(loaded_state_dict)

    # we can even load the analog state dict into the non-analog model
    model.load_state_dict(analog_model.state_dict())


if __name__ == "__main__":
    test_optimizer_state()
