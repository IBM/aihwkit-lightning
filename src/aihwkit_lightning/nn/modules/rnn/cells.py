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

""" Analog cells for RNNs. """

from typing import Optional, Tuple
from collections import namedtuple

from torch import Tensor, sigmoid, tanh, zeros, cat
from torch.nn import Module

from aihwkit_lightning.nn.modules.linear import AnalogLinear

from aihwkit_lightning.simulator.configs import TorchInferenceRPUConfig


LSTMState = namedtuple("LSTMState", ["hx", "cx"])


def _get_linear(
    in_size: int,
    out_size: int,
    bias: bool,
    device=None,
    dtype=None,
    rpu_config: Optional[TorchInferenceRPUConfig] = None,
) -> Module:
    """Return an analog linear module given the parameters."""
    return AnalogLinear(in_size, out_size, bias, device, dtype, rpu_config)


class AnalogVanillaRNNCell(Module):
    """Analog Vanilla RNN Cell.

    Args:
        input_size: in_features size for W_ih matrix
        hidden_size: in_features and out_features size for W_hh matrix
        bias: whether to use a bias row on the analog tile or not
        rpu_config: configuration for an analog resistive processing
            unit. If not given a native torch model will be
            constructed instead.

    """

    # pylint: disable=abstract-method
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool,
        device=None,
        dtype=None,
        rpu_config: Optional[TorchInferenceRPUConfig] = None,
    ):
        super().__init__()

        if rpu_config is None:
            raise ValueError("rpu_config must be provided. Try TorchInferenceRPUConfig()")

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = _get_linear(input_size, hidden_size, bias, device, dtype, rpu_config)
        self.weight_hh = _get_linear(hidden_size, hidden_size, bias, device, dtype, rpu_config)

    def get_zero_state(self, batch_size: int) -> LSTMState:
        """Returns a zeroed state.

        Args:
            batch_size: batch size of the input

        Returns:
           Zeroed state tensor
        """
        param = next(self.parameters())
        return LSTMState(
            zeros(batch_size, self.hidden_size, device=param.device, dtype=param.dtype),
            None,  # cx unused
        )

    def forward(self, input_: Tensor, state: Tensor) -> Tuple[Tensor, Tuple[Tensor, None]]:
        """Forward pass.

        Args:
            input_: input tensor
            state: LSTM state tensor

        Returns:
            output and output states (which is the same here)
        """
        # pylint: disable=arguments-differ
        h_x, _ = state
        igates = self.weight_ih(input_)
        hgates = self.weight_hh(h_x)

        out = tanh(igates + hgates)

        return out, (out, None)  # output will also be hidden state


class AnalogLSTMCell(Module):
    """Analog LSTM Cell.

    Args:
        input_size: in_features size for W_ih matrix
        hidden_size: in_features and out_features size for W_hh matrix
        bias: whether to use a bias row on the analog tile or not
        rpu_config: configuration for an analog resistive processing
            unit. If not given a native torch model will be
            constructed instead.
    """

    # pylint: disable=abstract-method

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool,
        device=None,
        dtype=None,
        rpu_config: Optional[TorchInferenceRPUConfig] = None,
    ):
        super().__init__()

        if rpu_config is None:
            raise ValueError("rpu_config must be provided. Try TorchInferenceRPUConfig()")

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = _get_linear(input_size, 4 * hidden_size, bias, device, dtype, rpu_config)
        self.weight_hh = _get_linear(hidden_size, 4 * hidden_size, bias, device, dtype, rpu_config)

    def get_zero_state(self, batch_size: int) -> LSTMState:
        """Returns a zeroed state.

        Args:
            batch_size: batch size of the input

        Returns:
           Zeroed state tensor
        """
        param = next(self.parameters())
        return LSTMState(
            zeros(batch_size, self.hidden_size, device=param.device, dtype=param.dtype),
            zeros(batch_size, self.hidden_size, device=param.device, dtype=param.dtype),
        )

    def forward(
        self, input_: Tensor, state: Tuple[Tensor, Tensor]
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """Forward pass.

        Args:
            input_: input tensor
            state: LSTM state tensor

        Returns:
            output h_y and output states tuple h_y and c_y
        """
        # pylint: disable=arguments-differ
        h_x, c_x = state
        gates = self.weight_ih(input_) + self.weight_hh(h_x)
        in_gate, forget_gate, cell_gate, out_gate = gates.chunk(4, 1)

        in_gate = sigmoid(in_gate)
        forget_gate = sigmoid(forget_gate)
        cell_gate = tanh(cell_gate)
        out_gate = sigmoid(out_gate)

        c_y = (forget_gate * c_x) + (in_gate * cell_gate)
        h_y = out_gate * tanh(c_y)

        return h_y, (h_y, c_y)


class AnalogLSTMCellCombinedWeight(Module):
    """Analog LSTM Cell that use a combined weight for storing gates and inputs.

    Args:
        input_size: The number of expected features in the input `x`
        hidden_size: The number of features in the hidden state `h`
        bias: whether to use a bias row on the analog tile or not.
        rpu_config: configuration for an analog resistive processing
            unit. If not given a native torch model will be
            constructed instead.
    """

    # pylint: disable=abstract-method

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool,
        device=None,
        dtype=None,
        rpu_config: Optional[TorchInferenceRPUConfig] = None,
    ):
        super().__init__()

        if rpu_config is None:
            raise ValueError("rpu_config must be provided. Try TorchInferenceRPUConfig()")

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight = _get_linear(
            input_size + hidden_size, 4 * hidden_size, bias, device, dtype, rpu_config
        )

    def get_zero_state(self, batch_size: int) -> LSTMState:
        """Returns a zeroed state.

        Args:
            batch_size: batch size of the input

        Returns:
           Zeroed state tensor
        """
        param = next(self.parameters())
        return LSTMState(
            zeros(batch_size, self.hidden_size, device=param.device, dtype=param.dtype),
            zeros(batch_size, self.hidden_size, device=param.device, dtype=param.dtype),
        )

    def forward(
        self, input_: Tensor, state: Tuple[Tensor, Tensor]
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """Forward pass.

        Args:
            input_: input tensor
            state: LSTM state tensor

        Returns:
            output h_y and output states tuple h_y and c_y
        """
        # pylint: disable=arguments-differ
        h_x, c_x = state
        x_input = cat((input_, h_x), 1)
        gates = self.weight(x_input)
        in_gate, forget_gate, cell_gate, out_gate = gates.chunk(4, 1)

        in_gate = sigmoid(in_gate)
        forget_gate = sigmoid(forget_gate)
        cell_gate = tanh(cell_gate)
        out_gate = sigmoid(out_gate)

        c_y = (forget_gate * c_x) + (in_gate * cell_gate)
        h_y = out_gate * tanh(c_y)

        return h_y, (h_y, c_y)


class AnalogGRUCell(Module):
    """Analog GRU Cell.

    Args:
        input_size: in_features size for W_ih matrix
        hidden_size: in_features and out_features size for W_hh matrix
        bias: whether to use a bias row on the analog tile or not
        rpu_config: configuration for an analog resistive processing
            unit. If not given a native torch model will be
            constructed instead.
    """

    # pylint: disable=abstract-method

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool,
        device=None,
        dtype=None,
        rpu_config: Optional[TorchInferenceRPUConfig] = None,
    ):
        super().__init__()

        if rpu_config is None:
            raise ValueError("rpu_config must be provided. Try TorchInferenceRPUConfig()")

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = AnalogLinear(input_size, 3 * hidden_size, bias, device, dtype, rpu_config)
        self.weight_hh = AnalogLinear(hidden_size, 3 * hidden_size, bias, device, dtype, rpu_config)

    def get_zero_state(self, batch_size: int) -> LSTMState:
        """Returns a zeroed state.

        Args:
            batch_size: batch size of the input

        Returns:
           Zeroed state tensor
        """
        param = next(self.parameters())
        return LSTMState(
            zeros(batch_size, self.hidden_size, device=param.device, dtype=param.dtype),
            None,  # cx unused
        )

    def forward(self, input_: Tensor, state: Tensor) -> Tuple[Tensor, Tuple[Tensor, None]]:
        """Forward pass.

        Args:
            input_: input tensor
            state: LSTM state tensor

        Returns:
            output h_y and output states h_y (which is the same here)
        """
        # pylint: disable=too-many-locals
        h_x, _ = state
        g_i = self.weight_ih(input_)
        g_h = self.weight_hh(h_x)
        i_r, i_i, i_n = g_i.chunk(3, 1)
        h_r, h_i, h_n = g_h.chunk(3, 1)

        reset_gate = sigmoid(i_r + h_r)
        input_gate = sigmoid(i_i + h_i)
        new_gate = tanh(i_n + reset_gate * h_n)
        hidden_y = new_gate + input_gate * (h_x - new_gate)

        return hidden_y, (hidden_y, None)  # output will also be hidden state
