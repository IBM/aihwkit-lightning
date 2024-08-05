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

""" Analog RNN modules. """

import warnings
import math
from copy import deepcopy
import re

from typing import Any, List, Optional, Tuple, Type, Callable
from torch import Tensor, jit
from torch.nn import Dropout, ModuleList, init, Module, Linear, LSTM
from torch.autograd import no_grad

from aihwkit_lightning.nn.modules.container import AnalogContainerBase
from aihwkit_lightning.nn.modules.linear import AnalogLinear
from aihwkit_lightning.nn.modules.rnn.cells import AnalogLSTMCell, AnalogGRUCell
from aihwkit_lightning.nn.modules.rnn.layers import AnalogRNNLayer, AnalogBidirRNNLayer
from aihwkit_lightning.simulator.configs import TorchInferenceRPUConfig
from .layers import AnalogRNNLayer, AnalogBidirRNNLayer


class ModularRNN(Module):
    """Helper class to create a Modular RNN

    Args:
        num_layers: number of serially connected RNN layers
        layer: RNN layer type (e.g. AnalogLSTMLayer)
        dropout: dropout applied to output of all RNN layers except last
        first_layer_args: RNNCell type, input_size, hidden_size, rpu_config, etc.
        other_layer_args: RNNCell type, hidden_size, hidden_size, rpu_config, etc.
    """

    # pylint: disable=abstract-method

    # Necessary for iterating through self.layers and dropout support
    __constants__ = ["layers", "num_layers"]

    def __init__(
        self,
        num_layers: int,
        layer: Type,
        dropout: float,
        first_layer_args: Any,
        other_layer_args: Any,
    ) -> None:
        super().__init__()
        self.layers = self.init_stacked_analog_lstm(
            num_layers, layer, first_layer_args, other_layer_args
        )
        self.num_layers = num_layers
        # Introduce a Dropout layer on the outputs of each RNN layer except
        # the last layer.
        if num_layers == 1 and dropout > 0:
            warnings.warn(
                "dropout lstm adds dropout layers after all but last "
                "recurrent layer, it expects num_layers greater than "
                "1, but got num_layers = 1"
            )
        self.dropout_layer = Dropout(dropout) if dropout > 0.0 else None

    @staticmethod
    def init_stacked_analog_lstm(
        num_layers: int, layer: Type, first_layer_args: Any, other_layer_args: Any
    ) -> ModuleList:
        """Construct a list of LSTMLayers over which to iterate.

        Args:
            num_layers: number of serially connected LSTM layers
            layer: RNN layer type (e.g. AnalogLSTMLayer)
            first_layer_args: RNNCell type, input_size, hidden_size, rpu_config, etc.
            other_layer_args: RNNCell type, hidden_size, hidden_size, rpu_config, etc.

        Returns:
            torch.nn.ModuleList, which is similar to a regular Python list,
            but where torch.nn.Module methods can be applied
        """
        layers = [layer(*first_layer_args)] + [
            layer(*other_layer_args) for _ in range(num_layers - 1)
        ]
        return ModuleList(layers)

    def get_zero_state(self, batch_size: int) -> List[Tensor]:
        """Returns a zeroed state.

        Args:
            batch_size: batch size of the input

        Returns:
           List of zeroed state tensors for each layer
        """
        return [lay.get_zero_state(batch_size) for lay in self.layers]

    def forward(  # pylint: disable=arguments-differ
        self, input: Tensor, states: List  # pylint: disable=redefined-builtin
    ) -> Tuple[Tensor, List]:
        """Forward pass.

        Args:
            input: input tensor
            states: list of LSTM state tensors

        Returns:
            outputs and states
        """
        # List[RNNState]: One state per layer.
        output_states = jit.annotate(List, [])
        output = input

        for i, rnn_layer in enumerate(self.layers):
            state = states[i]
            output, out_state = rnn_layer(output, state)
            # Apply the dropout layer except the last layer.
            if i < self.num_layers - 1 and self.dropout_layer is not None:
                output = self.dropout_layer(output)
            output_states += [out_state]

        return output, output_states


class AnalogRNN(AnalogContainerBase, Module):
    """Modular RNN that uses analog tiles.

    Args:
        cell: type of Analog RNN cell (AnalogLSTMCell/AnalogGRUCell/AnalogVanillaRNNCell)
        input_size: in_features to W_{ih} matrix of first layer
        hidden_size: in_features and out_features for W_{hh} matrices
        bias: whether to use a bias row on the analog tile or not
        num_layers: number of serially connected RNN layers
        bidir: if True, becomes a bidirectional RNN
        dropout: dropout applied to output of all RNN layers except last
        xavier: whether standard PyTorch LSTM weight
            initialization (default) or Xavier initialization
        rpu_config: configuration for an analog resistive processing
            unit. If not given a native torch model will be
            constructed instead.
    """

    # pylint: disable=abstract-method, too-many-arguments

    def __init__(
        self,
        cell: Type,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = False,
        dropout: float = 0.0,
        bidir: bool = False,
        proj_size: int = 0,
        xavier: bool = False,
        device=None,
        dtype=None,
        rpu_config: Optional[TorchInferenceRPUConfig] = None,
    ):
        super().__init__()

        if bidir:
            layer = AnalogBidirRNNLayer
            num_dirs = 2
        else:
            layer = AnalogRNNLayer
            num_dirs = 1
        # TODO: Implement batch_first
        if batch_first == True:
            raise RuntimeError("batch_first is not supported for AnalogRNN")
        # TODO: Implement proj_size > 0
        if proj_size != 0:
            raise RuntimeError("proj_size != 0 not supported for AnalogRNN")
        self.proj_size = 0

        self.rnn = ModularRNN(
            num_layers,
            layer,
            dropout,
            first_layer_args=[cell, input_size, hidden_size, bias, device, dtype, rpu_config],
            other_layer_args=[
                cell,
                num_dirs * hidden_size,
                hidden_size,
                bias,
                device,
                dtype,
                rpu_config,
            ],
        )
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bidirectional = bidir
        self.reset_parameters(xavier)
        self._register_load_state_dict_pre_hook(self.update_state_dict)
        self._register_state_dict_hook(self.return_pytorch_state_dict)

    @no_grad()
    def init_layers(
        self, weight_init_fn: Callable, bias_init_fn: Optional[Callable] = None
    ) -> None:
        """Init the analog layers with custom functions.

        Args:
            weight_init_fn: in-place tensor function applied to weight of
                ``AnalogLinear`` layers
            bias_init_fn: in-place tensor function applied to bias of
                ``AnalogLinear`` layers

        Note:
            If no bias init function is provided the weight init
            function is taken for the bias as well.
        """

        def init_weight_and_bias(weight: Tensor, bias: Optional[Tensor]) -> None:
            """Init the weight and bias"""
            weight_init_fn(weight.data)
            if bias is not None:
                if bias_init_fn is None:
                    weight_init_fn(bias.data)
                else:
                    bias_init_fn(bias.data)

        for module in self.modules():
            if isinstance(module, AnalogLinear):
                weight, bias = module.get_weights_and_biases()
                init_weight_and_bias(weight, bias)
                module.set_weights_and_biases(weight, bias)
            elif isinstance(module, Linear):
                # init torch layers if any
                init_weight_and_bias(module.weight, module.bias)

    def reset_parameters(self, xavier: bool = False) -> None:
        """Weight and bias initialization.

        Args:
            xavier: whether standard PyTorch LSTM weight
               initialization (default) or Xavier initialization
        """
        if xavier:
            self.init_layers(init.xavier_uniform_, init.zeros_)
        else:
            stdv = 1.0 / math.sqrt(self.hidden_size)
            self.init_layers(lambda x: x.uniform_(-stdv, stdv))

    def get_zero_state(self, batch_size: int) -> List[Tensor]:
        """Returns a zeroed RNN state based on cell type and layer type

        Args:
            batch_size: batch size of the input

        Returns:
           List of zeroed state tensors for each layer

        """
        return self.rnn.get_zero_state(batch_size)

    def forward(
        self, input: Tensor, states: Optional[List] = None  # pylint: disable=redefined-builtin
    ) -> Tuple[Tensor, List]:
        """Forward pass.

        Args:
            input: input tensor
            states: list of LSTM state tensors

        Returns:
            outputs and states
        """

        if input.dim() not in (2, 3):
            raise ValueError(
                f"RNN: Expected input to be 2D or 3D, got {input.dim()}D tensor instead"
            )
        batch_dim = 0 if self.batch_first else 1
        if states is not None and self.bidirectional:
            flattened_states = states if not self.bidirectional else [b for a in states for b in a]
        if states is not None and len(states) != self.num_layers:
            raise RuntimeError(f"Expecting {self.num_layers} states, but received {len(states)}")

        if not input.dim() == 3:
            input = input.unsqueeze(batch_dim)
            if states is not None:
                for state in flattened_states:
                    for t_name in ["hx", "cx"]:
                        d = getattr(state, t_name).dim()
                        if d != 1:
                            raise RuntimeError(
                                f"For unbatched 2-D input, {t_name} should be 1-D but got {d}-D tensor"
                            )
        else:
            if states is not None:
                for state in flattened_states:
                    for t_name in ["hx", "cx"]:
                        d = getattr(state, t_name).dim()
                        if d != 2:
                            raise RuntimeError(
                                f"For batched 3-D input, {t_name} should be 2-D but got {d}-D tensor"
                            )

        max_batch_size = input.size(1)
        if states is None:
            states = self.get_zero_state(max_batch_size)
            flattened_states = states if not self.bidirectional else [b for a in states for b in a]
        for state in flattened_states:
            self.check_forward_args(input, state, max_batch_size)

        return self.rnn(input, states)

    def check_input(self, input: Tensor, batch_size: int) -> None:
        if self.input_size != input.size(-1):
            raise RuntimeError(
                f"input.size(-1) must be equal to input_size. Expected {self.input_size}, got {input.size(-1)}"
            )

    def check_hidden_size(
        self,
        hx: Tensor,
        expected_hidden_size: Tuple[int, int, int],
        msg: str = "Expected hidden size {}, got {}",
    ) -> None:
        if hx.size() != expected_hidden_size:
            raise RuntimeError(msg.format(expected_hidden_size, list(hx.size())))

    def check_forward_args(self, input: Tensor, hidden: List, batch_size: int):
        self.check_input(input, batch_size)
        self.check_hidden_size(hidden.hx, (batch_size, self.hidden_size))
        if hidden.cx is not None:
            self.check_hidden_size(hidden.cx, (batch_size, self.hidden_size))

    def update_state_dict(module, state_dict, *args, **kwargs):
        if not any(["h_l" in a.rsplit(".", 1)[1] for a in state_dict.keys()]):
            return
        bidir = hasattr(module.rnn.layers[0], "directions")
        old_state_dict = deepcopy(state_dict)
        for key in old_state_dict.keys():
            stem, suffix = key.rsplit(".", 1)
            layer = re.search("h_l(\d+)", key).groups()[0]
            i_h = re.search("(hh|ih)", key).groups()[0]
            weight_bias = re.search("((weight|bias))", key).groups()[0]
            direction = 1 if "_reverse" in suffix else 0
            b_dir = f".directions.{direction}" if bidir else ""
            new_key = f"{stem}.rnn.layers.{layer}{b_dir}.cell.weight_{i_h}.{weight_bias}"
            state_dict[new_key] = state_dict.pop(key)

    def return_pytorch_state_dict(self, module, state_dict, prefix, local_metadata):
        keys = [key for key in state_dict.keys() if prefix in key]
        bidir = any(["directions" in key for key in keys])
        for key in keys:
            suffix = key.replace(f"{prefix}", "")
            if bidir:
                _, _, layer, _, dir, _, i_h, weight_bias = suffix.split(".")
            else:
                _, _, layer, _, i_h, weight_bias = suffix.split(".")
                dir = "0"
            i_h = i_h[-2:]
            reverse = "_reverse" if dir == "1" else ""
            pytorch_suffix = f"{prefix}{weight_bias}_{i_h}_l{layer}{reverse}"
            state_dict[pytorch_suffix] = state_dict.pop(key)

    @classmethod
    def from_digital(
        cls,
        module: LSTM,
        rpu_config: Optional[TorchInferenceRPUConfig] = None,
        realistic_read_write: bool = False,
    ) -> "AnalogRNN":
        analog_module = AnalogRNN(
            AnalogLSTMCell,
            module.input_size,
            module.hidden_size,
            module.num_layers,
            module.bias is not None,
            module.batch_first,
            module.dropout,
            module.bidirectional,
            module.proj_size,
            realistic_read_write,
            module.weight_hh_l0.device,
            module.weight_hh_l0.dtype,
            rpu_config,
        )
        for i, layer in enumerate(analog_module.rnn.layers):
            if analog_module.bidirectional == True:
                layer.directions[0].cell.weight_ih.set_weights_and_biases(
                    getattr(module, f"weight_ih_l{i}"), getattr(module, f"bias_ih_l{i}")
                )
                layer.directions[0].cell.weight_hh.set_weights_and_biases(
                    getattr(module, f"weight_hh_l{i}"), getattr(module, f"bias_hh_l{i}")
                )
                layer.directions[1].cell.weight_ih.set_weights_and_biases(
                    getattr(module, f"weight_ih_l{i}_reverse"),
                    getattr(module, f"bias_ih_l{i}_reverse"),
                )
                layer.directions[1].cell.weight_hh.set_weights_and_biases(
                    getattr(module, f"weight_hh_l{i}_reverse"),
                    getattr(module, f"bias_hh_l{i}_reverse"),
                )
            else:
                layer.cell.weight_ih.set_weights_and_biases(
                    getattr(module, f"weight_ih_l{i}"), getattr(module, f"bias_ih_l{i}")
                )
                layer.cell.weight_hh.set_weights_and_biases(
                    getattr(module, f"weight_hh_l{i}"), getattr(module, f"bias_hh_l{i}")
                )
        return analog_module
