# -*- coding: utf-8 -*-

# (C) Copyright 2026 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""Neural network module."""

# Convenience imports for easier access to the classes.

from aihwkit_lightning.nn.modules.container import AnalogSequential, AnalogWrapper
from aihwkit_lightning.nn.modules.linear import AnalogLinear
from aihwkit_lightning.nn.modules.conv import AnalogConv1d, AnalogConv2d, AnalogConv3d
from aihwkit_lightning.nn.modules.base import AnalogLayerBase
from aihwkit_lightning.nn.modules.torch_utils.torch_linear import TorchLinear
from aihwkit_lightning.nn.modules.rnn.rnn import AnalogRNN
from aihwkit_lightning.nn.modules.rnn.cells import (
    AnalogGRUCell,
    AnalogLSTMCell,
    AnalogVanillaRNNCell,
    AnalogLSTMCellCombinedWeight,
)

try:
    from aihwkit_lightning.nn.modules.triton_utils.triton_linear import TritonLinear
except ImportError:
    pass
    # if this fails and shouldn't fail, an exception will be raised down the road and not here
except RuntimeError as e:
    if str(e) != "0 active drivers ([]). There should only be one.":
        raise RuntimeError(e) from e
