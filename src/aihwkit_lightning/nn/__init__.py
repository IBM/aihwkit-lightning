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

"""Neural network module."""

# Convenience imports for easier access to the classes.

from aihwkit_lightning.nn.modules.container import AnalogSequential, AnalogWrapper
from aihwkit_lightning.nn.modules.linear import AnalogLinear
from aihwkit_lightning.nn.modules.conv import AnalogConv1d, AnalogConv2d
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
        raise RuntimeError(e)
