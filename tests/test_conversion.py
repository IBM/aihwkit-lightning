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

from torch import Tensor
from torch.nn import Module, Linear
from aihwkit_lightning.nn import AnalogLinear
from aihwkit_lightning.nn.conversion import convert_to_analog


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


def test_conversion():
    """
    Test the correctness of the conversion to analog.
    """
    # Create a model
    model = Model()

    # Convert the model to analog
    analog_model = convert_to_analog(
        model,
        rpu_config=None,
        conversion_map=None,
        specific_rpu_config_fun=None,
        module_name="",
        ensure_analog_root=True,
        exclude_modules=None,
        inplace=False,
        verbose=False,
    )

    # Check that the model has been converted to analog
    assert isinstance(analog_model.fc1, AnalogLinear)
    assert isinstance(analog_model.fc2, AnalogLinear)


if __name__ == "__main__":
    test_conversion()
