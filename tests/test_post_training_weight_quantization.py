# -*- coding: utf-8 -*-

# (C) Copyright 2025 IBM. All Rights Reserved.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=too-many-locals, too-many-public-methods, no-member
"""Test for post-training weight quantization."""

from unittest import SkipTest
from pytest import mark
from torch import Tensor, randn, allclose
from torch.nn import Module, Linear, Conv1d, Conv2d
from torch.optim import AdamW
from aihwkit_lightning.nn.conversion import convert_to_analog
from aihwkit_lightning.simulator.configs import (
    TorchInferenceRPUConfig,
    WeightQuantizationType,
    WeightClipType,
)
from aihwkit_lightning.optim import AnalogOptimizer


class WeightQuantTestModel(Module):
    """Test model."""

    def __init__(self):
        super().__init__()
        self.conv1 = Conv1d(in_channels=3, out_channels=3, kernel_size=3, padding=1)
        self.conv2 = Conv2d(in_channels=3, out_channels=1024, kernel_size=3)
        self.fc1 = Linear(1024, 512)
        self.fc2 = Linear(512, 10)

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
        x = x.reshape(x.shape[0], -1, 1024)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


@mark.parametrize(
    "weight_quantization_type",
    [
        WeightQuantizationType.DISCRETIZE,
        WeightQuantizationType.DISCRETIZE_PER_CHANNEL,
        WeightQuantizationType.NONE,
    ],
)
@mark.parametrize("num_weight_bits", [2, 4, 8])
@mark.parametrize("max_input_size", [256, 512])
@mark.parametrize(
    "clip_type",
    [
        WeightClipType.LAYER_GAUSSIAN,
        WeightClipType.LAYER_GAUSSIAN_PER_CHANNEL,
        WeightClipType.LEARNABLE_PER_CHANNEL,
        WeightClipType.NONE,
    ],
)
def test_post_training_weight_quantization(
    weight_quantization_type: WeightQuantizationType,
    num_weight_bits: int,
    max_input_size: int,
    clip_type: WeightClipType,
):
    """Test loading the optimizer state and compare to torch"""

    if clip_type != WeightClipType.NONE and weight_quantization_type != WeightQuantizationType.NONE:
        if (
            "CHANNEL" in str(clip_type)
            and "CHANNEL" not in str(weight_quantization_type)
            or "CHANNEL" not in str(clip_type)
            and "CHANNEL" in str(weight_quantization_type)
        ):
            raise SkipTest("Cannot mix channel and tensor")

    rpu_config = TorchInferenceRPUConfig()
    rpu_config.modifier.quantization_type = weight_quantization_type
    rpu_config.modifier.res = 2**num_weight_bits - 2  # 5 bits, from [-15, to 15]
    rpu_config.mapping.max_input_size = max_input_size
    rpu_config.clip.type = clip_type
    rpu_config.clip.sigma = 2.0

    inp = randn(2, 3, 100)

    model = WeightQuantTestModel()

    analog_model = convert_to_analog(model, rpu_config=rpu_config)

    # create an analog optimizer
    optim = AnalogOptimizer(
        AdamW, analog_model.analog_layers(), analog_model.parameters(), lr=0.001
    )
    for _ in range(1):
        optim.zero_grad()
        out = analog_model(inp)
        loss = out.sum()
        loss.backward()
        optim.step()

    training_loss = analog_model(inp).sum()
    analog_model.quantize_weights()
    training_loss_quantized = analog_model(inp).sum()
    analog_model.eval()
    eval_loss_quantized = analog_model(inp).sum()

    assert allclose(training_loss_quantized, eval_loss_quantized, atol=1e-5)
    assert allclose(training_loss, training_loss_quantized, atol=1e-5)
    assert allclose(training_loss, eval_loss_quantized, atol=1e-5)


if __name__ == "__main__":
    test_post_training_weight_quantization(
        weight_quantization_type=WeightQuantizationType.DISCRETIZE_PER_CHANNEL,
        num_weight_bits=4,
        max_input_size=512,
        clip_type=WeightClipType.NONE,
    )
