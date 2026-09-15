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

"""Example on how to evaluate an AIHWKIT-Lightning model with AIHWKIT."""

import torch
from aihwkit.inference.noise.pcm import PCMLikeNoiseModel
from aihwkit.inference.compensation.drift import GlobalDriftCompensation
from model import resnet32
from reporting import deviation, print_model, print_row, print_table_header, print_takeaways
from aihwkit_lightning.simulator.configs import TorchInferenceRPUConfig
from aihwkit_lightning.nn.conversion import convert_to_analog
from aihwkit_lightning.nn.export import export_to_aihwkit

# Drift times (in seconds) at which the analog model is evaluated.
T_INFERENCES = [("1 s", 1.0), ("1 h", 3600.0), ("1 day", 86400.0), ("1 year", 365 * 86400.0)]

# Programming and drift noise are stochastic, so every drift time is repeated.
N_REPETITIONS = 5

if __name__ == "__main__":
    torch.manual_seed(0)

    model = resnet32()
    model.eval()
    print_model("Digital (floating point) model", model)

    rpu_config = TorchInferenceRPUConfig()
    # `convert_to_analog` copies the model by default, so `model` stays digital.
    analog_model = convert_to_analog(model, rpu_config)
    analog_model.eval()
    print_model("Analog model (AIHWKIT-Lightning)", analog_model)

    aihwkit_model = export_to_aihwkit(model=analog_model, max_output_size=-1)
    aihwkit_model.to(torch.float32)
    for analog_tile in aihwkit_model.analog_tiles():
        new_rpu_config = analog_tile.rpu_config
        break

    new_rpu_config.noise_model = PCMLikeNoiseModel(g_max=25.0)
    new_rpu_config.drift_compensation = GlobalDriftCompensation()
    aihwkit_model.replace_rpu_config(new_rpu_config)
    aihwkit_model.eval()

    # Evaluation. The weights are random (the model is not trained), so the numbers below
    # report how much each analog model deviates from the floating point baseline. They are
    # not accuracies, and the top-1 agreement is only a rough proxy for how much the
    # hardware non-idealities perturb the decisions of this particular network.
    inputs = torch.randn(8, 3, 32, 32)
    with torch.no_grad():
        reference = model(inputs)

        print_table_header(n_inputs=inputs.shape[0], n_repetitions=N_REPETITIONS)
        print_row("AIHWKIT-Lightning", [deviation(reference, analog_model(inputs))])

        for label, t_inference in T_INFERENCES:
            devs = []
            for _ in range(N_REPETITIONS):
                # Programs the weights (first call only) and drifts them to `t_inference`.
                aihwkit_model.drift_analog_weights(t_inference)
                devs.append(deviation(reference, aihwkit_model(inputs)))
            print_row(f"AIHWKIT @ t = {label}", devs)

    print_takeaways()
