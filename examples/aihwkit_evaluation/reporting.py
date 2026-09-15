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

"""Printing and evaluation feedback helpers for the AIHWKIT evaluation example."""

from collections import Counter
from typing import Dict, List

import torch

SEPARATOR = "=" * 92
NAME_WIDTH = 28


def layer_summary(module: torch.nn.Module) -> str:
    """Count the leaf modules of ``module`` by type."""
    counts = Counter(
        type(child).__name__ for child in module.modules() if not list(child.children())
    )
    return ", ".join(f"{name}: {count}" for name, count in sorted(counts.items()))


def print_model(title: str, module: torch.nn.Module) -> None:
    """Print ``module`` together with a summary of its leaf modules."""
    print(f"\n{SEPARATOR}\n{title}\n{SEPARATOR}")
    print(module)
    print(f"\nLeaf modules -> {layer_summary(module)}")


def deviation(baseline: torch.Tensor, outputs: torch.Tensor) -> Dict[str, float]:
    """Measure how far the ``outputs`` logits deviate from the ``baseline`` logits."""
    return {
        "abs_error": (outputs - baseline).abs().mean().item(),
        "rel_l2": (torch.linalg.norm(outputs - baseline) / torch.linalg.norm(baseline)).item(),
        "agreement": (outputs.argmax(dim=-1) == baseline.argmax(dim=-1)).float().mean().item(),
    }


def print_table_header(n_inputs: int, n_repetitions: int) -> None:
    """Print the header of the evaluation table."""
    print(f"\n{SEPARATOR}")
    print(f"Evaluation against the floating point model ({n_inputs} random inputs,")
    print(f"{n_repetitions} repetitions per drift time). The model is untrained: these are")
    print("deviations from the baseline, not accuracies.")
    print(SEPARATOR)
    print(f"{'':<{NAME_WIDTH}} {'mean |err|':>10}   {'rel. L2 error':>18}   {'top-1 agr.':>8}")


def print_row(name: str, deviations: List[Dict[str, float]]) -> None:
    """Print one row of the evaluation table, averaged over the repetitions."""
    values = {key: torch.tensor([dev[key] for dev in deviations]) for key in deviations[0]}
    spread = values["rel_l2"].std().item() if len(deviations) > 1 else 0.0
    print(
        f"{name:<{NAME_WIDTH}} {values['abs_error'].mean():10.4f}   "
        f"{values['rel_l2'].mean():7.2%} +/- {spread:6.2%}   {values['agreement'].mean():8.1%}"
    )


def print_takeaways() -> None:
    """Print how to read the evaluation table."""
    print(
        "\nWhat to look for:"
        "\n  * AIHWKIT-Lightning matches the floating point model, as the default"
        "\n    TorchInferenceRPUConfig adds no inference-time noise. This checks that the"
        "\n    conversion and the export preserved the network."
        "\n  * The AIHWKIT rows add PCM programming noise and conductance drift. The deviation"
        "\n    is already sizeable right after programming and keeps growing with the inference"
        "\n    time, since the drift compensation only corrects the global weight scale."
        "\n  * The spread across repetitions is large, so single-shot numbers are noisy."
        "\n    Average over several repetitions before comparing configurations."
    )
