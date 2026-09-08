# -*- coding: utf-8 -*-

# (C) Copyright 2026 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""Functions related to computing max-abs for weight slices."""
from torch import Tensor, full


def sliced_abs_max(upper_end_of_slices: Tensor, weights: Tensor) -> Tensor:
    """TODO"""

    assert weights.ndim == 2, "Weight must be 2D"
    n_splits = upper_end_of_slices.numel()
    if n_splits == 1:
        return weights.abs().amax(dim=-1).unsqueeze(0)

    # allocate output tensors
    per_channel_amax = full(
        size=(n_splits, weights.size(0)),
        fill_value=float("-inf"),
        dtype=weights.dtype,
        device=weights.device,
    )

    for slice_idx, upper in enumerate(upper_end_of_slices):
        start = 0 if slice_idx == 0 else upper_end_of_slices[slice_idx - 1]
        end = upper
        slice_weights = weights[:, start:end]  # type: ignore[misc]
        per_channel_amax[slice_idx] = slice_weights.abs().amax(dim=1)

    return per_channel_amax
