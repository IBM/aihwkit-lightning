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

"""Helper function for computing sliced abs-max."""

# import torch
import triton
import triton.language as tl
from torch import full, Tensor, float32


# fmt: off
@triton.autotune(
    # pylint: disable=line-too-long
    configs=[
        triton.Config({"BLOCK_SIZE_N_COLS": 32, "BLOCK_SIZE_N_ROWS": 32}, num_stages=3, num_warps=1),  # noqa: E501
        triton.Config({"BLOCK_SIZE_N_COLS": 64, "BLOCK_SIZE_N_ROWS": 32}, num_stages=3, num_warps=8),  # noqa: E501
        triton.Config({"BLOCK_SIZE_N_COLS": 128, "BLOCK_SIZE_N_ROWS": 32}, num_stages=3, num_warps=8),  # noqa: E501
        triton.Config({"BLOCK_SIZE_N_COLS": 256, "BLOCK_SIZE_N_ROWS": 32}, num_stages=3, num_warps=8),  # noqa: E501
        triton.Config({"BLOCK_SIZE_N_COLS": 32, "BLOCK_SIZE_N_ROWS": 64}, num_stages=3, num_warps=8),  # noqa: E501
        triton.Config({"BLOCK_SIZE_N_COLS": 64, "BLOCK_SIZE_N_ROWS": 64}, num_stages=3, num_warps=8),  # noqa: E501
        triton.Config({"BLOCK_SIZE_N_COLS": 128, "BLOCK_SIZE_N_ROWS": 64}, num_stages=3, num_warps=8),  # noqa: E501
        triton.Config({"BLOCK_SIZE_N_COLS": 256, "BLOCK_SIZE_N_ROWS": 64}, num_stages=3, num_warps=8),  # noqa: E501
        triton.Config({"BLOCK_SIZE_N_COLS": 32, "BLOCK_SIZE_N_ROWS": 128}, num_stages=3, num_warps=8),  # noqa: E501
        triton.Config({"BLOCK_SIZE_N_COLS": 64, "BLOCK_SIZE_N_ROWS": 128}, num_stages=3, num_warps=8),  # noqa: E501
        triton.Config({"BLOCK_SIZE_N_COLS": 128, "BLOCK_SIZE_N_ROWS": 128}, num_stages=3, num_warps=8),  # noqa: E501
        triton.Config({"BLOCK_SIZE_N_COLS": 256, "BLOCK_SIZE_N_ROWS": 128}, num_stages=3, num_warps=8),  # noqa: E501
        triton.Config({"BLOCK_SIZE_N_COLS": 32, "BLOCK_SIZE_N_ROWS": 256}, num_stages=3, num_warps=8),  # noqa: E501
        triton.Config({"BLOCK_SIZE_N_COLS": 64, "BLOCK_SIZE_N_ROWS": 256}, num_stages=3, num_warps=8),  # noqa: E501
        triton.Config({"BLOCK_SIZE_N_COLS": 128, "BLOCK_SIZE_N_ROWS": 256}, num_stages=3, num_warps=8),  # noqa: E501
        triton.Config({"BLOCK_SIZE_N_COLS": 256, "BLOCK_SIZE_N_ROWS": 256}, num_stages=3, num_warps=8),  # noqa: E501
    ],
    key=["n_cols", "n_rows"],
    reset_to_zero=["per_channel_amax_ptr"],
)
@triton.jit
def fast_abs_max_kernel(
    weights_ptr,
    per_channel_amax_ptr,
    col_stride,
    row_stride,
    n_cols,
    n_rows,
    BLOCK_SIZE_N_COLS: tl.constexpr,
    BLOCK_SIZE_N_ROWS: tl.constexpr,
):
    """
    Computes the per-channel absolute maximum of the weights.

    Args:
        weights_ptr (pointer): pointer to the weights
        per_channel_amax_ptr (pointer): pointer to the per-channel amax output vector
        col_stride (int): stride for moving to the next row of the matrix (next column)
        row_stride (int): stride for moving to the next column of the matrix (next row)
        n_cols (int): number of columns in the weight matrix (assuming x @ W^T).
            So n_cols is n_rows of W
        n_rows (int): number of rows. - same as above -
        BLOCK_SIZE_N_COLS (tl.constexpr): block size for operating along the columns
        BLOCK_SIZE_N_ROWS (tl.constexpr): block size for operating along the rows
    """

    # pylint: disable=too-many-locals, invalid-name

    pid = tl.program_id(0)  # axis is 0 since we are on a 1D grid

    # n_pid_cols = tl.cdiv(n_cols, BLOCK_SIZE_N_COLS)
    n_pid_rows = tl.cdiv(n_rows, BLOCK_SIZE_N_ROWS)

    # what block are we in?
    col_block_idx = pid // n_pid_rows
    row_block_idx = pid % n_pid_rows

    # create the pointer array used for loading
    col_offs = (col_block_idx * BLOCK_SIZE_N_COLS + tl.arange(0, BLOCK_SIZE_N_COLS))
    row_offs = (row_block_idx * BLOCK_SIZE_N_ROWS + tl.arange(0, BLOCK_SIZE_N_ROWS))

    ptrs = weights_ptr + (col_offs[:, None] * col_stride + row_offs[None, :] * row_stride)
    # shape: [BLOCK_SIZE_N_COLS, BLOCK_SIZE_N_ROWS]
    block_weights = tl.load(
        ptrs,
        mask=(col_offs[:, None] < n_cols) & (row_offs[None, :] < n_rows),
        other=float("-inf")
    )

    block_weights_abs = tl.where(
        (col_offs[:, None] < n_cols) & (row_offs[None, :] < n_rows),
        tl.abs(block_weights),
        float("-inf")
    )

    # shape
    # [BLOCK_SIZE_N_COLS]
    abs_max_block_weights = tl.max(block_weights_abs, axis=1)

    # atomic max
    tl.atomic_max(
        per_channel_amax_ptr + col_offs,
        abs_max_block_weights.to(tl.float32),
        mask=col_offs < n_cols
    )
# fmt: on


def fast_abs_max(weights: Tensor):
    """
    Computest the per-channel absolute maximum of the weights.

    Args:
        weights (torch.Tensor): [n_cols, n_rows], x @ weights.T is assumed.
            Computes amax over the rows.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: [1],[n_cols], the per-tensor abs max
            and the per-channel abs max.
    """
    assert weights.ndim == 2, "weight matrix shape must have 2 dimensions"
    assert weights.is_contiguous(), "weight matrix must be contiguous"
    n_cols, n_rows = weights.shape

    # allocate output tensors
    per_channel_amax = full(
        size=(n_cols,), fill_value=float("-inf"), dtype=float32, device=weights.device
    )

    # invoke kernel
    def grid(meta):
        return (
            triton.cdiv(n_cols, meta["BLOCK_SIZE_N_COLS"])
            * triton.cdiv(n_rows, meta["BLOCK_SIZE_N_ROWS"]),
        )
    fast_abs_max_kernel[grid](
        weights, per_channel_amax, weights.stride(0), weights.stride(1), n_cols, n_rows
    )

    per_channel_amax = per_channel_amax.to(weights.dtype)
    return per_channel_amax.max(), per_channel_amax


if __name__ == "__main__":
    from torch import randn, allclose, float16

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["n_cols", "n_rows"],
            x_vals=[128 * i for i in range(2, 33)],
            line_arg="provider",
            line_vals=["torch", "triton"],
            line_names=["PyTorch", "Triton"],
            styles=[("green", "-"), ("blue", "-")],
            ylabel="Time [ms]",
            plot_name="abs-max performance",
            args={},
        )
    )
    def benchmark(n_cols: int, n_rows: int, provider: str):
        """
        Benchmark the absolute maximum computation.

        Args:
            n_cols (int): Number of columns
            n_rows (int): Number of rows
            provider (int): torch or triton

        Returns:
            Tuple[float]: Median, min, max of the runtimes in ms
        """
        weights = randn((n_cols, n_rows), device="cuda", dtype=float16)
        quantiles = [0.5, 0.2, 0.8]

        if provider == "torch":

            def bench():
                amax = weights.abs().amax(dim=1)
                return amax.max(), amax

            time_ms, min_ms, max_ms = triton.testing.do_bench(bench, quantiles=quantiles)
        if provider == "triton":
            time_ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: fast_abs_max(weights), quantiles=quantiles
            )
        return time_ms, max_ms, min_ms

    # TODO: pass the slices (per input region) and compute per-channel abs-max for each slice

    benchmark.run(print_data=True, save_path="debug/abs_max_perf")
    sizes = [1, 10, 25, 255, 257, 1023, 1025]
    for n_cols_ in sizes:
        for n_rows_ in sizes:
            print(f"n_cols: {n_cols_}, n_rows: {n_rows_}")
            weights_ = randn((n_cols_, n_rows_), device="cuda", dtype=float32)
            amax_ = weights_.abs().amax(dim=1)
            triton_amax = fast_abs_max(weights_)
            assert allclose(amax_, triton_amax[1])
