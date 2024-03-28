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

import triton  # type: ignore
import triton.language as tl  # type: ignore
from torch import full, Tensor, float32, int32, tensor


# fmt: off
@triton.autotune(
    # pylint: disable=line-too-long
    configs=[
        triton.Config({"BLOCK_SIZE_N_COLS": 32, "BLOCK_SIZE_N_ROWS": 32}, num_stages=3, num_warps=1),  # noqa: E501
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
def sliced_fast_abs_max_kernel(  # pylint: disable=too-many-arguments
    weights_ptr,
    split_upper_bounds_ptr,
    per_channel_amax_ptr,
    col_stride,
    row_stride,
    per_channel_amax_row_stride,
    n_splits,
    n_cols,
    n_rows,
    BLOCK_SIZE_N_COLS: tl.constexpr,
    BLOCK_SIZE_N_ROWS: tl.constexpr,
):
    """
    Computes the per-channel absolute maximum of the weights.

    Args:
        weights_ptr (pointer): pointer to the weights
        split_upper_bounds_ptr (pointer): pointer to the split upper bounds
        per_channel_amax_ptr (pointer): pointer to the per-channel amax output vector
        col_stride (int): stride for moving to the next row of the matrix (next column)
        row_stride (int): stride for moving to the next column of the matrix (next row)
        per_channel_amax_row_stride (int): stride for moving to the next row of the per-channel amax
        n_splits (int): number of splits
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

    # what is the min row we have?
    min_row = row_block_idx * BLOCK_SIZE_N_ROWS
    max_row = min(min_row + BLOCK_SIZE_N_ROWS, n_rows)

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

    lower_bound = 0
    slice_idx = 0
    for upper_bound_idx in range(0, n_splits):
        upper_bound = tl.load(split_upper_bounds_ptr + upper_bound_idx)

        start = max(lower_bound, min_row)  # this will be at least min-row
        end = min(upper_bound, max_row)  # this will be at most max-row
        if start < end:
            masked_weights = tl.where(
                # this selects all columns and masks out the rows that are not in the range
                (row_offs[None, :] >= start) & (row_offs[None, :] < end),
                block_weights_abs,
                float("-inf")
            )
            # shape
            # [BLOCK_SIZE_N_COLS]
            abs_max_block_weights = tl.max(masked_weights, axis=1)

            # atomic max
            tl.atomic_max(
                (per_channel_amax_ptr + slice_idx * per_channel_amax_row_stride) + col_offs,
                abs_max_block_weights.to(tl.float32),
                mask=col_offs < n_cols
            )

        slice_idx += 1
        lower_bound = upper_bound


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
    return per_channel_amax


def sliced_fast_abs_max(weights: Tensor, split_sizes: list[int]):
    """
    Computest the per-channel absolute maximum of the weights.

    Args:
        weights (torch.Tensor): [n_cols, n_rows], x @ weights.T is assumed.
            Computes amax over the rows.
        split_sizes (list[int]): list of sizes for the splits across input dimension.
            256 -> [128, 128], 257 -> [85, 85, 86]

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: [1],[n_cols], the per-tensor abs max
            and the per-channel abs max.
    """
    if len(split_sizes) == 1:
        return fast_abs_max(weights)

    assert weights.ndim == 2, "weight matrix shape must have 2 dimensions"
    assert weights.is_contiguous(), "weight matrix must be contiguous"
    n_cols, n_rows = weights.shape

    # [6, 5, 5] -> [6, 11, 16] shows the end of the blocks
    split_upper_bounds = tensor(split_sizes, device=weights.device).cumsum(dim=0, dtype=int32)

    # allocate output tensors
    per_channel_amax = full(
        size=(len(split_sizes), n_cols),
        fill_value=float("-inf"),
        dtype=float32,
        device=weights.device,
    )

    # invoke kernel
    def grid(meta):
        return (
            triton.cdiv(n_cols, meta["BLOCK_SIZE_N_COLS"])
            * triton.cdiv(n_rows, meta["BLOCK_SIZE_N_ROWS"]),
        )

    sliced_fast_abs_max_kernel[grid](
        weights,
        split_upper_bounds,
        per_channel_amax,
        weights.stride(0),
        weights.stride(1),
        per_channel_amax.stride(0),
        len(split_sizes),
        n_cols,
        n_rows,
    )

    per_channel_amax = per_channel_amax.to(weights.dtype)
    return per_channel_amax


if __name__ == "__main__":
    from torch import randn, allclose, float16

    def get_split_size(size: int, max_size: int):
        """Get the split sizes for the given size and max size."""
        n_splits = (size + max_size - 1) // max_size
        base, extra = divmod(size, n_splits)
        return [base + (i < extra) for i in range(n_splits)]

    def bench(weights: Tensor, split_sizes: list[int]):
        """Torch groundtruth function"""
        if len(split_sizes) == 1:
            return weights.abs().amax(dim=1)
        # [6, 5, 5] -> [6, 11, 16] shows the end of the blocks
        split_upper_bounds = tensor(split_sizes, device=weights.device).cumsum(dim=0, dtype=int32)
        # allocate output tensors
        per_channel_amax = full(
            size=(len(split_sizes), weights.size(0)),
            fill_value=float("-inf"),
            dtype=weights.dtype,
            device=weights.device,
        )
        for slice_idx, upper in enumerate(split_upper_bounds):
            start = 0 if slice_idx == 0 else split_upper_bounds[slice_idx - 1]
            end = upper
            slice_weights = weights[:, start:end]  # type: ignore[misc]
            per_channel_amax[slice_idx] = slice_weights.abs().amax(dim=1)
        return per_channel_amax

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
        split_sizes = get_split_size(n_rows, max_size=128)

        quantiles = [0.5, 0.2, 0.8]

        if provider == "torch":
            time_ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: bench(weights, split_sizes), quantiles=quantiles
            )
        if provider == "triton":
            time_ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: sliced_fast_abs_max(weights, split_sizes), quantiles=quantiles
            )
        return time_ms, max_ms, min_ms

    benchmark.run(print_data=True, save_path="debug/sliced_abs_max_perf")
    sizes = [1, 10, 25, 255, 257, 1023, 1025]
    for n_cols_ in sizes:
        for n_rows_ in sizes:
            split_sizes_ = get_split_size(n_rows_, max_size=128)
            print(f"n_cols: {n_cols_}, n_rows: {n_rows_}")
            weights_ = randn((n_cols_, n_rows_), device="cuda", dtype=float32)
            amax_ = weights_.abs().amax(dim=1)
            amax_ = bench(weights_, split_sizes_)
            triton_amax = sliced_fast_abs_max(weights_, split_sizes_)
            assert allclose(amax_, triton_amax)
