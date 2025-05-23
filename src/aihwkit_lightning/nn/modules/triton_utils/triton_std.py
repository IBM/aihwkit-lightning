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

"""Helper function for computing sliced standard deviation."""

import triton  # type: ignore
import triton.language as tl  # type: ignore
from torch import zeros, Tensor, float32, tensor, cat, sqrt, empty, cuda
from .triton_utils import lightning_autotune, requires_blocksizes


# fmt: off
@lightning_autotune(
    # pylint: disable=line-too-long
    configs=[
        triton.Config({"BLOCK_SIZE_INP": 32, "BLOCK_SIZE_HIDDEN": 32}, num_stages=3, num_warps=1),  # noqa: E501
        triton.Config({"BLOCK_SIZE_INP": 64, "BLOCK_SIZE_HIDDEN": 32}, num_stages=3, num_warps=8),  # noqa: E501
        triton.Config({"BLOCK_SIZE_INP": 128, "BLOCK_SIZE_HIDDEN": 32}, num_stages=3, num_warps=8),  # noqa: E501
        triton.Config({"BLOCK_SIZE_INP": 256, "BLOCK_SIZE_HIDDEN": 32}, num_stages=3, num_warps=8),  # noqa: E501
        triton.Config({"BLOCK_SIZE_INP": 32, "BLOCK_SIZE_HIDDEN": 64}, num_stages=3, num_warps=8),  # noqa: E501
        triton.Config({"BLOCK_SIZE_INP": 64, "BLOCK_SIZE_HIDDEN": 64}, num_stages=3, num_warps=8),  # noqa: E501
        triton.Config({"BLOCK_SIZE_INP": 128, "BLOCK_SIZE_HIDDEN": 64}, num_stages=3, num_warps=8),  # noqa: E501
        triton.Config({"BLOCK_SIZE_INP": 256, "BLOCK_SIZE_HIDDEN": 64}, num_stages=3, num_warps=8),  # noqa: E501
        triton.Config({"BLOCK_SIZE_INP": 32, "BLOCK_SIZE_HIDDEN": 128}, num_stages=3, num_warps=8),  # noqa: E501
        triton.Config({"BLOCK_SIZE_INP": 64, "BLOCK_SIZE_HIDDEN": 128}, num_stages=3, num_warps=8),  # noqa: E501
        triton.Config({"BLOCK_SIZE_INP": 128, "BLOCK_SIZE_HIDDEN": 128}, num_stages=3, num_warps=8),  # noqa: E501
        triton.Config({"BLOCK_SIZE_INP": 256, "BLOCK_SIZE_HIDDEN": 128}, num_stages=3, num_warps=8),  # noqa: E501
        triton.Config({"BLOCK_SIZE_INP": 32, "BLOCK_SIZE_HIDDEN": 256}, num_stages=3, num_warps=8),  # noqa: E501
        triton.Config({"BLOCK_SIZE_INP": 64, "BLOCK_SIZE_HIDDEN": 256}, num_stages=3, num_warps=8),  # noqa: E501
        triton.Config({"BLOCK_SIZE_INP": 128, "BLOCK_SIZE_HIDDEN": 256}, num_stages=3, num_warps=8),  # noqa: E501
        triton.Config({"BLOCK_SIZE_INP": 256, "BLOCK_SIZE_HIDDEN": 256}, num_stages=3, num_warps=8),  # noqa: E501
    ],
    key=["inp_size", "hidden_size"],
    reset_to_zero=["per_slice_sum_ptr"],
    enable=cuda.is_available()
)
@triton.jit
def per_slice_sum_kernel(  # pylint: disable=too-many-locals
    # pointers to tensors
    inp_ptr,  # 2D [inp_size, hidden_size]
    upper_end_of_slices_ptr,  # 1D [num_slices]
    per_slice_sum_ptr,  # 1D [num_slices]
    # sizes
    inp_size,
    hidden_size,
    num_slices,
    # strides
    stride_inp_inp_size,
    stride_inp_hidden_size,
    # block sizes
    BLOCK_SIZE_INP: tl.constexpr,  # pylint: disable=invalid-name
    BLOCK_SIZE_HIDDEN: tl.constexpr,  # pylint: disable=invalid-name
):
    """Compute the sum per slice."""
    pid = tl.program_id(axis=0)
    offs_am = pid * BLOCK_SIZE_INP + tl.arange(0, BLOCK_SIZE_INP)
    ir_range_lower = 0
    for slice_idx in range(0, num_slices):
        ir_range_upper = tl.load(upper_end_of_slices_ptr + slice_idx)
        current_lower = ir_range_lower

        if num_slices == 1:
            num_k = tl.cdiv(ir_range_upper - ir_range_lower, BLOCK_SIZE_HIDDEN)
        else:
            num_k = tl.cdiv(hidden_size, BLOCK_SIZE_HIDDEN)

        offs_k = current_lower + tl.arange(0, BLOCK_SIZE_HIDDEN)
        a_ptrs = inp_ptr + (
            offs_am[:, None] * stride_inp_inp_size + offs_k[None, :] * stride_inp_hidden_size
        )

        for k in range(0, num_k):
            current_upper = min(
                ir_range_upper, ir_range_lower + (k + 1) * BLOCK_SIZE_HIDDEN, hidden_size
            )
            inp_block = tl.load(
                a_ptrs,
                mask=(offs_am[:, None] < inp_size) & (offs_k[None, :] < current_upper),
                other=0.0,
            )
            inp_block = inp_block.to(tl.float32)
            tl.atomic_add(per_slice_sum_ptr + slice_idx, tl.sum(inp_block))

            # increment pointer
            offs_k += current_upper - current_lower
            a_ptrs += (current_upper - current_lower) * stride_inp_hidden_size

            current_lower = current_upper
        ir_range_lower = ir_range_upper


@lightning_autotune(
    # pylint: disable=line-too-long
    configs=[
        triton.Config({"BLOCK_SIZE_INP": 32, "BLOCK_SIZE_HIDDEN": 32}, num_stages=3, num_warps=1),  # noqa: E501
        triton.Config({"BLOCK_SIZE_INP": 64, "BLOCK_SIZE_HIDDEN": 32}, num_stages=3, num_warps=8),  # noqa: E501
        triton.Config({"BLOCK_SIZE_INP": 128, "BLOCK_SIZE_HIDDEN": 32}, num_stages=3, num_warps=8),  # noqa: E501
        triton.Config({"BLOCK_SIZE_INP": 256, "BLOCK_SIZE_HIDDEN": 32}, num_stages=3, num_warps=8),  # noqa: E501
        triton.Config({"BLOCK_SIZE_INP": 32, "BLOCK_SIZE_HIDDEN": 64}, num_stages=3, num_warps=8),  # noqa: E501
        triton.Config({"BLOCK_SIZE_INP": 64, "BLOCK_SIZE_HIDDEN": 64}, num_stages=3, num_warps=8),  # noqa: E501
        triton.Config({"BLOCK_SIZE_INP": 128, "BLOCK_SIZE_HIDDEN": 64}, num_stages=3, num_warps=8),  # noqa: E501
        triton.Config({"BLOCK_SIZE_INP": 256, "BLOCK_SIZE_HIDDEN": 64}, num_stages=3, num_warps=8),  # noqa: E501
        triton.Config({"BLOCK_SIZE_INP": 32, "BLOCK_SIZE_HIDDEN": 128}, num_stages=3, num_warps=8),  # noqa: E501
        triton.Config({"BLOCK_SIZE_INP": 64, "BLOCK_SIZE_HIDDEN": 128}, num_stages=3, num_warps=8),  # noqa: E501
        triton.Config({"BLOCK_SIZE_INP": 128, "BLOCK_SIZE_HIDDEN": 128}, num_stages=3, num_warps=8),  # noqa: E501
        triton.Config({"BLOCK_SIZE_INP": 256, "BLOCK_SIZE_HIDDEN": 128}, num_stages=3, num_warps=8),  # noqa: E501
        triton.Config({"BLOCK_SIZE_INP": 32, "BLOCK_SIZE_HIDDEN": 256}, num_stages=3, num_warps=8),  # noqa: E501
        triton.Config({"BLOCK_SIZE_INP": 64, "BLOCK_SIZE_HIDDEN": 256}, num_stages=3, num_warps=8),  # noqa: E501
        triton.Config({"BLOCK_SIZE_INP": 128, "BLOCK_SIZE_HIDDEN": 256}, num_stages=3, num_warps=8),  # noqa: E501
        triton.Config({"BLOCK_SIZE_INP": 256, "BLOCK_SIZE_HIDDEN": 256}, num_stages=3, num_warps=8),  # noqa: E501
    ],
    key=["inp_size", "hidden_size"],
    reset_to_zero=["per_slice_centered_and_squared_ptr"],
    enable=cuda.is_available()
)
@triton.jit
def center_and_square_kernel(  # pylint: disable=too-many-locals, too-many-arguments
    # pointers to tensors
    inp_ptr,  # 2D [inp_size, hidden_size]
    upper_end_of_slices_ptr,  # 1D [num_slices]
    per_slice_mean_ptr,  # 1D [num_slices]
    per_slice_centered_and_squared_ptr,  # 1D [num_slices]
    # sizes
    inp_size,
    hidden_size,
    num_slices,
    # strides
    stride_inp_inp_size,
    stride_inp_hidden_size,
    # block sizes
    BLOCK_SIZE_INP: tl.constexpr,  # pylint: disable=invalid-name
    BLOCK_SIZE_HIDDEN: tl.constexpr,  # pylint: disable=invalid-name
):
    """Center and square each slice."""
    pid = tl.program_id(axis=0)
    offs_am = pid * BLOCK_SIZE_INP + tl.arange(0, BLOCK_SIZE_INP)
    ir_range_lower = 0
    for slice_idx in range(0, num_slices):
        current_mean = tl.load(per_slice_mean_ptr + slice_idx)
        ir_range_upper = tl.load(upper_end_of_slices_ptr + slice_idx)
        current_lower = ir_range_lower
        if num_slices == 1:
            num_k = tl.cdiv(ir_range_upper - ir_range_lower, BLOCK_SIZE_HIDDEN)
        else:
            num_k = tl.cdiv(hidden_size, BLOCK_SIZE_HIDDEN)
        offs_k = current_lower + tl.arange(0, BLOCK_SIZE_HIDDEN)
        a_ptrs = inp_ptr + (
            offs_am[:, None] * stride_inp_inp_size + offs_k[None, :] * stride_inp_hidden_size
        )
        for k in range(0, num_k):
            current_upper = min(
                ir_range_upper, ir_range_lower + (k + 1) * BLOCK_SIZE_HIDDEN, hidden_size
            )

            inp_block = tl.load(
                a_ptrs,
                mask=(offs_am[:, None] < inp_size) & (offs_k[None, :] < current_upper),
                other=current_mean,
            )
            inp_block = inp_block.to(tl.float32)
            a_centered = inp_block - current_mean
            centered_and_squared = tl.sum((a_centered * a_centered))
            tl.atomic_add(per_slice_centered_and_squared_ptr + slice_idx, centered_and_squared)

            # increment pointer
            offs_k += current_upper - current_lower
            a_ptrs += (current_upper - current_lower) * stride_inp_hidden_size

            current_lower = current_upper
        ir_range_lower = ir_range_upper
# fmt: on


# pylint: disable=redefined-outer-name
def sliced_fast_std(inp: Tensor, upper_end_of_slices: Tensor):
    """
    Given inputs of shape [..., hidden_size] and upper_end_of_slices Tensor
    that has the upper index of each slice of the inputs,
    computes the per-slice std of the inputs.
    For example, upper_end_of_slices can be [16, 32].
    Then the stds would be calculated for ranges inps[..., :16],
    inps[..., 16:32].

    Args:
        inp: Tensor of shape [..., hidden_size]
        upper_end_of_slices: Tensor of shape [num_slices]
    Returns:
        stds: Tensor of shape [num_slices]
    """
    # shortcut if not sliced
    if upper_end_of_slices.numel() == 1:
        return inp.std().view(1, 1)

    inp = inp.flatten(end_dim=-2)
    inp_size, hidden_size = inp.shape
    upper_end_of_slices = upper_end_of_slices.flatten().contiguous()
    num_slices = upper_end_of_slices.numel()
    per_slice_sum = zeros((num_slices,), device=inp.device, dtype=float32)
    num_inputs_per_slice = inp_size * cat(
        [upper_end_of_slices[0].view((1,)), (upper_end_of_slices[1:] - upper_end_of_slices[:-1])]
    )

    def per_slice_grid(meta):
        return (triton.cdiv(inp_size, meta["BLOCK_SIZE_INP"]),)

    block_args = (32, 32) if requires_blocksizes(per_slice_sum_kernel, per_slice_grid) else tuple()
    per_slice_sum_kernel[per_slice_grid](
        inp,
        upper_end_of_slices,
        per_slice_sum,
        inp_size,
        hidden_size,
        num_slices,
        inp.stride(0),
        inp.stride(1),
        *block_args,
    )
    per_slice_mean = per_slice_sum / num_inputs_per_slice

    per_slice_centered_and_squared = zeros((num_slices,), device=inp.device, dtype=float32)

    center_and_square_kernel[per_slice_grid](
        inp,
        upper_end_of_slices,
        per_slice_mean,
        per_slice_centered_and_squared,
        inp_size,
        hidden_size,
        num_slices,
        inp.stride(0),
        inp.stride(1),
        *block_args,
    )
    per_slice_std = sqrt(per_slice_centered_and_squared / (num_inputs_per_slice - 1))
    per_slice_std = per_slice_std.to(dtype=inp.dtype)
    return per_slice_std


def naive_per_slice_std(inp: Tensor, upper_end_of_slices: Tensor):
    """
    Given inputs of shape [..., hidden_size] and upper_end_of_slices Tensor
    that has the upper index of each slice of the inputs,
    computes the per-slice std of the inputs.
    For example, upper_end_of_slices can be [16, 32].
    Then the stds would be calculated for ranges inps[..., :16],
    inps[..., 16:32].

    Args:
        inp: Tensor of shape [..., hidden_size]
        upper_end_of_slices: Tensor of shape [num_slices]
    Returns:
        stds: Tensor of shape [num_slices]
    """
    # shortcut if not sliced
    if upper_end_of_slices.numel() == 1:
        return inp.std().view(1, 1)
    lower = 0
    stds = empty((upper_end_of_slices.numel(),), device=inp.device, dtype=inp.dtype)
    for idx, upper in enumerate(upper_end_of_slices):
        stds[idx] = inp[:, lower:upper].std()
        lower = upper
    return stds


if __name__ == "__main__":
    from torch import randn, float16, allclose, manual_seed

    manual_seed(0)

    rand_inp = randn((1, 63), device="cuda", dtype=float32)
    split_sizes = [10, 12, 13, 2, 2, 24]
    upper_end_of_slices = (
        tensor(split_sizes, device=rand_inp.device, dtype=rand_inp.dtype)
        .cumsum(dim=0)
        .contiguous()
        .int()
    )

    per_slice_stds = sliced_fast_std(rand_inp, upper_end_of_slices)
    stds = naive_per_slice_std(rand_inp, upper_end_of_slices)

    assert allclose(stds, per_slice_stds, atol=1e-5)

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["hidden_size"],
            x_vals=[128 * i for i in range(8, 40, 4)],
            line_arg="provider",
            line_vals=["torch", "triton"],
            line_names=["PyTorch", "Triton"],
            styles=[("green", "-"), ("blue", "-")],
            ylabel="Time [ms]",
            plot_name="sliced-std performance",
            args={},
        )
    )
    def benchmark(hidden_size: int, provider: str):
        """
        Benchmarks sliced std computation

        Args:
            hidden_size (int): Hidden dimension
            provider (int): torch or triton

        Returns:
            Tuple[float]: Median, min, max of the runtimes in ms
        """
        print(f"{provider}: Hidden size {hidden_size}")
        inp_size = 1000
        inp = randn((inp_size, hidden_size), device="cuda", dtype=float16)
        interval = hidden_size // 4
        split_sizes = [interval for _ in range(3)] + [hidden_size - int(3 * interval)]
        upper_end_of_slices = (
            tensor(split_sizes, device=rand_inp.device, dtype=rand_inp.dtype)
            .cumsum(dim=0)
            .contiguous()
            .int()
        )

        quantiles = [0.5, 0.2, 0.8]

        if provider == "torch":
            time_ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: naive_per_slice_std(inp, upper_end_of_slices), quantiles=quantiles
            )
        if provider == "triton":
            time_ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: sliced_fast_std(inp, upper_end_of_slices), quantiles=quantiles
            )
        return time_ms, max_ms, min_ms

    benchmark.run(print_data=True, save_path="debug/sliced_std_perf")
