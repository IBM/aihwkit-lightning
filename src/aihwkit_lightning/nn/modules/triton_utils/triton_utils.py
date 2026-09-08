# -*- coding: utf-8 -*-

# (C) Copyright 2026 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.
"""Helper to disable autotuner in decorator."""

from inspect import signature
from triton.runtime.autotuner import Autotuner, autotune


def requires_blocksizes(fn, grid):
    """
    Does this function require passing block sizes?

    Args:
        fn (callable) : Function such as sliced_fast_abs_max_kernel
        grid (callable) : The grid used to launch the kernel
    Returns (bool) Whether this function requires passing block sizes.
    """
    if isinstance(fn, Autotuner):
        return False
    sig = signature(fn[grid].fn)
    return any("BLOCK_SIZE" in p.name for p in sig.parameters.values())


def lightning_autotune(*args, enable: bool = False, **kwargs):
    """
    If enable, will return the autotuner as usual.
    If not enable, will just return the jit'ed function.
    """
    if enable:
        return autotune(*args, **kwargs)

    def decorator(fn):
        return fn

    return decorator
