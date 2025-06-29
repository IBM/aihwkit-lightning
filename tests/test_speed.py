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

# pylint: disable=too-many-locals, too-many-public-methods, no-member
"""Test the speed."""

import os
from typing import Union
from contextlib import redirect_stdout
from unittest import SkipTest
from pytest import mark

from torch import dtype as torch_dtype
from torch import device as torch_device
from torch import cuda as torch_cuda
from torch import randn, float32, float16, bfloat16, Tensor
from torch import compile as torch_compile
from torch.optim import Optimizer, AdamW
from torch.nn import Linear
from torch.utils import benchmark

from aihwkit.nn import AnalogLinear as AIHWKITAnalogLinear
from aihwkit.simulator.configs import TorchInferenceRPUConfig as AIHWKITRPUConfig
from aihwkit.simulator.configs import NoiseManagementType, BoundManagementType
from aihwkit.simulator.configs import WeightModifierType as AIHWKITWeightModifierType
from aihwkit.simulator.configs import WeightClipType as AIHWKITWeightClipType
from aihwkit.simulator.configs import WeightRemapType
from aihwkit.optim.analog_optimizer import AnalogOptimizerMixin

from aihwkit_lightning.nn import AnalogLinear
from aihwkit_lightning.simulator.configs import TorchInferenceRPUConfig as RPUConfig
from aihwkit_lightning.simulator.configs import WeightNoiseInjectionType, WeightClipType
from aihwkit_lightning.optim import AnalogOptimizer


TRITON_AVAIL = False
try:
    import triton

    # pylint: disable=ungrouped-imports
    from aihwkit_lightning.nn.modules.linear import is_at_least_volta_gpu

    if not is_at_least_volta_gpu():
        raise ImportError("GPU must at least be Volta")
    TRITON_AVAIL = True
except ImportError:
    print("Could not import triton_utils.triton_linear. Using PyTorch variant.")


SKIP_CUDA_TESTS = os.getenv("SKIP_CUDA_TESTS") or not torch_cuda.is_available()


@mark.parametrize("is_test", [True, False])
@mark.parametrize("inp_size", [10])
@mark.parametrize("out_size", [10])
@mark.parametrize("bias", [True])
@mark.parametrize("inp_res", [2**8 - 2])
@mark.parametrize("max_inp_size", [256])
@mark.parametrize("ir_enable", [True, False])
@mark.parametrize("ir_learn_input_range", [True, False])
@mark.parametrize("ir_init_value", [2.0])
@mark.parametrize("ir_init_from_data", [-1, 0, 10])
@mark.parametrize("ir_init_std_alpha", [2.0])
@mark.parametrize("device", ["cpu", "cuda"])
@mark.parametrize("dtype", [float32, float16, bfloat16])
def test_torch_compile(  # pylint: disable=too-many-arguments
    is_test: bool,
    inp_size: int,
    out_size: int,
    bias: bool,
    inp_res: float,
    max_inp_size: int,
    ir_enable: bool,
    ir_learn_input_range: bool,
    ir_init_value: float,
    ir_init_from_data: int,
    ir_init_std_alpha: float,
    device: torch_device,
    dtype: torch_dtype,
):
    """Test the speed of the forward pass."""

    if device == "cuda" and SKIP_CUDA_TESTS:
        raise SkipTest("CUDA tests are disabled/ can't be performed")

    if not ir_enable and inp_res > 0:
        raise SkipTest("IR not enabled but inp_res > 0")

    if ir_enable:
        raise SkipTest("Compile doesn't work with IR learning. We're working on that.")

    def populate_rpu(rpu_config: RPUConfig):
        rpu_config.forward.inp_res = inp_res
        rpu_config.forward.out_res = -1
        rpu_config.forward.out_bound = -1
        rpu_config.forward.out_noise = 0.0
        rpu_config.mapping.max_input_size = max_inp_size
        rpu_config.pre_post.input_range.enable = ir_enable
        rpu_config.pre_post.input_range.learn_input_range = ir_learn_input_range
        rpu_config.pre_post.input_range.init_value = ir_init_value
        rpu_config.pre_post.input_range.init_from_data = ir_init_from_data
        rpu_config.pre_post.input_range.init_std_alpha = ir_init_std_alpha
        return rpu_config

    rpu = populate_rpu(RPUConfig())
    linear = AnalogLinear(in_features=inp_size, out_features=out_size, bias=bias, rpu_config=rpu)
    if is_test:
        linear = linear.eval()
    linear = linear.to(device=device, dtype=dtype)
    compiled_linear = torch_compile(linear)
    inp = randn(inp_size, device=device, dtype=dtype)
    linear(inp)  # pylint: disable=not-callable
    compiled_linear(inp)

    @torch_compile
    def forward_backward(model: AnalogLinear, inp: Tensor):
        out = model(inp)
        out.sum().backward()
        return out

    forward_backward(linear, inp)


def gen_rpu(
    ir_enable: bool,
    weight_noise_enable: bool,
    clip_enable: bool,
    out_noise_enable: bool,
    adc_enable: bool,
):
    """Generate the RPU configuration."""
    is_perfect = False
    if not (ir_enable or weight_noise_enable or out_noise_enable):
        is_perfect = True

    def rpu(rpu_config: Union[AIHWKITRPUConfig, RPUConfig]):
        rpu_config.forward.is_perfect = is_perfect
        rpu_config.mapping.max_input_size = -1
        rpu_config.forward.inp_res = 254 if ir_enable else -1
        rpu_config.forward.out_res = 254 if adc_enable else -1
        rpu_config.forward.out_bound = 12 if adc_enable else -1
        rpu_config.forward.out_noise = 0.02 if out_noise_enable else 0.0
        rpu_config.pre_post.input_range.enable = ir_enable
        rpu_config.pre_post.input_range.learn_input_range = True
        rpu_config.pre_post.input_range.init_from_data = 0
        rpu_config.clip.sigma = 2.0
        rpu_config.modifier.std_dev = 0.01

        if isinstance(rpu_config, AIHWKITRPUConfig):
            rpu_config.forward.is_perfect = is_perfect
            rpu_config.mapping.max_output_size = -1
            rpu_config.mapping.learn_out_scaling = False
            rpu_config.mapping.weight_scaling_omega = 1.0
            rpu_config.mapping.weight_scaling_columnwise = False
            rpu_config.mapping.out_scaling_columnwise = False
            rpu_config.forward.noise_management = (
                NoiseManagementType.ABS_MAX if not ir_enable else NoiseManagementType.NONE
            )
            rpu_config.forward.bound_management = BoundManagementType.NONE
            rpu_config.remap.type = WeightRemapType.LAYERWISE_SYMMETRIC

            rpu_config.modifier.noise_type = (
                AIHWKITWeightModifierType.ADD_NORMAL
                if weight_noise_enable
                else AIHWKITWeightModifierType.NONE
            )
            rpu_config.clip.type = (
                AIHWKITWeightClipType.LAYER_GAUSSIAN if clip_enable else AIHWKITWeightClipType.NONE
            )
        else:
            rpu_config.modifier.noise_type = (
                WeightNoiseInjectionType.ADD_NORMAL
                if weight_noise_enable
                else WeightNoiseInjectionType.NONE
            )
            rpu_config.clip.type = (
                WeightClipType.LAYER_GAUSSIAN if clip_enable else WeightClipType.NONE
            )
        return rpu_config

    lightning_config = rpu(RPUConfig())
    aihwkit_config = rpu(AIHWKITRPUConfig())

    return lightning_config, aihwkit_config


def linear_forward_backward_and_step(
    linear: Union[Linear, AnalogLinear], inp: Tensor, optimizer: Optimizer
):
    """Perform the forward, backward and step operations."""
    out = linear(inp)
    out.sum().backward()
    optimizer.step()


def benchmark_test(stmt, function, inp, optim, num_threads, label, sub_label, description, results):
    """Benchmark function with specified parameters"""
    results.append(
        benchmark.Timer(
            stmt=f"{stmt}({function[0]}, inp, optim)",
            setup=f"from __main__ import {stmt}",
            globals={f"{function[0]}": function[1], "inp": inp, "optim": optim},
            num_threads=num_threads,
            label=label,
            sub_label=sub_label,
            description=description,
        ).timeit(5000)
    )
    return results


def benchmark_linear_speed_and_peak_memory_of_fwd_bwd(
    lightning_rpu_config: RPUConfig, aihwkit_rpu_config: AIHWKITRPUConfig
):
    """Benchmark the speed and peak memory of the forward pass."""
    bsz = 512
    dtype = float16
    device = torch_device("cuda" if torch_cuda.is_available() else "cpu")
    assert device == torch_device("cuda"), "Running this on a CPU is not recommended."

    sizes = [128 * 2**i for i in range(5)]
    results = []

    for size in sizes:
        linear = AnalogLinear(
            in_features=size, out_features=size, bias=True, rpu_config=lightning_rpu_config
        )
        linear = linear.to(device=device, dtype=dtype)
        lightning_optim = AnalogOptimizer(
            AdamW, linear.analog_layers(), linear.parameters(), lr=0.0
        )

        aihwkit_linear = AIHWKITAnalogLinear(
            in_features=size, out_features=size, bias=True, rpu_config=aihwkit_rpu_config
        )
        aihwkit_linear.remap_analog_weights()
        aihwkit_linear = aihwkit_linear.to(device=device, dtype=dtype)

        class AihwkitAnalogAdamW(AnalogOptimizerMixin, AdamW):
            """AIHWKIT AdamW optimizer."""

        aihwkit_optim = AihwkitAnalogAdamW(aihwkit_linear.parameters(), lr=0.0)

        torch_linear = Linear(size, size, bias=True)
        torch_linear = torch_linear.to(device=device, dtype=dtype)
        torch_optim = AdamW(torch_linear.parameters(), lr=0.0)

        inp = randn((bsz, size), device=device, dtype=dtype, requires_grad=True)

        label = "LinearFwdBwdStep"
        sub_label = f"[{size}, {size}]"
        layers = [linear, aihwkit_linear, torch_linear]
        optims = [lightning_optim, aihwkit_optim, torch_optim]
        descriptions = ["AIHWKIT (lightning)", "AIHWKIT", "Torch"]
        for num_threads in [1]:
            for test, optim, description in zip(layers, optims, descriptions):
                results = benchmark_test(
                    "linear_forward_backward_and_step",
                    ("linear", test),
                    inp,
                    optim,
                    num_threads,
                    label,
                    sub_label,
                    description,
                    results,
                )

    compare = benchmark.Compare(results)
    redirect_print(str(compare))


def redirect_print(string: str):
    """Redirect the print to a file."""

    with open("debug/benchmarks/out.txt", "w") as file:  # pylint: disable=unspecified-encoding
        with redirect_stdout(file):
            print(string)
    print(string, flush=True)


def benchmark_aihwkit_lightning():
    """Benchmark the speed of the fwd bwd step for differenet rpu-configs."""

    os.makedirs("debug/benchmarks", exist_ok=True)
    redirect_print("-------------------------------------")
    redirect_print("----------Nothing turned on----------")
    lightning_rpu_config, aihwkit_rpu_config = gen_rpu(
        ir_enable=False,
        weight_noise_enable=False,
        clip_enable=False,
        out_noise_enable=False,
        adc_enable=False,
    )
    benchmark_linear_speed_and_peak_memory_of_fwd_bwd(lightning_rpu_config, aihwkit_rpu_config)
    redirect_print("=====================================\n\n")

    redirect_print("--------------------------------------")
    redirect_print("---------------Clipping---------------")
    lightning_rpu_clip, aihwkit_rpu_clip = gen_rpu(
        ir_enable=False,
        weight_noise_enable=False,
        clip_enable=True,
        out_noise_enable=False,
        adc_enable=False,
    )
    benchmark_linear_speed_and_peak_memory_of_fwd_bwd(lightning_rpu_clip, aihwkit_rpu_clip)
    redirect_print("======================================\n\n")

    redirect_print("-----------------------------------------")
    redirect_print("---------------WeightNoise---------------")
    lightning_rpu_weight_noise, aihwkit_rpu_weight_noise = gen_rpu(
        ir_enable=False,
        weight_noise_enable=True,
        clip_enable=False,
        out_noise_enable=False,
        adc_enable=False,
    )
    benchmark_linear_speed_and_peak_memory_of_fwd_bwd(
        lightning_rpu_weight_noise, aihwkit_rpu_weight_noise
    )
    redirect_print("=========================================\n\n")

    redirect_print("--------------------------------")
    redirect_print("---------------IR---------------")
    lightning_rpu_ir, aihwkit_rpu_ir = gen_rpu(
        ir_enable=True,
        weight_noise_enable=False,
        clip_enable=False,
        out_noise_enable=False,
        adc_enable=False,
    )
    benchmark_linear_speed_and_peak_memory_of_fwd_bwd(lightning_rpu_ir, aihwkit_rpu_ir)
    redirect_print("================================\n\n")

    redirect_print("-----------------------------------------------------")
    redirect_print("---------------Clipping+WeightNoise+IR---------------")
    lightning_rpu_all, aihwkit_rpu_all = gen_rpu(
        ir_enable=True,
        weight_noise_enable=True,
        clip_enable=True,
        out_noise_enable=False,
        adc_enable=False,
    )
    benchmark_linear_speed_and_peak_memory_of_fwd_bwd(lightning_rpu_all, aihwkit_rpu_all)
    redirect_print("=====================================================")


def benchmark_triton_implementation(max_input_size: int):
    """Test the speed of the triton implementation compared to AIHWKIT."""
    assert TRITON_AVAIL, "Triton is not available"

    def bench(layer: AnalogLinear, inp: Tensor, optim: AdamW):
        """Pass the inputs through the layer"""
        out = layer(inp)
        loss = out.mean()
        loss.backward()
        optim.step()

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["n_cols", "n_rows"],
            x_vals=[128 * i for i in range(8, 40, 4)],
            line_arg="provider",
            line_vals=["aihwkit", "lightning", "triton"],
            line_names=["AIHWKIT", "AIHWKIT-Lightning", "Triton"],
            styles=[("green", "-"), ("blue", "-"), ("red", "-")],
            ylabel="Time [ms]",
            plot_name="linear performance fwd",
            args={"max_input_size": max_input_size},
        )
    )
    def layer_benchmark(n_cols: int, n_rows: int, provider: str, max_input_size: int):
        """
        Benchmark the linear layer.

        Args:
            n_cols (int): Number of columns
            n_rows (int): Number of rows
            provider (int): torch or triton
            max_input_size (int): Maximum input size

        Returns:
            Tuple[float]: Median, min, max of the runtimes in ms
        """
        quantiles = [0.5, 0.2, 0.8]
        bsz = 512
        dtype = float16
        device = torch_device("cuda" if torch_cuda.is_available() else "cpu")
        assert device == torch_device("cuda"), "Running this on a CPU is not recommended."
        rpu_config, aihwkit_rpu_config = gen_rpu(
            ir_enable=True,
            weight_noise_enable=True,
            clip_enable=True,
            out_noise_enable=False,
            adc_enable=False,
        )
        rpu_config.mapping.max_input_size = max_input_size
        aihwkit_rpu_config.mapping.max_input_size = max_input_size

        layer = AnalogLinear(
            in_features=n_rows,
            out_features=n_cols,
            bias=False,
            rpu_config=rpu_config,
            device=device,
            dtype=dtype,
        )
        lightning_optim = AnalogOptimizer(AdamW, layer.analog_layers(), layer.parameters(), lr=0.0)
        aihwkit_layer = AIHWKITAnalogLinear(
            in_features=n_rows, out_features=n_cols, bias=False, rpu_config=aihwkit_rpu_config
        )
        aihwkit_layer.remap_analog_weights()
        aihwkit_layer = aihwkit_layer.to(dtype=dtype, device=device)

        class AihwkitAnalogAdamW(AnalogOptimizerMixin, AdamW):
            """AIHWKIT AdamW optimizer."""

        aihwkit_optim = AihwkitAnalogAdamW(aihwkit_layer.parameters(), lr=0.0)

        inp = randn(bsz, n_rows, device=device, dtype=dtype, requires_grad=True)
        if provider == "triton":
            os.environ["AIHWKIT_USE_TRITON"] = "1"
            time_ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: bench(layer, inp, lightning_optim), quantiles=quantiles
            )
            del os.environ["AIHWKIT_USE_TRITON"]
        elif provider == "aihwkit":
            time_ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: bench(aihwkit_layer, inp, aihwkit_optim), quantiles=quantiles
            )
        elif provider == "lightning":
            time_ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: bench(layer, inp, lightning_optim), quantiles=quantiles
            )
        return time_ms, max_ms, min_ms

    save_path = f"debug/linear_performance_fwd_bwd_torch_vs_triton_max_input_size_{max_input_size}"
    layer_benchmark.run(print_data=True, save_path=save_path)


if __name__ == "__main__":
    benchmark_triton_implementation(max_input_size=-1)
    benchmark_triton_implementation(max_input_size=512)
    benchmark_aihwkit_lightning()
