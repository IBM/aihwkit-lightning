<div style="text-align: center;">
<img src="docs/assets/cover.png" alt="cover" height="500"/>
</div>

# AIHWKIT-Lightning ⚡
[![Documentation Status](https://readthedocs.org/projects/aihwkit_lightning/badge/?version=latest)](https://aihwkit_lightning.readthedocs.io/en/latest/?badge=latest)
[![Build Status](https://app.travis-ci.com/IBM/aihwkit-lightning.svg?token=nYQJ7muZkSyoDYxxh5yc&branch=main)](https://travis-ci.com/IBM/aihwkit-lightning)

## What is AIHWKIT-Lightning?
**A fast and scalable toolkit for hardware-aware training of large neural networks on Analog In-Memory Computing (AIMC) hardware.**

AIHWKIT-Lightning is a streamlined, performance-optimized toolkit designed for efficient hardware-aware training of large language models and neural networks deployed on analog in-memory computing systems. Built by IBM Research, it addresses the computational demands of training billion-parameter models with analog hardware non-idealities.

## Key Capabilities

- **🚀 High Performance**: Up to 3.7× faster training compared to existing frameworks with lower memory consumption
- **⚡ GPU-Accelerated**: Dedicated CUDA and Triton kernels for optimal performance on modern GPUs (V100, A100, H100)
- **🎯 Hardware-Aware Training**: Simulates analog hardware non-idealities including:
 - Weight and output noise injection
 - Input/output quantization (DAC/ADC effects)
 - Weight clipping and modification
- **📈 Scalable**: Successfully demonstrated on models up to 3.8B parameters trained on billions of tokens
- **🔧 Easy Integration**: Drop-in replacement for PyTorch layers with minimal code changes
- **🎛️ Tiling Support**: Efficient handling of large weight matrices across multiple analog tiles

## Relationship to AIHWKIT

AIHWKIT-Lightning is a specialized, performance-focused toolkit that complements IBM's [AIHWKIT](https://github.com/IBM/aihwkit) - the comprehensive analog AI hardware simulation framework. While AIHWKIT offers extensive features for research and development with rich simulation capabilities, AIHWKIT-Lightning prioritizes speed and scalability for production-scale training:

- **AIHWKIT**: Feature-rich, comprehensive simulation, ideal for research and prototyping
- **AIHWKIT-Lightning**: Streamlined, fast, optimized for large-scale training

Models can be easily converted between the two frameworks, allowing you to train efficiently with Lightning and then export to AIHWKIT for detailed analysis and inference simulation.

## Installing the nightly version (recommended)
```bash
pip install git+https://github.com/IBM/aihwkit-lightning.git
```
or with

```bash
git clone git@github.com:IBM/aihwkit-lightning.git
cd aihwkit-lightning
pip install -e .
```

## Installing a previous version
> Note: For previous versions, make sure your setuptools is at version `setuptools==75.1.0`.

For version `v1.0.1`, you can do
```bash
pip install scikit-build
pip install git+https://github.com/IBM/aihwkit-lightning.git@v1.0.1
```
or with

```bash
git clone git@github.com:IBM/aihwkit-lightning.git
pip install scikit-build
git checkout v1.0.1
cd aihwkit-lightning
pip install -e .
```

## Examples

### Basic Training
```python
from torch import Tensor
from torch.nn.functional import mse_loss
from torch.optim import SGD

# Import the aihwkit constructs.
from aihwkit_lightning.nn import AnalogLinear
from aihwkit_lightning.optim import AnalogOptimizer
from aihwkit_lightning.simulator.configs import TorchInferenceRPUConfig

x = Tensor([[0.1, 0.2, 0.4, 0.3], [0.2, 0.1, 0.1, 0.3]])
y = Tensor([[1.0, 0.5], [0.7, 0.3]])

# Define a network using a single Analog layer.
model = AnalogLinear(4, 2, rpu_config=TorchInferenceRPUConfig())

# Use the analog-aware stochastic gradient descent optimizer.
opt = AnalogOptimizer(SGD, model.analog_layers(), model.parameters(), lr=0.01)

# Train the network.
for epoch in range(10):
    pred = model(x)
    loss = mse_loss(pred, y)
    loss.backward()
    opt.step()
    print(f"Loss error: {loss:.4f}")
```

### Advanced Training

In the [examples] folder, we have some examples that show how to use the AIHWKIT-Lightning:
- [Huggingface] shows how to train a network with Huggingface and AIHWKIT-Lightning.
- [DeepSpeed + AIHWKIT-Lightning] shows how to integrate AIHWKIT-Lightning with DeepSpeed.
- [SLURM + DeepSpeed + Huggingface Accelerate + AIHWKIT-Lightning] shows how to do multi-node training of a language model using DeepSpeed, Slurm, Huggingface Accelerate and AIHWKIT-Lightning.

### Exporting to AIHWKIT
One can easily convert any model trained with AIHWKIT-Lightning to AIHWKIT.

```python
from aihwkit_lightning.nn.export import export_to_aihwkit
# `model` is a model from AIHWKIT-Lightning
# `max_output_size` <= 0 means that we do not
# split layers along the output dimension
aihwkit_model = export_to_aihwkit(model=model, max_output_size=-1)
```

## Contributing
Before starting to write code for a possible contribution, please get in touch with us on Slack or by opening an issue. We can then discuss whether the proposed feature makes sense for this toolkit and how
to proceed.

For bug-fixes, please follow the instructions below.

Install the development requirements.
```bash
pip install -r requirements_dev.txt
mypy --install-types
```
Create a fork from the `main` branch and make a well-documented PR. Make sure to run the following before submitting the PR:
```bash
make pytest
make black
make mypy
make pycodestyle
make pylint
```
All of these should pass.

## `triton` mode

For [triton](https://triton-lang.org/main/index.html) on a CPU, you can build it using
```bash
git clone https://github.com/triton-lang/triton.git;
cd triton/python;
pip install ninja cmake wheel; # build-time dependencies
pip install -e .
```
and then when you want to run/test stuff in triton on a CPU, you can prepend `TRITON_CPU_BACKEND=1` before your script.
This is a feature added from [Triton-CPU](https://github.com/triton-lang/triton-cpu). This mode is very handy for debugging.

For [triton](https://triton-lang.org/main/index.html) on a GPU, you need to have >= GPU compute capability 7.0 (e.g. V100, A100, H100) and have
triton installed. Currently, only the nightly version works, which can be installed using
```bash
pip install -U --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/Triton-Nightly/pypi/simple/ triton-nightly
```

AIHWKIT-Lightning can be accelerated using [triton](https://triton-lang.org/main/index.html). This generally only makes sense when your layer sizes are in the thousands and when you want to split the layer into multiple tiles (only across the input dimension is supported).
To enable triton for `AnalogConv2d` and `AnalogLinear`, either `export AIHWKIT_USE_TRITON="1"` or execute your script as such `AIHWKIT_USE_TRITON="1" python your_script.py`. This feature is off by default.

### Current limitations of the `triton` mode
The triton kernel is generally faster than the normal PyTorch implementation, and much faster when you set `mapping.max_input_size` to something other than `0 or -1`, i.e. you split your matrix into tiles.
However, some things are still not optimal. Until these points are resolved, we consider the `triton` mode experimental.

- The sliced `std()` kernel that calculates the `std()` for slices of a tensor is not very fast. Fixing this, would speed up the scenario where we chunk the weight matrix along the input dimension significantly.


## Further notes
- Tests checking the correctness against AIHWKIT are passing. Becuase we don't normalize inputs and weights, tests for `float16` and `bfloat16` only pass for high `atol`. When normalizing the input (which is not needed and adds extra FLOPs, tests are also passing in half precision).
- Currently, `torch.compile` doesn't work when input range learning is activated, because a leaf variable requiring gradients gets updated in the forward pass.
- Input range learning is made up of three gradients. Our own gradient + the gradients resulting from the operations `inp_slice = inp_slice / input_range[slice_idx]` and `out_slice *= input_range[slice_idx]`. These two gradients are not accessible in `triton` mode. We verified that the downstrean accuracy was not affected by this. The pure PyTorch version also leaves out these gradients, but the custom gradient is correct compared the AIHWKIT.


## Authors
IBM Research has developed AIHWKIT-Lightning with Julian Büchel as the initial core author.
You can contact us by opening a new issue in the repository.

## Cite
```
@inproceedings{
aihwkitlightning,
    title={AIHWKIT-Lightning: A Scalable HW-Aware Training Toolkit for Analog In-Memory Computing},
    author={Julian Büchel and William Andrew Simon and Corey Lammie and Giovanni Acampa and Kaoutar El Maghraoui and Manuel Le Gallo and Abu Sebastian},
    booktitle={NeurIPS 2024 Workshop Machine Learning with new Compute Paradigms},
    year={2024},
    url={https://openreview.net/forum?id=QNdxOgGmhR}
}
```

## License
This project is licensed under the [MIT License].

[MIT License]: LICENSE.txt
[examples]: examples/
[Huggingface]: examples/basic_huggingface/test_huggingface.py
[SLURM + DeepSpeed + Huggingface Accelerate + AIHWKIT-Lightning]: examples/deepspeed_and_huggingface/
[DeepSpeed + AIHWKIT-Lightning]: examples/deepspeed_cifar10
