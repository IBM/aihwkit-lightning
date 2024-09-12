<img src="docs/assets/cover.png" alt="mamba header image" height="200"/>

# AIHWKIT-Lightning ⚡
[![Documentation Status](https://readthedocs.org/projects/aihwkit_lightning/badge/?version=latest)](https://aihwkit_lightning.readthedocs.io/en/latest/?badge=latest)
[![Build Status](https://app.travis-ci.com/IBM/aihwkit-lightning.svg?token=nYQJ7muZkSyoDYxxh5yc&branch=main)](https://travis-ci.com/IBM/aihwkit-lightning)

## Installation
```bash
git clone git@github.ibm.com:AIHW/aihwkit-lightning.git
cd aihwkit-lightning
pip install -e .
```

For the `triton` mode to work, you need to have >= GPU compute capability 7.0 (e.g. V100, A100, H100) and have `triton` installed.
Currently, only the nightly version works, which can be installed using
```bash
pip install -U --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/Triton-Nightly/pypi/simple/ triton-nightly
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


## License
This project is licensed under the [MIT License].

[MIT License]: LICENSE.txt
[examples]: examples/
[Huggingface]: examples/basic_huggingface/test_huggingface.py
[SLURM + DeepSpeed + Huggingface Accelerate + AIHWKIT-Lightning]: examples/deepspeed_and_huggingface/
[DeepSpeed + AIHWKIT-Lightning]: examples/deepspeed_cifar10
