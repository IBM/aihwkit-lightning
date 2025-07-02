# Training of a simple CNN using DeepSpeed
This example shows the code changes you have to make when using AIHWKIT Lightning with DeepSpeed.
The network used here is a very basic two-layer CNN that doesn't achieve good accuracy. This
is not a guide on HW-aware training, but more of an example code showing how you can
do multi-GPU training using only DeepSpeed.

## Prerequisites
First, install PyTorch.
To install DeepSpeed, you can clone it from `https://github.com/microsoft/DeepSpeed`, step into the repo and execute `pip install -e .`.
then, install triton from here:
```bash
pip install -U --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/Triton-Nightly/pypi/simple/ triton-nightly
[OPTIONAL] pip install deepspeed
```

> Note: The following used to be necessary for old versions of AIHWKIT-Lightning, but should
not be needed anymore. We recomment to try running the example before without making
the following changes.

We had to make another change for the example to run:
In
```bash
path/to/conda/environment/<env-name>/bin/deepspeed
```
we commented out some requirements that were not really necessary.
```python
#!.../miniconda3/envs/torch-nightly/bin/python
# EASY-INSTALL-DEV-SCRIPT: 'deepspeed==0.14.3+488a823','deepspeed'
# __requires__ = 'deepspeed==0.14.3+488a823'
__import__('pkg_resources')
__file__ = '.../DeepSpeed/bin/deepspeed'
with open(__file__) as f:
    exec(compile(f.read(), __file__, 'exec'))
```
In `path/to/DeepSpeed/requirements/requirements-triton.txt` we also removed the version
of `triton`.

## Running the example
First, find the definition of the `data_path` in the `cifar10_deepspeed.py`:
```bash
data_path = os.path.expanduser("~/scratch/aihwkit-lightning-example/")
```
and change it to a directory where the Cifar10 data can be stored.
Then, step into this example folder, and execute

```bash
bash run_ds.sh --dtype=fp16 --log-interval=100 --epochs=5 --use-triton
```

Note that in `run_ds.sh`, we also pass the `--master_port`. If the port `29501` is busy for you
just change it.
