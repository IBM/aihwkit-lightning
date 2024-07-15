# SLURM + DeepSpeed + Huggingface Accelerate
Training on larger batch sizes significantly reduces the training time of LLMs. When the point is reached where the peak memory of one training iteration exceeds the DRAM capacity of a single GPU, mulit-GPU training is employed. Standard clusters typically comprise of many nodes. Each node typically has 8 GPUs. This tutorial shows how you can train an Analog LLM in a multi-node setup using SLURM, AIHWKIT Lightning, DeepSpeed, and Huggingface Accelerate on 2 nodes, each with 8 GPUs.

This tutorial will assume that you are in a very memory-limited setup. We therefore use gradient + optimizer state sharding (DeepSpeed ZeRO-2), optimizer state offloading to CPU, gradient/activation checkpointing, and FP16.
The model we use, however, is just a small Bert model, so you don't really need this. This is just to show you the possibilities of DeepSpeed and how well it integrates with AIHWKIT Lightning.

## Prerequisites
First, install PyTorch.
We had to make some changes in DeepSpeed, so it might be better to clone DeepSpeed from `https://github.com/microsoft/DeepSpeed`, step into the repo and execute `pip install -e .`.
```bash
pip install -U --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/Triton-Nightly/pypi/simple/ triton-nightly
[OPTIONAL] pip install deepspeed
pip install transformers
pip install accelerate
pip install datasets
```

There is two paths that you need:
```bash
/path/to/aihwkit-lightning/
/path/to/examples/folder/aihwkit-lightning-example/
```
The `/path/to/examples/folder/aihwkit-lightning-example` is a folder that will store the dataset, model, and tokenizer for this example. It will also store cached files etc. Therefore, it shouldn't be restricted by a storage limit. It should also not be a path to a location that is on a slow drive.

Next, setup a directory where you save the model, dataset etc. from this example. Let's say that path is `/home/path/to/example/`.
In `slurm.sh` there is a section showing
```bash
#SBATCH --output=/home/aihwkit-lightning-example/%j.out
#SBATCH --error=/home/aihwkit-lightning-example/%j.err
```
Adjust these paths to a
```bash
#SBATCH --output=/path/to/examples/folder/aihwkit-lightning-example/%j.out
#SBATCH --error=/path/to/examples/folder/aihwkit-lightning-example/%j.err
```

There is also a path `$HOME/scratch` that points to caches for triton and torch extensions etc. Also, change this path to something you like. It can be just `$HOME`.

You also need to ensure that `HOST_FILE_PATH` points to `hostfile` in this example. Also, make sure that `--deepspeed_config_file` points to the correct path of your `aihwkit-lightning` installation. The same goes for `train.py --config`.

In `config.yaml`, change these directories as well:
```bash
example_directory: ~/scratch/aihwkit-lightning-example/ --> /path/to/examples/folder/aihwkit-lightning-example/
ds_config_path: ~/scratch/aihwkit-lightning/examples/deepspeed_and_huggingface/ds_config.json 
    --> /path/to/aihwkit-lightning/examples/deepspeed_and_huggingface/ds_config.json
```

## Explanations
### CustomTrainer
As can be seen from `utils.py`, we define our own `CustomTrainer` that overwrites the `training_step` function.
We do this, so that we can call `clip_weights` on the analog layers after every actual update to the weights.
Normally, the analog optimizer takes care of this, but here, DeepSpeed uses its own optimizer, which is why we have to do that.

### DeepSpeed
In the `ds_config.json`, whenever you see `"auto"`, it means that the value from Huggingface Transformers is used.
```bash
"deepspeed_multinode_launcher": "standard"
```
This can be changed to `c10d`, but then you also have to change it in the `slurm.sh` script at `--rdzv_backend c10d`.

We also enable FP16 training:
```bash
"fp16": {
    "enabled": true,
    "auto_cast": true,
    ...
}
```

ZeRO optimizer state. We use stage 2, which means that we shard the gradients and the optimizer state. We further offload the optimizer to CPU.
```bash
"zero_optimization": {
    "stage": 2,
    "offload_optimizer": {
        "device": "cpu"
    },
    ...
}
```

In the `TrainingArguments` in `train.py`, you can also see that we specify `gradient_accumulation_steps=args.gradient_accumulation_steps`. In `config.yaml`, this is specified as `true`,
so we do gradient checkpointing.


### RPU-Config
The part in the `config.yaml` dictating the `RPUConfig` used for the HW-aware training is this:
```bash
rpu_config:
  clip_sigma: 2.5  # we clip to 2.5 x std() of the weights, per tensor
  clip_type: gaussian  # using the std(). This can also be gaussian_channel
  forward_inp_res: 254  # 2**n-bits - 2 where n-bits is the number of bits for your input
  forward_out_noise: 0.0
  forward_out_noise_per_channel: false
  mapping_max_input_size: -1
  modifier_enable_during_test: false
  modifier_res: -1
  modifier_std_dev: 0.023  # we inject 2.3% noise per-tensor
  modifier_type: add_gauss  # gaussian noise per-tensor
  input_range_decay: 0.001
  input_range_enable: true  # we learn input ranges for static input quantization
  input_range_fast_mode: false
  input_range_init_from_data: 500  # for the first 500 batches, we update the input ranges from data instead of learning them
  input_range_init_std_alpha: 3.0
  input_range_init_value: 3.0
  input_range_init_with_max: false
  input_range_input_min_percentage: 0.95
  input_range_learn_input_range: true
```

## Changes to DeepSpeed (you might not need to do them)
In `DeepSpeed/deepspeed/ops/transformer/inference/triton/matmul_ext.py`, we changed the `put` function in `class AutotuneCacheManager:` to
```python
def put(self, table):
    if self.file_path:
        assert self.lock_path is not None
    # with FileLock(self.lock_path):
    #     with open(self.file_path + ".tmp", 'wb') as handle:
    #         pickle.dump(table, handle)
    #     os.rename(self.file_path + ".tmp", self.file_path)
```
For us, this led to race conditions, and commenting it out seemed to work.


Also, in `DeepSpeed/deepspeed/runtime/fp16/loss_scaler.py`
we changed parts of the `update_scale` function to
```python
def update_scale(self, overflow):
    if overflow:
        # self.cur_scale /= self.scale_factor
        if self.delayed_shift == 1 or self.cur_hysteresis == 1:
            if (self.cur_scale == self.min_scale) and self.raise_error_at_min_scale:
                # raise Exception(
                #     "Current loss scale already at minimum - cannot decrease scale anymore. Exiting run.") 
                logger.info("Loss has overflown and exception would have been raised.")
        [OMITTED]
```

meaning that we don't throw an exception if we can't reduce the scale anymore for FP16 training.
