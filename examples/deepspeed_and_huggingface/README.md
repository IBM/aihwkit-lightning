# DeepSpeed + HuggingFace Accelerate Example

Trains a BERT masked language model on WikiText-103 using AIHWKIT-Lightning with HuggingFace Trainer, optionally with DeepSpeed for multi-GPU/multi-node training.

## Setup & Run

```bash
cd examples/deepspeed_and_huggingface
uv sync
uv run train.py --config config.yaml
```

For multi-GPU with Accelerate:

```bash
uv run bash launch_accelerate.sh
```

For SLURM multi-node, see `slurm.sh`.

## Configuration

Edit `config.yaml` to adjust:

- `example_directory`: where model/data/checkpoints are stored (needs sufficient disk space)
- `ds_config_path`: path to `ds_config.json` for DeepSpeed settings
- `rpu_config`: hardware-aware training parameters (noise, clipping, input ranges)

Set `fp: true` in the config to use standard PyTorch (no analog simulation).

## Notes

- Requires CUDA for GPU training
- Logs to Weights & Biases by default (`report_to: wandb` in config)

### CustomTrainer

In `utils.py`, a `CustomTrainer` overwrites `training_step` to call `clip_weights` on analog layers after each update. This is needed because DeepSpeed uses its own optimizer.

### DeepSpeed

- In `ds_config.json`, `"auto"` values are inherited from HuggingFace TrainingArguments
- Uses ZeRO-2 (gradient + optimizer state sharding) with optional CPU offloading
- FP16 training is enabled by default

### RPU Config

The `rpu_config` section in `config.yaml` controls hardware-aware training:

```yaml
rpu_config:
  clip_sigma: 2.5         # clip to 2.5x std() of weights
  clip_type: gaussian      # per-tensor (or gaussian_channel for per-channel)
  forward_inp_res: 254     # 2^n_bits - 2
  modifier_std_dev: 0.023  # 2.3% weight noise
  modifier_type: add_gauss # gaussian noise per-tensor
  input_range_enable: true # learn input ranges for static quantization
```

## Changes to DeepSpeed (you might not need to do them)

In `DeepSpeed/deepspeed/runtime/zero/stage_1_and_2.py`, find the section
```python
# Use different parallel to do all_to_all_reduce related things
# padding on each partition for alignment purposes
self.groups_padding = []
```
and add below:
```python
self.groups_is_ir_param = [[] for _ in range(len(self.optimizer.param_groups))]
```
Then, still in the `__init__` of the Zero optimizer, find
```python
# free temp CPU params
for param in self.bit16_groups[i]:
    del param.cpu_data
```
and below add
```python
is_ir_param = torch.zeros_like(flattened_buffer)
offset = 0
for p_tensor, name in zip(self.round_robin_bit16_meta[i], self.param_names.values()):
    if "input_range" in name:
        is_ir_param[offset] = 1.0
    offset += p_tensor.numel()
```
Further down, find `self.parallel_partitioned_bit16_groups.append(data_parallel_partitions)`. Below, add
```python
partitions = self.get_data_parallel_partitions(is_ir_param, i)
for part in partitions:
    self.groups_is_ir_param[i].append(torch.arange(part.numel(), device=get_accelerator().current_device_name())[part == 1.0])
del is_ir_param
del partitions
```
Now, in the `step` function, find the line `fp32_partition = self.single_partition_of_fp32_groups[i]`. Note that this appears in both the `if self.cpu_offload:` clause and the `else` clause. Do the following in both clauses.

Before the following block
```python
bit16_partitions[partition_id].data.copy_(
    fp32_partition.to(get_accelerator().current_device_name()).data)
```
insert
```python
ir_indices = self.groups_is_ir_param[i][partition_id]
fp32_partition.data.index_put_((ir_indices.cpu(),), bit16_partitions[partition_id][ir_indices].data.float().cpu())
```

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

Also, in `DeepSpeed/deepspeed/runtime/fp16/loss_scaler.py` we changed parts of the `update_scale` function to not throw an exception if the scale can't be reduced anymore for FP16 training.
