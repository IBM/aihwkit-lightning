# DeepSpeed CIFAR-10 Example

Trains a simple CNN on CIFAR-10 using DeepSpeed with AIHWKIT-Lightning's analog layers. Based on the [DeepSpeed CIFAR example](https://github.com/microsoft/DeepSpeedExamples/tree/master/training/cifar).

## Setup & Run

```bash
cd examples/deepspeed_cifar10
uv sync
uv run deepspeed cifar10_deepspeed.py --dtype=fp16 --log-interval=100 --epochs=5
```

Add `--use-triton` to use Triton kernels for AIHWKIT-Lightning.

## Notes

- Requires CUDA (uses `deepspeed.init_distributed` with NCCL backend)
- The `data_path` in the script defaults to `~/scratch/aihwkit-lightning-example/` — change it if needed
- DeepSpeed uses its own optimizer, so weight clipping is done manually after each step
- See `run_ds.sh` for a multi-GPU launch script (adjust `--master_port` if needed)
