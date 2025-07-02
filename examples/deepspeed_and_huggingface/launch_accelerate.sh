#!/bin/bash

module load gcc/9.3.0/1
export WANDB_CACHE_DIR=$HOME/scratch/.cache
export WANDB_DATA_DIR=$HOME/scratch/.cache
export WANDB_DIR=$HOME/scratch/.cache
export WANDB_CONFIG_DIR=$HOME/scratch/.cache
export TMPDIR=$HOME/scratch/.cache
export MKL_SERVICE_FORCE_INTEL=1
export MAX_JOBS=8
export TRITON_HOME=$HOME/scratch/.triton
export TRITON_CACHE_DIR=$HOME/scratch/.cache
export TORCH_EXTENSIONS_DIR=$HOME/scratch/.cache/torch-extensions

# # For using triton version of AIHWKIT
# export TRITON_PRINT_AUTOTUNING=1
# export AIHWKIT_USE_TRITON=1

# # For DEBUG
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# export TORCH_CPP_LOG_LEVEL=INFO
# export TORCH_DISTRIBUTED_DEBUG=INFO
# export NCCL_DEBUG=INFO
# export ACCELERATE_DEBUG_MODE="1"

NUM_PROCESSES=8  # 1 node with 8 GPUs

rm -rf $TORCH_EXTENSIONS_DIR
accelerate launch \
    --multi_gpu \
    --dynamo_backend no \
    --mixed_precision fp16 \
    --num_processes $NUM_PROCESSES \
    --gpu_ids all \
    --num_machines 1 \
    train.py --config ./config.yaml