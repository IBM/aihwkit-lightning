#!/bin/bash
#SBATCH --output=/gpfs/u/home/ANFM/ANFMbchl/scratch/aihwkit-lightning-example/%j.out
#SBATCH --error=/gpfs/u/home/ANFM/ANFMbchl/scratch/aihwkit-lightning-example/%j.err
#SBATCH --time=30
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=2
#SBATCH --partition=npl-2024
#SBATCH --gres=gpu:8

nvidia-smi

echo "Running on node: $(hostname)"
echo "In directory:    $(pwd)"
echo "Starting on:     $(date)"

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

export HOSTNAMES=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))

echo "WORLD_SIZE="$WORLD_SIZE
echo "MASTER_ADDR="$MASTER_ADDR
echo "MASTER_PORT="$MASTER_PORT

NUM_PROCESSES=16  # 2 nodes a 8 GPUs

rm -rf $TORCH_EXTENSIONS_DIR

function makehostfile() {
perl -e '$slots=split /,/, $ENV{"SLURM_STEP_GPUS"};
$slots=8 if $slots==0; # workaround 8 gpu machines
@nodes = split /
/, qx[scontrol show hostnames $ENV{"SLURM_JOB_NODELIST"}];
print map { "$b$_ slots=$slots
" } @nodes'
}
HOST_FILE_PATH=$HOME/scratch/aihwkit-lightning-example/hostfile
makehostfile > $HOST_FILE_PATH
# you can prepend `mprof run --multiprocess` to accelerate launch to monitor the RAM usage
srun bash -c "accelerate launch \
    --use_deepspeed \
    --deepspeed_hostfile $HOST_FILE_PATH \
    --deepspeed_multinode_launcher standard \
    --dynamo_backend no \
    --mixed_precision fp16 \
    --num_processes $NUM_PROCESSES \
    --gpu_ids all \
    --num_machines 2 \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --machine_rank \$SLURM_NODEID \
    --rdzv_backend static \
    --deepspeed_config_file $HOME/scratch/aihwkit-lightning/examples/deepspeed_and_huggingface/ds_config.json \
    train.py --config $HOME/scratch/aihwkit-lightning/examples/deepspeed_and_huggingface/config.yaml"

echo "Finished at: $(date)"