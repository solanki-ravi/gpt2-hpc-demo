#!/bin/bash
#SBATCH --job-name=gpt2-small
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --partition=compute

export MASTER_ADDR=$(scontrol show hostnames $SLURM_NODELIST | head -n 1)
export MASTER_PORT=29500

module load cuda/12.1

srun torchrun \
  --nproc_per_node=8 \
  --nnodes=$SLURM_JOB_NUM_NODES \
  --node_rank=$SLURM_NODEID \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  train.py --deepspeed --deepspeed_config deepspeed_config.json

