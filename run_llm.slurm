#!/bin/bash
#SBATCH --job-name=gpt2-train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1       # Should match number of GPUs per node
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-node=1         # Request 1 GPUs per node (matches g2-standard-4)
#SBATCH --cpus-per-task=2         # Adjust based on instance type and data loading needs
#SBATCH --time=06:00:00           # Adjust job time limit as needed
#SBATCH --partition=g2gpu         # Matches the queue name in cluster-config.yaml
#SBATCH --output=slurm-%j.out     # Standard output file
#SBATCH --error=slurm-%j.err      # Standard error file

# --- Environment Setup ---
# Change to the directory where the job was submitted from
cd $SLURM_SUBMIT_DIR

echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "Node list: $SLURM_NODELIST"
echo "Master Node: $(hostname)"
echo "Running in directory: $(pwd)"

# Activate conda environment if needed (adjust path if necessary)
# source /shared/home/ubuntu/miniconda/bin/activate llm 
# Or use virtualenv:
# source /shared/home/ubuntu/my_venv/bin/activate

# Install dependencies on each node (only runs once per node allocation)
# Using srun ensures pip runs in parallel across nodes if needed, 
# but typically dependencies are shared via the filesystem (/home/ubuntu)
# A simple pip install might suffice if the shared filesystem is reliable.
echo "Installing dependencies..."
# Check Python version
echo "Python version:"
python3 --version
# Use pip3 and install to user directory
pip3 install --user -r requirements.txt
# Add user's local bin to PATH for scripts installed by pip
export PATH=$HOME/.local/bin:$PATH
echo "Dependencies installed."

# --- Distributed Training Setup ---
# Get the master node's IP address
export MASTER_ADDR=$(scontrol show hostnames $SLURM_NODELIST | head -n 1)
# Get a free port (use python3)
export MASTER_PORT=$(python3 -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')

echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"

# --- Run the Training Script ---
# torchrun handles rank assignment based on srun context
# Remove --node_rank and potentially unnecessary --deepspeed flag
echo "Launching training..."
# Use python3 -m torch.distributed.run instead of just torchrun
srun --label python3 -m torch.distributed.run \
  --nproc_per_node $SLURM_GPUS_PER_NODE \
  --nnodes $SLURM_JOB_NUM_NODES \
  --rdzv_id $SLURM_JOB_ID \
  --rdzv_backend c10d \
  --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
  train.py --deepspeed_config deepspeed_config.json

echo "Training script finished."

