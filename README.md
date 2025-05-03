# GPT-2 Training with DeepSpeed on Slurm

This project contains scripts to train a GPT-2 language model using DeepSpeed for efficient distributed training on a Slurm cluster managed by the Google Cloud HPC Toolkit.

## Files

*   `train.py`: The main Python script for loading data, configuring the model, and running the training loop using DeepSpeed.
*   `run_llm.slurm`: The Slurm batch script used to submit the training job. It sets up the environment and launches `train.py` using `srun` and `torchrun`.
*   `deepspeed_config.json`: Configuration file for DeepSpeed, specifying settings like batch sizes, gradient accumulation, FP16, and ZeRO optimization stages.
*   `requirements.txt`: Lists the required Python packages.
*   `hpc-slurm.yaml`: The HPC Toolkit blueprint defining the Slurm cluster infrastructure (nodesets, partitions, controller, etc.).

## Prerequisites

1.  **HPC Cluster:** A Slurm cluster deployed using the HPC Toolkit (based on `hpc-slurm.yaml`).
2.  **Shared Filesystem:** A shared filesystem (like NFS or Filestore, defined as `homefs` in `hpc-slurm.yaml`) mounted at `/home` (or adjusted paths in scripts) accessible by the login node and compute nodes.
3.  **Code:** Clone or copy this project directory to the shared filesystem.
4.  **System Dependencies on Compute Nodes:** The compute nodes in the target partition (e.g., `g2gpu`) need the following system packages installed (using `sudo dnf install -y ...` on Rocky Linux 8):
    *   `ninja-build`
    *   `python38-devel` (or the version matching the Python used)
    *   `"Development Tools"` package group (for `g++`, `make`)
    *   `gcc-toolset-12` (or the version required by the PyTorch/DeepSpeed build)
    *(Manual installation via SSH is currently required as per the setup steps)*
5.  **Python Dependencies:** Install the Python packages on the login node (and potentially within the job using the script) in your user environment:
    ```bash
    pip3 install --user -r requirements.txt
    ```

## Running the Training Job

The training job is submitted using the `run_llm.slurm` script.

```bash
sbatch run_llm.slurm
```

### Configuration

You can configure the training run by setting environment variables when submitting the job using `sbatch --export=...`.

*   **`EPOCHS`**: Number of training epochs.
    *   Default: `10`
    *   Example: `sbatch --export=ALL,EPOCHS=5 run_llm.slurm`
*   **`CHECKPOINT_DIR`**: Directory path where DeepSpeed checkpoints will be saved. This should be on the shared filesystem.
    *   Default: `./my_gpt2_checkpoint` (relative to where `sbatch` is run)
    *   Example: `sbatch --export=ALL,CHECKPOINT_DIR="/home/user/gpt2_run1/checkpoints" run_llm.slurm`
*   **`DATA_PERCENTAGE`**: Percentage of the "openwebtext" training split to use.
    *   Default: `10` (meaning 10%)
    *   Example: `sbatch --export=ALL,DATA_PERCENTAGE=50 run_llm.slurm`

You can combine these flags:

```bash
sbatch --export=ALL,EPOCHS=3,DATA_PERCENTAGE=25,CHECKPOINT_DIR="/fsx/my_training_run" run_llm.slurm
```

*(Note: The requirement for `ALL` in `--export=ALL,...` depends on your specific Slurm configuration regarding environment variable propagation.)*

## Monitoring

*   Check job status: `squeue -u $USER`
*   Check output: `cat slurm-<job_id>.out`
*   Check errors: `cat slurm-<job_id>.err`

## Running on a Multi-GPU Node

The current configuration requests and assumes a single GPU per node (`g2-standard-4`). To run on a node with multiple GPUs (e.g., an `a2-highgpu-4g` with 4 GPUs), you need to make the following adjustments:

1.  **Modify `hpc-slurm.yaml`:**
    *   Change the `machine_type` for the relevant nodeset (e.g., `g2_gpu_nodeset`) to a multi-GPU instance type (e.g., `a2-highgpu-4g`).
    *   Redeploy the cluster infrastructure using the HPC Toolkit (`./ghpc deploy ...`) for the change to take effect.

2.  **Modify `run_llm.slurm`:**
    *   Update the SBATCH directive to request the correct number of GPUs per node. For example, for a 4-GPU node:
        ```bash
        #SBATCH --gpus-per-node=4
        ```
    *   You might need to remove or adjust the `#SBATCH --gres=gpu:1` line if present and conflicting.
    *   The rest of the script (`srun ... --nproc_per_node $SLURM_GPUS_PER_NODE ...`) should automatically adapt because `$SLURM_GPUS_PER_NODE` will be set by Slurm based on your `--gpus-per-node` request.

3.  **Modify `deepspeed_config.json` (Optional but Recommended):**
    *   To maintain the same *global batch size* and training dynamics, you usually need to adjust `gradient_accumulation_steps`.
    *   The relationship is: `global_batch_size = micro_batch_per_gpu * num_gpus * gradient_accumulation_steps`.
    *   If you change the number of GPUs (`num_gpus`) but want to keep `global_batch_size` and `micro_batch_per_gpu` the same, calculate the new `gradient_accumulation_steps`.
    *   *Example:* If the original config was `micro_batch=2`, `num_gpus=1`, `accum=16` (Global=32), and you switch to `num_gpus=4`, the new accumulation steps should be `32 / (2 * 4) = 4`.
        ```json
        {
          "train_global_batch_size": 32,
          "train_micro_batch_size_per_gpu": 2, 
          "gradient_accumulation_steps": 4, // Adjusted for 4 GPUs
          ...
        }
        ```
    *   If you don't adjust this, your global batch size will increase proportionally to the number of GPUs. 