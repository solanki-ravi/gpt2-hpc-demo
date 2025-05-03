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