# GPT-2 Training with DeepSpeed on Slurm and GCP HPC Clusters

This project contains scripts to train a GPT-2 language model using DeepSpeed for efficient distributed training on a Slurm cluster managed by the Google Cloud HPC Toolkit.

For more information about GCP HPC Clusters, please refer to: https://cloud.google.com/solutions/hpc?hl=en
For a tutorial on GCP HPC Cluster with Slurm, please refer to: https://codelabs.developers.google.com/codelabs/hpc-slurm-on-gcp#0

For information on the training architecture, please refer to the following diagram.

The model was trained on 8 g2-standard-4 cluster instances (NVIDIA L4s). More information on GCP G4 instances can be found here: https://cloud.google.com/compute/docs/gpus/#l4-gpus

The model could easily be launched on more powerful H{1|2}00 instances on GCP, by defiing the appropriate node configuration (hpc-slurm.yaml).

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

## Inference (`inference.py`)

An `inference.py` script is provided to load a trained DeepSpeed checkpoint and generate text based on a prompt.

**Prerequisites:**

*   Ensure Python dependencies are installed (`pip3 install --user -r requirements.txt`).
*   Have access to a saved DeepSpeed checkpoint directory (e.g., `./my_gpt2_checkpoint/global_step65110`).
*   Ensure `deepspeed_config.json` used during training is present in the current directory (as the inference script uses it to initialize the model engine structure for loading).
*   If running on GPU, ensure drivers and CUDA are set up.
*   If using DeepSpeed CPUAdam/Offloading during training, the inference environment might still need build tools (`ninja-build`, `python38-devel`, `gcc-toolset-12`) installed, although inference is often done without DeepSpeed optimizations active.

**Usage:**

```bash
python3 inference.py <path_to_checkpoint_tag_directory> [options]
```

**Arguments:**

*   `checkpoint_dir` (Positional): The full path to the *specific checkpoint tag directory* you want to load (e.g., `./my_gpt2_checkpoint/global_step65110`).
*   `--prompt` (Optional): The starting text prompt.
    *   Default: `"DeepSpeed is"`
*   `--model_name` (Optional): Base model name for loading the tokenizer.
    *   Default: `gpt2`
*   `--device` (Optional): Device to run on (`cuda:0`, `cpu`, etc.).
    *   Default: `cuda:0` if available, else `cpu`.
*   `--max_new_tokens` (Optional): Max number of tokens to generate after the prompt.
    *   Default: `50`

**Example:**

```bash
python3 inference.py my_gpt2_checkpoint/global_step65110 --prompt "The future of AI is"
```

This command will load the specified checkpoint, initialize the DeepSpeed engine using `deepspeed_config.json`, generate text based on the prompt, and print the result.

## Cluster Management Script (`manage_cluster.sh`)

A helper script `manage_cluster.sh` is provided to simplify creating and destroying the cluster using the HPC Toolkit.

**Prerequisites:**

*   Ensure the `ghpc` executable is in your current directory or PATH.
*   Ensure the blueprint file (`hpc-slurm.yaml`) is in the current directory.
*   Make the script executable:
    ```bash
    chmod +x manage_cluster.sh
    ```
*   Set your Google Cloud Project ID as an environment variable:
    ```bash
    export PROJECT_ID="your-gcp-project-id"
    ```

**Usage:**

*   **Create Cluster:**
    ```bash
    ./manage_cluster.sh create <your_deployment_name>
    ```
    *(Replace `<your_deployment_name>` with the `deployment_name` from `hpc-slurm.yaml`, e.g., `hpc-slurm-gpt2demo-gpu-g2-deepspeed`)*

*   **Destroy Cluster:**
    ```bash
    ./manage_cluster.sh destroy <your_deployment_name>
    ```
    *(Replace `<your_deployment_name>` accordingly)*

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
   
## Optimizations:
Apply the following optimizations to maximize MFU.

### Batch size:

1. **Adjust `train_micro_batch_size_per_gpu`:**

   ```
   nvidia-smi 
   Sat May  3 16:50:52 2025       
   +-----------------------------------------------------------------------------------------+
   | NVIDIA-SMI 550.90.12              Driver Version: 550.90.12      CUDA Version: 12.4     |
   |-----------------------------------------+------------------------+----------------------+
   | GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
   | Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
   |                                         |                        |               MIG M. |
   |=========================================+========================+======================|
   |   0  NVIDIA L4                      On  |   00000000:00:03.0 Off |                    0 |
   | N/A   76C    P0             62W /   72W |    6483MiB /  23034MiB |     82%      Default |
   |                                         |                        |                  N/A |
   +-----------------------------------------+------------------------+----------------------+
                                                                                            
   +-----------------------------------------------------------------------------------------+
   | Processes:                                                                              |
   |  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
   |        ID   ID                                                               Usage      |
   |=========================================================================================|
   |    0   N/A  N/A      2156      C   /usr/bin/python3                             6474MiB |
   +-----------------------------------------------------------------------------------------+

   train_micro_batch_size_per_gpu = 2 > train_micro_batch_size_per_gpu = 6
   ```
