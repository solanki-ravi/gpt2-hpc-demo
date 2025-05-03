#!/bin/bash

# Script to manage the HPC Slurm cluster deployment using ghpc

set -e # Exit immediately if a command exits with a non-zero status.

# --- Configuration ---
BLUEPRINT_FILE="hpc-slurm.yaml"
# Extract deployment name manually or pass as arg if yq not available
# DEPLOYMENT_NAME=$(yq e '.vars.deployment_name' "$BLUEPRINT_FILE")
# --- End Configuration ---

# --- Usage Instructions ---
usage() {
    echo "Usage: $0 <command> <deployment_name>"
    echo ""
    echo "Commands:"
    echo "  create   <deployment_name>  Deploys the cluster defined in $BLUEPRINT_FILE."
    echo "                            Requires PROJECT_ID environment variable to be set."
    echo "  destroy  <deployment_name>  Destroys the specified cluster deployment."
    echo ""
    echo "Example:"
    echo "  export PROJECT_ID=your-gcp-project-id"
    echo "  $0 create hpc-slurm-gpt2demo-gpu-g2-deepspeed"
    echo "  $0 destroy hpc-slurm-gpt2demo-gpu-g2-deepspeed"
    exit 1
}

# --- Argument Parsing ---
COMMAND=$1
DEPLOYMENT_NAME=$2

if [ -z "$COMMAND" ] || [ -z "$DEPLOYMENT_NAME" ]; then
    echo "Error: Command and deployment name are required."
    usage
fi

# --- Command Logic ---
case "$COMMAND" in
    create)
        echo "Starting cluster creation for deployment: $DEPLOYMENT_NAME..."
        if [ -z "$PROJECT_ID" ]; then
            echo "Error: PROJECT_ID environment variable is not set."
            exit 1
        fi
        if [ ! -f "$BLUEPRINT_FILE" ]; then
            echo "Error: Blueprint file '$BLUEPRINT_FILE' not found in current directory."
            exit 1
        fi
        echo "Running: ./ghpc deploy $BLUEPRINT_FILE -v project_id=$PROJECT_ID"
        ./ghpc deploy "$BLUEPRINT_FILE" -v project_id="$PROJECT_ID"
        echo "Cluster creation command finished."
        ;;

    destroy)
        echo "Starting cluster destruction for deployment: $DEPLOYMENT_NAME..."
        # ghpc destroy uses the deployment name, not the blueprint file
        echo "Running: ./ghpc destroy $DEPLOYMENT_NAME"
        ./ghpc destroy "$DEPLOYMENT_NAME"
        echo "Cluster destruction command finished."
        ;;

    *)
        echo "Error: Invalid command '$COMMAND'"
        usage
        ;;
esac

exit 0 