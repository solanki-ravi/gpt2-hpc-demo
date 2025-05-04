#!/bin/bash

# Script to manage the HPC Slurm cluster deployment using gcluster (HPC Toolkit)

set -e # Exit immediately if a command exits with a non-zero status.

# --- Configuration ---
BLUEPRINT_FILE="hpc-slurm.yaml"
# Extract deployment name manually or pass as arg if yq not available
# DEPLOYMENT_NAME=$(yq e '.vars.deployment_name' "$BLUEPRINT_FILE")
# --- End Configuration ---

# --- Usage Instructions ---
usage() {
    echo "Usage: $0 <command> <deployment_name> [options]"
    echo ""
    echo "Commands:"
    echo "  create   <deployment_name>  Generates deployment directory from $BLUEPRINT_FILE using gcluster."
    echo "                            Requires PROJECT_ID environment variable to be set."
    echo "  deploy   <deployment_name>  Deploys resources for the specified deployment directory using gcluster."
    echo "  destroy  <deployment_name>  Destroys the specified cluster deployment using gcluster."
    echo "                            Use --auto-approve to skip confirmation."
    echo ""
    echo "Example:"
    echo "  export PROJECT_ID=your-gcp-project-id"
    echo "  $0 create hpc-slurm-gpt2demo-gpu-g2-deepspeed"
    echo "  $0 deploy hpc-slurm-gpt2demo-gpu-g2-deepspeed"
    echo "  $0 destroy hpc-slurm-gpt2demo-gpu-g2-deepspeed --auto-approve"
    exit 1
}

# --- Argument Parsing ---
COMMAND=$1
DEPLOYMENT_NAME=$2
AUTO_APPROVE=""

# Check for --auto-approve flag for destroy command
if [ "$COMMAND" == "destroy" ] && [ "$3" == "--auto-approve" ]; then
    AUTO_APPROVE="--auto-approve"
fi

if [ -z "$COMMAND" ] || [ -z "$DEPLOYMENT_NAME" ]; then
    echo "Error: Command and deployment name are required."
    usage
fi

# --- Command Logic ---
case "$COMMAND" in
    create)
        echo "Generating deployment directory: $DEPLOYMENT_NAME from $BLUEPRINT_FILE..."
        if [ -z "$PROJECT_ID" ]; then
            echo "Error: PROJECT_ID environment variable is not set."
            exit 1
        fi
        if [ ! -f "$BLUEPRINT_FILE" ]; then
            echo "Error: Blueprint file '$BLUEPRINT_FILE' not found in current directory."
            exit 1
        fi
        # Ensure deployment name in blueprint matches argument (optional check)
        blueprint_dep_name=$(grep "deployment_name:" "$BLUEPRINT_FILE" | awk '{print $2}')
        if [ "$blueprint_dep_name" != "$DEPLOYMENT_NAME" ]; then
             echo "Warning: Deployment name '$DEPLOYMENT_NAME' provided does not match name in $BLUEPRINT_FILE ('$blueprint_dep_name'). Using name from blueprint for creation."
             # Use name from blueprint for consistency during create
             # gcluster create uses the name embedded in the blueprint
        fi
        echo "Running: ./gcluster create $BLUEPRINT_FILE -v project_id=$PROJECT_ID"
        # gcluster create uses the vars within the blueprint file, including deployment_name
        ./gcluster create "$BLUEPRINT_FILE" -v project_id="$PROJECT_ID"
        echo "Deployment directory generation finished."
        ;;

    deploy)
        echo "Starting deployment for directory: $DEPLOYMENT_NAME..."
        if [ ! -d "$DEPLOYMENT_NAME" ]; then
            echo "Error: Deployment directory '$DEPLOYMENT_NAME' not found. Run 'create' first."
            exit 1
        fi
        echo "Running: ./gcluster deploy $DEPLOYMENT_NAME"
        ./gcluster deploy "$DEPLOYMENT_NAME"
        echo "Deployment command finished."
        ;;

    destroy)
        echo "Starting cluster destruction for deployment: $DEPLOYMENT_NAME..."
        if [ ! -d "$DEPLOYMENT_NAME" ]; then
             echo "Warning: Deployment directory '$DEPLOYMENT_NAME' not found. Proceeding with destroy command anyway."
             # Allow destroy even if dir is missing, gcluster might handle state
        fi
        echo "Running: ./gcluster destroy $DEPLOYMENT_NAME $AUTO_APPROVE"
        ./gcluster destroy "$DEPLOYMENT_NAME" $AUTO_APPROVE
        echo "Cluster destruction command finished."
        ;;

    *)
        echo "Error: Invalid command '$COMMAND'"
        usage
        ;;
esac

exit 0 