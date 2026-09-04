#!/bin/bash
#SBATCH --job-name=speech_feature_extraction
#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --mem=20G
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4

# ---USER REQUIRED---
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --output=logs/extract_features/%x_%j.txt

# ---Configuration ---
MODE=${1:-"speech"}

# --- Source Shared Environment Setup ---
source slurm/scripts/setup_env.sh

# Ensure output log folder exists
mkdir -p logs/extract_features

# --- Task logic ---
PYTHON_SCRIPT="scripts.extract_speech_features"
EXTRA_ARGS=""

case "$MODE" in
    "speech") 
        EXTRA_ARGS="--force"
        ;;
    "static_speech") 
        EXTRA_ARGS="--skip-embeddings --force"
        ;;
    *)
        echo "Invalid mode: $MODE. Valid options are 'speech' or 'static_speech'."
        exit 1
        ;;
esac

# --- Execution ---
python -m "$PYTHON_SCRIPT" $EXTRA_ARGS