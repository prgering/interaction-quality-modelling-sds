#!/bin/bash
#SBATCH --job-name=preprocess_system_features
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --mem=20G
#SBATCH --cpus-per-task=1
#SBATCH --array=0-3
#SBATCH --output=logs/preprocess_features/%x_%A_%a.txt

# --- Source Shared Environment Setup ---
source slurm/scripts/setup_env.sh

# Ensure output log folder exists
mkdir -p logs/preprocess_features

# ---Parse Parameters---
PCA_VALUES=( "0.5" "0.6" "0.7" "0.8" )
SYSTEMF_TEXT_PCA="${PCA_VALUES[$SLURM_ARRAY_TASK_ID]}"

# --- Execution ---
python -m scripts.prepare_features_for_modelling \
    system \
    --systemf_text_pca ${SYSTEMF_TEXT_PCA} \
    --remove-first-turn

