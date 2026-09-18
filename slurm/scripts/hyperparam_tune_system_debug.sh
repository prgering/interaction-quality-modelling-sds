#!/bin/bash
#SBATCH --job-name=system_lstm_tuning
#SBATCH --time=1:00:00
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --mem=40G
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=1
#
# ---USER REQUIRED---
#SBATCH --output=logs/system_lstm_tuning/%x_%A_%a.txt

# --- Source Shared Environment Setup ---
source slurm/scripts/setup_env.sh

# Ensure output log folder exists
mkdir -p logs/system_lstm_tuning

# --- Execution ---
python -m scripts.train_lstm \
    system \
    --use_attention \
    --hidden_size 256 \
    --systemf_text_pca 0.6 \
    --pretrained_text_model sbert