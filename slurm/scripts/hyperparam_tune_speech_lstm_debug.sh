#!/bin/bash
#SBATCH --job-name=speech_lstm_tuning
#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --mem=90G
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#
# ---USER REQUIRED---
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --output=logs/speech_lstm_tuning/%x_%A_%a.txt

# --- Source Shared Environment Setup ---
source slurm/scripts/setup_env.sh

# Ensure output log folder exists
mkdir -p logs/speech_lstm_tuning

# Execute your Python training script, redirecting stdout and stderr
python -m scripts.hyperparam_tuning_lstm \
    speech \
    --use_attention \
    --hidden_size 384 \
    --speechf_text_pca 0.8 \
    --speechf_wav_pca 0.6 \
    --pretrained_text_model "roberta" \
    --pretrained_speech_model "wavlm"
