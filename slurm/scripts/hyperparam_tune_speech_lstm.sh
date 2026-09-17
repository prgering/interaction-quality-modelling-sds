#!/bin/bash
#SBATCH --job-name=speech_lstm_tuning
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --mem=40G
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --array=1-192
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

# --- Parse Parameters ---
PARAMS=$(sed -n "${SLURM_ARRAY_TASK_ID}p" $PROJECT_ROOT/slurm/configs/speech_lstm_hyperparam_combs.txt)
read -r USE_ATTENTION BIDIRECTIONAL HIDDEN_SIZE TEXT_PCA SPEECH_PCA <<< "$PARAMS"

TEXT_MODEL="roberta"
SPEECH_MODEL="wavlm"

BIDIRECTIONAL_FLAG=""
if [ "${BIDIRECTIONAL}" == "True" ]; then
    BIDIRECTIONAL_FLAG="--bidirectional"
fi

USE_ATTENTION_FLAG=""
if [ "${USE_ATTENTION}" == "True" ]; then
    USE_ATTENTION_FLAG="--use_attention"
else
    USE_ATTENTION_FLAG=""
fi

# --- Echoing Run Information ---
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
echo "SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}"
echo "Running experiment with:"
echo "  Model Type: LSTM"
echo "  Dataset Type: Speech"
echo "  Use Attention: ${USE_ATTENTION}"
echo "  Bidirectional: ${BIDIRECTIONAL}"
echo "  Hidden Size: ${HIDDEN_SIZE}"
echo "  PCA Speechf TEXT: ${TEXT_PCA}"
echo "  PCA Speechf SPEECH: ${SPEECH_PCA}"
echo "  Pretrained Text Model: ${TEXT_MODEL}"
echo "  Pretrained Speech Model: ${SPEECH_MODEL}"

# Execute your Python training script, redirecting stdout and stderr
python -m scripts.hyperparam_tuning_lstm \
    speech \
    ${USE_ATTENTION_FLAG} \
    ${BIDIRECTIONAL_FLAG} \
    --hidden_size "${HIDDEN_SIZE}" \
    --speechf_text_pca "${SPEECHF_TEXT_PCA}" \
    --speechf_wav_pca "${SPEECHF_SPEECH_PCA}" \
    --pretrained_text_model "${TARGET_TEXT_MODEL}" \
    --pretrained_speech_model "${TARGET_SPEECH_MODEL}"
