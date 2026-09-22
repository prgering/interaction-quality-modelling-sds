#!/bin/bash
#SBATCH --job-name=eval_best_models
#SBATCH --time=5:00:00
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --mem=80G
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=1
#SBATCH --array=1-2
#
# ---USER REQUIRED---
#SBATCH --output=logs/eval_best_models/%x_%A_%a.txt

# --- Source Shared Environment Setup ---
source slurm/scripts/setup_env.sh

# Ensure output log folder exists
mkdir -p logs/eval_best_models

# --- Parse Parameters ---
PARAMS=$(sed -n "${SLURM_ARRAY_TASK_ID}p" $PROJECT_ROOT/slurm/configs/best_model_params.txt)

read -r DATASET TEXT_MODEL SPEECH_MODEL USE_ATTENTION BIDIRECTIONAL \
        HIDDEN_SIZE SPEECHF_TEXT_PCA SPEECHF_SPEECH_PCA SYSTEMF_PCA \
        NUM_LAYERS LEARNING_RATE BATCH_SIZE <<< "$PARAMS"

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
echo "Running final training and evaluation with:"
echo "  Model Type: LSTM"
echo "  Dataset Type: ${DATASET}"
echo "  Use Attention: ${USE_ATTENTION}"
echo "  Bidirectional: ${BIDIRECTIONAL}"
echo "  Hidden Size: ${HIDDEN_SIZE}"
echo "  Number of Layers: ${NUM_LAYERS}"
echo "  Learning Rate: ${LEARNING_RATE}"
echo "  Batch Size: ${BATCH_SIZE}"
echo "  PCA Speechf TEXT: ${SPEECHF_TEXT_PCA}"
echo "  PCA Speechf SPEECH: ${SPEECHF_SPEECH_PCA}"
echo "  PCA Systemf: ${SYSTEMF_PCA}"
echo "  Pretrained Text Model: ${TEXT_MODEL}"
echo "  Pretrained Speech Model: ${SPEECH_MODEL}"

# Execute your Python training script, redirecting stdout and stderr
python -m scripts.train_lstm \
    ${DATASET} \
    ${USE_ATTENTION_FLAG} \
    ${BIDIRECTIONAL_FLAG} \
    --hidden_size "${HIDDEN_SIZE}" \
    --speechf_text_pca "${SPEECHF_TEXT_PCA}" \
    --speechf_wav_pca "${SPEECHF_SPEECH_PCA}" \
    --systemf_text_pca "${SYSTEMF_PCA}" \
    --num_layers "${NUM_LAYERS}" \
    --lr "${LEARNING_RATE}" \
    --batch_size "${BATCH_SIZE}" \
    --pretrained_text_model "${TARGET_TEXT_MODEL}" \
    --pretrained_speech_model "${TARGET_SPEECH_MODEL}"
