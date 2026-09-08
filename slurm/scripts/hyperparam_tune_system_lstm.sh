#!/bin/bash
#SBATCH --job-name=system_lstm_tuning
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --mem=40G
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --array=1-144
#
# ---USER REQUIRED---
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --output=logs/system_lstm_tuning/%x_%A_%a.txt

# --- Source Shared Environment Setup ---
source slurm/scripts/setup_env.sh

# Ensure output log folder exists
mkdir -p logs/system_lstm_tuning

# --- Parse Parameters ---
PARAMS=$(sed -n "${SLURM_ARRAY_TASK_ID}p" $PROJECT_ROOT/slurm/configs/system_lstm_hyperparam_combs.txt)
read -r USE_ATTENTION BIDIRECTIONAL HIDDEN_SIZE SYSTEMF_PCA TEXT_MODEL <<< "$PARAMS"

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
echo "  Dataset Type: System"
echo "  Use Attention: ${USE_ATTENTION}"
echo "  Bidirectional: ${BIDIRECTIONAL}"
echo "  Hidden Size: ${HIDDEN_SIZE}"
echo "  PCA System TEXT: ${SYSTEMF_PCA}"
echo "  Pretrained Text Model: ${TEXT_MODEL}"

# --- Execution ---
python -m scripts.hyperparam_tuning_lstm \
    system \
    ${USE_ATTENTION_FLAG} \
    ${BIDIRECTIONAL_FLAG} \
    --hidden_size ${HIDDEN_SIZE} \
    --systemf_text_pca ${SYSTEMF_PCA} \
    --pretrained_text_model ${TEXT_MODEL}