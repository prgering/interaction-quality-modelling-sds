#!/bin/bash
#SBATCH --job-name=end2end_tuning
#SBATCH --time=10:00:00
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --mem=80G
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=1
#SBATCH --array=1-27
#
# ---USER REQUIRED---
#SBATCH --output=logs/end2end_tuning/%x_%A_%a.txt

# ---Configuration ---
MODE=${MODE:-"speech"}

# --- Source Shared Environment Setup ---
source slurm/scripts/setup_env.sh

# Ensure output log folder exists
mkdir -p logs/end2end_tuning

case "$MODE" in
    "speech") 
        DATASET_TYPE="speech"
        HIDDEN_SIZE=384
        NUM_LAYERS=2
        HEAD_LR=5e-4
        USE_ATTENTION="True"
        BIDIRECTIONAL="False"

        CONFIG_FILE="$PROJECT_ROOT/slurm/configs/speech_end2end_hyperparam_combs.txt"
        ;;
    "system") 
        DATASET_TYPE="system"
        HIDDEN_SIZE=256
        NUM_LAYERS=2
        HEAD_LR=5e-4
        USE_ATTENTION="True"
        BIDIRECTIONAL="False"

        CONFIG_FILE="$PROJECT_ROOT/slurm/configs/system_end2end_hyperparam_combs.txt"
        ;;
    *)
        echo "Invalid mode: $MODE. Valid options are 'speech' or 'system'."
        exit 1
        ;;
esac

# --- Parameter Mapping ---
PARAMS=$(sed -n "${SLURM_ARRAY_TASK_ID}p" $CONFIG_FILE)
read -r WINDOW_SIZE TRANSFORMER_LR FROZEN_LAYERS <<< "$PARAMS"

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
echo "  Model Type: End-to-End"
echo "  Dataset Type: ${DATASET_TYPE}"
echo "  Use Attention: ${USE_ATTENTION}"
echo "  Bidirectional: ${BIDIRECTIONAL}"
echo "  Hidden Size: ${HIDDEN_SIZE}"
echo "  Number of Layers: ${NUM_LAYERS}"
echo "  Window Size: ${WINDOW_SIZE}"
echo "  Head Learning Rate: ${HEAD_LR}"
echo "  Transformer Learning Rate: ${TRANSFORMER_LR}"
echo "  Number of Frozen Layers: ${FROZEN_LAYERS}"

# --- Execution ---
python -m scripts.run_fine_tuning \
    "${DATASET_TYPE}" \
    ${BIDIRECTIONAL_FLAG} \
    ${USE_ATTENTION_FLAG} \
    --hidden_size "${HIDDEN_SIZE}" \
    --num_layers "${NUM_LAYERS}" \
    --window_size "${WINDOW_SIZE}" \
    --head_lr "${HEAD_LR}" \
    --transformer_lr "${TRANSFORMER_LR}" \
    --num_frozen_layers "${FROZEN_LAYERS}"
