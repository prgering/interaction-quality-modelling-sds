#!/bin/bash
#SBATCH --job-name=eval_best_end2end
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
#SBATCH --output=logs/eval_best_end2end/%x_%A_%a.txt

# --- Source Shared Environment Setup ---
source slurm/scripts/setup_env.sh

# Ensure output log folder exists
mkdir -p logs/eval_best_end2end

# --- Parse Parameters ---
PARAMS=$(sed -n "${SLURM_ARRAY_TASK_ID}p" $PROJECT_ROOT/slurm/configs/best_end2end_model_params.txt)

read -r DATASET_TYPE HIDDEN_SIZE AUTO_FEATURES_ONLY \
        WINDOW_SIZE FROZEN_LAYERS TRANSFORMER_LR <<< "$PARAMS"

# Fixed Hyperparameters
USE_ATTENTION="True"
NUM_LAYERS="2"
BIDIRECTIONAL="False"
HEAD_LR="5e-4"

# --- Flag Logic ---
FLAGS=()
[[ "$BIDIRECTIONAL" == "True" ]]    && FLAGS+=("--bidirectional")
[[ "$USE_ATTENTION" == "True" ]]    && FLAGS+=("--use_attention")


# --- Logging Run Info ---
cat <<EOF
---------------------------------------------------------
SLURM_JOB_ID: ${SLURM_JOB_ID}
SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}
Dataset:      ${DATASET_TYPE}
Model:        End-to-End
---------------------------------------------------------
EOF

# --- Execution ---
python -m "${PYTHON_SCRIPT}" \
    "${DATASET_TYPE}" \
    "${FLAGS[@]}" \
    --hidden_size "${HIDDEN_SIZE}" \
    --num_layers "${NUM_LAYERS}" \
    --window_size "${WINDOW_SIZE}" \
    --head_lr "${HEAD_LR}" \
    --transformer_lr "${TRANSFORMER_LR}" \
    --num_frozen_layers "${FROZEN_LAYERS}" \
