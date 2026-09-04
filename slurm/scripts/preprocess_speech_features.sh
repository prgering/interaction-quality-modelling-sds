#!/bin/bash
#SBATCH --job-name=preprocess_speech_features
#SBATCH --time=90:00:00
#SBATCH --nodes=1
#SBATCH --mem=50G
#SBATCH --cpus-per-task=16
#SBATCH --array=1-16
#SBATCH --output=logs/preprocess_features/%x_%A_%a.txt

# --- Source Shared Environment Setup ---
source slurm/scripts/setup_env.sh

# Ensure output log folder exists
mkdir -p logs/preprocess_features

# --- Parse Parameters ---
PARAMS=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$PROJECT_ROOT/slurm/configs/speech_pca_combos.txt")
read -r SPEECHF_TEXT_PCA SPEECHF_WAV_PCA <<< "$PARAMS"

# --- Execution ---
python -m scripts.prepare_features_for_modelling \
    speech \
    --speechf_text_pca ${SPEECHF_TEXT_PCA} \
    --speechf_wav_pca ${SPEECHF_WAV_PCA}
