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
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
export LEGO_BASE_PATH="$PROJECT_ROOT"
cd "$PROJECT_ROOT"

# Ensure output log folder exists
mkdir -p logs/extract_features

# --- Conda Environment Setup ---
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
fi

conda activate iq_predict_env

# --- Task logic ---
case "$MODE" in
    "static_speech") 
        PYTHON_SCRIPT="scripts.extract_static_speech_features" 
        ;;
    "speech") 
        module load CUDA/12.4.0 cuDNN/9.1.1.17-CUDA-12.4.0 2>/dev/null || true
        PYTHON_SCRIPT="scripts.extract_speech_features" 
        ;;
    *)
        echo "Invalid mode: $MODE. Please choose system, static, or speech."
        exit 1
        ;;
esac

# --- Execution ---
python -m "$PYTHON_SCRIPT"