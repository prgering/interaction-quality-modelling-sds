#!/bin/bash
# Shared Setup for HPC Jobs

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
export LEGO_BASE_PATH="$PROJECT_ROOT"
cd "$PROJECT_ROOT"

# --- Conda Environment Setup ---
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
fi

conda activate iq_predict_env