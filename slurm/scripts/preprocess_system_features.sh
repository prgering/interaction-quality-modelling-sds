#!/bin/bash
#SBATCH --job-name=preprocess_system_features
#SBATCH --time=90:00:00
#SBATCH --nodes=1
#SBATCH --mem=1G
#SBATCH --cpus-per-task=1
#SBATCH --array=1-4
#
# ---USER REQUIRED---
#SBATCH --output=/mnt/parscratch/users/acp23prg/iq_lego_repo/experiments/slurm_output/2026-01-26_preprocess_features/slurm_job_output_%A_%a.txt
#SBATCH --mail-user=prgering1@sheffield.ac.uk
#SBATCH --mail-type=END,FAIL

# Users MUST verify these module specifications are correct for their specific HPC system.
module load CUDA/12.4.0
module load cuDNN/9.1.1.17-CUDA-12.4.0

# --- Conda Environment Setup ---
source ~/miniconda3/etc/profile.d/conda.sh
conda activate iq_predict_env
echo "Conda environment active: $CONDA_DEFAULT_ENV"

# Specify Date, Dataset Type
DATE=$(date +%Y-%m-%d)
DATASET_TYPE="system_features"

# --- Specify code directory ---
PROJECT_ROOT="/mnt/parscratch/users/acp23prg/iq_lego_repo/" # Adjust this path as necessary to the script
export LEGO_BASE_PATH=$PROJECT_ROOT

BASE_EXP_DIR="$PROJECT_ROOT/experiments"
EXPERIMENT_BATCH_NAME="${DATE}_preprocess_dataset"
EXPERIMENT_ROOT_DIR="${BASE_EXP_DIR}/${EXPERIMENT_BATCH_NAME}"

THIS_GROUP_DIR="${EXPERIMENT_ROOT_DIR}/${DATASET_TYPE}_preprocess"

mkdir -p "${EXPERIMENT_ROOT_DIR}"
mkdir -p "${THIS_GROUP_DIR}/logs"

# ---Hyperparams and Array Mapping---
PCA_VALUES=( "0.5" "0.6" "0.7" "0.8" )
SYSTEMF_TEXT_PCA="${PCA_VALUES[$SLURM_ARRAY_TASK_ID]}"

# Define dynamic output file name based on parameters
DYNAMIC_OUTPUT_FILE="output_preprocess_${DATASET_TYPE}_sytxtpca${SYSTEMF_TEXT_PCA//./p}_sptxtpca_spwpca.HM.txt"

# --- Specify Python script ---
PYTHON_SCRIPT="code.modelling_iq.preprocess_feature_data" # Adjust this path as necessary to the script

# --- Change to project code directory ---
cd $PROJECT_ROOT || exit 1

# --- Echoing Run Information ---
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
echo "SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}"
echo "Preprocessing ${DATASET_TYPE} with:"
echo " SYSTEMF TEXT PCA: ${SYSTEMF_TEXT_PCA}"
echo "Output will be logged to: ${DYNAMIC_OUTPUT_FILE}"

# Execute your Python training script, redirecting stdout and stderr
python -m "${PYTHON_SCRIPT}" \
    "${DATASET_TYPE}" \
    --systemf_text_pca "${SYSTEMF_TEXT_PCA}" \
    > "${THIS_GROUP_DIR}/logs/${DYNAMIC_OUTPUT_FILE}" 2>&1

# --- Check Job Status ---
if [ $? -eq 0 ]; then
    echo "Job completed successfully."
else
    echo "Job failed with exit status $?."
fi

