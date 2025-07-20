#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --partition=a100-grind
#SBATCH --job-name=medmnist_cnn
#SBATCH --mail-user=s.salem@student.uq.edu.au
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=/home/Student/s4964547/medmnist_project/logs/%j.out
#SBATCH --time=0-00:40:00

# ============================================================
# Runtime configuration (override at submit: sbatch --export=ALL,EPOCHS=5,BATCH=64,LR=0.001,...)
# NOTE: DO **NOT** paste your raw W&B API key in this file if the repo is shared.
#       Preferred: store the key in a protected text file (0600 perms) or export at submit.
# ============================================================
: ${EPOCHS:=5}
: ${BATCH:=64}
: ${LR:=0.001}
: ${DATA_FLAG:=pathmnist}
: ${MODEL:=simplecnn}         # model choice: simplecnn, resnet18, densenet121, efficientnet_b0
: ${WANDB_PROJECT:=medmnist}
: ${WANDB_ENTITY:=}            # optional
: ${WANDB_MODE:=online}       # set to online if cluster has outbound internet
: ${WANDB_RUNNAME:=run_${SLURM_JOB_ID}}
: ${WANDB_GROUP:=slurm_${SLURM_JOB_ID}}
: ${WANDB_API_KEY:=}           # will auto-populate below if secret file exists

# --- Secure W&B key injection -----------------------------------------------
# If you want to avoid exposing your API key in your SLURM command history,
# create a *private* file:  chmod 600 /home/Student/s4964547/medmnist_project/.secrets/wandb_api_key
# that contains the key on a single line.
# The block below will read it automatically if WANDB_API_KEY is empty.
PROJECT_DIR=/home/Student/s4964547/medmnist_project   # >>> USER EDIT PATH if moved
SECRET_KEY_FILE=${PROJECT_DIR}/.secrets/wandb_api_key
if [[ -z "${WANDB_API_KEY}" && -f "${SECRET_KEY_FILE}" ]]; then
    WANDB_API_KEY=$(<"${SECRET_KEY_FILE}")
    export WANDB_API_KEY
    echo "[W&B] Loaded API key from ${SECRET_KEY_FILE}"  # will appear in log but not the key itself
fi

# Project directories ----------------------------------------------------------
PROJECT_DIR=/home/Student/s4964547/medmnist_project
PKG_DIR=${PROJECT_DIR}/medmnist_train
IMG_PATH=${PROJECT_DIR}/medmnist.sif
LOG_DIR=${PROJECT_DIR}/logs
mkdir -p "${LOG_DIR}"

cd "${PROJECT_DIR}" || { echo "ERROR: PROJECT_DIR not found"; exit 1; }

# Confirm Singularity exists ---------------------------------------------------
if ! command -v singularity >/dev/null 2>&1; then
    echo "ERROR: Singularity not found in PATH" >&2
    exit 2
fi

# Show environment -------------------------------------------------------------
echo "=== Runtime Info ==="
echo "Hostname: $(hostname)"
echo "SLURM_JOBID: ${SLURM_JOB_ID}"
echo "Node list: ${SLURM_NODELIST}"
echo "GPUs requested: ${SLURM_GPUS}"
echo "Working dir: $(pwd)"
echo "Image: ${IMG_PATH}"
echo "EPOCHS=${EPOCHS} BATCH=${BATCH} LR=${LR} DATA_FLAG=${DATA_FLAG} MODEL=${MODEL}"
echo "WANDB_MODE=${WANDB_MODE} WANDB_PROJECT=${WANDB_PROJECT} WANDB_RUNNAME=${WANDB_RUNNAME}"
echo "--------------------"

# Export env so they pass into container ---------------------------------------
export WANDB_MODE WANDB_PROJECT WANDB_ENTITY WANDB_RUNNAME WANDB_GROUP WANDB_API_KEY

# Pre-run GPU diagnostics on host (optional; may be blank if no driver on host node wrapper)
echo "=== Host GPU Info (may be empty on some systems) ==="
which nvidia-smi && nvidia-smi || echo "nvidia-smi not available on host"

# Run container & training -----------------------------------------------------
srun singularity exec --nv \
    -B ${PROJECT_DIR}:/workspace \
    ${IMG_PATH} \
    bash -c "cd /workspace && python -m src.train \
        --epochs ${EPOCHS} \
        --batch_size ${BATCH} \
        --lr ${LR} \
        --data_flag ${DATA_FLAG} \
        --model ${MODEL} \
        --wandb_project \"${WANDB_PROJECT}\" \
        --wandb_entity \"${WANDB_ENTITY}\" \
        --wandb_runname \"${WANDB_RUNNAME}\" \
        --wandb_group \"${WANDB_GROUP}\""