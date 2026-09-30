#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --partition=a100-grind
#SBATCH --job-name=medmnist_cnn
#SBATCH --mail-user=-
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output=[add your dir]/%j.out
#SBATCH --time=0-00:40:00

cd [add your dir]/medmnist_project || { echo "PROJECT_DIR not found"; exit 1; }

command -v singularity >/dev/null 2>&1 || { echo "ERROR: Singularity not installed"; exit 1; }
module load singularity

echo "=== Runtime Info ==="
echo "Hostname: $(hostname)"
echo "SLURM_JOBID: $SLURM_JOB_ID"
echo "Node list: $SLURM_NODELIST"
echo "GPUs requested: $SLURM_GPUS"
echo "Working dir: $(pwd)"
echo "Image: [add your dir]/medmnist_project/medmnist.sif"
echo "--------------------"

srun singularity exec --nv -B [add your dir]/medmnist_project:/workspace [add your dir]/medmnist_project/medmnist.sif python /workspace/train.py
