#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --partition=a100-grind
#SBATCH --job-name=medmnist_cnn
#SBATCH --mail-user=s.salem@student.uq.edu.au
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output=/home/Student/s4964547/medmnist_project/logs/%j.out
#SBATCH --time=0-00:40:00

cd /home/Student/s4964547/medmnist_project || { echo "PROJECT_DIR not found"; exit 1; }

command -v singularity >/dev/null 2>&1 || { echo "ERROR: Singularity not installed"; exit 1; }
module load singularity

echo "=== Runtime Info ==="
echo "Hostname: $(hostname)"
echo "SLURM_JOBID: $SLURM_JOB_ID"
echo "Node list: $SLURM_NODELIST"
echo "GPUs requested: $SLURM_GPUS"
echo "Working dir: $(pwd)"
echo "Image: /home/Student/s4964547/medmnist_project/medmnist.sif"
echo "--------------------"
