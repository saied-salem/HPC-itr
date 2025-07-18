#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --partition=a100-test
#SBATCH --job-name=medmnist_cnn
#SBATCH --mail-user=sa.salem.saied@gmail.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output=/home/s4964547/medmnist_project/logs/%j.out
#SBATCH --time=0-00:05:00

mkdir -p /home/s4964547/medmnist_project/logs
cd /home/s4964547/medmnist_project || { echo "PROJECT_DIR not found"; exit 1; }

module load singularity

echo "=== Runtime Info ==="
echo "Hostname: $(hostname)"
echo "SLURM_JOBID: $SLURM_JOB_ID"
echo "Node list: $SLURM_NODELIST"
echo "GPUs requested: $SLURM_GPUS"
echo "Working dir: $(pwd)"
echo "Image: /home/s4964547/medmnist_project/medmnist.sif"
echo "--------------------"

srun singularity exec --nv -B /home/s4964547/medmnist_project:/workspace /home/s4964547/medmnist_project/medmnist.sif python /workspace/train.py
