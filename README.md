# MedMnist_HPC

A reproducible deep learning pipeline for MedMNIST datasets, supporting both local and HPC (Slurm + Singularity/Apptainer) execution, with integrated Weights & Biases (W&B) experiment tracking and hyperparameter sweeps.

---

## Project Structure

```
MedMnist_HPC/
├── src/                # Source code (train.py, model.py, etc.)
├── sweep/              # Sweep configs and batch scripts
├── misclassified_samples/  # Output images
├── .secrets/           # Store your W&B API key here (not tracked)
├── medmnist.sif        # Singularity/Apptainer container image
├── requirements.txt    # Python dependencies
├── run_medmnist.sbatch # Example single-run Slurm script
├── sweep.yaml          # W&B sweep config
└── ...
```

---

## Prerequisites

- Python 3.8+
- [Weights & Biases](https://wandb.ai/) account
- (Local) Install dependencies: `pip install -r requirements.txt`
- (HPC) Singularity/Apptainer and Slurm access
- W&B API key in `.secrets/wandb_api_key` (chmod 600 for security)
- (HPC) Container image: `medmnist.sif` (build or request from admin)

---

## 1. Single Training Run


### **Inside Singularity/Apptainer (Local)**

```bash
singularity exec --nv -B $(pwd):/workspace medmnist.sif \
  bash -c "cd /workspace && WANDB_API_KEY=\$(<.secrets/wandb_api_key) python src/train.py --data_flag dermamnist --epochs 50 --batch_size 64 --lr 0.001 --model resnet18 --wandb_project medmnist --wandb_entity concept-based-x"
```

### **On HPC (Slurm + Singularity)**

Edit `run_medmnist.sbatch` as needed, then submit:

```bash
sbatch run_medmnist.sbatch
```

---

## 2. Hyperparameter Sweep with W&B

### **A. Start the Container Locally**

```bash
singularity shell --nv -B $(pwd):/workspace medmnist.sif
```

- This opens an interactive shell inside the container with your project directory mounted at `/workspace`.

### **B. Start the Sweep Inside the Container**

```bash
cd /workspace
WANDB_API_KEY=$(<.secrets/wandb_api_key) wandb sweep sweep/sweep.yaml --name "my_sweep_name"
```
- Copy the sweep ID from the output (e.g., `abcd1234`).

### **C. Run Sweep Agent(s) on HPC**

Submit a sweep agent (runs up to `run_cap` experiments sequentially):

```bash
sbatch --export=ALL,SWEEP_ID=abcd1234 sweep/run_sweep_medmnist.sbatch
```


### **C. Run Sweep Agent Locally (for testing)**

```bash
WANDB_API_KEY=$(<.secrets/wandb_api_key) wandb agent concept-based-x/medmnist/abcd1234
```

---

## 3. Notes & Best Practices

- **W&B API Key:** Never hardcode. Always use `.secrets/wandb_api_key` and load as shown above.
- **Containerization:** All HPC runs use `medmnist.sif` for reproducibility.
- **Sweep Config:** Edit `sweep/sweep.yaml` to change parameters, project/entity, or sweep method.
- **Logs:** Check output logs in `logs/` directory on HPC.
- **Results:** Monitor all runs and sweeps on your [W&B dashboard](https://wandb.ai/concept-based-x/medmnist).
- **Security:** `.secrets/wandb_api_ksey` should be `chmod 600` and never committed to git.

---