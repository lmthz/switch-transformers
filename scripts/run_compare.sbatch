#!/bin/bash -l
#SBATCH -J switchtr
#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 06:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs

# Environment: pick ONE approach
# Approach A: ORCD module + venv (common)
module load miniforge  

python -c "import torch; print('cuda available:', torch.cuda.is_available()); print('device count:', torch.cuda.device_count())"

# Skip if already generated data
python data_generation.py

# Runs transformer training on CUDA (your run_compare.py sets device="cuda")
python run_compare.py