#!/bin/bash -l
#SBATCH -J sw_data
#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t 01:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs

source /home/lizthorn/miniforge3/etc/profile.d/conda.sh
conda activate switchgpu

echo "=== Generating evaluation datasets (30 instances per type) ==="
echo "=== Generates 630 .npz files: 21 dataset types x 30 random instances ==="
python data_generation.py --n_instances 30
echo "=== Done. Run next: sbatch scripts/run_msar.sbatch ==="