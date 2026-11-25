#!/bin/bash -l
#SBATCH -J switch_pool
#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs

source /home/lizthorn/miniforge3/etc/profile.d/conda.sh
conda activate switchgpu

echo "=== Generating single-switch AR pool ==="
python generate_switch_pool.py \
    --n_series 200000 \
    --series_len 512 \
    --burn_in 100 \
    --ar_order 2 \
    --ar_coeff_scale 1.2 \
    --sigma_lo 0.15 \
    --sigma_hi 0.70 \
    --seed 42 \
    --out switch_pool.npz \
    --chunk_size 512

echo "=== Done — pool saved to switch_pool.npz ==="
