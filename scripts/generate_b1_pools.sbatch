#!/bin/bash -l
#SBATCH -J sw_b1pools
#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t 02:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs

source /home/lizthorn/miniforge3/etc/profile.d/conda.sh
conda activate switchgpu

echo "=== Generating B1 pools (CPU only) ==="
echo ""

if [ -f "pool_b1_ar_only.npz" ]; then
    echo "pool_b1_ar_only.npz already exists, skipping"
else
    echo "=== ar_only pool ==="
    python generate_pool.py \
        --n_series 500000 \
        --ar_coeff_scale 0.6 \
        --family_preset ar_only \
        --out pool_b1_ar_only.npz
fi

if [ -f "pool_b1_ar_arma.npz" ]; then
    echo "pool_b1_ar_arma.npz already exists, skipping"
else
    echo "=== ar_arma pool ==="
    python generate_pool.py \
        --n_series 500000 \
        --ar_coeff_scale 0.6 \
        --family_preset ar_arma \
        --out pool_b1_ar_arma.npz
fi

if [ -f "pool_b1_ar_arma_arima.npz" ]; then
    echo "pool_b1_ar_arma_arima.npz already exists, skipping"
else
    echo "=== ar_arma_arima pool ==="
    python generate_pool.py \
        --n_series 500000 \
        --ar_coeff_scale 0.6 \
        --family_preset ar_arma_arima \
        --out pool_b1_ar_arma_arima.npz
fi

if [ -f "series_pool.npz" ]; then
    echo "series_pool.npz (full) already exists, skipping"
else
    echo "=== full pool ==="
    python generate_pool.py \
        --n_series 500000 \
        --ar_coeff_scale 0.6 \
        --family_preset full \
        --out series_pool.npz
fi

echo ""
echo "=== Done. Run next: sbatch scripts/run_density.sbatch ==="