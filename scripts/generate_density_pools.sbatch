#!/bin/bash -l
#SBATCH -J sw_pools
#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t 06:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs

source /home/lizthorn/miniforge3/etc/profile.d/conda.sh
conda activate switchgpu

echo "=== Generating all series pools (CPU only) ==="
echo ""

if [ -f "series_pool.npz" ]; then
    echo "series_pool.npz already exists, skipping"
else
    echo "=== full pool ==="
    python generate_pool.py --n_series 500000 --ar_coeff_scale 0.6 \
        --family_preset full --ar_order_lo 1 --ar_order_hi 10 \
        --out series_pool.npz
fi

if [ -f "pool_b1_ar_only.npz" ]; then
    echo "pool_b1_ar_only.npz already exists, skipping"
else
    echo "=== B1: ar_only ==="
    python generate_pool.py --n_series 500000 --ar_coeff_scale 0.6 \
        --family_preset ar_only --ar_order_lo 1 --ar_order_hi 10 \
        --out pool_b1_ar_only.npz
fi

if [ -f "pool_b1_ar_arma.npz" ]; then
    echo "pool_b1_ar_arma.npz already exists, skipping"
else
    echo "=== B1: ar_arma ==="
    python generate_pool.py --n_series 500000 --ar_coeff_scale 0.6 \
        --family_preset ar_arma --ar_order_lo 1 --ar_order_hi 10 \
        --out pool_b1_ar_arma.npz
fi

if [ -f "pool_b1_ar_arma_arima.npz" ]; then
    echo "pool_b1_ar_arma_arima.npz already exists, skipping"
else
    echo "=== B1: ar_arma_arima ==="
    python generate_pool.py --n_series 500000 --ar_coeff_scale 0.6 \
        --family_preset ar_arma_arima --ar_order_lo 1 --ar_order_hi 10 \
        --out pool_b1_ar_arma_arima.npz
fi

if [ -f "pool_b2_order2.npz" ]; then
    echo "pool_b2_order2.npz already exists, skipping"
else
    echo "=== B2: ar_order_hi=2 ==="
    python generate_pool.py --n_series 500000 --ar_coeff_scale 0.6 \
        --family_preset full --ar_order_lo 1 --ar_order_hi 2 \
        --out pool_b2_order2.npz
fi

if [ -f "pool_b2_order4.npz" ]; then
    echo "pool_b2_order4.npz already exists, skipping"
else
    echo "=== B2: ar_order_hi=4 ==="
    python generate_pool.py --n_series 500000 --ar_coeff_scale 0.6 \
        --family_preset full --ar_order_lo 1 --ar_order_hi 4 \
        --out pool_b2_order4.npz
fi

if [ -f "pool_b2_order6.npz" ]; then
    echo "pool_b2_order6.npz already exists, skipping"
else
    echo "=== B2: ar_order_hi=6 ==="
    python generate_pool.py --n_series 500000 --ar_coeff_scale 0.6 \
        --family_preset full --ar_order_lo 1 --ar_order_hi 6 \
        --out pool_b2_order6.npz
fi

for SCALE in 0.1 0.2 0.3 0.4 0.5 0.8 1.0 1.2; do
    FNAME="pool_b3_${SCALE}.npz"
    if [ -f "$FNAME" ]; then
        echo "$FNAME already exists, skipping"
    else
        echo "=== B3: ar_coeff_scale=${SCALE} ==="
        python generate_pool.py --n_series 500000 --ar_coeff_scale "$SCALE" \
            --family_preset full --ar_order_lo 1 --ar_order_hi 10 \
            --out "$FNAME"
    fi
done

echo ""
echo "=== All pools generated ==="
echo "Run next: sbatch scripts/run_density.sbatch"