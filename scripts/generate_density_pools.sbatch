#!/bin/bash -l
#SBATCH -J sw_pools
#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 06:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs pools_de

source /home/lizthorn/miniforge3/etc/profile.d/conda.sh
conda activate switchgpu

echo "=== Generating all density experiment pools (CPU only) ==="
echo "=== AR order fixed at 2 for all experiments except B2 ==="
echo ""

# ── Full pool (used by C, D/E full, B3/0.6) ────────────────────
# ar_order_hi=2: order fixed, not a variable in these experiments
if [ -f "series_pool.npz" ]; then
    echo "series_pool.npz already exists, skipping"
else
    echo "=== full pool (ar_order_hi=2) ==="
    python generate_pool.py --n_series 500000 --ar_coeff_scale 0.6 \
        --family_preset full --ar_order_lo 1 --ar_order_hi 2 \
        --out series_pool.npz
fi

# ── B1 pools — family preset, ar_order_hi=2 ───────────────────
if [ -f "pool_b1_ar_only.npz" ]; then
    echo "pool_b1_ar_only.npz already exists, skipping"
else
    echo "=== B1: ar_only (ar_order_hi=2) ==="
    python generate_pool.py --n_series 500000 --ar_coeff_scale 0.6 \
        --family_preset ar_only --ar_order_lo 1 --ar_order_hi 2 \
        --out pool_b1_ar_only.npz
fi

if [ -f "pool_b1_ar_arma.npz" ]; then
    echo "pool_b1_ar_arma.npz already exists, skipping"
else
    echo "=== B1: ar_arma (ar_order_hi=2) ==="
    python generate_pool.py --n_series 500000 --ar_coeff_scale 0.6 \
        --family_preset ar_arma --ar_order_lo 1 --ar_order_hi 2 \
        --out pool_b1_ar_arma.npz
fi

if [ -f "pool_b1_ar_arma_arima.npz" ]; then
    echo "pool_b1_ar_arma_arima.npz already exists, skipping"
else
    echo "=== B1: ar_arma_arima (ar_order_hi=2) ==="
    python generate_pool.py --n_series 500000 --ar_coeff_scale 0.6 \
        --family_preset ar_arma_arima --ar_order_lo 1 --ar_order_hi 2 \
        --out pool_b1_ar_arma_arima.npz
fi

# ── B2 pools — AR order variation (only experiment where order varies) ──
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

# ── B3 pools — coefficient scale, ar_order_hi=2 ───────────────
# Note: delete old B3 pools before running if they used ar_order_hi=10:
#   rm pool_b3_0.1.npz pool_b3_0.2.npz pool_b3_0.3.npz pool_b3_0.4.npz
#   rm pool_b3_0.5.npz pool_b3_0.8.npz pool_b3_1.0.npz pool_b3_1.2.npz

for SCALE in 0.1 0.2 0.3 0.4 0.5 0.8 1.0 1.2; do
    FNAME="pool_b3_${SCALE}.npz"
    if [ -f "$FNAME" ]; then
        echo "$FNAME already exists, skipping"
    else
        echo "=== B3: ar_coeff_scale=${SCALE} (ar_order_hi=2) ==="
        python generate_pool.py --n_series 500000 --ar_coeff_scale "$SCALE" \
            --family_preset full --ar_order_lo 1 --ar_order_hi 2 \
            --out "$FNAME"
    fi
done

# ── D/E pools — task diversity, ar_order_hi=2 ──────────────────
M_VALUES="128 256 512 1024 2048 4096 8192 16384 32768 65536 131072 262144 500000"

echo ""
echo "=== D/E pools (full preset, ar_order_hi=2) ==="
for M in $M_VALUES; do
    FNAME="pools_de/pool_d_full_${M}.npz"
    if [ -f "$FNAME" ]; then
        echo "$FNAME already exists, skipping"
    elif [ "$M" -eq 500000 ] && [ -f "series_pool.npz" ]; then
        echo "M=500000 full: symlinking series_pool.npz"
        ln -sf "$(pwd)/series_pool.npz" "$FNAME"
    else
        echo "=== D/E full: M=${M} (ar_order_hi=2) ==="
        python generate_pool.py --n_series "$M" --ar_coeff_scale 0.6 \
            --family_preset full --ar_order_lo 1 --ar_order_hi 2 \
            --out "$FNAME"
    fi
done

echo ""
echo "=== D/E pools (ar_only preset, ar_order_hi=2) ==="
for M in $M_VALUES; do
    FNAME="pools_de/pool_e_ar_only_${M}.npz"
    if [ -f "$FNAME" ]; then
        echo "$FNAME already exists, skipping"
    else
        echo "=== E ar_only: M=${M} (ar_order_hi=2) ==="
        python generate_pool.py --n_series "$M" --ar_coeff_scale 0.6 \
            --family_preset ar_only --ar_order_lo 1 --ar_order_hi 2 \
            --out "$FNAME"
    fi
done

echo ""
echo "=== All pools generated (ar_order_hi=2 everywhere except B2) ==="
echo "Run next: sbatch scripts/run_density.sbatch"