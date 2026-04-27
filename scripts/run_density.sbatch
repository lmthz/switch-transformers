#!/bin/bash -l
#SBATCH -J sw_density
#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 06:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs

source /home/lizthorn/miniforge3/etc/profile.d/conda.sh
conda activate switchgpu

echo "=== Environment ==="
python -c "import torch; print('cuda:', torch.cuda.is_available(), '| devices:', torch.cuda.device_count())"
python -c "import wandb; print('wandb:', wandb.__version__)"
echo ""

if [ ! -f "msar_results.csv" ]; then
    echo "[error] msar_results.csv not found. Run: sbatch scripts/run_msar.sbatch"
    exit 1
fi
N_FOUND=$(ls generated_data/*_r*.npz 2>/dev/null | wc -l)
if [ "$N_FOUND" -lt 63 ]; then
    echo "[error] Only $N_FOUND dataset files found (need 63). Run: sbatch scripts/generate_data.sbatch"
    exit 1
fi

POOL=series_pool.npz
if [ ! -f "$POOL" ]; then
    echo "=== Generating full series pool ==="
    python generate_pool.py --n_series 500000 --ar_coeff_scale 0.6 \
        --family_preset full --out "$POOL"
fi

echo "=== Prerequisites OK ($N_FOUND dataset files) ==="
echo ""

# B1 pool args
B1_ARGS=""
if [ -f "pool_b1_ar_only.npz" ];       then B1_ARGS="$B1_ARGS --pool_b1_ar_only pool_b1_ar_only.npz"; fi
if [ -f "pool_b1_ar_arma.npz" ];       then B1_ARGS="$B1_ARGS --pool_b1_ar_arma pool_b1_ar_arma.npz"; fi
if [ -f "pool_b1_ar_arma_arima.npz" ]; then B1_ARGS="$B1_ARGS --pool_b1_ar_arma_arima pool_b1_ar_arma_arima.npz"; fi
if [ -f "series_pool.npz" ];            then B1_ARGS="$B1_ARGS --pool_b1_full series_pool.npz"; fi

# B3 pool args — scale=0.6 reuses series_pool.npz
B3_ARGS=""
if [ -f "pool_b3_0.1.npz" ];  then B3_ARGS="$B3_ARGS --pool_b3_0_1 pool_b3_0.1.npz"; fi
if [ -f "pool_b3_0.2.npz" ];  then B3_ARGS="$B3_ARGS --pool_b3_0_2 pool_b3_0.2.npz"; fi
if [ -f "pool_b3_0.3.npz" ];  then B3_ARGS="$B3_ARGS --pool_b3_0_3 pool_b3_0.3.npz"; fi
if [ -f "pool_b3_0.4.npz" ];  then B3_ARGS="$B3_ARGS --pool_b3_0_4 pool_b3_0.4.npz"; fi
if [ -f "pool_b3_0.5.npz" ];  then B3_ARGS="$B3_ARGS --pool_b3_0_5 pool_b3_0.5.npz"; fi
if [ -f "series_pool.npz" ];   then B3_ARGS="$B3_ARGS --pool_b3_0_6 series_pool.npz"; fi
if [ -f "pool_b3_0.8.npz" ];  then B3_ARGS="$B3_ARGS --pool_b3_0_8 pool_b3_0.8.npz"; fi
if [ -f "pool_b3_1.0.npz" ];  then B3_ARGS="$B3_ARGS --pool_b3_1_0 pool_b3_1.0.npz"; fi
if [ -f "pool_b3_1.2.npz" ];  then B3_ARGS="$B3_ARGS --pool_b3_1_2 pool_b3_1.2.npz"; fi

echo "=== Running density experiments ==="
python run_density_experiment.py \
    --experiments A B1 B2 B3 C \
    --pool_path "$POOL" \
    --msar_csv msar_results.csv \
    --n_instances 3 \
    --exp_a_steps 10000 \
    --exp_b_steps 10000 \
    $B1_ARGS \
    $B3_ARGS

echo ""
echo "=== Complete ==="
echo "Results:"
echo "  results_density_exp_a.csv   (linear regression)"
echo "  results_density_exp_b1.csv  (process family coverage)"
echo "  results_density_exp_b2.csv  (AR order coverage)"
echo "  results_density_exp_b3.csv  (coefficient magnitude)"
echo "  results_density_exp_c.csv   (training steps sweep)"
echo "W&B: ls wandb/ then: wandb sync wandb/offline-run-<id>"