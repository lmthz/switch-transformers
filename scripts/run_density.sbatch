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
    echo "[error] series_pool.npz not found. Run: sbatch scripts/generate_density_pools.sbatch"
    exit 1
fi

echo "=== Prerequisites OK ($N_FOUND dataset files) ==="
echo ""

B1_ARGS=""
[ -f "pool_b1_ar_only.npz" ]       && B1_ARGS="$B1_ARGS --pool_b1_ar_only pool_b1_ar_only.npz"
[ -f "pool_b1_ar_arma.npz" ]       && B1_ARGS="$B1_ARGS --pool_b1_ar_arma pool_b1_ar_arma.npz"
[ -f "pool_b1_ar_arma_arima.npz" ] && B1_ARGS="$B1_ARGS --pool_b1_ar_arma_arima pool_b1_ar_arma_arima.npz"
[ -f "series_pool.npz" ]            && B1_ARGS="$B1_ARGS --pool_b1_full series_pool.npz"

B2_ARGS=""
[ -f "pool_b2_order2.npz" ] && B2_ARGS="$B2_ARGS --pool_b2_lo_order pool_b2_order2.npz"
[ -f "pool_b2_order4.npz" ] && B2_ARGS="$B2_ARGS --pool_b2_mid_order pool_b2_order4.npz"
[ -f "pool_b2_order6.npz" ] && B2_ARGS="$B2_ARGS --pool_b2_hi_order pool_b2_order6.npz"
[ -f "series_pool.npz" ]     && B2_ARGS="$B2_ARGS --pool_b2_full series_pool.npz"

B3_ARGS=""
[ -f "pool_b3_0.1.npz" ] && B3_ARGS="$B3_ARGS --pool_b3_0_1 pool_b3_0.1.npz"
[ -f "pool_b3_0.2.npz" ] && B3_ARGS="$B3_ARGS --pool_b3_0_2 pool_b3_0.2.npz"
[ -f "pool_b3_0.3.npz" ] && B3_ARGS="$B3_ARGS --pool_b3_0_3 pool_b3_0.3.npz"
[ -f "pool_b3_0.4.npz" ] && B3_ARGS="$B3_ARGS --pool_b3_0_4 pool_b3_0.4.npz"
[ -f "pool_b3_0.5.npz" ] && B3_ARGS="$B3_ARGS --pool_b3_0_5 pool_b3_0.5.npz"
[ -f "series_pool.npz" ]  && B3_ARGS="$B3_ARGS --pool_b3_0_6 series_pool.npz"
[ -f "pool_b3_0.8.npz" ] && B3_ARGS="$B3_ARGS --pool_b3_0_8 pool_b3_0.8.npz"
[ -f "pool_b3_1.0.npz" ] && B3_ARGS="$B3_ARGS --pool_b3_1_0 pool_b3_1.0.npz"
[ -f "pool_b3_1.2.npz" ] && B3_ARGS="$B3_ARGS --pool_b3_1_2 pool_b3_1.2.npz"

DE_ARGS=""
if [ -d "pools_de" ]; then
    DE_ARGS="--pool_d_dir pools_de --pool_e_dir_full pools_de --pool_e_dir_ar_only pools_de"
    echo "=== D/E will use pre-generated pools from pools_de/ ==="
else
    echo "=== D/E will use on-the-fly generation (run generate_density_pools.sbatch to speed up reruns) ==="
fi
echo ""

echo "=== Running density experiments ==="
# Default: run A B3 D E only (B1 B2 C already have good results)
# To run all: change --experiments line to: --experiments A B1 B2 B3 C D E
python run_density_experiment.py \
    --experiments A B3 D E \
    --pool_path "$POOL" \
    --msar_csv msar_results.csv \
    --n_instances 3 \
    --exp_a_steps 50000 \
    --exp_b_steps 10000 \
    $B1_ARGS \
    $B2_ARGS \
    $B3_ARGS \
    $DE_ARGS

echo ""
echo "=== Complete ==="
echo "Results:"
echo "  results_density_exp_a.csv   (linear regression)"
echo "  results_density_exp_b1.csv  (process family coverage)"
echo "  results_density_exp_b2.csv  (AR order coverage)"
echo "  results_density_exp_b3.csv  (coefficient magnitude)"
echo "  results_density_exp_c.csv   (training steps sweep)"
echo "  results_density_exp_d.csv   (task diversity sweep, Raventos replication)"
echo "  results_density_exp_e.csv   (task diversity x model class, 2D sweep)"
echo "W&B: ls wandb/ then: wandb sync wandb/offline-run-<id>"