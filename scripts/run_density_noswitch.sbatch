#!/bin/bash -l
#SBATCH -J sw_density_ns
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
echo ""

# Prerequisites: eval datasets
N_NS=$(find generated_data_noswitch/ -name "*_r*.npz" 2>/dev/null | wc -l)
if [ "$N_NS" -lt 510 ]; then
    echo "[error] Only $N_NS no-switch dataset files found (need 510)."
    echo "Run first: sbatch scripts/generate_noswitch.sbatch"
    exit 1
fi

# Prerequisites: training pool (C_NS)
if [ ! -f "pool_noswitch.npz" ]; then
    echo "[error] pool_noswitch.npz not found."
    echo "Run first: sbatch scripts/generate_noswitch_pools.sbatch"
    exit 1
fi

# Prerequisites: B1/B2/B3/D/E pools
N_POOLS_NS=$(find pools_ns/ -name "*.npz" 2>/dev/null | wc -l)
if [ "$N_POOLS_NS" -lt 25 ]; then
    echo "[error] Only $N_POOLS_NS pool files found in pools_ns/ (need at least 25)."
    echo "Run first: sbatch scripts/generate_noswitch_pools.sbatch"
    exit 1
fi

echo "=== Prerequisites OK ($N_NS eval files, $N_POOLS_NS pool files) ==="
echo ""

# Build optional B1 pool args
B1_ARGS=""
[ -f "pool_b1_ns_ar_only.npz" ]       && B1_ARGS="$B1_ARGS --pool_b1_ar_only pool_b1_ns_ar_only.npz"
[ -f "pool_b1_ns_ar_arma.npz" ]       && B1_ARGS="$B1_ARGS --pool_b1_ar_arma pool_b1_ns_ar_arma.npz"
[ -f "pool_b1_ns_ar_arma_arima.npz" ] && B1_ARGS="$B1_ARGS --pool_b1_ar_arma_arima pool_b1_ns_ar_arma_arima.npz"
[ -f "pool_noswitch.npz" ]             && B1_ARGS="$B1_ARGS --pool_b1_full pool_noswitch.npz"

B2_ARGS=""
[ -f "pool_b2_ns_order2.npz" ]  && B2_ARGS="$B2_ARGS --pool_b2_lo_order pool_b2_ns_order2.npz"
[ -f "pool_b2_ns_order4.npz" ]  && B2_ARGS="$B2_ARGS --pool_b2_mid_order pool_b2_ns_order4.npz"
[ -f "pool_b2_ns_order6.npz" ]  && B2_ARGS="$B2_ARGS --pool_b2_hi_order pool_b2_ns_order6.npz"
[ -f "pool_b2_ns_order10.npz" ] && B2_ARGS="$B2_ARGS --pool_b2_full pool_b2_ns_order10.npz"

B3_ARGS=""
[ -f "pool_b3_ns_0.1.npz" ] && B3_ARGS="$B3_ARGS --pool_b3_0_1 pool_b3_ns_0.1.npz"
[ -f "pool_b3_ns_0.2.npz" ] && B3_ARGS="$B3_ARGS --pool_b3_0_2 pool_b3_ns_0.2.npz"
[ -f "pool_b3_ns_0.3.npz" ] && B3_ARGS="$B3_ARGS --pool_b3_0_3 pool_b3_ns_0.3.npz"
[ -f "pool_b3_ns_0.4.npz" ] && B3_ARGS="$B3_ARGS --pool_b3_0_4 pool_b3_ns_0.4.npz"
[ -f "pool_b3_ns_0.5.npz" ] && B3_ARGS="$B3_ARGS --pool_b3_0_5 pool_b3_ns_0.5.npz"
[ -f "pool_b3_ns_0.6.npz" ] && B3_ARGS="$B3_ARGS --pool_b3_0_6 pool_b3_ns_0.6.npz"
[ -f "pool_b3_ns_0.8.npz" ] && B3_ARGS="$B3_ARGS --pool_b3_0_8 pool_b3_ns_0.8.npz"
[ -f "pool_b3_ns_1.0.npz" ] && B3_ARGS="$B3_ARGS --pool_b3_1_0 pool_b3_ns_1.0.npz"
[ -f "pool_b3_ns_1.2.npz" ] && B3_ARGS="$B3_ARGS --pool_b3_1_2 pool_b3_ns_1.2.npz"

DE_ARGS="--pool_d_dir pools_ns --pool_e_dir_full pools_ns --pool_e_dir_ar_only pools_ns"

# Accept EXPS env variable to run specific experiments, e.g.:
#   sbatch --export=ALL,EXPS="C" scripts/run_density_noswitch.sbatch
EXPERIMENTS="${EXPS:-B1 B2 B3 C D E}"
echo "=== Running no-switch density experiments: $EXPERIMENTS ==="
python run_density_experiment.py \
    --noswitch \
    --experiments $EXPERIMENTS \
    --data_dir_noswitch generated_data_noswitch \
    --pool_noswitch pool_noswitch.npz \
    --n_instances 3 \
    --exp_b_steps 10000 \
    $B1_ARGS \
    $B2_ARGS \
    $B3_ARGS \
    $DE_ARGS

echo ""
echo "=== Complete ==="
echo "Results:"
echo "  results_density_exp_b1_ns.csv  (process family coverage, no-switch)"
echo "  results_density_exp_b2_ns.csv  (AR order coverage, no-switch)"
echo "  results_density_exp_b3_ns.csv  (coefficient magnitude, no-switch)"
echo "  results_density_exp_c_ns.csv   (training steps sweep, no-switch)"
echo "  results_density_exp_d_ns.csv   (task diversity sweep, no-switch)"
echo "  results_density_exp_e_ns.csv   (task diversity x model class, no-switch)"
echo "W&B: ls wandb/ then: wandb sync wandb/offline-run-<id>"
