#!/bin/bash -l
#SBATCH -J sw_noswitch
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

# ── Step 1: Generate no-switch evaluation datasets ──────────────────────────
N_NS=$(ls generated_data_noswitch/*_r*.npz 2>/dev/null | wc -l)
EXPECTED_NS=510   # 17 datasets × 30 instances
if [ "$N_NS" -lt "$EXPECTED_NS" ]; then
    echo "=== Generating no-switch datasets ($N_NS / $EXPECTED_NS found) ==="
    python generate_noswitch_data.py --n_instances 30 --n 1000 --burn 600
else
    echo "=== No-switch datasets already present ($N_NS files) ==="
fi
echo ""

# ── Step 2: Generate no-switch training pool ────────────────────────────────
POOL_NS=pool_noswitch.npz
if [ ! -f "$POOL_NS" ]; then
    echo "=== Generating no-switch training pool ==="
    python generate_pool.py \
        --no_switch \
        --n_series 524288 \
        --ar_coeff_scale 1.2 \
        --family_preset full \
        --out "$POOL_NS"
else
    echo "=== No-switch pool already present: $POOL_NS ==="
fi
echo ""

# ── Step 3: Run Experiment F ─────────────────────────────────────────────────
echo "=== Running Experiment F (no-switch training steps sweep) ==="
python run_density_experiment.py \
    --experiments F \
    --data_dir_noswitch generated_data_noswitch \
    --pool_noswitch "$POOL_NS" \
    --n_instances 3 \
    --exp_b_steps 10000

echo ""
echo "=== Complete ==="
echo "Results:"
echo "  results_density_exp_f.csv  (no-switch training steps sweep)"
