#!/bin/bash -l
#SBATCH -J sw_noswitch_gen
#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 04:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs

source /home/lizthorn/miniforge3/etc/profile.d/conda.sh
conda activate switchgpu

echo "=== Step 1: No-switch evaluation datasets ==="
N_NS=$(ls generated_data_noswitch/*_r*.npz 2>/dev/null | wc -l)
EXPECTED_NS=510   # 17 datasets × 30 instances
if [ "$N_NS" -lt "$EXPECTED_NS" ]; then
    echo "Generating ($N_NS / $EXPECTED_NS found)..."
    python generate_noswitch_data.py --n_instances 30 --n 1000 --burn 600
else
    echo "Already present ($N_NS files) — skipping."
fi
echo ""

echo "=== Step 2: No-switch training pool ==="
POOL_NS=pool_noswitch.npz
if [ ! -f "$POOL_NS" ]; then
    echo "Generating pool_noswitch.npz (524k series)..."
    python generate_pool.py \
        --no_switch \
        --n_series 524288 \
        --ar_coeff_scale 1.2 \
        --family_preset full \
        --out "$POOL_NS"
else
    echo "Already present: $POOL_NS — skipping."
fi

echo ""
echo "=== Generation complete ==="
echo "  generated_data_noswitch/  ($(ls generated_data_noswitch/*_r*.npz 2>/dev/null | wc -l) files)"
echo "  pool_noswitch.npz"
echo "Now run: sbatch scripts/run_noswitch.sbatch"
