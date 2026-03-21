#!/bin/bash -l
#SBATCH -J switchtr
#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t 06:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs

source /home/${USER}/.conda/etc/profile.d/conda.sh 2>/dev/null || \
    source /orcd/home/002/lizthorn/.conda/etc/profile.d/conda.sh 2>/dev/null || \
    { module load miniforge && source "$(conda info --base)/etc/profile.d/conda.sh"; }

conda activate switchgpu

echo "=== Environment ==="
python -c "import torch; print('cuda:', torch.cuda.is_available(), '| devices:', torch.cuda.device_count())"
echo ""

if [ -d "generated_data" ] && [ "$(ls generated_data/*.npz 2>/dev/null | wc -l)" -ge 21 ]; then
    echo "=== Evaluation datasets already exist, skipping ==="
else
    echo "=== Generating evaluation datasets ==="
    python data_generation.py
fi
echo ""

POOL=series_pool.npz
if [ -f "$POOL" ]; then
    echo "=== Pool already exists at $POOL, skipping generation ==="
else
    echo "=== Generating series pool (CPU, ~15 min) ==="
    python generate_pool.py --n_series 500000 --out "$POOL"
fi
echo ""

echo "=== EXPERIMENT 1: baseline ==="
python run_compare.py \
    --pool_path "$POOL" \
    --steps 22000 \
    --ar_coeff_scale 0.6 \
    --experiment_name baseline
echo ""

echo "=== EXPERIMENT 2: dense_supervision ==="
python run_compare.py \
    --pool_path "$POOL" \
    --steps 22000 \
    --ar_coeff_scale 0.6 \
    --dense_supervision \
    --experiment_name dense_supervision
echo ""

echo "=== EXPERIMENT 3: wider_coeffs (ar_coeff_scale=1.2) ==="
python run_compare.py \
    --pool_path "$POOL" \
    --steps 22000 \
    --ar_coeff_scale 1.2 \
    --experiment_name wider_coeffs
echo ""

echo "=== All experiments complete ==="
echo "Results saved to: results_baseline.csv, results_dense_supervision.csv, results_wider_coeffs.csv"