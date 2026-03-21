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

# ── Activate conda environment ──────────────────────────────────
source /home/${USER}/.conda/etc/profile.d/conda.sh 2>/dev/null || \
    source /orcd/home/002/lizthorn/.conda/etc/profile.d/conda.sh 2>/dev/null || \
    { module load miniforge && source "$(conda info --base)/etc/profile.d/conda.sh"; }

conda activate switchgpu

# Confirm GPU is visible
echo "=== Environment ==="
python -c "import torch; print('cuda:', torch.cuda.is_available(), '| devices:', torch.cuda.device_count())"
echo ""

# ── Step 1: Generate evaluation datasets (once only) ────────────
if [ -d "generated_data" ] && [ "$(ls generated_data/*.npz 2>/dev/null | wc -l)" -ge 21 ]; then
    echo "=== Step 1: Evaluation datasets already exist, skipping ==="
else
    echo "=== Step 1: Generating evaluation datasets ==="
    python data_generation.py
fi

# ── Step 2: Generate series pool (CPU only, no GPU used here) ───
POOL=series_pool.npz
if [ -f "$POOL" ]; then
    echo "=== Step 2: Pool already exists at $POOL, skipping generation ==="
else
    echo "=== Step 2: Generating series pool (CPU, ~15 min) ==="
    python data/generate_pool.py --n_series 500000 --out "$POOL"
fi
echo ""

# ── Step 3: Train transformer + run full comparison ─────────────
echo "=== Step 3: Training transformer + MSAR comparison ==="
python run_compare.py --pool_path "$POOL"