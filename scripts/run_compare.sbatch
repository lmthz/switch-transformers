#!/bin/bash -l
#SBATCH -J sw_transformer
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
echo "=== Prerequisites OK ($N_FOUND dataset files, msar_results.csv found) ==="
echo ""

echo "=== Plotting training data samples ==="
python plot_training_samples.py --n_per_family 2 --out training_samples.png
echo ""

POOL=series_pool.npz
if [ -f "$POOL" ]; then
    echo "=== Pool already exists, skipping ==="
else
    echo "=== Generating series pool on compute node CPUs (~15 min) ==="
    python generate_pool.py --n_series 500000 --ar_coeff_scale 0.6 --out "$POOL"
fi
echo ""

echo "=== Training transformer and evaluating ==="
python run_compare.py \
    --pool_path "$POOL" \
    --msar_csv msar_results.csv \
    --steps 100000 \
    --ar_coeff_scale 0.6 \
    --n_instances 3 \
    --experiment_name decoder_3instances

echo ""
echo "=== Complete ==="
echo "Results: results_decoder_3instances.csv"
echo "Plot:    training_samples.png"
echo "W&B:     run 'wandb sync wandb/' to upload logs"