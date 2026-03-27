#!/bin/bash -l
#SBATCH -J switchtr
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

source /home/${USER}/.conda/etc/profile.d/conda.sh 2>/dev/null || \
    source /orcd/home/002/lizthorn/.conda/etc/profile.d/conda.sh 2>/dev/null || \
    { module load miniforge && source "$(conda info --base)/etc/profile.d/conda.sh"; }

conda activate switchgpu

echo "=== Environment ==="
python -c "import torch; print('cuda:', torch.cuda.is_available(), '| devices:', torch.cuda.device_count())"
echo ""

N_EXPECTED=63
N_FOUND=$(ls generated_data/*_r*.npz 2>/dev/null | wc -l)
if [ "$N_FOUND" -ge "$N_EXPECTED" ]; then
    echo "=== Evaluation datasets already exist ($N_FOUND files), skipping ==="
else
    echo "=== Generating evaluation datasets (3 instances per type) ==="
    python data_generation.py --n_instances 3
fi
echo ""

echo "=== Plotting training data samples ==="
python plot_training_samples.py --n_per_family 2 --out training_samples.png
echo ""

POOL=series_pool.npz
if [ -f "$POOL" ]; then
    echo "=== Pool already exists at $POOL, skipping ==="
else
    echo "=== Generating series pool on compute node CPUs ==="
    python generate_pool.py --n_series 500000 --ar_coeff_scale 0.6 --out "$POOL"
fi
echo ""

echo "=== EXPERIMENT: decoder-only, 3 instances per dataset ==="
python run_compare.py \
    --pool_path "$POOL" \
    --steps 100000 \
    --ar_coeff_scale 0.6 \
    --n_instances 3 \
    --experiment_name decoder_3instances

echo ""
echo "=== Complete ==="
echo "Results:       results_decoder_3instances.csv"
echo "Training plot: training_samples.png"