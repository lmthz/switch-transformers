#!/bin/bash -l
#SBATCH -J sw_probe
#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 02:00:00
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

python run_probe_analysis.py --pool_path series_pool.npz

echo ""
echo "=== Complete ==="
echo "Results:"
echo "  results_probe.csv       (error-proxy vs hidden-state probe accuracy)"
echo "  results_per_switch.csv  (per-switch metrics by switches_in_context)"
