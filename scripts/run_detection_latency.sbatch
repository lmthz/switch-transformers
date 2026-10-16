#!/bin/bash -l
#SBATCH -J detect_latency
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

python run_detection_latency.py \
    --data_dir generated_data \
    --train_steps 10000 \
    --n_instances 3 \
    --seed 0 \
    --pool_path series_pool.npz \
    --cusum_threshold 5.0 \
    --cusum_drift 0.5 \
    --latency_csv results_detection_latency.csv \
    --reset_csv results_context_reset.csv \
    --rls_csv results_rls.csv \
    "$@"
