#!/bin/bash -l
#SBATCH -J change_aware_train
#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 06:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs runs/change_aware

source /home/lizthorn/miniforge3/etc/profile.d/conda.sh
conda activate switchgpu

echo "=== Environment ==="
python -c "import torch; print('cuda:', torch.cuda.is_available(), '| devices:', torch.cuda.device_count())"

echo "=== Training all datasets (resumes from checkpoints if present) ==="
python run_change_aware_train.py \
    --all_datasets \
    --n_instances 3 \
    --data_dir generated_data \
    --ckpt_dir runs/change_aware \
    --context_len 64 \
    --val_frac 0.3 \
    --steps 30000 \
    --batch_size 128 \
    --lr 3e-4 \
    --save_every 500 \
    --seed 0 \
    --device cuda \
    "$@"

echo "=== Evaluating trained models ==="
python run_change_aware_train.py \
    --eval_only \
    --all_datasets \
    --n_instances 3 \
    --data_dir generated_data \
    --ckpt_dir runs/change_aware \
    --context_len 64 \
    --val_frac 0.3 \
    --cusum_threshold 5.0 \
    --cusum_drift 0.5 \
    --eval_csv results_change_aware.csv \
    --device cuda

echo "=== Done ==="
