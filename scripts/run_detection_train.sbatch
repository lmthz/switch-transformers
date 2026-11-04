#!/bin/bash -l
#SBATCH -J detection_train
#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 06:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs runs/detection_multitask runs/detection_detection_only

source /home/lizthorn/miniforge3/etc/profile.d/conda.sh
conda activate switchgpu

echo "=== Environment ==="
python -c "import torch; print('cuda:', torch.cuda.is_available(), '| devices:', torch.cuda.device_count())"

echo "=== Training: multitask variant ==="
python run_detection_train.py \
    --all_datasets \
    --variant multitask \
    --data_dir generated_data \
    --ckpt_dir runs/detection_multitask \
    --context_len 64 \
    --val_frac 0.3 \
    --steps 30000 \
    --batch_size 128 \
    --lr 3e-4 \
    --save_every 500 \
    --d_model 128 \
    --n_heads 4 \
    --n_layers 4 \
    --split_k 8 \
    --det_weight 0.1 \
    --alpha 0.5 \
    --seed 42 \
    --device cuda

echo "=== Training: detection_only variant ==="
python run_detection_train.py \
    --all_datasets \
    --variant detection_only \
    --data_dir generated_data \
    --ckpt_dir runs/detection_detection_only \
    --context_len 64 \
    --val_frac 0.3 \
    --steps 30000 \
    --batch_size 128 \
    --lr 3e-4 \
    --save_every 500 \
    --d_model 128 \
    --n_heads 4 \
    --n_layers 4 \
    --split_k 8 \
    --alpha 0.5 \
    --seed 42 \
    --device cuda

echo "=== Done ==="
