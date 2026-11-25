#!/bin/bash -l
#SBATCH -J detection_pool
#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH -t 06:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs runs/detection_pool

source /home/lizthorn/miniforge3/etc/profile.d/conda.sh
conda activate switchgpu

echo "=== Environment ==="
python -c "import torch; print('cuda:', torch.cuda.is_available(), '| devices:', torch.cuda.device_count())"

echo "=== Training pool detector (multitask variant) ==="
python run_detection_train.py \
    --pool_path switch_pool.npz \
    --variant multitask \
    --ckpt_dir runs/detection_pool \
    --context_len 64 \
    --steps 100000 \
    --batch_size 128 \
    --lr 3e-4 \
    --save_every 2000 \
    --d_model 128 \
    --n_heads 4 \
    --n_layers 4 \
    --split_k 8 \
    --det_weight 0.1 \
    --alpha 0.15 \
    --seed 42 \
    --device cuda

echo "=== Done — checkpoint in runs/detection_pool/pool_detector_final.pt ==="
