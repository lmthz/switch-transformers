#!/bin/bash -l
#SBATCH -J multisystem_train
#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 04:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

# Train per-regime CausalTransformerForecasters for the thesis switch-filter architecture.
# For each dataset × regime k, trains a transformer on contiguous segments where
# states == k, then estimates per-regime sigma_hat from val residuals.
# Saves to runs/per_regime/{dataset}_regime{k}_final.pt

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs

source /home/lizthorn/miniforge3/etc/profile.d/conda.sh
conda activate switchgpu

echo "=== Environment ==="
python -c "import torch; print('cuda:', torch.cuda.is_available(), '| devices:', torch.cuda.device_count())"

echo "=== Training per-regime transformers (all datasets, K=2 regimes) ==="
python run_multisystem_train.py \
    --all_datasets \
    --data_dir generated_data \
    --ckpt_dir runs/per_regime \
    --context_len 64 \
    --val_frac 0.3 \
    --steps 15000 \
    --batch_size 128 \
    --lr 3e-4 \
    --d_model 128 \
    --n_heads 4 \
    --n_layers 4 \
    --device cuda

echo "=== Done — checkpoints in runs/per_regime/ ==="
