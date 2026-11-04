#!/bin/bash -l
#SBATCH -J detection_eval
#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

# Run after run_detection_train.sbatch has completed.
# Evaluates all 5 detection methods on the val split of every dataset.
# Saves results_detection_eval.csv.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs

source /home/lizthorn/miniforge3/etc/profile.d/conda.sh
conda activate switchgpu

echo "=== Environment ==="
python -c "import torch; print('cuda:', torch.cuda.is_available(), '| devices:', torch.cuda.device_count())"

echo "=== Evaluating all methods ==="
python run_detection_eval.py \
    --all_datasets \
    --methods all \
    --data_dir generated_data \
    --det_only_ckpt_dir runs/detection_detection_only \
    --multitask_ckpt_dir runs/detection_multitask \
    --base_ckpt_dir runs/base_transformer \
    --context_len 64 \
    --val_frac 0.3 \
    --cusum_drift 0.5 \
    --bocpd_lambda 100.0 \
    --out results_detection_eval.csv \
    --device cuda

echo "=== Done — results in results_detection_eval.csv ==="
