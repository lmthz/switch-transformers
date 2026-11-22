#!/bin/bash -l
#SBATCH -J multisystem_eval
#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

# Run after run_multisystem_train.sbatch has completed.
# Evaluates four methods on the val split of every dataset:
#   oracle         — exact AR likelihoods + Bayesian switch filter
#   xformer_filter — transformer likelihoods + Bayesian switch filter
#   bocpd          — classical BOCPD baseline
#   cusum          — classical CUSUM baseline
# Saves results_multisystem_eval.csv.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs

source /home/lizthorn/miniforge3/etc/profile.d/conda.sh
conda activate switchgpu

echo "=== Environment ==="
python -c "import torch; print('cuda:', torch.cuda.is_available(), '| devices:', torch.cuda.device_count())"

echo "=== Evaluating all methods ==="
python run_multisystem_eval.py \
    --all_datasets \
    --methods all \
    --data_dir generated_data \
    --ckpt_dir runs/per_regime \
    --context_len 64 \
    --val_frac 0.3 \
    --bocpd_lambda 100.0 \
    --cusum_drift 0.5 \
    --out results_multisystem_eval.csv \
    --device cuda

echo "=== Done — results in results_multisystem_eval.csv ==="
