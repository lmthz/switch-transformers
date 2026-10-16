#!/bin/bash
#SBATCH --job-name=detect_latency
#SBATCH --output=slurm_detect_latency_%j.out
#SBATCH --error=slurm_detect_latency_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --partition=mit_normal_gpu

cd ~/switch-transformers
source activate transformer_env 2>/dev/null || conda activate transformer_env 2>/dev/null || true

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
