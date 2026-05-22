#!/bin/bash -l
#SBATCH -J sw_noswitch
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

source /home/lizthorn/miniforge3/etc/profile.d/conda.sh
conda activate switchgpu

echo "=== Environment ==="
python -c "import torch; print('cuda:', torch.cuda.is_available(), '| devices:', torch.cuda.device_count())"
echo ""

# Prerequisites check
N_NS=$(ls generated_data_noswitch/*_r*.npz 2>/dev/null | wc -l)
if [ "$N_NS" -lt 510 ]; then
    echo "[error] Only $N_NS no-switch dataset files found (need 510)."
    echo "Run first: sbatch scripts/generate_noswitch.sbatch"
    exit 1
fi
if [ ! -f "pool_noswitch.npz" ]; then
    echo "[error] pool_noswitch.npz not found."
    echo "Run first: sbatch scripts/generate_noswitch.sbatch"
    exit 1
fi

echo "=== Prerequisites OK ($N_NS no-switch dataset files) ==="
echo ""

echo "=== Running Experiment F (no-switch training steps sweep) ==="
python run_density_experiment.py \
    --experiments F \
    --data_dir_noswitch generated_data_noswitch \
    --pool_noswitch pool_noswitch.npz \
    --n_instances 3 \
    --exp_b_steps 10000

echo ""
echo "=== Complete ==="
echo "Results:"
echo "  results_density_exp_f.csv  (no-switch training steps sweep)"
echo "W&B: ls wandb/ then: wandb sync wandb/offline-run-<id>"
