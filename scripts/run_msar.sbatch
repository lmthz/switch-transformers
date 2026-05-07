#!/bin/bash -l
#SBATCH -J sw_msar
#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t 06:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs

source /home/lizthorn/miniforge3/etc/profile.d/conda.sh
conda activate switchgpu

if [ -f "msar_results.csv" ]; then
    echo "=== Deleting old msar_results.csv ==="
    rm msar_results.csv
fi

echo "=== Running MSAR on all 21 datasets x 30 instances ==="
echo "=== Order selection on r0 only, reused for r1-r29 ==="
echo "=== 5 random restarts per fit (10 for hard datasets) ==="
python run_msar_all.py \
    --n_instances 30 \
    --val_frac 0.3 \
    --n_restarts 5 \
    --out msar_results.csv
echo "=== Done. Run next: sbatch scripts/run_compare.sbatch ==="