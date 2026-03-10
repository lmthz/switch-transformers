#!/bin/bash
set -euo pipefail

GPUS=${GPUS:-1}
CPUS=${CPUS:-8}
MEM=${MEM:-32G}
TIME=${TIME:-06:00:00}

echo "requesting interactive session"
salloc -p mit_normal_gpu --gres=gpu:${GPUS} --cpus-per-task=${CPUS} --mem=${MEM} --time=${TIME}
srun --pty bash -l