#!/bin/bash
set -e

# you may need to edit partition or qos on engaging
GPUS=${GPUS:-1}
CPUS=${CPUS:-4}
MEM=${MEM:-16G}
TIME=${TIME:-02:00:00}

echo "requesting interactive session"
salloc --gres=gpu:$GPUS --cpus-per-task=$CPUS --mem=$MEM --time=$TIME
srun --pty bash -l