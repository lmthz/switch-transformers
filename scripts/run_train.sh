#!/bin/bash
set -e

CONFIG=${1:-configs/default.yaml}

python train_transformer.py --config "$CONFIG"