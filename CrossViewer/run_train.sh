#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${SCRIPT_DIR}"

NUM_GPUS=4
CONFIG_INPUT="${1:-configs/default.yaml}"
PORT=12355

export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES=0,1,2,3

case "${CONFIG_INPUT}" in
    /*) CONFIG_PATH="${CONFIG_INPUT}" ;;
    *) CONFIG_PATH="${PROJECT_DIR}/${CONFIG_INPUT}" ;;
esac

cd "${PROJECT_DIR}"

echo "========================================="
echo "CrossViewer Training"
echo "========================================="
echo "Config: $CONFIG_PATH"
echo "GPUs: $NUM_GPUS"
echo "========================================="

if [ $NUM_GPUS -eq 1 ]; then
    echo "Starting single-GPU training..."
    python scripts/train.py --config "$CONFIG_PATH"
else
    echo "Starting multi-GPU training with DDP..."
    torchrun \
        --nproc_per_node=$NUM_GPUS \
        --master_port=$PORT \
        scripts/train.py \
        --config "$CONFIG_PATH"
fi

echo "Training complete!"
