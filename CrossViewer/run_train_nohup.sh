#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${SCRIPT_DIR}"
CONFIG_INPUT="${1:-configs/default.yaml}"
MASTER_PORT="${MASTER_PORT:-12365}"
CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"
NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-DETAIL}"
NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
TORCH_NCCL_TRACE_BUFFER_SIZE="${TORCH_NCCL_TRACE_BUFFER_SIZE:-1048576}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

case "${CONFIG_INPUT}" in
  /*) CONFIG_PATH="${CONFIG_INPUT}" ;;
  *) CONFIG_PATH="${PROJECT_DIR}/${CONFIG_INPUT}" ;;
esac

export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

LOG_DIR="${LOG_DIR:-$("${PYTHON_BIN}" -c '
import sys
from pathlib import Path
import yaml

config_path = Path(sys.argv[1]).expanduser().resolve()
cfg = yaml.safe_load(config_path.read_text())
raw_log_dir = cfg.get("training", {}).get("log_dir", "../logs")
log_dir = Path(raw_log_dir).expanduser()
if not log_dir.is_absolute():
    log_dir = (config_path.parent / log_dir).resolve()
print(log_dir)
' "${CONFIG_PATH}")}"

mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/train_nohup_$(date +%Y%m%d_%H%M%S).log"

echo "Config: ${CONFIG_PATH}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_DEVICES}"
echo "MASTER_PORT=${MASTER_PORT}"
if [ -z "${NPROC_PER_NODE}" ]; then
  NPROC_PER_NODE=$(echo "${CUDA_DEVICES}" | awk -F',' '{print NF}')
fi

echo "Log: ${LOG_FILE}"
echo "NPROC_PER_NODE=${NPROC_PER_NODE}"
echo "NCCL_DEBUG=${NCCL_DEBUG}"
echo "TORCH_DISTRIBUTED_DEBUG=${TORCH_DISTRIBUTED_DEBUG}"
echo "NCCL_ASYNC_ERROR_HANDLING=${NCCL_ASYNC_ERROR_HANDLING}"
echo "TORCH_NCCL_TRACE_BUFFER_SIZE=${TORCH_NCCL_TRACE_BUFFER_SIZE}"
echo "PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}"

nohup env CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" \
  PYTHONPATH="${PYTHONPATH}" \
  NCCL_DEBUG="${NCCL_DEBUG}" \
  TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG}" \
  NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING}" \
  TORCH_NCCL_TRACE_BUFFER_SIZE="${TORCH_NCCL_TRACE_BUFFER_SIZE}" \
  PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF}" \
  "${TORCHRUN_BIN}" --nproc_per_node="${NPROC_PER_NODE}" --master_port="${MASTER_PORT}" \
  "${PROJECT_DIR}/scripts/train.py" --config "${CONFIG_PATH}" \
  > "${LOG_FILE}" 2>&1 &

echo "Started. Tail logs with:"
echo "  tail -f ${LOG_FILE}"
