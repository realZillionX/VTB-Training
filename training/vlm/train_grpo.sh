#!/bin/bash
# ============================================================
# Qwen3-VL GRPO Training Launcher (Shell Wrapper)
# ============================================================
# Usage:
#   bash train_grpo.sh --model_path /path/to/Qwen3-VL-32B --data_path train_grpo.jsonl

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}"
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export MS_OFFLINE=1

python "${SCRIPT_DIR}/train_grpo.py" "$@"
