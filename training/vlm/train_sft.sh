#!/bin/bash
# ============================================================
# VLM SFT Training Script (Qwen3-VL)
# Using ms-swift CLI with DeepSpeed ZeRO-3
# ============================================================
#
# Usage:
#   bash train_sft.sh --model_path /path/to/Qwen3-VL-32B
#
# Required:
#   MODEL_PATH: Path to Qwen3-VL model weights
#
# Optional Environment Variables:
#   NUM_GPUS: Number of GPUs (default: 8)
#   OUTPUT_DIR: Output directory (default: output/sft_qwen3_vl)
#   DATASET: Training dataset path (default: train_sft.jsonl)

set -e

# ============================================================
# Offline Mode
# ============================================================
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# ============================================================
# Parse Arguments
# ============================================================
MODEL_PATH=""
DATASET="${DATASET:-train_sft.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-output/sft_qwen3_vl}"
NUM_GPUS="${NUM_GPUS:-8}"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --num_gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 --model_path <path> [--dataset <path>] [--output_dir <path>] [--num_gpus <n>]"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "$MODEL_PATH" ]; then
    echo "Error: --model_path is required"
    echo "Usage: $0 --model_path <path> [--dataset <path>] [--output_dir <path>] [--num_gpus <n>]"
    exit 1
fi

if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model path does not exist: $MODEL_PATH"
    exit 1
fi

if [ ! -f "$DATASET" ]; then
    echo "Error: Dataset file not found: $DATASET"
    echo "Please run: python -m vtb_training.data.prepare_vlm_data --data_root <path>"
    exit 1
fi

# ============================================================
# Config Display
# ============================================================
echo "============================================================"
echo "VLM SFT Training (Qwen3-VL)"
echo "============================================================"
echo ""
echo "Configuration:"
echo "  Model Path: $MODEL_PATH"
echo "  Dataset: $DATASET"
echo "  Output Dir: $OUTPUT_DIR"
echo "  Num GPUs: $NUM_GPUS"
echo ""

# Generate GPU list
GPU_LIST=$(seq -s, 0 $((NUM_GPUS - 1)))

# ============================================================
# Run Training
# ============================================================
mkdir -p "$OUTPUT_DIR"

CUDA_VISIBLE_DEVICES=$GPU_LIST \
swift sft \
    --model "$MODEL_PATH" \
    --dataset "$DATASET" \
    --output_dir "$OUTPUT_DIR" \
    --deepspeed "zero3" \
    --learning_rate 1e-5 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing true \
    --save_steps 100 \
    --logging_steps 10 \
    --max_length 2048 \
    --bf16 true \
    --sft_type full \
    --report_to tensorboard

echo ""
echo "============================================================"
echo "SFT Training finished!"
echo "Output saved to: $OUTPUT_DIR"
echo "============================================================"
