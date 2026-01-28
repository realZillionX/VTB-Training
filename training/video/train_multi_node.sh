#!/bin/bash
# ============================================================
# Wan2.2-TI2V LoRA Training Script (Multi-Node with Slurm)
# Using DiffSynth-Studio with DeepSpeed ZeRO-2
# ============================================================
#
# Usage:
#   MODEL_BASE_PATH=/path/to/Wan2.2-TI2V-5B \
#   DIFFSYNTH_PATH=/path/to/DiffSynth-Studio \
#   bash train_multi_node.sh --dataset /path/to/dataset.csv
#
# Required Environment Variables:
#   MODEL_BASE_PATH: Path to Wan2.2-TI2V-5B model weights
#   DIFFSYNTH_PATH: Path to DiffSynth-Studio installation

set -e

# ============================================================
# Environment Setup
# ============================================================
export DIFFSYNTH_SKIP_DOWNLOAD=True
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ============================================================
# Default Parameters
# ============================================================
DATASET_BASE_PATH=""
DATASET_PATH="${SCRIPT_DIR}/dataset.csv"
OUTPUT_PATH="${SCRIPT_DIR}/output/wan_lora_multi"

# Video configuration
NUM_FRAMES=81
HEIGHT=896
WIDTH=480

# Training parameters
LEARNING_RATE=1e-4
NUM_EPOCHS=3
LORA_RANK=32
GRADIENT_ACCUMULATION=1
DATASET_REPEAT=1
SAVE_STEPS=250

# Resume training (optional)
LORA_CHECKPOINT=""

# ============================================================
# Parse Arguments
# ============================================================
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)
            DATASET_PATH="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_PATH="$2"
            shift 2
            ;;
        --num_frames)
            NUM_FRAMES="$2"
            shift 2
            ;;
        --height)
            HEIGHT="$2"
            shift 2
            ;;
        --width)
            WIDTH="$2"
            shift 2
            ;;
        --lora_rank)
            LORA_RANK="$2"
            shift 2
            ;;
        --num_epochs)
            NUM_EPOCHS="$2"
            shift 2
            ;;
        --learning_rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --save_steps)
            SAVE_STEPS="$2"
            shift 2
            ;;
        --lora_checkpoint)
            LORA_CHECKPOINT="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# ============================================================
# Validate Environment
# ============================================================
if [ -z "${MODEL_BASE_PATH}" ]; then
    echo "Error: MODEL_BASE_PATH environment variable not set"
    exit 1
fi

if [ -z "${DIFFSYNTH_PATH}" ]; then
    echo "Error: DIFFSYNTH_PATH environment variable not set"
    exit 1
fi

# ============================================================
# Build Model Paths
# ============================================================
TOKENIZER_PATH="${MODEL_BASE_PATH}/google/umt5-xxl"

MODEL_PATHS='[
    [
        "'${MODEL_BASE_PATH}'/diffusion_pytorch_model-00001-of-00003.safetensors",
        "'${MODEL_BASE_PATH}'/diffusion_pytorch_model-00002-of-00003.safetensors",
        "'${MODEL_BASE_PATH}'/diffusion_pytorch_model-00003-of-00003.safetensors"
    ],
    "'${MODEL_BASE_PATH}'/models_t5_umt5-xxl-enc-bf16.pth",
    "'${MODEL_BASE_PATH}'/Wan2.2_VAE.pth"
]'

# ============================================================
# Config Display
# ============================================================
echo "========================================"
echo "Wan2.2-TI2V-5B LoRA 多机分布式训练 (Rank=${LORA_RANK})"
echo "========================================"
echo "  LoRA Rank: ${LORA_RANK}"
echo "  Epochs: ${NUM_EPOCHS}"
echo "  Resolution: ${WIDTH}x${HEIGHT}"
echo "  Frames: ${NUM_FRAMES}"
echo "  输出路径: ${OUTPUT_PATH}"
echo "========================================"

mkdir -p "${OUTPUT_PATH}"

# ============================================================
# Run Training (Slurm manages node coordination)
# ============================================================
CONFIG_FILE="${SCRIPT_DIR}/../../configs/accelerate_config_multi.yaml"

cd "${DIFFSYNTH_PATH}"

accelerate launch \
    --config_file "${CONFIG_FILE}" \
    "${SCRIPT_DIR}/train.py" \
    --dataset_base_path "${DATASET_BASE_PATH}" \
    --dataset_metadata_path "${DATASET_PATH}" \
    --height ${HEIGHT} \
    --width ${WIDTH} \
    --num_frames ${NUM_FRAMES} \
    --dataset_repeat ${DATASET_REPEAT} \
    --model_paths "${MODEL_PATHS}" \
    --tokenizer_path "${TOKENIZER_PATH}" \
    --learning_rate ${LEARNING_RATE} \
    --num_epochs ${NUM_EPOCHS} \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION} \
    --remove_prefix_in_ckpt "pipe.dit." \
    --output_path "${OUTPUT_PATH}" \
    --lora_base_model "dit" \
    --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
    --lora_rank ${LORA_RANK} \
    --extra_inputs "input_image" \
    --use_gradient_checkpointing \
    --save_steps ${SAVE_STEPS} \
    ${LORA_CHECKPOINT:+--lora_checkpoint "${LORA_CHECKPOINT}"}

echo "训练完成！模型保存在: ${OUTPUT_PATH}"
