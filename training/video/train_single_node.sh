#!/bin/bash
# ============================================================
# Wan2.2-TI2V LoRA Training Script (Single Node)
# Using DiffSynth-Studio with DeepSpeed ZeRO-2
# ============================================================
#
# Usage:
#   MODEL_BASE_PATH=/path/to/Wan2.2-TI2V-5B \
#   DIFFSYNTH_PATH=/path/to/DiffSynth-Studio \
#   bash train_single_node.sh --dataset /path/to/dataset.csv
#
# Required Environment Variables:
#   MODEL_BASE_PATH: Path to Wan2.2-TI2V-5B model weights
#   DIFFSYNTH_PATH: Path to DiffSynth-Studio installation
#
# Optional Arguments:
#   --dataset: Dataset CSV path (default: ./dataset.csv)
#   --output_dir: Output directory (default: ./output/wan_lora)
#   --num_frames: Number of frames (default: 81, must satisfy (n-1)%4==0)
#   --height: Video height (default: 896, must be divisible by 32)
#   --width: Video width (default: 480, must be divisible by 32)
#   --lora_rank: LoRA rank (default: 32)
#   --num_epochs: Number of epochs (default: 3)

set -e

# ============================================================
# Environment Setup
# ============================================================
export DIFFSYNTH_SKIP_DOWNLOAD=True
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export NCCL_P2P_DISABLE=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ============================================================
# Default Parameters
# ============================================================
DATASET_PATH="${SCRIPT_DIR}/dataset.csv"
OUTPUT_PATH="${SCRIPT_DIR}/output/wan_lora"

# Video configuration (CRITICAL: must follow alignment rules)
NUM_FRAMES=81      # Must satisfy (n-1) % 4 == 0
HEIGHT=896         # Must be divisible by 32
WIDTH=480          # Must be divisible by 32

# Training parameters
LEARNING_RATE=1e-4
NUM_EPOCHS=3
LORA_RANK=32
GRADIENT_ACCUMULATION=1
DATASET_REPEAT=1
SAVE_STEPS=250

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
    echo ""
    echo "Please set it to the path of Wan2.2-TI2V-5B model weights:"
    echo "  export MODEL_BASE_PATH=/path/to/Wan2.2-TI2V-5B"
    exit 1
fi

if [ -z "${DIFFSYNTH_PATH}" ]; then
    echo "Error: DIFFSYNTH_PATH environment variable not set"
    echo ""
    echo "Please set it to the path of DiffSynth-Studio installation:"
    echo "  export DIFFSYNTH_PATH=/path/to/DiffSynth-Studio"
    exit 1
fi

if [ ! -d "${MODEL_BASE_PATH}" ]; then
    echo "Error: Model path not found: ${MODEL_BASE_PATH}"
    exit 1
fi

if [ ! -d "${DIFFSYNTH_PATH}" ]; then
    echo "Error: DiffSynth-Studio not found: ${DIFFSYNTH_PATH}"
    exit 1
fi

if [ ! -f "${DATASET_PATH}" ]; then
    echo "Error: Dataset CSV not found: ${DATASET_PATH}"
    exit 1
fi

# Validate video configuration
if [ $(( (NUM_FRAMES - 1) % 4 )) -ne 0 ]; then
    echo "Error: NUM_FRAMES must satisfy (n-1) % 4 == 0"
    echo "  Current: ${NUM_FRAMES}, Valid examples: 81, 49, 25"
    exit 1
fi

if [ $((HEIGHT % 32)) -ne 0 ] || [ $((WIDTH % 32)) -ne 0 ]; then
    echo "Error: HEIGHT and WIDTH must be divisible by 32"
    echo "  Current: ${HEIGHT}x${WIDTH}"
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
echo "============================================================"
echo "Wan2.2-TI2V LoRA Training (Single Node)"
echo "============================================================"
echo ""
echo "Environment:"
echo "  Model Base: ${MODEL_BASE_PATH}"
echo "  DiffSynth: ${DIFFSYNTH_PATH}"
echo ""
echo "Dataset:"
echo "  Path: ${DATASET_PATH}"
echo "  Output: ${OUTPUT_PATH}"
echo ""
echo "Video Configuration:"
echo "  Resolution: ${WIDTH}x${HEIGHT}"
echo "  Frames: ${NUM_FRAMES}"
echo ""
echo "Training Parameters:"
echo "  Learning Rate: ${LEARNING_RATE}"
echo "  Epochs: ${NUM_EPOCHS}"
echo "  LoRA Rank: ${LORA_RANK}"
echo "  Save Steps: ${SAVE_STEPS}"
echo ""

mkdir -p "${OUTPUT_PATH}"

# ============================================================
# Run Training
# ============================================================
cd "${DIFFSYNTH_PATH}"

accelerate launch "${SCRIPT_DIR}/train.py" \
    --dataset_base_path "" \
    --dataset_metadata_path "${DATASET_PATH}" \
    --height ${HEIGHT} --width ${WIDTH} --num_frames ${NUM_FRAMES} \
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
    --save_steps ${SAVE_STEPS}

echo ""
echo "============================================================"
echo "Training finished!"
echo "Model saved to: ${OUTPUT_PATH}"
echo "============================================================"
