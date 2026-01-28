#!/bin/bash
# ============================================================
# Wan2.2-TI2V LoRA Training Script (Multi-Node)
# Using DiffSynth-Studio with DeepSpeed ZeRO-2
# ============================================================
#
# Usage:
#   # On each node:
#   MODEL_BASE_PATH=/path/to/Wan2.2-TI2V-5B \
#   DIFFSYNTH_PATH=/path/to/DiffSynth-Studio \
#   MASTER_ADDR=<master_ip> \
#   MASTER_PORT=29500 \
#   NUM_NODES=5 \
#   NODE_RANK=<0-4> \
#   bash train_multi_node.sh --dataset /path/to/dataset.csv
#
# Required Environment Variables:
#   MODEL_BASE_PATH: Path to Wan2.2-TI2V-5B model weights
#   DIFFSYNTH_PATH: Path to DiffSynth-Studio installation
#   MASTER_ADDR: IP address of the master node
#   MASTER_PORT: Port for distributed communication (default: 29500)
#   NUM_NODES: Total number of nodes
#   NODE_RANK: Rank of current node (0-indexed)
#
# Optional Arguments:
#   Same as train_single_node.sh

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
# Multi-Node Configuration
# ============================================================
MASTER_ADDR="${MASTER_ADDR:?Error: MASTER_ADDR not set}"
MASTER_PORT="${MASTER_PORT:-29500}"
NUM_NODES="${NUM_NODES:?Error: NUM_NODES not set}"
NODE_RANK="${NODE_RANK:?Error: NODE_RANK not set}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"

TOTAL_GPUS=$((NUM_NODES * GPUS_PER_NODE))

# ============================================================
# Default Parameters
# ============================================================
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
# Config Display (only on rank 0)
# ============================================================
if [ "${NODE_RANK}" == "0" ]; then
    echo "============================================================"
    echo "Wan2.2-TI2V LoRA Training (Multi-Node)"
    echo "============================================================"
    echo ""
    echo "Cluster Configuration:"
    echo "  Master: ${MASTER_ADDR}:${MASTER_PORT}"
    echo "  Nodes: ${NUM_NODES}"
    echo "  GPUs per Node: ${GPUS_PER_NODE}"
    echo "  Total GPUs: ${TOTAL_GPUS}"
    echo ""
    echo "Video Configuration:"
    echo "  Resolution: ${WIDTH}x${HEIGHT}"
    echo "  Frames: ${NUM_FRAMES}"
    echo ""
    echo "Training Parameters:"
    echo "  LoRA Rank: ${LORA_RANK}"
    echo "  Epochs: ${NUM_EPOCHS}"
    echo "  Learning Rate: ${LEARNING_RATE}"
    echo ""
fi

mkdir -p "${OUTPUT_PATH}"

# ============================================================
# Create Dynamic Accelerate Config
# ============================================================
ACCELERATE_CONFIG="/tmp/accelerate_config_${NODE_RANK}.yaml"
cat > "${ACCELERATE_CONFIG}" << EOF
compute_environment: LOCAL_MACHINE
distributed_type: DEEPSPEED
num_machines: ${NUM_NODES}
num_processes: ${TOTAL_GPUS}
machine_rank: ${NODE_RANK}
main_process_ip: ${MASTER_ADDR}
main_process_port: ${MASTER_PORT}
mixed_precision: bf16
deepspeed_config:
  zero_stage: 2
  offload_optimizer_device: 'none'
  offload_param_device: 'none'
  gradient_accumulation_steps: ${GRADIENT_ACCUMULATION}
EOF

# ============================================================
# Run Training
# ============================================================
cd "${DIFFSYNTH_PATH}"

accelerate launch \
    --config_file "${ACCELERATE_CONFIG}" \
    "${SCRIPT_DIR}/train.py" \
    --dataset_base_path "" \
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
    --save_steps ${SAVE_STEPS}

if [ "${NODE_RANK}" == "0" ]; then
    echo ""
    echo "============================================================"
    echo "Training finished!"
    echo "Model saved to: ${OUTPUT_PATH}"
    echo "============================================================"
fi
