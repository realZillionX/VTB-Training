#!/bin/bash
# ============================================================
# Qwen-Image-Edit SFT Training Script
# Using DiffSynth-Studio with DeepSpeed ZeRO-2
# ============================================================
#
# Usage:
#   DIFFSYNTH_PATH=/path/to/DiffSynth-Studio bash train_sft.sh \
#       --dataset_root /path/to/VLMPuzzle/dataset \
#       --output_dir ./outputs/train
#
# Required Environment Variables:
#   DIFFSYNTH_PATH: Path to DiffSynth-Studio installation
#
# Optional Arguments:
#   --dataset_root: VLMPuzzle dataset root
#   --output_dir: Output directory for checkpoints
#   --metadata_path: Path to metadata.json (default: ./data/metadata.json)
#   --num_nodes: Total nodes (default: 1)
#   --gpus_per_node: GPUs per node (default: 8)
#   --machine_rank: Current node rank (default: 0)
#   --main_process_ip: Main node IP (required if num_nodes > 1)
#   --main_process_port: Main node port (default: 29500)
#   --accelerate_config: Use custom accelerate config (skip auto-generate)
#   --learning_rate: Learning rate (default: 1e-4)
#   --num_epochs: Number of epochs (default: 5)
#   --lora_rank: LoRA rank (default: 32)

set -e

# ============================================================
# Parse Arguments
# ============================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
METADATA_PATH="${SCRIPT_DIR}/data/metadata.json"
OUTPUT_PATH="${SCRIPT_DIR}/outputs/train/Qwen-Image-Edit-2511_lora"
DATASET_ROOT=""
DATASET_BASE_PATH=""
ACCELERATE_CONFIG=""

# Distributed parameters (defaults)
NUM_NODES=1
GPUS_PER_NODE=8
MACHINE_RANK=0
MAIN_PROCESS_IP=""
MAIN_PROCESS_PORT=29500

# Training parameters (defaults)
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
NUM_EPOCHS="${NUM_EPOCHS:-5}"
LORA_RANK="${LORA_RANK:-32}"
BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=1
DATASET_REPEAT=1
MAX_PIXELS=1048576  # 1024x1024
NUM_INFERENCE_STEPS=40

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset_root)
            DATASET_ROOT="$2"
            shift 2
            ;;
        --metadata_path)
            METADATA_PATH="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_PATH="$2"
            shift 2
            ;;
        --accelerate_config)
            ACCELERATE_CONFIG="$2"
            shift 2
            ;;
        --num_nodes)
            NUM_NODES="$2"
            shift 2
            ;;
        --gpus_per_node)
            GPUS_PER_NODE="$2"
            shift 2
            ;;
        --machine_rank)
            MACHINE_RANK="$2"
            shift 2
            ;;
        --main_process_ip)
            MAIN_PROCESS_IP="$2"
            shift 2
            ;;
        --main_process_port)
            MAIN_PROCESS_PORT="$2"
            shift 2
            ;;
        --learning_rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --num_epochs)
            NUM_EPOCHS="$2"
            shift 2
            ;;
        --lora_rank)
            LORA_RANK="$2"
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
if [ -z "${DIFFSYNTH_PATH}" ]; then
    echo "Error: DIFFSYNTH_PATH environment variable not set"
    echo "Please set it to the path of DiffSynth-Studio installation"
    echo ""
    echo "Example:"
    echo "  export DIFFSYNTH_PATH=/path/to/DiffSynth-Studio"
    echo "  bash train_sft.sh --dataset_root /path/to/VLMPuzzle/dataset"
    exit 1
fi

if [ ! -d "${DIFFSYNTH_PATH}" ]; then
    echo "Error: DiffSynth-Studio directory not found: ${DIFFSYNTH_PATH}"
    exit 1
fi

# Check if metadata exists, if not try to generate it
if [ ! -f "${METADATA_PATH}" ]; then
    if [ -z "${DATASET_ROOT}" ]; then
        echo "Error: Metadata file not found: ${METADATA_PATH}"
        echo "Please provide --dataset_root to generate it, or provide --metadata_path"
        exit 1
    fi
    
    echo "Generating metadata from dataset..."
    python -m data.tools.prepare_image_data \
        --dataset_root "${DATASET_ROOT}" \
        --output_path "${METADATA_PATH}"
fi

if [ -n "${DATASET_ROOT}" ]; then
    DATASET_BASE_PATH="${DATASET_ROOT}"
else
    echo "Warning: --dataset_root not provided. Assume metadata.json contains absolute paths."
fi

# ============================================================
# Config Display
# ============================================================
echo "============================================================"
echo "Qwen-Image-Edit SFT Training"
echo "============================================================"
echo ""
echo "Configuration:"
echo "  DiffSynth-Studio: ${DIFFSYNTH_PATH}"
echo "  Dataset Root: ${DATASET_ROOT:-'(using metadata directly)'}"
echo "  Dataset Base Path: ${DATASET_BASE_PATH:-'(empty)'}"
echo "  Metadata Path: ${METADATA_PATH}"
echo "  Output Path: ${OUTPUT_PATH}"
echo "  Num Nodes: ${NUM_NODES}"
echo "  GPUs/Node: ${GPUS_PER_NODE}"
echo "  Machine Rank: ${MACHINE_RANK}"
if [ -n "${ACCELERATE_CONFIG}" ]; then
  echo "  Accelerate Config: ${ACCELERATE_CONFIG}"
fi
echo ""
echo "Training Parameters:"
echo "  Learning Rate: ${LEARNING_RATE}"
echo "  Num Epochs: ${NUM_EPOCHS}"
echo "  LoRA Rank: ${LORA_RANK}"
echo "  Max Pixels: ${MAX_PIXELS}"
echo ""

# Create output directory
mkdir -p "${OUTPUT_PATH}"

# ============================================================
# Run Training
# ============================================================
echo "Starting training..."
echo ""

# Change to DiffSynth-Studio directory
cd "${DIFFSYNTH_PATH}"

# Use accelerate config (auto-generate if not provided)
if [ -z "${ACCELERATE_CONFIG}" ]; then
    ACCELERATE_CONFIG="${SCRIPT_DIR}/accelerate_config.yaml"
    if [ "${NUM_NODES}" -gt 1 ] && [ -z "${MAIN_PROCESS_IP}" ]; then
        echo "Error: --main_process_ip is required when --num_nodes > 1"
        exit 1
    fi

    TOTAL_PROCS=$((NUM_NODES * GPUS_PER_NODE))
    cat > "${ACCELERATE_CONFIG}" << EOF
compute_environment: LOCAL_MACHINE
distributed_type: DEEPSPEED
num_machines: ${NUM_NODES}
num_processes: ${TOTAL_PROCS}
machine_rank: ${MACHINE_RANK}
mixed_precision: bf16
deepspeed_config:
  zero_stage: 2
  offload_optimizer_device: 'cpu'
  offload_param_device: 'cpu'
  gradient_accumulation_steps: 1
EOF
    if [ "${NUM_NODES}" -gt 1 ]; then
        cat >> "${ACCELERATE_CONFIG}" << EOF
main_process_ip: ${MAIN_PROCESS_IP}
main_process_port: ${MAIN_PROCESS_PORT}
EOF
    fi
fi

accelerate launch \
    --config_file "${ACCELERATE_CONFIG}" \
    examples/qwen_image/model_training/train.py \
    --dataset_base_path "${DATASET_BASE_PATH}" \
    --dataset_metadata_path "${METADATA_PATH}" \
    --data_file_keys "image,edit_image" \
    --extra_inputs "edit_image" \
    --max_pixels ${MAX_PIXELS} \
    --dataset_repeat ${DATASET_REPEAT} \
    --model_id_with_origin_paths "Qwen/Qwen-Image-Edit-2511:transformer/diffusion_pytorch_model*.safetensors,Qwen/Qwen-Image:text_encoder/model*.safetensors,Qwen/Qwen-Image:vae/diffusion_pytorch_model.safetensors" \
    --learning_rate ${LEARNING_RATE} \
    --num_epochs ${NUM_EPOCHS} \
    --remove_prefix_in_ckpt "pipe.dit." \
    --output_path "${OUTPUT_PATH}" \
    --lora_base_model "dit" \
    --lora_target_modules "to_q,to_k,to_v,add_q_proj,add_k_proj,add_v_proj,to_out.0,to_add_out,img_mlp.net.2,img_mod.1,txt_mlp.net.2,txt_mod.1" \
    --lora_rank ${LORA_RANK} \
    --use_gradient_checkpointing \
    --dataset_num_workers 8 \
    --find_unused_parameters \
    --zero_cond_t

echo ""
echo "============================================================"
echo "Training finished!"
echo "Model saved to: ${OUTPUT_PATH}"
echo "============================================================"
