#!/bin/bash
# ============================================================
# Wan2.2-TI2V LoRA Training Script (Unified)
# Supports single-node and multi-node (via accelerate config)
# ============================================================
#
# Usage:
#   MODEL_BASE_PATH=/path/to/Wan2.2-TI2V-5B \
#   DIFFSYNTH_PATH=/path/to/DiffSynth-Studio \
#   bash train_sft.sh --dataset /path/to/dataset.csv --num_nodes 1
#
# Optional Arguments:
#   --dataset: Dataset CSV path
#   --dataset_root: Dataset root for relative paths
#   --output_dir: Output directory
#   --num_nodes: Number of nodes (default: 1)
#   --gpus_per_node: GPUs per node (default: 8)
#   --machine_rank: Machine rank (default: 0)
#   --num_frames / --height / --width / --lora_rank / --num_epochs / --learning_rate / --save_steps

set -e

export DIFFSYNTH_SKIP_DOWNLOAD=True
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATASET_PATH="${SCRIPT_DIR}/dataset.csv"
DATASET_BASE_PATH=""
OUTPUT_PATH="${SCRIPT_DIR}/output/wan_lora"

NUM_NODES=1
GPUS_PER_NODE=8
MACHINE_RANK=0

NUM_FRAMES=81
HEIGHT=896
WIDTH=480

LEARNING_RATE=1e-4
NUM_EPOCHS=3
LORA_RANK=32
GRADIENT_ACCUMULATION=1
DATASET_REPEAT=1
SAVE_STEPS=250
LORA_CHECKPOINT=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)
            DATASET_PATH="$2"; shift 2;;
        --dataset_root)
            DATASET_BASE_PATH="$2"; shift 2;;
        --output_dir)
            OUTPUT_PATH="$2"; shift 2;;
        --num_nodes)
            NUM_NODES="$2"; shift 2;;
        --gpus_per_node)
            GPUS_PER_NODE="$2"; shift 2;;
        --machine_rank)
            MACHINE_RANK="$2"; shift 2;;
        --num_frames)
            NUM_FRAMES="$2"; shift 2;;
        --height)
            HEIGHT="$2"; shift 2;;
        --width)
            WIDTH="$2"; shift 2;;
        --lora_rank)
            LORA_RANK="$2"; shift 2;;
        --num_epochs)
            NUM_EPOCHS="$2"; shift 2;;
        --learning_rate)
            LEARNING_RATE="$2"; shift 2;;
        --save_steps)
            SAVE_STEPS="$2"; shift 2;;
        --lora_checkpoint)
            LORA_CHECKPOINT="$2"; shift 2;;
        *)
            echo "Unknown option: $1"; exit 1;;
    esac
done

if [ -z "${MODEL_BASE_PATH}" ]; then
    echo "Error: MODEL_BASE_PATH not set"; exit 1
fi
if [ -z "${DIFFSYNTH_PATH}" ]; then
    echo "Error: DIFFSYNTH_PATH not set"; exit 1
fi
if [ ! -f "${DATASET_PATH}" ]; then
    echo "Error: Dataset CSV not found: ${DATASET_PATH}"; exit 1
fi

if [ $(( (NUM_FRAMES - 1) % 4 )) -ne 0 ]; then
    echo "Error: NUM_FRAMES must satisfy (n-1) % 4 == 0"; exit 1
fi
if [ $((HEIGHT % 32)) -ne 0 ] || [ $((WIDTH % 32)) -ne 0 ]; then
    echo "Error: HEIGHT and WIDTH must be divisible by 32"; exit 1
fi

if [[ "${NUM_NODES}" -le 1 ]]; then
    export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}
fi

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

mkdir -p "${OUTPUT_PATH}"

CONFIG_ARGS=()
TMP_CONFIG=""
if [[ "${NUM_NODES}" -gt 1 ]]; then
    NUM_PROCESSES=$((NUM_NODES * GPUS_PER_NODE))
    TMP_CONFIG="$(mktemp /tmp/accelerate_config_XXXX.yaml)"
    cat > "${TMP_CONFIG}" << CONFIG
compute_environment: LOCAL_MACHINE
debug: false
deepspeed_config:
  gradient_accumulation_steps: ${GRADIENT_ACCUMULATION}
  offload_optimizer_device: 'none'
  offload_param_device: 'none'
  zero3_init_flag: false
  zero_stage: 2
distributed_type: DEEPSPEED
downcast_bf16: 'no'
enable_cpu_affinity: false
machine_rank: ${MACHINE_RANK}
main_training_function: main
mixed_precision: bf16
num_machines: ${NUM_NODES}
num_processes: ${NUM_PROCESSES}
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
CONFIG
    CONFIG_ARGS=(--config_file "${TMP_CONFIG}")
    trap 'rm -f "${TMP_CONFIG}"' EXIT
fi

LORA_ARGS=()
if [[ -n "${LORA_CHECKPOINT}" ]]; then
    LORA_ARGS=(--lora_checkpoint "${LORA_CHECKPOINT}")
fi

cd "${DIFFSYNTH_PATH}"

accelerate launch \
    ${CONFIG_ARGS[@]} \
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
    ${LORA_ARGS[@]}
