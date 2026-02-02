"""
VLM GRPO Training Script (Qwen3-VL)
Using ms-swift GRPOTrainer with custom reward functions

Usage:
    python train_grpo.py \
        --model_path /path/to/Qwen3-VL-32B \
        --data_path train_grpo.jsonl \
        --output_dir output/grpo_qwen3_vl
"""

import os
import argparse

# Force offline mode
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['MS_OFFLINE'] = '1'

import torch
from datasets import load_dataset
from swift.llm import get_model_tokenizer
from swift.rlhf_trainers import GRPOTrainer, GRPOConfig
from swift.utils import get_logger
from peft import LoraConfig, TaskType

# Import reward functions from data package
from training.vlm.rewards.vlm_rewards import reward_eyeballing, reward_maze, reward_format

logger = get_logger()


def custom_reward_manager(completions, solution, **kwargs):
    """
    Dispatch rewards based on solution format.
    
    Auto-detects task type:
    - If solution looks like a list "[...]", it's Maze.
    - If solution is a single letter "A"-"E", it's Eyeballing.
    """
    rewards = []
    
    for completion, sol in zip(completions, solution):
        sol = sol.strip()
        
        if sol.startswith('[') and sol.endswith(']'):
            # Maze Task
            r = reward_maze([completion], [sol])[0]
        else:
            # Eyeballing Task
            r = reward_eyeballing([completion], [sol])[0]
            
        rewards.append(r)
            
    return rewards


def parse_args():
    parser = argparse.ArgumentParser(description="GRPO Training for Qwen3-VL")
    
    # Required
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to Qwen3-VL model weights")
    
    # Optional
    parser.add_argument("--data_path", type=str, default="train_grpo.jsonl",
                        help="Path to GRPO training data (JSONL)")
    parser.add_argument("--output_dir", type=str, default="output/grpo_qwen3_vl",
                        help="Output directory for checkpoints")
    parser.add_argument("--learning_rate", type=float, default=1e-6,
                        help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=1,
                        help="Number of training epochs")
    parser.add_argument("--num_generations", type=int, default=8,
                        help="Number of completions per prompt for GRPO")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Per-device batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8,
                        help="Gradient accumulation steps")
    parser.add_argument("--lora_rank", type=int, default=16,
                        help="LoRA rank")
    parser.add_argument("--use_vllm", action="store_true", default=True,
                        help="Use vLLM for faster generation")
    parser.add_argument("--vllm_gpu_util", type=float, default=0.5,
                        help="vLLM GPU memory utilization")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 60)
    print("VLM GRPO Training (Qwen3-VL)")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Model Path: {args.model_path}")
    print(f"  Data Path: {args.data_path}")
    print(f"  Output Dir: {args.output_dir}")
    print(f"  Num Generations: {args.num_generations}")
    print(f"  LoRA Rank: {args.lora_rank}")
    print()
    
    # Validate paths
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model path not found: {args.model_path}")
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Data path not found: {args.data_path}")
    
    # Configuration
    training_args = GRPOConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_generations=args.num_generations,
        max_prompt_length=1024,
        max_completion_length=1024,
        num_train_epochs=args.num_epochs,
        save_steps=100,
        logging_steps=10,
        bf16=True,
        report_to="tensorboard",
        use_vllm=args.use_vllm,
        vllm_gpu_memory_utilization=args.vllm_gpu_util,
    )

    print(f"Loading dataset from {args.data_path}...")
    dataset = load_dataset('json', data_files=args.data_path, split='train')
    
    print("Loading Model...")
    model, tokenizer = get_model_tokenizer(args.model_path)
    
    # LoRA config
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_rank * 2,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                       "gate_proj", "up_proj", "down_proj"]
    )
    
    # Trainer
    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        reward_funcs=[custom_reward_manager, reward_format],
        peft_config=lora_config,
    )
    
    print("Starting Training...")
    trainer.train()
    
    print("=" * 60)
    print(f"Training finished! Model saved to {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
