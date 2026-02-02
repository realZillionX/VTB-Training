import os
# Force offline mode
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import json
import argparse
import subprocess
from swift.llm import get_model_tokenizer, get_template, inference
from swift.utils import seed_everything
import torch

def main():
    parser = argparse.ArgumentParser(description="Pre-check inference using ms-swift")
    parser.add_argument("--model_path", type=str, required=True, help="Path to Qwen3-VL-32B-Thinking model")
    parser.add_argument("--data_path", type=str, default="train_sft.jsonl", help="Path to prepared jsonl data")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples to test")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")
    
    args = parser.parse_args()
    
    # Check if data exists
    if not os.path.exists(args.data_path):
        print(f"Data file {args.data_path} not found. Please run prepare_data.py first.")
        return

    # Load first N samples
    samples = []
    with open(args.data_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= args.num_samples:
                break
            samples.append(json.loads(line))
            
    print(f"Loaded {len(samples)} samples.")
    
    # Load Model and Template
    print(f"Loading model from {args.model_path}...")
    # NOTE: Adjust 'model_type' if needed, or let swift auto-detect
    try:
        model, tokenizer = get_model_tokenizer(args.model_path, model_kwargs={'device_map': args.device})
        template = get_template(model.model_meta.template, tokenizer)
    except Exception as e:
        print(f"Error loading model with swift API: {e}")
        print("Please ensure ms-swift is installed and model path is correct.")
        return

    seed_everything(42)
    
    for i, sample in enumerate(samples):
        print(f"\n=== Sample {i+1} ===")
        print(f"Task: {sample.get('task_type', 'unknown')}")

        # 兼容旧格式：query/response
        if 'messages' in sample:
            messages = sample['messages']
            query = messages[0]['content'] if messages else ''
            images = sample.get('images', [])
            print(f"Prompt: {query}")
        else:
            query = sample.get('query', '')
            images = sample.get('images', [])
            print(f"Prompt: {query}")

        response, _ = inference(model, template, query, images=images)

        print(f"Model Output: {response}")
        if 'solution' in sample:
            print(f"Ground Truth (Solution): {sample['solution']}")
        elif 'response' in sample:
            print(f"Ground Truth (Response): {sample['response']}")

if __name__ == "__main__":
    main()
