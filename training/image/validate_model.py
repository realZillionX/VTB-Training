#!/usr/bin/env python3
"""
验证训练后的模型效果。

加载训练好的 LoRA 权重，在验证集上生成图片进行对比。
"""

import os
import json
import argparse
from PIL import Image
import torch
import sys

# 添加 DiffSynth-Studio 到路径
DIFFSYNTH_PATH = os.environ.get(
    "DIFFSYNTH_PATH",
    "/inspire/hdd/project/embodied-multimodality/tongjingqi-CZXS25110029/chj_code/DiffSynth-Studio"
)
if DIFFSYNTH_PATH not in sys.path:
    sys.path.insert(0, DIFFSYNTH_PATH)

from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig


def process_eyeballing_prompt(prompt: str) -> str:
    """处理 eyeballing 任务的 prompt"""
    import re
    patterns_to_remove = [
        r'\s*Speak out[^.]*\.[^.]*\.',
        r'\s*In portrait[^.]*\.',
        r'\s*Static camera\.',
    ]
    result = prompt
    for pattern in patterns_to_remove:
        result = re.sub(pattern, '', result, flags=re.IGNORECASE)
    result = ' '.join(result.split())
    if result and not result.endswith('.'):
        result += '.'
    return result.strip()


def process_maze_prompt(prompt: str) -> str:
    """处理 maze 任务的 prompt"""
    return "Draw a red path connecting two red dots without touching the black walls."


def load_model_with_lora(lora_path: str, model_dir: str = None, low_vram: bool = False):
    """
    加载训练后的模型（带 LoRA 权重）
    """
    print(f"加载模型...")
    print(f"LoRA 权重: {lora_path}")
    
    if model_dir:
        transformer_path = os.path.join(model_dir, "Qwen/Qwen-Image-Edit-2511")
        base_model_path = os.path.join(model_dir, "Qwen/Qwen-Image")
        processor_path = os.path.join(model_dir, "Qwen/Qwen-Image-Edit")
        
        model_configs = [
            ModelConfig(model_path=transformer_path, origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors"),
            ModelConfig(model_path=base_model_path, origin_file_pattern="text_encoder/model*.safetensors"),
            ModelConfig(model_path=base_model_path, origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
        ]
        processor_config = ModelConfig(model_path=processor_path, origin_file_pattern="processor/")
    else:
        model_configs = [
            ModelConfig(model_id="Qwen/Qwen-Image-Edit-2511", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors"),
            ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="text_encoder/model*.safetensors"),
            ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
        ]
        processor_config = ModelConfig(model_id="Qwen/Qwen-Image-Edit", origin_file_pattern="processor/")
    
    if low_vram:
        vram_config = {
            "offload_dtype": "disk",
            "offload_device": "disk",
            "onload_dtype": torch.float8_e4m3fn,
            "onload_device": "cpu",
            "preparing_dtype": torch.float8_e4m3fn,
            "preparing_device": "cuda",
            "computation_dtype": torch.bfloat16,
            "computation_device": "cuda",
        }
        for config in model_configs:
            config.offload_dtype = vram_config["offload_dtype"]
            config.offload_device = vram_config["offload_device"]
            config.onload_dtype = vram_config["onload_dtype"]
            config.onload_device = vram_config["onload_device"]
            config.preparing_dtype = vram_config["preparing_dtype"]
            config.preparing_device = vram_config["preparing_device"]
            config.computation_dtype = vram_config["computation_dtype"]
            config.computation_device = vram_config["computation_device"]
    
    pipe = QwenImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=model_configs,
        processor_config=processor_config,
    )
    
    # 加载 LoRA 权重
    print(f"加载 LoRA 权重: {lora_path}")
    pipe.load_lora(pipe.dit, lora_path)
    
    print("模型加载完成！")
    return pipe


def validate_task(
    pipe,
    dataset_root: str,
    task_type: str,
    output_dir: str,
    start_idx: int = 0,
    num_samples: int = 10,
    seed: int = 42,
):
    """
    验证单个任务类型
    """
    task_config = {
        "eyeballing": "dataset_eyeballing",
        "maze": "dataset_maze",
    }
    
    dataset_subdir = task_config[task_type]
    data_json_path = os.path.join(dataset_root, dataset_subdir, "data.json")
    
    print(f"\n{'='*60}")
    print(f"验证任务: {task_type}")
    print(f"数据文件: {data_json_path}")
    print(f"样本范围: [{start_idx}, {start_idx + num_samples})")
    print(f"{'='*60}")
    
    with open(data_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    task_output_dir = os.path.join(output_dir, task_type)
    os.makedirs(task_output_dir, exist_ok=True)
    
    for i, item in enumerate(data[start_idx:start_idx + num_samples]):
        idx = start_idx + i
        item_id = item.get("id", f"item_{idx}")
        original_prompt = item.get("prompt", "")
        puzzle_path = item.get("image", "")
        solution_path = item.get("solution_image_path", "")
        
        if task_type == "eyeballing":
            processed_prompt = process_eyeballing_prompt(original_prompt)
        else:
            processed_prompt = process_maze_prompt(original_prompt)
        
        puzzle_full_path = os.path.join(dataset_root, dataset_subdir, puzzle_path)
        solution_full_path = os.path.join(dataset_root, dataset_subdir, solution_path)
        
        print(f"\n[{i+1}/{num_samples}] ID: {item_id}")
        print(f"  Prompt: {processed_prompt[:60]}...")
        
        if not os.path.exists(puzzle_full_path):
            print(f"  ✗ 输入图片不存在")
            continue
        
        try:
            puzzle_image = Image.open(puzzle_full_path).convert("RGB")
            width, height = puzzle_image.size
            
            generated_image = pipe(
                prompt=processed_prompt,
                edit_image=[puzzle_image],
                seed=seed,
                num_inference_steps=40,
                height=height,
                width=width,
                edit_image_auto_resize=True,
                zero_cond_t=True,
            )
            
            sample_dir = os.path.join(task_output_dir, f"sample_{idx:04d}_{item_id[:8]}")
            os.makedirs(sample_dir, exist_ok=True)
            
            puzzle_image.save(os.path.join(sample_dir, "input_puzzle.png"))
            generated_image.save(os.path.join(sample_dir, "generated.png"))
            
            if os.path.exists(solution_full_path):
                solution_image = Image.open(solution_full_path)
                solution_image.save(os.path.join(sample_dir, "ground_truth.png"))
            
            with open(os.path.join(sample_dir, "info.txt"), 'w', encoding='utf-8') as f:
                f.write(f"ID: {item_id}\n")
                f.write(f"Task Type: {task_type}\n")
                f.write(f"Prompt: {processed_prompt}\n")
            
            print(f"  ✓ 已保存到: {sample_dir}")
            
        except Exception as e:
            print(f"  ✗ 推理失败: {e}")


def main():
    parser = argparse.ArgumentParser(description="验证训练后的模型效果")
    parser.add_argument(
        "--lora_path",
        type=str,
        required=True,
        help="LoRA 权重文件路径"
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="/inspire/hdd/project/embodied-multimodality/public/VLMPuzzle/dataset",
        help="VLMPuzzle 数据集根目录"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs/validate",
        help="输出目录"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=None,
        help="模型目录"
    )
    parser.add_argument(
        "--task_types",
        type=str,
        nargs="+",
        default=["eyeballing", "maze"],
        choices=["eyeballing", "maze"],
        help="要验证的任务类型"
    )
    parser.add_argument(
        "--start_idx",
        type=int,
        default=0,
        help="起始样本索引"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="每个任务类型验证的样本数"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子"
    )
    parser.add_argument(
        "--low_vram",
        action="store_true",
        help="使用低显存模式"
    )
    args = parser.parse_args()
    
    pipe = load_model_with_lora(
        lora_path=args.lora_path,
        model_dir=args.model_dir,
        low_vram=args.low_vram
    )
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    for task_type in args.task_types:
        validate_task(
            pipe=pipe,
            dataset_root=args.dataset_root,
            task_type=task_type,
            output_dir=args.output_dir,
            start_idx=args.start_idx,
            num_samples=args.num_samples,
            seed=args.seed,
        )
    
    print("\n" + "="*60)
    print("验证完成！")
    print(f"结果已保存到: {args.output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
