#!/usr/bin/env python3
"""
推理预检脚本：测试原始 Qwen-Image-Edit-2511 模型在 VLMPuzzle 数据集上的效果。

功能：
1. 加载原始 Qwen-Image-Edit-2511 模型
2. 分别在 eyeballing 和 maze 任务的前 N 条数据上进行推理
3. 保存输入图片、生成图片和答案图片用于对比
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
    """处理 eyeballing 任务的 prompt，删除后两句话"""
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


def load_model(model_dir: str = None, low_vram: bool = False):
    """
    加载 Qwen-Image-Edit-2511 模型
    
    Args:
        model_dir: 模型目录，如果为 None 则从 ModelScope 在线加载
        low_vram: 是否使用低显存模式
    """
    print("加载 Qwen-Image-Edit-2511 模型...")
    
    if model_dir:
        # 从本地加载
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
        # 从 ModelScope 在线加载
        model_configs = [
            ModelConfig(model_id="Qwen/Qwen-Image-Edit-2511", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors"),
            ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="text_encoder/model*.safetensors"),
            ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
        ]
        processor_config = ModelConfig(model_id="Qwen/Qwen-Image-Edit", origin_file_pattern="processor/")
    
    # 低显存模式配置
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
    
    print("模型加载完成！")
    return pipe


def run_inference(
    pipe,
    puzzle_image_path: str,
    prompt: str,
    seed: int = 42,
    num_inference_steps: int = 40,
):
    """
    运行单次推理
    
    Args:
        pipe: QwenImagePipeline
        puzzle_image_path: 输入图片路径
        prompt: 处理后的提示词
        seed: 随机种子
        num_inference_steps: 推理步数
    
    Returns:
        生成的 PIL Image
    """
    # 加载输入图片
    puzzle_image = Image.open(puzzle_image_path).convert("RGB")
    
    # 获取原始图片尺寸
    width, height = puzzle_image.size
    
    # 运行推理
    generated_image = pipe(
        prompt=prompt,
        edit_image=[puzzle_image],  # Qwen-Image-Edit-2511 需要列表格式
        seed=seed,
        num_inference_steps=num_inference_steps,
        height=height,
        width=width,
        edit_image_auto_resize=True,
        zero_cond_t=True,  # Qwen-Image-Edit-2511 特有参数
    )
    
    return generated_image


def precheck_task(
    pipe,
    dataset_root: str,
    task_type: str,
    metadata_path: str,
    output_dir: str,
    num_samples: int = 5,
    seed: int = 42,
):
    """
    对单个任务类型进行预检
    
    Args:
        pipe: QwenImagePipeline
        dataset_root: 数据集根目录
        task_type: 任务类型或子目录名
        metadata_path: data.json 路径（优先使用）
        output_dir: 输出目录
        num_samples: 测试样本数
        seed: 随机种子
    """
    data_json_path = metadata_path
    if data_json_path is None:
        candidates = []
        if dataset_root and task_type:
            candidates.append(os.path.join(dataset_root, task_type, "data.json"))
            candidates.append(os.path.join(dataset_root, f"dataset_{task_type}", "data.json"))
        for candidate in candidates:
            if os.path.exists(candidate):
                data_json_path = candidate
                break
    if data_json_path is None or not os.path.exists(data_json_path):
        raise FileNotFoundError("Metadata path not found. Please provide --metadata_path.")
    
    print(f"\n{'='*60}")
    print(f"预检任务: {task_type}")
    print(f"数据文件: {data_json_path}")
    print(f"{'='*60}")
    
    # 加载数据
    with open(data_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 创建输出目录
    task_output_dir = os.path.join(output_dir, task_type)
    os.makedirs(task_output_dir, exist_ok=True)
    
    # 处理前 N 条数据
    for i, item in enumerate(data[:num_samples]):
        item_id = item.get("id", f"item_{i}")
        original_prompt = item.get("prompt", "")
        puzzle_path = item.get("image", "")
        solution_path = item.get("solution_image_path", "")
        
        # 处理 prompt
        if "correct_option" in item:
            processed_prompt = process_eyeballing_prompt(original_prompt)
        elif "solution_path_cell_ids" in item or task_type.startswith("maze"):
            processed_prompt = process_maze_prompt(original_prompt)
        else:
            processed_prompt = original_prompt
        
        # 构建完整路径
        meta_dir = os.path.dirname(os.path.abspath(data_json_path))
        puzzle_full_path = os.path.join(meta_dir, puzzle_path)
        solution_full_path = os.path.join(meta_dir, solution_path)
        
        print(f"\n[{i+1}/{num_samples}] ID: {item_id}")
        print(f"  原始 Prompt: {original_prompt[:60]}...")
        print(f"  处理后 Prompt: {processed_prompt[:60]}...")
        print(f"  输入图片: {puzzle_path}")
        print(f"  答案图片: {solution_path}")
        
        # 检查文件是否存在
        if not os.path.exists(puzzle_full_path):
            print(f"  ✗ 输入图片不存在: {puzzle_full_path}")
            continue
        
        if not os.path.exists(solution_full_path):
            print(f"  ✗ 答案图片不存在: {solution_full_path}")
            continue
        
        # 运行推理
        print(f"  运行推理中...")
        try:
            generated_image = run_inference(
                pipe,
                puzzle_full_path,
                processed_prompt,
                seed=seed,
            )
            
            # 保存结果
            sample_dir = os.path.join(task_output_dir, f"sample_{i:02d}_{item_id[:8]}")
            os.makedirs(sample_dir, exist_ok=True)
            
            # 复制输入图片
            puzzle_image = Image.open(puzzle_full_path)
            puzzle_image.save(os.path.join(sample_dir, "input_puzzle.png"))
            
            # 保存生成图片
            generated_image.save(os.path.join(sample_dir, "generated.png"))
            
            # 复制答案图片
            solution_image = Image.open(solution_full_path)
            solution_image.save(os.path.join(sample_dir, "ground_truth.png"))
            
            # 保存 prompt 信息
            with open(os.path.join(sample_dir, "info.txt"), 'w', encoding='utf-8') as f:
                f.write(f"ID: {item_id}\n")
                f.write(f"Task Type: {task_type}\n")
                f.write(f"Original Prompt: {original_prompt}\n")
                f.write(f"Processed Prompt: {processed_prompt}\n")
            
            print(f"  ✓ 结果已保存到: {sample_dir}")
            
        except Exception as e:
            print(f"  ✗ 推理失败: {e}")
            import traceback
            traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(
        description="测试原始 Qwen-Image-Edit-2511 模型在 VLMPuzzle 数据集上的效果"
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default=None,
        help="VLMPuzzle 数据集根目录（可选）"
    )
    parser.add_argument(
        "--metadata_path",
        type=str,
        default=None,
        help="data.json 路径（优先使用）"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs/precheck",
        help="输出目录"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=None,
        help="模型目录，如果不指定则从 ModelScope 在线加载"
    )
    parser.add_argument(
        "--task_types",
        type=str,
        nargs="+",
        default=["maze_square"],
        help="要测试的任务类型或子目录名"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5,
        help="每个任务类型测试的样本数"
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
        help="使用低显存模式 (FP8 量化 + CPU offload)"
    )
    args = parser.parse_args()
    
    # 加载模型
    pipe = load_model(args.model_dir, args.low_vram)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 对每个任务类型进行预检
    for task_type in args.task_types:
        precheck_task(
            pipe=pipe,
            dataset_root=args.dataset_root,
            task_type=task_type,
            metadata_path=args.metadata_path,
            output_dir=args.output_dir,
            num_samples=args.num_samples,
            seed=args.seed,
        )
    
    print("\n" + "="*60)
    print("预检完成！")
    print(f"结果已保存到: {args.output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
