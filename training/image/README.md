# Qwen-Image Training (Image Editing Model)

使用 DiffSynth-Studio 框架对 Qwen-Image-Edit-2511 模型进行 LoRA SFT 训练。

## 环境要求

```bash
# 安装 DiffSynth-Studio
cd /path/to/DiffSynth-Studio
pip install -e .

pip install accelerate deepspeed
```

## 数据准备

```bash
# 将 VLMPuzzle 数据集转换为 DiffSynth-Studio 格式
python -m vtb_training.data.prepare_image_data \
    --dataset_root /path/to/VLMPuzzle/dataset \
    --output_path ./data/metadata.json

# 输出: data/metadata.json
```

## 训练

```bash
# 设置 DiffSynth-Studio 路径
export DIFFSYNTH_PATH=/path/to/DiffSynth-Studio

# 启动训练
bash train_sft.sh --dataset_root /path/to/VLMPuzzle/dataset

# 可选参数:
#   --output_dir ./outputs/train
#   --learning_rate 1e-4
#   --num_epochs 5
#   --lora_rank 32
```

## 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| learning_rate | 1e-4 | 学习率 |
| num_epochs | 5 | 训练轮数 |
| lora_rank | 32 | LoRA Rank |
| max_pixels | 1048576 | 最大像素数 (1024×1024) |

## 验证

```bash
python validate_model.py \
    --lora_path ./outputs/train/Qwen-Image-Edit-2511_lora/epoch-4.safetensors \
    --output_dir ./outputs/validate \
    --num_samples 10
```

## 输出格式

训练数据格式（metadata.json）:
```json
{
    "prompt": "Draw a red path connecting two red dots...",
    "image": "dataset_maze/solutions/xxx.png",
    "edit_image": "dataset_maze/puzzles/xxx.png"
}
```
