# Wan2.2 Training (Video Generation Model)

使用 DiffSynth-Studio 框架对 Wan2.2-TI2V-5B 模型进行 LoRA SFT 训练。

## 环境要求

```bash
# 安装 DiffSynth-Studio
cd /path/to/DiffSynth-Studio
pip install -e .

pip install accelerate deepspeed pandas
```

## 模型权重

需要预先下载以下文件到 `MODEL_BASE_PATH`:
- `diffusion_pytorch_model-*.safetensors` (DiT 模型，3个文件)
- `models_t5_umt5-xxl-enc-bf16.pth` (T5 编码器)
- `Wan2.2_VAE.pth` (VAE)
- `google/umt5-xxl/` (Tokenizer)

## 数据准备

训练脚本需要 CSV 文件，包含两列：
- `video`: 视频文件绝对路径
- `prompt`: 文本描述

```csv
video,prompt
"/path/to/video1.mp4","Draw a red path..."
"/path/to/video2.mp4","Connect the dots..."
```

**重要**: 使用 `QUOTE_ALL` 格式化 CSV 以避免解析错误。

## 训练

### 单节点训练 (8 GPU)

```bash
export MODEL_BASE_PATH=/path/to/Wan2.2-TI2V-5B
export DIFFSYNTH_PATH=/path/to/DiffSynth-Studio

bash train_single_node.sh --dataset ./dataset.csv

# 可选参数:
#   --output_dir ./output/wan_lora
#   --num_frames 81
#   --height 896
#   --width 480
#   --lora_rank 32
#   --num_epochs 3
```

### 多节点训练

在每个节点上运行：
```bash
export MODEL_BASE_PATH=/path/to/Wan2.2-TI2V-5B
export DIFFSYNTH_PATH=/path/to/DiffSynth-Studio
export MASTER_ADDR=192.168.1.100
export MASTER_PORT=29500
export NUM_NODES=5
export NODE_RANK=0  # 每个节点不同: 0, 1, 2, 3, 4

bash train_multi_node.sh --dataset /shared/dataset.csv
```

## 视频配置约束 (CRITICAL)

| 参数 | 约束 | 示例 |
|------|------|------|
| num_frames | `(n-1) % 4 == 0` | 81, 49, 25 |
| height/width | 被 32 整除 | 896, 480 |

**违反约束会导致静默失败或崩溃！**

## 自动续训

脚本支持自动从中断处恢复：
- 自动检测 `output_path` 下最新的 checkpoint
- 自动加载权重并恢复进度
- 直接重新运行脚本即可

## 推理

```bash
python inference.py \
    --lora_path ./output/wan_lora/epoch-2.safetensors \
    --input_image puzzle.png \
    --prompt "Draw a red path connecting two red dots"
```

## 评估

```bash
python parallel_eval.py \
    --lora_path ./output/wan_lora/epoch-2.safetensors \
    --dataset_root /path/to/VLMPuzzle/dataset
```
