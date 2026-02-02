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
\"/path/to/video1.mp4\",\"Draw a red path...\"
\"/path/to/video2.mp4\",\"Connect the dots...\"
```

**重要**: 使用 `QUOTE_ALL` 格式化 CSV 以避免解析错误。

## 训练

### 统一训练脚本（单机 / 多机）

```bash
export MODEL_BASE_PATH=/path/to/Wan2.2-TI2V-5B
export DIFFSYNTH_PATH=/path/to/DiffSynth-Studio

# 先转换 VLMPuzzle 数据为 CSV
python -m data.tools.prepare_video_data \
    --dataset_root /path/to/VLMPuzzle/dataset \
    --output_path ./dataset/train_video.csv

# 单机
bash train_sft.sh --dataset ./dataset/train_video.csv --dataset_root /path/to/VLMPuzzle/dataset --num_nodes 1

# 可选参数:
#   --output_dir ./output/wan_lora
#   --num_frames 81
#   --height 896
#   --width 480
#   --lora_rank 32
#   --num_epochs 3
```

```bash
# 多机（示例：15 节点 × 8 GPU）
bash train_sft.sh --dataset ./dataset/train_video.csv --dataset_root /path/to/VLMPuzzle/dataset --num_nodes 15 --gpus_per_node 8 --machine_rank 0
```

**可用参数**:

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--num_nodes` | 1 | 节点数 |
| `--gpus_per_node` | 8 | 每节点 GPU 数 |
| `--machine_rank` | 0 | 机器 rank（多机必填） |
| `--dataset` | `./dataset.csv` | 数据集 CSV 路径 |
| `--dataset_root` | - | 数据集根目录（相对路径时需要） |
| `--output_dir` | `./output/wan_lora` | 输出目录 |
| `--lora_rank` | 32 | LoRA Rank |
| `--num_epochs` | 3 | 训练轮数 |
| `--learning_rate` | 1e-4 | 学习率 |
| `--num_frames` | 81 | 视频帧数 |
| `--height` | 896 | 视频高度 |
| `--width` | 480 | 视频宽度 |
| `--save_steps` | 250 | 保存间隔 |
| `--lora_checkpoint` | - | 续训 checkpoint 路径 |

> **Note**: 多机模式会自动生成临时 accelerate 配置文件。

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

## 推理（建议使用预检脚本）

单样本推理建议直接复用 `tests/video/infer_precheck.py`，将 CSV 控制为 1 条样本即可。

## 预检与验证

```bash
python tests/video/infer_precheck.py \
    --dataset ./dataset/train_video.csv \
    --model_base_path /path/to/Wan2.2-TI2V-5B \
    --lora_ckpt ./output/wan_lora/epoch-2.safetensors \
    --output_dir ./outputs/precheck_video

python tests/video/validate_model.py \
    --dataset ./dataset/train_video.csv \
    --metadata_path /path/to/VLMPuzzle/dataset/maze_square/data.json \
    --model_base_path /path/to/Wan2.2-TI2V-5B \
    --lora_ckpt ./output/wan_lora/epoch-2.safetensors \
    --output_dir ./outputs/validate_video
```

## 评估

```bash
python tests/video/parallel_eval.py \
    --input_dir /path/to/VLMPuzzle/dataset/maze_square/puzzles \
    --lora_ckpt ./output/wan_lora/epoch-2.safetensors \
    --metadata_path /path/to/VLMPuzzle/dataset/maze_square/data.json
```
