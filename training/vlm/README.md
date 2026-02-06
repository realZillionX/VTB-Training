# Qwen3-VL Training (VLM)

使用 ms-swift 框架对 Qwen3-VL 模型进行 SFT 和 GRPO 训练。

## 环境要求

```bash
pip install ms-swift peft vllm datasets
```

## 数据准备

```bash
# 将 VLMPuzzle 数据集转换为 ms-swift 格式
python -m data.tools.prepare_vlm_data \
    --data_root /path/to/VLMPuzzle/dataset \
    --output_dir ./data

# 输出:
#   data/train_sft.jsonl   - SFT 训练数据（默认使用绝对图片路径）
#   data/train_grpo.jsonl  - GRPO 训练数据
```

## Phase 1: SFT 监督微调

```bash
bash train_sft.sh --model_path /path/to/Qwen3-VL-32B-Thinking

# 可选参数:
#   --dataset data/train_sft.jsonl
#   --output_dir output/sft_qwen3_vl
#   --num_gpus 8
```

## Phase 2: GRPO 强化学习

在 SFT 完成后，加载 checkpoint 继续 GRPO 训练：

```bash
bash train_grpo.sh \
    --model_path output/sft_qwen3_vl/checkpoint-100 \
    --data_path data/train_grpo.jsonl \
    --output_dir output/grpo_qwen3_vl

# 可选参数:
#   --learning_rate 1e-6
#   --num_generations 8
#   --lora_rank 16
```

## 奖励函数

GRPO 使用自定义奖励函数（`training/vlm/rewards/vlm_rewards.py`）：

| 任务类型 | 评分规则 |
|---------|---------|
| Eyeballing | 1.0=正确, 0.0=错误, -1.0=格式错误 |
| Maze | 0.0~1.0=部分匹配, -1.0=格式错误 |

## 验证

```bash
python evaluators/vlm/validate_model.py \
    --model_path /path/to/Qwen3-VL-32B \
    --data_path data/train_sft.jsonl \
    --output_dir output/validate_vlm \
    --num_samples 50
```

## 预检

```bash
python evaluators/vlm/infer_precheck.py \
    --model_path /path/to/Qwen3-VL-32B \
    --data_path data/train_sft.jsonl \
    --num_samples 5
```
