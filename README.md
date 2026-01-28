# VTB-Training

**VideoThinkBench 可复现训练代码库**

使用 VideoThinkBench 提供的训练集，对三类模型进行后训练：
- **VLM (Qwen3-VL)**: SFT + GRPO 强化学习
- **图像编辑模型 (Qwen-Image)**: LoRA SFT
- **视频生成模型 (Wan2.2)**: LoRA SFT

## 项目结构

```
VTB-Training/
├── vtb_training/              # 核心 Python 包
│   ├── puzzle/                # VideoThinkBench Puzzle 库
│   ├── data/                  # 数据处理工具
│   ├── rewards/               # VLM GRPO 奖励函数
│   └── utils/                 # 通用工具
├── training/                  # 训练入口脚本
│   ├── vlm/                   # Qwen3-VL 训练
│   ├── image/                 # Qwen-Image 训练
│   └── video/                 # Wan2.2 训练
├── scripts/                   # 评估/生成脚本
└── configs/                   # 配置模板
```

## 环境要求

- Python 3.10+
- CUDA 12.0+
- **VLM 训练**: `ms-swift`, `peft`, `vllm`
- **图像/视频训练**: `DiffSynth-Studio`, `accelerate`, `deepspeed`

```bash
pip install -r requirements.txt
```

## 快速开始

### 1. VLM 训练 (Qwen3-VL)

```bash
# 准备数据
python -m vtb_training.data.prepare_vlm_data \
    --data_root /path/to/VLMPuzzle/dataset

# SFT 训练
cd training/vlm && bash train_sft.sh --model_path /path/to/Qwen3-VL-32B

# GRPO 强化学习
python train_grpo.py --model_path /path/to/sft_checkpoint
```

### 2. 图像编辑模型训练 (Qwen-Image)

```bash
# 准备数据
python -m vtb_training.data.prepare_image_data \
    --dataset_root /path/to/VLMPuzzle/dataset \
    --output_path ./data/metadata.json

# 训练
cd training/image
DIFFSYNTH_PATH=/path/to/DiffSynth-Studio bash train_sft.sh
```

### 3. 视频生成模型训练 (Wan2.2)

```bash
# 单节点训练
cd training/video
MODEL_BASE_PATH=/path/to/Wan2.2-TI2V-5B \
DIFFSYNTH_PATH=/path/to/DiffSynth-Studio \
bash train_single_node.sh

# 多节点训练
bash train_multi_node.sh
```

## 引用

如果您使用了本代码库，请引用 "Thinking with Video" 论文：

```bibtex
@article{tong2025thinking,
  title={Thinking with video: Video generation as a promising multimodal reasoning paradigm},
  author={Tong, Jingqi and Mou, Yurong and Li, Hangcheng and Li, Mingzhe and Yang, Yongzhuo and Zhang, Ming and Chen, Qiguang and Liang, Tianyi and Hu, Xiaomeng and Zheng, Yining and others},
  journal={arXiv preprint arXiv:2511.04570},
  year={2025}
}
```

## 许可证

MIT License
