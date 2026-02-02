# VTB-Training

**VideoThinkBench 可复现训练代码库**

使用 VideoThinkBench 提供的训练集，对三类模型进行后训练：
- **VLM (Qwen3-VL)**: SFT + GRPO 强化学习
- **图像编辑模型 (Qwen-Image)**: LoRA SFT
- **视频生成模型 (Wan2.2)**: LoRA SFT

> **最新状态**: 本项目目前的策略是 **Visual Centric First**，优先验证 Visual Reasoning 任务。详情见下方 "Project Status" 章节。

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
# 准备数据（支持直接读取 VLMPuzzle 的 data.json）
python -m data.tools.prepare_vlm_data \
    --data_root /path/to/VLMPuzzle/dataset \
    --output_dir ./dataset

# SFT 训练
cd training/vlm && bash train_sft.sh --model_path /path/to/Qwen3-VL-32B

# GRPO 强化学习
python train_grpo.py --model_path /path/to/sft_checkpoint
```

### 2. 图像编辑模型训练 (Qwen-Image)

```bash
# 准备数据
python -m data.tools.prepare_image_data \
    --dataset_root /path/to/VLMPuzzle/dataset \
    --output_path ./dataset/metadata.json

# 训练
cd training/image
DIFFSYNTH_PATH=/path/to/DiffSynth-Studio bash train_sft.sh
```

### 3. 视频生成模型训练 (Wan2.2)

```bash
# 数据准备（生成 CSV）
python -m data.tools.prepare_video_data \
    --dataset_root /path/to/VLMPuzzle/dataset \
    --output_path ./dataset/train_video.csv

# 统一训练脚本（单机 / 多机）
cd training/video
MODEL_BASE_PATH=/path/to/Wan2.2-TI2V-5B \
DIFFSYNTH_PATH=/path/to/DiffSynth-Studio \
bash train_sft.sh --dataset ./dataset/train_video.csv --num_nodes 1
```

---

## Project Status & Plan

> **核心原则**: Video Generation Reasoning 的探索目前优先采用 **Visual Centric First** 和 **Single-Task Training** 路径。

### 1. 训练数据状态 (Training Data Status)

Visual Centric 任务的生成器代码大多**已存在**（`data/puzzle/`），主要 Gap 在于**训练数据的生成配置**。

| 任务类型 | 子任务 | 代码状态 | 训练数据配置状态 |
| :--- | :--- | :--- | :--- |
| **Maze** | **Square Maze** | ✅ Ready | ✅ **Finalized**: 已验证。配置为 7×7, 9×9, 11×11 三种尺寸，单方块 32px，各生成 10k 组。 |
| | Hexagon Maze | ✅ Ready | 🚧 **Pending**: 代码就绪，可以将边长/像素大小作为参数暴露，待确定最佳训练参数。 |
| | Circle Maze | ✅ Ready | 🚧 **Pending**: 同上，需调试生成参数。 |
| **Eyeballing** | 全量 21 类 | ✅ Ready | 🚧 **Pending**: 代码就绪。需编写批量生成脚本，并将难度/视觉参数（如线条粗细、点大小）参数化，以便生成多样化数据。 |
| **Visual Puzzles** | Color/Shape | ✅ Ready | ⏸️ **Deferred**: 暂不作为训练重点。 |
| **ARC-AGI-2** | Abstract | ✅ Ready | ⏸️ **Deferred**: 暂不作为训练重点。 |

### 2. 待办事项 (TODO List)

#### Phase 1: 代码库重构 - ✅ DONE
- [x] **目录结构整理**: 完成 `eyeballing`, `maze`, `visual` 的分类整理。
- [x] **Import 修复**: 修正了重构带来的相对路径引用问题。

#### Phase 2: 核心任务数据生成 (Data Generation)
- [x] **Eyeballing (High Priority)**: 编写 `generate_eyeballing_video.py`，支持 CLI 参数配置难度/画布尺寸。
- [x] **Maze**: 编写 Hexagon/Circle 的参数化生成脚本。

#### Phase 3: 模型训练 (Training)
- [ ] **Wan2.2 Single-Task**: 待 Eyeballing 数据生成后，启动单任务训练。
- [ ] **VLM / Image Model**: 同步启动 SFT。

## 项目结构

```
VTB-Training/
├── data/                      # 核心库 (Python Package)
│   ├── puzzle/                # Generator 源码 (Maze, Eyeballing)
│   ├── tools/                 # 批量数据生成工具
│   └── utils/                 # 通用工具
├── dataset/                   # 数据产物 (已忽略)
├── training/                  # 训练脚本
│   ├── vlm/                   # Qwen3-VL
│   ├── image/                 # Qwen-Image
│   └── video/                 # Wan2.2
├── code_test/                 # 旧版单元测试（建议逐步迁移到 tests/）
└── tests/                     # 评测脚本与轻量单测（含 evaluator/vlm/video/image 四个分区）
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
