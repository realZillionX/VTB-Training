# Tests 目录说明

本目录用于**评测脚本**与**轻量单测**，结构如下：

```
tests/
├── evaluator/   # 评测器与数据工具相关（含 legacy 单测与汇总脚本）
├── vlm/         # VLM 并行推理与评测脚本
├── image/       # 图像模型并行推理与评测脚本
└── video/       # 视频模型并行推理与评测脚本
```

## evaluator/
- `maze_summary.py`、`multiple_choice_summary.py`、`reevaluate_vote_output.py`：评测结果汇总与复评。
- `test_*`：评测器与数据转换相关的单测。
- `legacy/`：历史单测，逐步清理与合并。

## vlm/ / image/ / video/
建议放置：
- 并行推理脚本（多 GPU、多卡分工）。
- 评测脚本（加载评测器，计算准确率/成功率）。

> 注：训练脚本仍放在 `training/`，tests 只负责验证与评测，不负责任务训练本身。
