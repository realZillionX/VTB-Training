# VTB-Training Developer Guide & Code of Conduct

## 1. 核心理念 (Core Philosophy)

### 1.1 Multi-Modal Reasoning Exploration
本项目致力于复现 "Thinking with Video" 等前沿工作，探索不同模态模型在**视觉/空间推理任务**上的能力边界。

### 1.2 Structure Overview
项目结构已重构为四大核心模块：

*   **`data/` (Core Library)**: 
    *   **定位**: 数据生成逻辑的核心库 (Python Package)。包含所有 Puzzle Generator 的源码。
    *   **内容**: `data/puzzle/` (生成器), `data/tools/` (批量工具), `data/utils/`.
    *   **引用**: `import data.puzzle.eyeballing...`
*   **`dataset/` (Artifacts)**:
    *   **定位**: 存放生成的数据产物。
    *   **内容**: JSONL, MP4, PNG 等文件。此目录被 gitignore。
*   **`training/` (Runners)**:
    *   **定位**: 训练脚本入口。
    *   **内容**: Qwen3-VL, Qwen-Image, Wan2.2 的训练代码。
*   **`code_test/` (Unit Tests)**:
    *   **定位**: 代码单元测试。
    *   **内容**: Unittest / Pytest 脚本，用于验证 Generator/Evaluator 逻辑的正确性。

## 2. 工作流

1.  **开发**: 在 `data/puzzle/` 修改生成逻辑。
2.  **测试**: 在 `code_test/` 编写测试用例验证修改。
3.  **生成**: 使用 `data/tools/video/` 下的脚本批量生成数据到 `dataset/`。
4.  **训练**: 使用 `training/` 下的脚本读取 `dataset/` 进行训练。
