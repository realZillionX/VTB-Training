"""
Image Data Preparation for Qwen-Image SFT Training (DiffSynth-Studio format).

将 VLMPuzzle 生成的 data.json 转换为 DiffSynth-Studio 训练所需的 metadata.json。

用法：
    python -m data.tools.prepare_image_data \
        --dataset_root /path/to/VLMPuzzle/dataset \
        --output_path ./dataset/metadata.json
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional


def iter_metadata_files(data_root: Path) -> Iterable[Path]:
    for meta in sorted(data_root.rglob("data.json")):
        yield meta


def resolve_path(base_dir: Path, value: str) -> Path:
    candidate = Path(str(value))
    if not candidate.is_absolute():
        candidate = base_dir / candidate
    return candidate


def detect_task_group(record: Dict[str, object], puzzle_name: str) -> Optional[str]:
    if "correct_option" in record:
        return "eyeballing"
    if "solution_path_cell_ids" in record or puzzle_name.startswith("maze"):
        return "maze"
    return None


def process_eyeballing_prompt(prompt: str) -> str:
    patterns_to_remove = [
        r"\s*Speak out[^.]*\.[^.]*\.",
        r"\s*In portrait[^.]*\.",
        r"\s*Static camera\.",
    ]

    result = prompt
    for pattern in patterns_to_remove:
        result = re.sub(pattern, "", result, flags=re.IGNORECASE)

    result = " ".join(result.split())
    if result and not result.endswith("."):
        result += "."
    return result.strip()


def process_maze_prompt(_: str) -> str:
    return "Draw a red path connecting two red dots without touching the black walls."


def build_item(
    record: Dict[str, object],
    puzzle_name: str,
    meta_dir: Path,
    dataset_root: Path,
    task_group: Optional[str],
    path_mode: str,
) -> Optional[Dict[str, str]]:
    puzzle_path_value = record.get("image")
    solution_path_value = record.get("solution_image_path")
    if not puzzle_path_value or not solution_path_value:
        return None

    puzzle_path = resolve_path(meta_dir, str(puzzle_path_value))
    solution_path = resolve_path(meta_dir, str(solution_path_value))

    if path_mode == "relative":
        try:
            puzzle_out = puzzle_path.relative_to(dataset_root).as_posix()
        except ValueError:
            puzzle_out = puzzle_path.as_posix()
        try:
            solution_out = solution_path.relative_to(dataset_root).as_posix()
        except ValueError:
            solution_out = solution_path.as_posix()
    else:
        puzzle_out = puzzle_path.as_posix()
        solution_out = solution_path.as_posix()

    task_name = str(record.get("task_type") or puzzle_name)
    original_prompt = str(record.get("prompt") or "").strip()
    if task_group == "eyeballing":
        prompt = process_eyeballing_prompt(original_prompt)
    elif task_group == "maze":
        prompt = process_maze_prompt(original_prompt)
    else:
        prompt = original_prompt

    return {
        "prompt": prompt,
        "image": solution_out,
        "edit_image": puzzle_out,
        "task_type": task_name,
    }


def collect_items(
    dataset_root: Path,
    task_groups: List[str],
    path_mode: str,
) -> List[Dict[str, str]]:
    items: List[Dict[str, str]] = []

    for meta_path in iter_metadata_files(dataset_root):
        try:
            payload = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print(f"Warning: invalid JSON, skip: {meta_path}")
            continue
        if not isinstance(payload, list):
            print(f"Warning: metadata is not a list, skip: {meta_path}")
            continue

        puzzle_name = meta_path.parent.name
        for record in payload:
            if not isinstance(record, dict):
                continue
            task_group = detect_task_group(record, puzzle_name)
            if task_group is not None and task_group not in task_groups:
                continue
            item = build_item(
                record=record,
                puzzle_name=puzzle_name,
                meta_dir=meta_path.parent,
                dataset_root=dataset_root,
                task_group=task_group,
                path_mode=path_mode,
            )
            if item is not None:
                items.append(item)

    return items


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert VLMPuzzle datasets to DiffSynth-Studio format")
    parser.add_argument("--dataset_root", type=str, required=True, help="VLMPuzzle dataset root directory")
    parser.add_argument("--output_path", type=str, default="./dataset/metadata.json", help="Output metadata.json path")
    parser.add_argument(
        "--task_groups",
        type=str,
        nargs="+",
        default=["eyeballing", "maze"],
        choices=["eyeballing", "maze"],
        help="Task groups to include",
    )
    parser.add_argument(
        "--path_mode",
        type=str,
        default="relative",
        choices=["relative", "absolute"],
        help="Image path output mode",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print info only, don't write file")

    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"dataset_root not found: {dataset_root}")

    output_path = Path(args.output_path).expanduser().resolve()

    print("=" * 60)
    print("VLMPuzzle -> DiffSynth 数据转换")
    print(f"Dataset Root: {dataset_root}")
    print(f"Output File: {output_path}")
    print(f"Task Groups: {args.task_groups}")
    print(f"Path Mode: {args.path_mode}")
    print("=" * 60)

    items = collect_items(dataset_root, args.task_groups, args.path_mode)

    print(f"Total: {len(items)} samples")
    if items:
        print("First 3 samples:")
        for i, item in enumerate(items[:3], start=1):
            print(f"[{i}] prompt: {item['prompt'][:80]}...")
            print(f"    image: {item['image']}")
            print(f"    edit_image: {item['edit_image']}")

    if args.dry_run:
        print("[Dry Run] Not writing file")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(items, handle, ensure_ascii=False, indent=2)

    print(f"✓ Data saved to: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
