"""
Video Data Preparation for Wan2.2 (DiffSynth-Studio CSV format).

将 VLMPuzzle 生成的 data.json 转换为 Wan2.2 训练所需的 CSV：
  - video: 视频文件路径
  - prompt: 文本描述

用法：
    python -m data.tools.prepare_video_data \
        --dataset_root /path/to/VLMPuzzle/dataset \
        --output_path ./dataset/train_video.csv
"""

import argparse
import csv
import json
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


def process_prompt(record: Dict[str, object], task_group: Optional[str]) -> str:
    if task_group == "maze":
        return "Draw a red path connecting two red dots without touching the black walls."
    prompt = str(record.get("prompt") or "").strip()
    return prompt


def guess_solution_video(solution_image: Path) -> Optional[Path]:
    if solution_image.suffix:
        candidate = solution_image.with_suffix(".mp4")
        if candidate.exists():
            return candidate
        candidate = solution_image.with_suffix(".webm")
        if candidate.exists():
            return candidate
    return None


def build_row(
    record: Dict[str, object],
    puzzle_name: str,
    meta_dir: Path,
    dataset_root: Path,
    task_group: Optional[str],
    path_mode: str,
) -> Optional[Dict[str, str]]:
    solution_value = record.get("solution_video_path") or record.get("solution_image_path")
    if not solution_value:
        return None

    solution_path = resolve_path(meta_dir, str(solution_value))
    if solution_path.suffix.lower() not in {".mp4", ".webm", ".mov"}:
        guessed = guess_solution_video(solution_path)
        if guessed is None:
            return None
        solution_path = guessed

    if path_mode == "relative":
        try:
            video_out = solution_path.relative_to(dataset_root).as_posix()
        except ValueError:
            video_out = solution_path.as_posix()
    else:
        video_out = solution_path.as_posix()

    prompt = process_prompt(record, task_group)
    task_name = str(record.get("task_type") or puzzle_name)
    return {"video": video_out, "prompt": prompt, "task_type": task_name}


def collect_rows(
    dataset_root: Path,
    task_groups: List[str],
    path_mode: str,
) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
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
            row = build_row(
                record=record,
                puzzle_name=puzzle_name,
                meta_dir=meta_path.parent,
                dataset_root=dataset_root,
                task_group=task_group,
                path_mode=path_mode,
            )
            if row is not None:
                rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert VLMPuzzle datasets to Wan2.2 CSV format")
    parser.add_argument("--dataset_root", type=str, required=True, help="VLMPuzzle dataset root directory")
    parser.add_argument("--output_path", type=str, default="./dataset/train_video.csv", help="Output CSV path")
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
        help="Video path output mode",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print info only, don't write file")

    args = parser.parse_args()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"dataset_root not found: {dataset_root}")

    output_path = Path(args.output_path).expanduser().resolve()

    print("=" * 60)
    print("VLMPuzzle -> Wan2.2 CSV 数据转换")
    print(f"Dataset Root: {dataset_root}")
    print(f"Output File: {output_path}")
    print(f"Task Groups: {args.task_groups}")
    print(f"Path Mode: {args.path_mode}")
    print("=" * 60)

    rows = collect_rows(dataset_root, args.task_groups, args.path_mode)

    print(f"Total: {len(rows)} samples")
    if rows:
        print(f"First sample: {rows[0]['video']}")

    if args.dry_run:
        print("[Dry Run] Not writing file")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["video", "prompt"], quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for row in rows:
            writer.writerow({"video": row["video"], "prompt": row["prompt"]})

    print(f"✓ Data saved to: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
