"""
VLM Data Preparation for SFT and GRPO (ms-swift format).

将 VLMPuzzle 生成的 data.json 转换为 ms-swift 可用的 JSONL 格式。

用法：
    python -m data.tools.prepare_vlm_data \
        --data_root /path/to/VLMPuzzle/dataset \
        --output_dir ./dataset
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

# Task -> Prompt Mapping for Eyeballing
EYEBALLING_PROMPTS = {
    "circle_center": "Which point looks like the center of the circle? Choose the best option.",
    "circumcenter": "Which point looked like the circumcenter of the triangle? Choose the best option.",
    "fermat_point": "Which point looks like the Fermat point of the triangle? Choose the best option.",
    "incenter": "Which point looks like the incenter of the triangle? Choose the best option.",
    "midpoint": "Which point looks like the midpoint of the segment? Choose the best option.",
    "orthocenter": "Which point looks like the orthocenter of the triangle? Choose the best option.",
    "point_reflection": "Which point looks like the reflection of the source point? Choose the best option.",
    "ray_intersection": "Which point looks like the intersection of the rays? Choose the best option.",
    "triangle_center": "Which point looks like the center of the triangle? Choose the best option.",
    "angle_bisector": "Which line looks like the angle bisector? Choose the best option.",
    "arc_connect": "Which line correctly connects the arcs? Choose the best option.",
    "circle_tangent_line": "Which line looks tangent to the circle? Choose the best option.",
    "circle_tangent_point": "Which point looks like the point of tangency? Choose the best option.",
    "parallel": "Which line looks parallel to the reference line? Choose the best option.",
    "perpendicular": "Which line looks perpendicular to the reference line? Choose the best option.",
    "perpendicular_bisector": "Which line looks like the perpendicular bisector? Choose the best option.",
    "ray_reflect": "Which ray looks like the correct reflection? Choose the best option.",
    "isosceles_trapezoid": "Which point completes the isosceles trapezoid? Choose the best option.",
    "parallelogram": "Which point completes the parallelogram? Choose the best option.",
    "right_triangle": "Which point forms a right triangle? Choose the best option.",
    "square_outlier": "Which point is the outlier that does not fit the square pattern? Choose the best option.",
}

MAZE_PROMPT = (
    "Find a path connecting two red dots without touching the black walls in the maze. "
    "Movement is between adjacent cells through shared edges only (no diagonal corner moves). "
    "Each cell has its ID printed on it. Present your answer as a list of cell IDs. "
    "Example: [1, 4, 3, 2]. Must answer now without asking for clarifications."
)


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


def get_prompt(task_group: str, puzzle_name: str, record: Dict[str, object]) -> str:
    if task_group == "maze":
        return str(record.get("gpt5_prompt") or MAZE_PROMPT).strip()
    if task_group == "eyeballing":
        if record.get("gpt5_prompt"):
            return str(record.get("gpt5_prompt")).strip()
        if puzzle_name in EYEBALLING_PROMPTS:
            return EYEBALLING_PROMPTS[puzzle_name]
        return str(record.get("prompt") or "Select the correct option from the image.").strip()
    return str(record.get("prompt") or "").strip()


def format_prompt(prompt: str, mode: str) -> str:
    if mode == "sft":
        suffix = "\nDo not output the thinking process. Output the answer directly."
    else:
        suffix = "\nPlease think step by step and output your final answer within <answer>...</answer> tags."
    return prompt + suffix


def build_entry(
    record: Dict[str, object],
    puzzle_name: str,
    meta_dir: Path,
    data_root: Path,
    task_group: str,
    mode: str,
    path_mode: str,
) -> Optional[Dict[str, object]]:
    image_value = record.get("image")
    if not image_value:
        return None

    image_path = resolve_path(meta_dir, str(image_value))
    if path_mode == "relative":
        try:
            image_value_out = image_path.relative_to(data_root).as_posix()
        except ValueError:
            image_value_out = image_path.as_posix()
    else:
        image_value_out = image_path.as_posix()

    task_name = str(record.get("task_type") or puzzle_name)
    prompt = format_prompt(get_prompt(task_group, task_name, record), mode)
    user_content = f"<image> {prompt}".strip()

    if task_group == "eyeballing":
        solution = str(record.get("correct_option", "")).strip()
    else:
        solution_list = record.get("solution_path_cell_ids", [])
        solution = json.dumps(solution_list)

    entry: Dict[str, object] = {
        "messages": [{"role": "user", "content": user_content}],
        "images": [image_value_out],
        "solution": solution,
        "task_type": task_name,
        "task_group": task_group,
    }

    if mode == "sft":
        entry["messages"].append({"role": "assistant", "content": solution})

    record_id = record.get("id")
    if record_id is not None:
        entry["id"] = str(record_id)
    return entry


def collect_entries(
    data_root: Path,
    task_groups: List[str],
    mode: str,
    path_mode: str,
) -> List[Dict[str, object]]:
    entries: List[Dict[str, object]] = []
    for meta_path in iter_metadata_files(data_root):
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
            if task_group is None or task_group not in task_groups:
                continue
            entry = build_entry(
                record=record,
                puzzle_name=puzzle_name,
                meta_dir=meta_path.parent,
                data_root=data_root,
                task_group=task_group,
                mode=mode,
                path_mode=path_mode,
            )
            if entry is not None:
                entries.append(entry)
    return entries


def write_jsonl(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert VLMPuzzle datasets to ms-swift JSONL format")
    parser.add_argument("--data_root", type=str, required=True, help="VLMPuzzle dataset root directory")
    parser.add_argument("--output_dir", type=str, default=".", help="Output directory for JSONL files")
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
        default="absolute",
        choices=["absolute", "relative"],
        help="Image path output mode",
    )

    args = parser.parse_args()

    data_root = Path(args.data_root).expanduser().resolve()
    if not data_root.exists():
        raise FileNotFoundError(f"data_root not found: {data_root}")

    output_dir = Path(args.output_dir).expanduser().resolve()

    print("=" * 60)
    print("VLMPuzzle -> ms-swift 数据转换")
    print(f"Data Root: {data_root}")
    print(f"Output Dir: {output_dir}")
    print(f"Task Groups: {args.task_groups}")
    print(f"Path Mode: {args.path_mode}")
    print("=" * 60)

    sft_rows = collect_entries(data_root, args.task_groups, mode="sft", path_mode=args.path_mode)
    grpo_rows = collect_entries(data_root, args.task_groups, mode="grpo", path_mode=args.path_mode)

    sft_out = output_dir / "train_sft.jsonl"
    grpo_out = output_dir / "train_grpo.jsonl"

    write_jsonl(sft_out, sft_rows)
    write_jsonl(grpo_out, grpo_rows)

    print(f"Saved {len(sft_rows)} SFT samples to {sft_out}")
    print(f"Saved {len(grpo_rows)} GRPO samples to {grpo_out}")
    print("=" * 60)


if __name__ == "__main__":
    main()
