"""Unified dataset generator with optional multi-CPU parallelism.

This script wraps per-task generators under data/puzzle and supports:
- Selecting task groups or explicit tasks.
- Generating data in parallel across CPU workers.
- Merging per-worker metadata into a single data.json per task.
"""

import argparse
import importlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@dataclass
class TaskSpec:
    name: str
    group: str
    module: str
    class_name: Optional[str] = None
    requires: Tuple[str, ...] = ()


EYEBALLING_TASKS = [
    "angle_bisector",
    "arc_connect",
    "arc_connect_point_ver",
    "circle_center",
    "circle_tangent_line",
    "circle_tangent_point",
    "circumcenter",
    "fermat_point",
    "incenter",
    "isosceles_trapezoid",
    "midpoint",
    "orthocenter",
    "parallel",
    "parallelogram",
    "perpendicular",
    "perpendicular_bisector",
    "ray",
    "ray_intersection",
    "ray_reflect",
    "reflection",
    "right_triangle",
    "square_outlier",
    "triangle_center",
]

VISUAL_TASKS = [
    "arcagi",
    "circle_count",
    "jigsaw",
    "mirror",
    "rects",
    "sudoku",
]

MAZE_TASKS = [
    "maze_square",
    "maze_hexagon",
    "maze_labyrinth",
]


def _camel_task(task_name: str) -> str:
    return "".join(chunk.capitalize() for chunk in task_name.split("_"))


def build_task_specs() -> Dict[str, TaskSpec]:
    specs: Dict[str, TaskSpec] = {}
    for task in EYEBALLING_TASKS:
        specs[task] = TaskSpec(
            name=task,
            group="eyeballing",
            module=f"data.puzzle.eyeballing.{task}.generator",
            class_name=f"{_camel_task(task)}Generator",
        )

    specs["maze_square"] = TaskSpec(
        name="maze_square",
        group="maze",
        module="data.puzzle.maze.maze_square.generator",
        class_name="MazeGenerator",
    )
    specs["maze_hexagon"] = TaskSpec(
        name="maze_hexagon",
        group="maze",
        module="data.puzzle.maze.maze_hexagon.generator",
        class_name="MazeHexagonGenerator",
    )
    specs["maze_labyrinth"] = TaskSpec(
        name="maze_labyrinth",
        group="maze",
        module="data.puzzle.maze.maze_labyrinth.generator",
        class_name="MazeLabyrinthGenerator",
    )

    specs["circle_count"] = TaskSpec(
        name="circle_count",
        group="visual",
        module="data.puzzle.visual.circle_count.generator",
        class_name="CircleCountGenerator",
    )
    specs["rects"] = TaskSpec(
        name="rects",
        group="visual",
        module="data.puzzle.visual.rects.generator",
        class_name="RectsGenerator",
    )
    specs["mirror"] = TaskSpec(
        name="mirror",
        group="visual",
        module="data.puzzle.visual.mirror.generator",
        class_name="MirrorGenerator",
    )
    specs["jigsaw"] = TaskSpec(
        name="jigsaw",
        group="visual",
        module="data.puzzle.visual.jigsaw.generator",
        class_name="JigsawGenerator",
    )
    specs["sudoku"] = TaskSpec(
        name="sudoku",
        group="visual",
        module="data.puzzle.visual.sudoku.generator",
        class_name="SudokuGenerator",
    )
    specs["arcagi"] = TaskSpec(
        name="arcagi",
        group="visual",
        module="data.puzzle.visual.arcagi.generator",
        class_name="ArcPuzzleGenerator",
        requires=("arcagi_dataset_dir",),
    )
    return specs


TASK_SPECS = build_task_specs()


def load_generator_class(spec: TaskSpec):
    module = importlib.import_module(spec.module)
    if spec.class_name and hasattr(module, spec.class_name):
        return getattr(module, spec.class_name)

    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        if isinstance(attr, type) and attr_name.endswith("Generator"):
            return attr

    raise ImportError(f"Generator class not found in {spec.module}")


def parse_task_config(config_path: Optional[str], config_json: Optional[str]) -> Dict[str, Dict[str, Any]]:
    if config_path:
        payload = json.loads(Path(config_path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("task_config_path must contain a JSON object")
        return payload
    if config_json:
        payload = json.loads(config_json)
        if not isinstance(payload, dict):
            raise ValueError("task_config must be a JSON object")
        return payload
    return {}


def split_counts(total: int, parts: int) -> List[int]:
    if parts <= 0:
        return []
    base = total // parts
    extra = total % parts
    return [base + (1 if idx < extra else 0) for idx in range(parts)]


def _prefix_path(value: Any, prefix: str) -> Any:
    if not isinstance(value, str):
        return value
    candidate = Path(value)
    if candidate.is_absolute():
        return value
    if value.startswith(prefix + "/"):
        return value
    return f"{prefix}/{value}"


def prefix_record_paths(record: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    keys = [
        "image",
        "solution_image_path",
        "solution_video_path",
        "video",
        "puzzle",
        "solution",
    ]
    for key in keys:
        if key in record:
            record[key] = _prefix_path(record[key], prefix)
    if "images" in record and isinstance(record["images"], list):
        record["images"] = [_prefix_path(item, prefix) for item in record["images"]]
    return record


def build_default_kwargs(spec: TaskSpec, args: argparse.Namespace, seed: Optional[int]) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {}
    if spec.group == "eyeballing":
        kwargs.update(
            canvas_width=args.canvas_width,
            seed=seed,
            record_video=args.video,
        )
        if args.point_radius is not None:
            kwargs["point_radius"] = args.point_radius
        if args.line_width is not None:
            kwargs["line_width"] = args.line_width
    elif spec.name == "maze_square":
        kwargs.update(
            rows=args.maze_rows,
            cols=args.maze_cols,
            cell_size=args.maze_cell_size,
            canvas_width=args.canvas_width,
            seed=seed,
            video=args.video,
        )
    elif spec.name == "maze_hexagon":
        kwargs.update(
            radius=args.hex_radius,
            cell_radius=args.hex_cell_size,
            wall_thickness=args.hex_wall_thickness,
            canvas_width=args.canvas_width,
            seed=seed,
            video=args.video,
        )
    elif spec.name == "maze_labyrinth":
        kwargs.update(
            rings=args.lab_rings,
            segments=args.lab_segments,
            ring_width=args.lab_cell_size,
            wall_thickness=args.lab_wall_thickness,
            canvas_width=args.canvas_width,
            seed=seed,
            video=args.video,
        )
    elif spec.name == "arcagi":
        kwargs.update(
            dataset_dir=args.arcagi_dataset_dir,
            canvas_width=args.canvas_width,
            seed=seed,
        )
    elif spec.name == "circle_count":
        kwargs.update(
            canvas_width=args.canvas_width,
            seed=seed,
        )
    else:
        kwargs.update(seed=seed)
    return kwargs


def generate_worker(job: Dict[str, Any]) -> Dict[str, Any]:
    spec: TaskSpec = job["spec"]
    output_dir: Path = job["output_dir"]
    count: int = job["count"]
    seed: Optional[int] = job["seed"]
    kwargs: Dict[str, Any] = job["kwargs"]

    output_dir.mkdir(parents=True, exist_ok=True)
    GenClass = load_generator_class(spec)
    import inspect

    sig = inspect.signature(GenClass.__init__)
    accepts_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in sig.parameters.values())
    if not accepts_kwargs:
        allowed = {name for name in sig.parameters if name not in {"self", "output_dir"}}
        kwargs = {key: value for key, value in kwargs.items() if key in allowed}

    generator = GenClass(output_dir=output_dir, **kwargs)
    records = [generator.create_random_puzzle() for _ in range(count)]
    metadata_path = output_dir / "data.json"
    generator.write_metadata(records, metadata_path, append=False)
    return {
        "task": spec.name,
        "count": count,
        "metadata": metadata_path.as_posix(),
        "worker_dir": output_dir.as_posix(),
    }


def merge_metadata(task_dir: Path, worker_dirs: List[Path]) -> Path:
    merged: List[Dict[str, Any]] = []
    for worker_dir in worker_dirs:
        meta_path = worker_dir / "data.json"
        if not meta_path.exists():
            continue
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            continue
        prefix = worker_dir.relative_to(task_dir).as_posix()
        for record in payload:
            if isinstance(record, dict):
                merged.append(prefix_record_paths(record, prefix))

    out_path = task_dir / "data.json"
    out_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified dataset generator with multi-CPU support")
    parser.add_argument("--output_dir", type=str, required=True, help="Root output directory")
    parser.add_argument("--tasks", nargs="+", default=["all"], help="Tasks to generate or 'all'")
    parser.add_argument("--task_groups", nargs="+", default=["eyeballing", "maze", "visual"], help="Task groups to include")
    parser.add_argument("--exclude_tasks", nargs="+", default=[], help="Tasks to exclude")
    parser.add_argument("--count", type=int, default=10, help="Samples per task")
    parser.add_argument("--num_workers", type=int, default=4, help="CPU workers per task")
    parser.add_argument("--seed", type=int, default=42, help="Base seed")
    parser.add_argument("--video", action="store_true", help="Generate video if supported")
    parser.add_argument("--canvas_width", type=int, default=480, help="Canvas width in pixels")

    # Eyeballing params
    parser.add_argument("--point_radius", type=int, default=None)
    parser.add_argument("--line_width", type=int, default=None)

    # Maze params
    parser.add_argument("--maze_rows", type=int, default=9)
    parser.add_argument("--maze_cols", type=int, default=9)
    parser.add_argument("--maze_cell_size", type=int, default=32)

    parser.add_argument("--hex_radius", type=int, default=4)
    parser.add_argument("--hex_cell_size", type=int, default=24)
    parser.add_argument("--hex_wall_thickness", type=int, default=None)

    parser.add_argument("--lab_rings", type=int, default=6)
    parser.add_argument("--lab_segments", type=int, default=18)
    parser.add_argument("--lab_cell_size", type=int, default=18)
    parser.add_argument("--lab_wall_thickness", type=int, default=None)

    # ArcAGI params
    parser.add_argument("--arcagi_dataset_dir", type=str, default=None, help="Required for arcagi task")

    # Custom task config
    parser.add_argument("--task_config_path", type=str, default=None, help="JSON file for per-task kwargs")
    parser.add_argument("--task_config", type=str, default=None, help="Inline JSON for per-task kwargs")

    args = parser.parse_args()

    task_config = parse_task_config(args.task_config_path, args.task_config)

    requested_tasks = set(args.tasks)
    if "all" in requested_tasks:
        requested_tasks = set(TASK_SPECS.keys())
    requested_tasks = {t for t in requested_tasks if t in TASK_SPECS}

    if args.task_groups:
        requested_tasks = {
            task for task in requested_tasks
            if TASK_SPECS[task].group in args.task_groups
        }

    for exclude in args.exclude_tasks:
        requested_tasks.discard(exclude)

    if not requested_tasks:
        raise ValueError("No tasks selected. Check --tasks / --task_groups / --exclude_tasks")

    output_root = Path(args.output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Unified Dataset Generator")
    print(f"Output Dir: {output_root}")
    print(f"Tasks: {sorted(requested_tasks)}")
    print(f"Count per task: {args.count}")
    print(f"Workers per task: {args.num_workers}")
    print("=" * 60)

    import multiprocessing as mp
    ctx = mp.get_context("spawn")

    for task in sorted(requested_tasks):
        spec = TASK_SPECS[task]
        missing = [req for req in spec.requires if not getattr(args, req, None)]
        if missing:
            print(f"[Skip] {task}: missing required args {missing}")
            continue

        task_dir = output_root / task
        task_dir.mkdir(parents=True, exist_ok=True)

        workers = max(1, min(args.num_workers, args.count))
        counts = split_counts(args.count, workers)

        jobs: List[Dict[str, Any]] = []
        for worker_idx, count in enumerate(counts):
            if count <= 0:
                continue
            worker_dir = task_dir / f"worker_{worker_idx:02d}"
            base_seed = args.seed if args.seed is not None else None
            seed = (base_seed + worker_idx) if base_seed is not None else None

            kwargs = build_default_kwargs(spec, args, seed)
            override_kwargs = task_config.get(task, {})
            if override_kwargs:
                kwargs.update(override_kwargs)

            job = {
                "spec": spec,
                "output_dir": worker_dir,
                "count": count,
                "seed": seed,
                "kwargs": kwargs,
            }
            jobs.append(job)

        print(f"[Task] {task}: {len(jobs)} workers")

        results = []
        with ctx.Pool(processes=workers) as pool:
            for result in pool.imap_unordered(generate_worker, jobs):
                results.append(result)

        worker_dirs = [Path(item["worker_dir"]) for item in results]
        merged_path = merge_metadata(task_dir, worker_dirs)
        total = sum(item["count"] for item in results)
        print(f"  -> merged {total} samples into {merged_path}")

    print("Done.")


if __name__ == "__main__":
    main()
