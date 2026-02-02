
import argparse
import importlib
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Type

from tqdm import tqdm

# Add project root to path
sys.path.append(os.getcwd())

from data.puzzle.point_target_base import PointTargetPuzzleGenerator
from data.puzzle.base import AbstractPuzzleGenerator

# Tasks to include
ALL_TASKS = [
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

def load_generator_class(task_name: str) -> Type[AbstractPuzzleGenerator]:
    """Dynamically load generator class for a task."""
    module_path = f"data.puzzle.eyeballing.{task_name}.generator"
    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        raise ImportError(f"Failed to load module for task '{task_name}': {e}")
    
    # Heuristic: convert snake_case to CamelCase + Generator
    # e.g. angle_bisector -> AngleBisectorGenerator
    class_name = "".join(x.capitalize() for x in task_name.split("_")) + "Generator"
    
    if hasattr(module, class_name):
        return getattr(module, class_name)
    
    # Fallback: look for any class ending in 'Generator' that inherits from AbstractPuzzleGenerator
    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        if (isinstance(attr, type) and 
            issubclass(attr, AbstractPuzzleGenerator) and 
            attr is not AbstractPuzzleGenerator and
            attr is not PointTargetPuzzleGenerator and
            attr_name.endswith("Generator")):
            return attr
            
    raise ImportError(f"Could not find generator class in {module_path}")

def generate_task_data(
    task_name: str,
    args: argparse.Namespace,
    base_output_dir: Path
) -> List[dict]:
    """Generate data for a single task."""
    
    output_dir = base_output_dir / task_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        GenClass = load_generator_class(task_name)
    except Exception as e:
        logging.error(f"Skipping {task_name}: {e}")
        return []

    # Instantiate generator
    # Note: PointTargetPuzzleGenerator uses kwargs for some configs
    generator = GenClass(
        output_dir=output_dir,
        canvas_width=args.canvas_width,
        seed=args.seed,
        record_video=True, # Always record video for this script
    )
    
    # Inject Difficulty / Visual Parameters
    if isinstance(generator, PointTargetPuzzleGenerator):
        if args.point_radius is not None:
            generator.point_radius = args.point_radius
            generator.POINT_RADIUS = args.point_radius # Class attr sometimes used
        if args.line_width is not None:
            generator.LINE_WIDTH = args.line_width

    logging.info(f"Generating {args.count} samples for {task_name}...")
    
    records = []
    for _ in tqdm(range(args.count), desc=f"{task_name}"):
        try:
            record = generator.create_random_puzzle()
            records.append(record)
        except Exception as e:
            logging.warning(f"Failed to generate a sample for {task_name}: {e}")

    # Write per-task metadata
    metadata_path = output_dir / "data.json"
    generator.write_metadata(records, metadata_path)
    
    # Return dicts for global aggregation
    return [r.to_dict() for r in records]

def main():
    parser = argparse.ArgumentParser(description="Generate Eyeballing Video Dataset")
    parser.add_argument("--tasks", nargs="+", default=["all"], help=f"Tasks to generate. Choices: {ALL_TASKS}")
    parser.add_argument("--count", type=int, default=10, help="Number of samples per task")
    parser.add_argument("--output-dir", type=str, required=True, help="Root directory for output")
    parser.add_argument("--canvas-width", type=int, default=480, help="Canvas width in pixels")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Difficulty / Visual Params
    parser.add_argument("--point-radius", type=int, default=None, help="Radius of points (difficulty control)")
    parser.add_argument("--line-width", type=int, default=None, help="Width of lines (visual style)")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    if "all" in args.tasks:
        target_tasks = ALL_TASKS
    else:
        target_tasks = args.tasks

    root_output = Path(args.output_dir)
    root_output.mkdir(parents=True, exist_ok=True)
    
    all_metadata = []
    
    for task in target_tasks:
        if task not in ALL_TASKS:
            logging.warning(f"Unknown task: {task}, skipping.")
            continue
            
        task_records = generate_task_data(task, args, root_output)
        
        # Add task type to metadata
        for r in task_records:
            r["task_type"] = task
            all_metadata.append(r)

    # Save global metadata
    global_meta_path = root_output / "all_eyeballing_metadata.jsonl"
    logging.info(f"Saving global metadata to {global_meta_path}")
    with open(global_meta_path, "w") as f:
        for item in all_metadata:
            f.write(json.dumps(item) + "\n")

    logging.info("Done.")

if __name__ == "__main__":
    main()
