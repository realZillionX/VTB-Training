
import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import List

from tqdm import tqdm

# Add project root to path
sys.path.append(os.getcwd())

from data.puzzle.maze.maze_hexagon import MazeHexagonGenerator
from data.puzzle.maze.maze_labyrinth import MazeLabyrinthGenerator

def generate_hexagon_data(args: argparse.Namespace, base_output_dir: Path) -> List[dict]:
    output_dir = base_output_dir / "maze_hexagon"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logging.info(f"Generating {args.count} Hexagon Mazes (Radius={args.hex_radius})...")
    
    generator = MazeHexagonGenerator(
        output_dir=output_dir,
        radius=args.hex_radius,
        cell_radius=args.cell_size, # Map cell_size to cell_radius for hexagon
        wall_thickness=args.wall_thickness,
        canvas_width=args.canvas_width,
        seed=args.seed,
        video=True,
    )
    
    records = []
    for _ in tqdm(range(args.count), desc="Hexagon"):
        try:
            record = generator.create_random_puzzle()
            records.append(record)
        except Exception as e:
            logging.warning(f"Failed to generate hexagon maze: {e}")
            
    # Write metadata
    generator.write_metadata(records, output_dir / "data.json")
    return [r.to_dict() for r in records]

def generate_labyrinth_data(args: argparse.Namespace, base_output_dir: Path) -> List[dict]:
    output_dir = base_output_dir / "maze_labyrinth"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logging.info(f"Generating {args.count} Labyrinth Mazes (Rings={args.lab_rings})...")
    
    generator = MazeLabyrinthGenerator(
        output_dir=output_dir,
        rings=args.lab_rings,
        segments=args.lab_segments,
        ring_width=args.cell_size, # Map cell_size to ring_width for labyrinth
        wall_thickness=args.wall_thickness,
        canvas_width=args.canvas_width,
        seed=args.seed,
        video=True,
    )
    
    records = []
    for _ in tqdm(range(args.count), desc="Labyrinth"):
        try:
            record = generator.create_random_puzzle()
            records.append(record)
        except Exception as e:
            logging.warning(f"Failed to generate labyrinth maze: {e}")
            
    generator.write_metadata(records, output_dir / "data.json")
    return [r.to_dict() for r in records]

def main():
    parser = argparse.ArgumentParser(description="Generate Maze Variant Video Dataset")
    parser.add_argument("--variants", nargs="+", default=["all"], choices=["all", "hexagon", "labyrinth"], help="Variants to generate")
    parser.add_argument("--count", type=int, default=10, help="Number of samples per variant")
    parser.add_argument("--output-dir", type=str, required=True, help="Root directory for output")
    parser.add_argument("--canvas-width", type=int, default=480, help="Canvas width in pixels")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--wall-thickness", type=int, default=None, help="Wall thickness in pixels")
    parser.add_argument("--cell-size", type=int, default=None, help="Cell radius (Hexagon) or Ring width (Labyrinth)")
    
    # Hexagon Specific
    parser.add_argument("--hex-radius", type=int, default=4, help="Hex: Number of rings from center (Grid Radius)")
    
    # Labyrinth Specific
    parser.add_argument("--lab-rings", type=int, default=6, help="Lab: Number of rings")
    parser.add_argument("--lab-segments", type=int, default=18, help="Lab: Number of angular segments")
    
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    root_output = Path(args.output_dir)
    root_output.mkdir(parents=True, exist_ok=True)
    
    targets = []
    if "all" in args.variants:
        targets = ["hexagon", "labyrinth"]
    else:
        targets = args.variants
        
    all_metadata = []
    
    if "hexagon" in targets:
        hex_records = generate_hexagon_data(args, root_output)
        for r in hex_records:
            r["variant"] = "hexagon"
            all_metadata.append(r)
            
    if "labyrinth" in targets:
        lab_records = generate_labyrinth_data(args, root_output)
        for r in lab_records:
            r["variant"] = "labyrinth"
            all_metadata.append(r)
            
    # Save global metadata
    global_meta_path = root_output / "all_maze_variants_metadata.jsonl"
    logging.info(f"Saving global metadata to {global_meta_path}")
    with open(global_meta_path, "w") as f:
        for item in all_metadata:
            f.write(json.dumps(item) + "\n")
            
    logging.info("Done.")

if __name__ == "__main__":
    main()
