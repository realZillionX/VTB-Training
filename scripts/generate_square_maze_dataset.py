
import argparse
import sys
import multiprocessing
import csv
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Tuple, List, Optional

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from vtb_training.puzzle.maze_square.generator import MazeGenerator

class FixedSizeMazeGenerator(MazeGenerator):
    def __init__(self, target_width: int, target_height: int, **kwargs):
        self.target_width = target_width
        self.target_height = target_height
        # Initialize super with target width and safe aspect ratio
        # We will override the padding computation anyway
        super().__init__(
            canvas_width=target_width, 
            aspect=target_width / target_height if target_height > 0 else 1.0, 
            **kwargs
        )

    def _compute_padding(self) -> Tuple[int, int, int, int, Tuple[int, int]]:
        """
        Override padding computation to center content within the exact target dimensions.
        """
        content_width = self.cols * self.cell_size
        content_height = self.rows * self.cell_size
        
        if content_width > self.target_width:
             raise ValueError(f"Content width {content_width} exceeds target width {self.target_width}")
        if content_height > self.target_height:
             raise ValueError(f"Content height {content_height} exceeds target height {self.target_height}")

        pad_left = (self.target_width - content_width) // 2
        pad_top = (self.target_height - content_height) // 2
        pad_right = self.target_width - content_width - pad_left
        pad_bottom = self.target_height - content_height - pad_top
        
        return pad_left, pad_top, pad_right, pad_bottom, (self.target_width, self.target_height)

    def save_video(
        self,
        record_id: str,
        puzzle_image: "Image.Image",
        points: List[Tuple[float, float]],
        thickness: int = 5,
        color: Tuple[int, int, int] = (220, 0, 0),
        fps: int = 30,
        duration: float = 6.4,
    ) -> Optional[Path]:
        # Force FPS=10 and duration=7.0 to achieve fixed 81 frames
        # Logic: int(10 * 7.0) + 1 + int(10 * 1.0) = 70 + 1 + 10 = 81
        return super().save_video(
            record_id, 
            puzzle_image, 
            points, 
            thickness, 
            color, 
            fps=10, 
            duration=7.0
        )

def generate_one(args: Tuple) -> Tuple[str, str]:
    # Unpack arguments for the worker
    idx, output_dir, rows, cols, cell_size, width, height, prompt = args
    
    # Instantiate generator per process
    generator = FixedSizeMazeGenerator(
        target_width=width,
        target_height=height,
        rows=rows,
        cols=cols,
        cell_size=cell_size,
        output_dir=output_dir,
        prompt=prompt,
        video=True, # Always generate video
        show_cell_id=False
    )
    
    # Create the puzzle and video
    record = generator.create_random_puzzle()
    
    # Construct the absolute path to the video file
    # MazeGenerator saves video as {record_id}_solution.mp4 in the 'solutions' subdirectory
    video_path = generator.solution_dir / f"{record.id}_solution.mp4"
    
    # Return absolute path and prompt
    return str(video_path.absolute()), record.prompt

def main():
    parser = argparse.ArgumentParser(description="Generate square maze dataset in parallel.")
    parser.add_argument("--rows", type=int, required=True, help="Number of rows in the maze grid")
    parser.add_argument("--cols", type=int, required=True, help="Number of cols in the maze grid")
    parser.add_argument("--count", type=int, required=True, help="Total number of puzzles to generate")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to save outputs")
    parser.add_argument("--workers", type=int, default=multiprocessing.cpu_count(), help="Number of parallel workers")
    parser.add_argument("--cell-size", type=int, default=32, help="Pixel size of each maze cell")
    parser.add_argument("--width", type=int, default=480, help="Target canvas width")
    parser.add_argument("--height", type=int, default=873, help="Target canvas height")
    parser.add_argument("--csv-out", type=Path, required=True, help="Path for the output CSV chunk")
    
    args = parser.parse_args()
    
    # Ensure output directories exist
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prompt is fixed as per requirements
    prompt = "Draw a red path connecting two red dots without touching the black walls. Static camera."
    
    # Prepare tasks
    tasks = []
    for i in range(args.count):
        tasks.append((i, args.output_dir, args.rows, args.cols, args.cell_size, args.width, args.height, prompt))
        
    results = []
    print(f"Starting generation of {args.count} mazes ({args.rows}x{args.cols}) with {args.workers} workers...")
    
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(generate_one, task) for task in tasks]
        for future in as_completed(futures):
            try:
                res = future.result()
                results.append(res)
                if len(results) % 100 == 0:
                    print(f"Generated {len(results)}/{args.count}", end='\r')
            except Exception as e:
                print(f"Error generating puzzle: {e}")
                
    print(f"\nFinished generating {len(results)} puzzles.")
    
    # Write results to CSV (without header to allow easy concatenation later, or with header? 
    # Best practice for batched scripts: Write header if file new. But here we output to specific partial csv.
    # We will write header and let the merger handle it.)
    
    with open(args.csv_out, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["video", "prompt"])
        for vid, prmt in results:
            writer.writerow([vid, prmt])
    
    print(f"Saved metadata to {args.csv_out}")

if __name__ == "__main__":
    main()
