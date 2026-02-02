import json
import tempfile
import unittest
from pathlib import Path

import sys
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.puzzle.maze.maze_square.evaluator import MazeEvaluator


class MazeEndpointSafetyTestCase(unittest.TestCase):
    def test_endpoint_blocks_not_counted_as_wall_collision(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir) / "maze_square"
            base_dir.mkdir(parents=True, exist_ok=True)

            width, height = 40, 20
            puzzle_path = base_dir / "puzzle.png"
            candidate_path = base_dir / "candidate.png"

            puzzle = Image.new("RGB", (width, height), (0, 0, 0))
            puzzle.save(puzzle_path)

            candidate = puzzle.copy()
            draw = ImageDraw.Draw(candidate)
            draw.rectangle([0, 0, 19, 19], fill=(255, 0, 0))
            draw.rectangle([20, 0, 39, 19], fill=(255, 0, 0))
            candidate.save(candidate_path)

            record = {
                "id": "test",
                "prompt": "",
                "gpt5_prompt": "",
                "canvas_dimensions": [width, height],
                "start": [0, 0],
                "goal": [0, 1],
                "cell_bboxes": [[(0, 0, 20, 20), (20, 0, 40, 20)]],
                "image": "puzzle.png",
                "solution_image_path": "solution.png",
            }
            (base_dir / "data.json").write_text(json.dumps([record]), encoding="utf-8")

            evaluator = MazeEvaluator(base_dir / "data.json", base_dir=base_dir)
            result = evaluator.evaluate("test", candidate_path)

            self.assertFalse(result.overlaps_walls)
            self.assertTrue(result.touches_start)
            self.assertTrue(result.touches_goal)


if __name__ == "__main__":
    unittest.main()
