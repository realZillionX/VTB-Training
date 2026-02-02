import json
import tempfile
import unittest
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.tools import prepare_image_data, prepare_vlm_data, prepare_video_data


class DataToolsTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmpdir.name)
        self._build_dataset()

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def _write_json(self, path: Path, payload: object) -> None:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _touch(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")

    def _build_dataset(self) -> None:
        maze_dir = self.root / "maze_square"
        maze_dir.mkdir(parents=True, exist_ok=True)
        self._touch(maze_dir / "puzzle.png")
        self._touch(maze_dir / "solution.png")
        self._touch(maze_dir / "solution.mp4")

        maze_record = {
            "id": "maze1",
            "task_type": "maze_square",
            "prompt": "Draw a red path connecting two red dots.",
            "gpt5_prompt": "Find the path.",
            "image": "puzzle.png",
            "solution_image_path": "solution.png",
            "solution_video_path": "solution.mp4",
            "solution_path_cell_ids": [1, 2, 3],
        }
        self._write_json(maze_dir / "data.json", [maze_record])

        eye_dir = self.root / "circle_center"
        eye_dir.mkdir(parents=True, exist_ok=True)
        self._touch(eye_dir / "puzzle.png")
        self._touch(eye_dir / "solution.png")

        eye_record = {
            "id": "eye1",
            "task_type": "circle_center",
            "prompt": "Which point looks like the center of the circle? In portrait. Static camera.",
            "image": "puzzle.png",
            "solution_image_path": "solution.png",
            "correct_option": "B",
        }
        self._write_json(eye_dir / "data.json", [eye_record])

    def test_prepare_vlm_data(self) -> None:
        entries = prepare_vlm_data.collect_entries(
            self.root,
            task_groups=["eyeballing", "maze"],
            mode="sft",
            path_mode="relative",
        )
        self.assertEqual(len(entries), 2)

        maze_entry = next(entry for entry in entries if entry["task_group"] == "maze")
        self.assertEqual(maze_entry["solution"], "[1, 2, 3]")
        self.assertEqual(len(maze_entry["messages"]), 2)
        self.assertTrue(maze_entry["images"][0].startswith("maze_square/"))

        eye_entry = next(entry for entry in entries if entry["task_group"] == "eyeballing")
        self.assertEqual(eye_entry["solution"], "B")
        self.assertTrue(eye_entry["images"][0].startswith("circle_center/"))

    def test_prepare_image_data(self) -> None:
        items = prepare_image_data.collect_items(
            self.root,
            task_groups=["eyeballing", "maze"],
            path_mode="relative",
        )
        self.assertEqual(len(items), 2)

        maze_item = next(item for item in items if item["task_type"] == "maze_square")
        self.assertEqual(
            maze_item["prompt"],
            "Draw a red path connecting two red dots without touching the black walls.",
        )

        eye_item = next(item for item in items if item["task_type"] == "circle_center")
        self.assertNotIn("Static camera", eye_item["prompt"])

    def test_prepare_video_data(self) -> None:
        rows = prepare_video_data.collect_rows(
            self.root,
            task_groups=["eyeballing", "maze"],
            path_mode="relative",
        )
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["prompt"], "Draw a red path connecting two red dots without touching the black walls.")
        self.assertEqual(row["video"], "maze_square/solution.mp4")


if __name__ == "__main__":
    unittest.main()
