import os
import sys
from pathlib import Path
import csv
import glob
import json
import argparse
from typing import Optional

import torch
from PIL import Image
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from diffsynth.utils.data import save_video

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.puzzle.maze.maze_square.evaluator import MazeEvaluator


def build_pipeline(model_base_path: str, tokenizer_path: str, device: str, lora_ckpt: Optional[str]) -> WanVideoPipeline:
    dit_files = sorted(glob.glob(os.path.join(model_base_path, "diffusion_pytorch_model*.safetensors")))
    if not dit_files:
        raise FileNotFoundError(f"No diffusion checkpoints found under: {model_base_path}")

    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=[
            ModelConfig(path=os.path.join(model_base_path, "models_t5_umt5-xxl-enc-bf16.pth")),
            ModelConfig(path=dit_files),
            ModelConfig(path=os.path.join(model_base_path, "Wan2.2_VAE.pth")),
        ],
        tokenizer_config=ModelConfig(path=tokenizer_path),
        audio_processor_config=None,
    )

    if lora_ckpt:
        pipe.load_lora(pipe.dit, lora_ckpt, alpha=1.0)

    return pipe


def load_first_frame(video_path: str, width: int, height: int) -> Optional[Image.Image]:
    try:
        import cv2
    except ImportError:
        print("OpenCV 未安装，无法读取视频首帧。请安装 opencv-python 或提供 --input_image。")
        return None

    cap = cv2.VideoCapture(video_path)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame)
    if width and height:
        image = image.resize((width, height), resample=Image.NEAREST)
    return image


def parse_puzzle_id(row: dict) -> Optional[str]:
    for key in ("puzzle_id", "id"):
        if row.get(key):
            return str(row.get(key))
    video_path = row.get('video', '')
    if not video_path:
        return None
    stem = Path(video_path).stem
    for suffix in ("_solution", "_solution_video", "_video"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return stem


def main():
    parser = argparse.ArgumentParser(description="Wan2.2 LoRA 验证脚本（可选 Maze 评测）")
    parser.add_argument("--dataset", type=str, required=True, help="CSV path (video,prompt)")
    parser.add_argument("--lora_ckpt", type=str, default=None, help="LoRA checkpoint")
    parser.add_argument("--model_base_path", type=str, required=True, help="Wan2.2 base model path")
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Tokenizer path")
    parser.add_argument("--metadata_path", type=str, default=None, help="Maze metadata.json (optional)")
    parser.add_argument("--output_dir", type=str, default="outputs/validate_video", help="Output directory")
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--height", type=int, default=896)
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--input_image", type=str, default=None, help="Optional fixed input image")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    tokenizer_path = args.tokenizer_path or os.path.join(args.model_base_path, "google/umt5-xxl")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pipe = build_pipeline(args.model_base_path, tokenizer_path, args.device, args.lora_ckpt)

    evaluator = None
    if args.metadata_path:
        meta_path = Path(args.metadata_path)
        if meta_path.exists():
            evaluator = MazeEvaluator(meta_path, base_dir=meta_path.parent)

    with dataset_path.open('r', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        rows = [row for _, row in zip(range(args.num_samples), reader)]

    results_path = output_dir / "validate.jsonl"
    total = 0
    valid = 0

    with results_path.open('w', encoding='utf-8') as handle:
        for idx, row in enumerate(rows, start=1):
            video_path = row.get('video', '').strip()
            prompt = row.get('prompt', '').strip()
            if not video_path or not prompt:
                continue

            puzzle_id = parse_puzzle_id(row)
            if args.input_image:
                input_image = Image.open(args.input_image).convert("RGB").resize((args.width, args.height))
            else:
                input_image = load_first_frame(video_path, args.width, args.height)
                if input_image is None:
                    continue

            video = pipe(
                prompt=prompt,
                negative_prompt="",
                input_image=input_image,
                num_frames=args.num_frames,
                height=args.height,
                width=args.width,
                seed=args.seed,
                tiled=True,
            )

            sample_dir = output_dir / f"sample_{idx:02d}"
            sample_dir.mkdir(parents=True, exist_ok=True)
            video_path_out = sample_dir / "generated.mp4"
            save_video(video, video_path_out.as_posix(), fps=15, quality=5)

            last_frame_path = sample_dir / "last_frame.png"
            if isinstance(video, (list, tuple)) and video:
                video[-1].save(last_frame_path)

            eval_result = None
            if evaluator and puzzle_id and last_frame_path.exists():
                try:
                    eval_result = evaluator.evaluate(puzzle_id, last_frame_path)
                except Exception as exc:
                    eval_result = {"error": str(exc)}

            payload = {
                "puzzle_id": puzzle_id,
                "video": video_path,
                "prompt": prompt,
                "generated_video": video_path_out.as_posix(),
            }
            if eval_result is not None:
                if hasattr(eval_result, "to_dict"):
                    payload["evaluation"] = eval_result.to_dict()
                    if not eval_result.overlaps_walls and eval_result.touches_start and eval_result.touches_goal and eval_result.connected:
                        valid += 1
                else:
                    payload["evaluation"] = eval_result
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            total += 1

    accuracy = (valid / total) if total else 0.0
    print("=" * 60)
    print(f"Total: {total}")
    if evaluator:
        print(f"Valid Maze Paths: {valid} ({accuracy:.1%})")
    print(f"Results saved to: {results_path}")
    print("=" * 60)


if __name__ == '__main__':
    main()
