import os
import sys
import glob
import json
import argparse
import logging
import multiprocessing
from pathlib import Path
from typing import Optional

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from diffsynth.utils.data import save_video

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.puzzle.maze.maze_square.evaluator import MazeEvaluator

DEFAULT_MODEL_BASE_PATH = os.environ.get(
    "WAN_MODEL_BASE_PATH",
    "/inspire/hdd/project/embodied-multimodality/public/downloaded_ckpts/Wan2.2-TI2V-5B",
)
DEFAULT_PROMPT = "Draw a red path connecting two red dots without touching the black walls. Static camera."

pipe = None
evaluator = None


def setup_logger(output_dir: str) -> None:
    logging.basicConfig(
        filename=os.path.join(output_dir, "evaluation.log"),
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger("").addHandler(console)


def init_worker(gpu_id: int, model_base_path: str, tokenizer_path: str, lora_ckpt: Optional[str], metadata_path: Optional[str]):
    global pipe, evaluator
    device = f"cuda:{gpu_id}"
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

    if lora_ckpt and os.path.exists(lora_ckpt):
        pipe.load_lora(pipe.dit, lora_ckpt, alpha=1.0)

    if metadata_path and os.path.exists(metadata_path):
        evaluator = MazeEvaluator(metadata_path, base_dir=os.path.dirname(metadata_path))


def process_item(args):
    puzzle_path, output_dir, prompt, width, height, num_frames = args
    global pipe, evaluator

    puzzle_id = os.path.basename(puzzle_path).replace("_puzzle.png", "")
    video_filename = f"{puzzle_id}_solution.mp4"
    video_path = os.path.join(output_dir, video_filename)
    frame_path = os.path.join(output_dir, f"{puzzle_id}_last_frame.png")

    try:
        if not (os.path.exists(video_path) and os.path.exists(frame_path)):
            input_image = Image.open(puzzle_path).convert("RGB").resize((width, height), resample=Image.NEAREST)
            video = pipe(
                prompt=prompt,
                negative_prompt="",
                input_image=input_image,
                num_frames=num_frames,
                height=height,
                width=width,
                seed=42,
                tiled=True,
            )
            save_video(video, video_path, fps=15, quality=5)
            if isinstance(video, (list, tuple)) and video:
                video[-1].save(frame_path)

        result_dict = {"status": "generated", "puzzle_id": puzzle_id}
        if evaluator and os.path.exists(frame_path):
            result = evaluator.evaluate(puzzle_id, frame_path)
            result_dict.update({
                "status": "evaluated",
                "connected": result.connected,
                "overlaps_walls": result.overlaps_walls,
                "touches_start": result.touches_start,
                "touches_goal": result.touches_goal,
                "message": result.message,
            })
        return result_dict
    except Exception as e:
        return {"status": "error", "puzzle_id": puzzle_id, "message": str(e)}


def worker_process(gpu_id, puzzle_paths, args, model_base, tokenizer_path, metadata_path, result_queue):
    init_worker(gpu_id, model_base, tokenizer_path, args.lora_ckpt, metadata_path)
    for puzzle_path in puzzle_paths:
        task_args = (puzzle_path, args.output_dir, args.prompt, args.width, args.height, args.num_frames)
        res = process_item(task_args)
        result_queue.put(res)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing *_puzzle.png files")
    parser.add_argument("--output_dir", type=str, default="evaluation_results", help="Output directory")
    parser.add_argument("--lora_ckpt", type=str, default=None, help="Path to LoRA checkpoint (Optional)")
    parser.add_argument(
        "--model_base_path",
        type=str,
        default=None,
        help="Wan2.2 base model path (defaults to WAN_MODEL_BASE_PATH env)",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=None,
        help="Tokenizer path (defaults to <model_base_path>/google/umt5-xxl)",
    )
    parser.add_argument("--gpu_ids", type=str, default="0", help="Comma separated GPU IDs")
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--height", type=int, default=896)
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--metadata_path", type=str, default=None, help="Maze metadata.json (optional)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    setup_logger(args.output_dir)

    puzzle_files = sorted(glob.glob(os.path.join(args.input_dir, "*_puzzle.png")))
    if not puzzle_files:
        print("No puzzles found!")
        return

    metadata_path = args.metadata_path
    if metadata_path is None:
        parent_dir = os.path.dirname(os.path.normpath(args.input_dir))
        candidate_meta = os.path.join(parent_dir, "data.json")
        if os.path.exists(candidate_meta):
            metadata_path = candidate_meta

    model_base = args.model_base_path or DEFAULT_MODEL_BASE_PATH
    tokenizer_path = args.tokenizer_path or os.path.join(model_base, "google/umt5-xxl")

    gpu_list = [int(x) for x in args.gpu_ids.split(",") if x.strip()]
    if not gpu_list:
        raise ValueError("gpu_ids is empty. Please provide at least one GPU id.")
    num_gpus = len(gpu_list)
    chunk_size = int(np.ceil(len(puzzle_files) / num_gpus))
    chunks = [puzzle_files[i:i + chunk_size] for i in range(0, len(puzzle_files), chunk_size)]

    ctx = multiprocessing.get_context('spawn')
    processes = []
    results_queue = ctx.Queue()

    for i, gpu_id in enumerate(gpu_list):
        if i >= len(chunks):
            break
        chunk = chunks[i]
        p = ctx.Process(
            target=worker_process,
            args=(gpu_id, chunk, args, model_base, tokenizer_path, metadata_path, results_queue),
        )
        p.start()
        processes.append(p)

    results = []
    total_tasks = len(puzzle_files)
    with tqdm(total=total_tasks, desc="Evaluating") as pbar:
        for _ in range(total_tasks):
            res = results_queue.get()
            results.append(res)
            pbar.update(1)

    for p in processes:
        p.join()

    summary_path = Path(args.output_dir) / "summary.json"
    summary = {
        "total": total_tasks,
        "evaluated": sum(1 for r in results if r.get("status") == "evaluated"),
        "errors": [r for r in results if r.get("status") == "error"],
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    main()
