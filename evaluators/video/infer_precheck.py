import os
import csv
import glob
import argparse
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from diffsynth.utils.data import save_video


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


def main():
    parser = argparse.ArgumentParser(description="Wan2.2 LoRA 推理预检（CSV）")
    parser.add_argument("--dataset", type=str, required=True, help="CSV path (video,prompt)")
    parser.add_argument("--lora_ckpt", type=str, default=None, help="LoRA checkpoint")
    parser.add_argument("--model_base_path", type=str, required=True, help="Wan2.2 base model path")
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Tokenizer path (default: <model_base_path>/google/umt5-xxl)")
    parser.add_argument("--output_dir", type=str, default="outputs/precheck_video", help="Output directory")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples")
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

    with dataset_path.open('r', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        rows = [row for _, row in zip(range(args.num_samples), reader)]

    for idx, row in enumerate(rows, start=1):
        video_path = row.get('video', '').strip()
        prompt = row.get('prompt', '').strip()
        if not video_path or not prompt:
            print(f"[{idx}] 跳过空记录")
            continue

        print(f"[{idx}/{len(rows)}] {video_path}")
        if args.input_image:
            input_image = Image.open(args.input_image).convert("RGB").resize((args.width, args.height))
        else:
            input_image = load_first_frame(video_path, args.width, args.height)
            if input_image is None:
                print("  无法读取首帧，跳过。")
                continue

        result = pipe(
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
        save_video(result, (sample_dir / "generated.mp4").as_posix(), fps=15, quality=5)
        input_image.save(sample_dir / "input.png")
        (sample_dir / "info.txt").write_text(
            f"video={video_path}\nprompt={prompt}\n", encoding="utf-8"
        )

    print(f"Done. Outputs saved to: {output_dir}")


if __name__ == '__main__':
    main()
