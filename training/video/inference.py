import torch, os, argparse, glob
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from diffsynth.utils.data import save_video, VideoData
from PIL import Image

# Default Offline Paths
MODEL_BASE_PATH = "/inspire/hdd/project/embodied-multimodality/public/downloaded_ckpts/Wan2.2-TI2V-5B"
TOKENIZER_PATH = f"{MODEL_BASE_PATH}/google/umt5-xxl"

def parse_args():
    parser = argparse.ArgumentParser(description="Wan2.2 LoRA Inference Script")
    parser.add_argument("--lora_ckpt", type=str, default="/inspire/hdd/project/embodied-multimodality/tongjingqi-CZXS25110029/chj_code/Wan2.2-Train/output/maze_square/step-10000.safetensors", help="Path to LoRA checkpoint (.safetensors)")
    parser.add_argument("--prompt", type=str, default="Draw a red path connecting two red dots without touching the black walls. Static camera.", help="Text prompt")
    parser.add_argument("--negative_prompt", type=str, default="", help="Negative prompt")
    parser.add_argument("--input_image", type=str, default="/inspire/hdd/project/embodied-multimodality/public/hjchen/MazeSquare/puzzles/5e2d0bfc-2305-4210-b327-a031831fd507_puzzle.png", help="Input image path (optional, for I2V)")
    parser.add_argument("--height", type=int, default=896, help="Video height (must be multiple of 32)")
    parser.add_argument("--width", type=int, default=480, help="Video width (must be multiple of 32)")
    parser.add_argument("--num_frames", type=int, default=81, help="Number of frames ((n-1)%4==0)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default="output.mp4", help="Output filename")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    
    # Advanced paths
    parser.add_argument("--model_base_path", type=str, default=MODEL_BASE_PATH, help="Base model path")
    parser.add_argument("--tokenizer_path", type=str, default=TOKENIZER_PATH, help="Tokenizer path")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"Initializing Pipeline...")
    print(f"  Model Base: {args.model_base_path}")
    print(f"  Tokenizer: {args.tokenizer_path}")
    
    # Resolve split checkpoint files for DiT
    dit_files = sorted(glob.glob(os.path.join(args.model_base_path, "diffusion_pytorch_model*.safetensors")))
    if not dit_files:
        print("Warning: No diffusion checkpoints found via glob! Check path.")

    # 1. Initialize Pipeline (Offline Mode - Explicit Paths)
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=args.device,
        model_configs=[
            ModelConfig(path=os.path.join(args.model_base_path, "models_t5_umt5-xxl-enc-bf16.pth")),
            ModelConfig(path=dit_files),
            ModelConfig(path=os.path.join(args.model_base_path, "Wan2.2_VAE.pth")),
        ],
        tokenizer_config=ModelConfig(path=args.tokenizer_path),
        audio_processor_config=None, # Disable audio processor download
    )
    
    # 2. Load LoRA
    if args.lora_ckpt:
        print(f"Loading LoRA from: {args.lora_ckpt}")
        pipe.load_lora(pipe.dit, args.lora_ckpt, alpha=1.0)
    
    # 3. Prepare Inputs
    input_image = None
    if args.input_image:
        print(f"Loading input image: {args.input_image}")
        # Wan2.2-TI2V usually takes a PIL image as conditioning
        input_image = Image.open(args.input_image).convert("RGB")
        # Resize to target resolution to avoid internal resizing mismatch
        input_image = input_image.resize((args.width, args.height))
    
    # 4. Generate
    print(f"Generating video...")
    print(f"  Prompt: {args.prompt}")
    print(f"  Resolution: {args.width}x{args.height}")
    print(f"  Frames: {args.num_frames}")
    
    video = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        input_image=input_image, 
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        seed=args.seed,
        tiled=True, # Enable tiled decoding for memory efficiency
    )
    
    # 5. Save
    print(f"Saving to {args.output}")
    save_video(video, args.output, fps=15, quality=5)
    print("Done!")

if __name__ == "__main__":
    main()
