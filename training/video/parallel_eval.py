
import os
import glob
import json
import torch
import argparse
import logging
import multiprocessing
import numpy as np
from PIL import Image
from tqdm import tqdm
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from diffsynth.utils.data import save_video

# ================= VLMPuzzle Import Setup =================
import sys
# Server path to VLMPuzzle
VLMPUZZLE_PATH = "/inspire/hdd/project/embodied-multimodality/tongjingqi-CZXS25110029/chj_code/VLMPuzzle"
sys.path.append(VLMPUZZLE_PATH)
try:
    from puzzle.maze_square.evaluator import MazeEvaluator
except ImportError:
    # Fallback or try adding subdirectory if needed, but sys.path.append should work
    pass

# ================= Configuration =================
MODEL_BASE_PATH = "/inspire/hdd/project/embodied-multimodality/public/downloaded_ckpts/Wan2.2-TI2V-5B"
TOKENIZER_PATH = f"{MODEL_BASE_PATH}/google/umt5-xxl"
DEFAULT_PROMPT = "Draw a red path connecting two red dots without touching the black walls. Static camera."

# Global worker variables
pipe = None
evaluator = None

def setup_logger(output_dir):
    logging.basicConfig(
        filename=os.path.join(output_dir, "evaluation.log"),
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)

def init_worker(gpu_id, model_base_path, tokenizer_path, lora_ckpt, metadata_path):
    global pipe, evaluator
    device = f"cuda:{gpu_id}"
    print(f"[Worker {gpu_id}] Initializing on {device}...")

    # 1. Initialize Pipeline
    dit_files = sorted(glob.glob(os.path.join(model_base_path, "diffusion_pytorch_model*.safetensors")))
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

    # 2. Load LoRA
    if lora_ckpt and os.path.exists(lora_ckpt):
        print(f"[Worker {gpu_id}] Loading LoRA: {lora_ckpt}")
        pipe.load_lora(pipe.dit, lora_ckpt, alpha=1.0)
    
# ... (Imports remain the same)

class DebugMazeEvaluator(MazeEvaluator):
    def evaluate(self, puzzle_id, candidate_path):
        # 1. Run standard evaluation (this might report false positive overlaps)
        try:
            result = super().evaluate(puzzle_id, candidate_path)
        except Exception as e:
            print(f"Evaluation failed internally for {puzzle_id}: {e}")
            raise e

        # 2. Patch Logic: Ignore overlap at start/end points due to resize artifacts
        if result.overlaps_walls:
            # We need to re-verify if the overlap is ONLY at the endpoints
            candidate_path = os.path.abspath(candidate_path)
            record = self.get_record(puzzle_id)
            
            # Load images again to compute masks
            with Image.open(candidate_path) as cand_image:
                candidate_image_rgb = cand_image.convert("RGB")
            candidate_pixels = np.array(candidate_image_rgb)
            
            source_path = self.resolve_path(record["image"])
            with Image.open(source_path) as src_image:
                puzzle_image = src_image.convert("RGB")
            
            if candidate_image_rgb.size != puzzle_image.size:
                 puzzle_image = puzzle_image.resize(candidate_image_rgb.size, Image.NEAREST)
            puzzle_pixels = np.array(puzzle_image)

            # Re-compute masks
            red_mask = self._red_mask(candidate_pixels)
            wall_mask = self._wall_mask(puzzle_pixels)
            
            # Identify Start/End locations (pixels)
            try:
                # We need the scaled coordinates
                source_size = src_image.size # Original size before resize
                target_size = candidate_image_rgb.size
                
                start_pt = self._resolve_endpoint(record, "start", source_size, target_size)
                goal_pt = self._resolve_endpoint(record, "goal", source_size, target_size)
                
                # Create a "Safe Mask" (Ignore overlaps in these regions)
                safe_radius = 8 # 8 pixels tolerance
                h, w = red_mask.shape
                y_grid, x_grid = np.ogrid[:h, :w]
                
                # Dist from start
                dist_start = np.sqrt((x_grid - start_pt[0])**2 + (y_grid - start_pt[1])**2)
                # Dist from goal
                dist_goal = np.sqrt((x_grid - goal_pt[0])**2 + (y_grid - goal_pt[1])**2)
                
                safe_mask = (dist_start <= safe_radius) | (dist_goal <= safe_radius)
                
                # Actual collision = (Red & Wall) AND (NOT Safe Zone)
                true_overlap = (red_mask & wall_mask) & (~safe_mask)
                
                if not np.any(true_overlap):
                    # False Alarm! It was just start/end clipping.
                    print(f"Puzzle {puzzle_id}: Initial overlap ignored (Safe Zone logic).")
                    result.overlaps_walls = False
                    # Re-evaluate validity based on new overlap status
                    # Is it connected? The original 'connected' check might have failed if it relies on 'overlaps_walls' logic?
                    # No, maze_base.py checks connected independent of overlaps_walls in _connected call, 
                    # BUT the final .connected attr implies logic "touches_start and touches_goal and not overlaps_walls" in some versions?
                    # Checking maze_base.py: 
                    # connected = False
                    # if touches_start and touches_goal and not overlaps_walls:
                    #    connected = self._connected(...)
                    
                    # Ah! If overlap was True, 'connected' logic was skipped in base class!
                    # So we MUST re-run connectivity check if we overturned the overlap verdict.
                    
                    start_seed = self._nearest_red(red_mask, start_pt)
                    goal_seed = self._nearest_red(red_mask, goal_pt)
                    touches_start = start_seed is not None
                    touches_goal = goal_seed is not None
                    
                    if touches_start and touches_goal:
                         # Re-run connectivity check (private method _connected)
                         is_connected = self._connected(red_mask, start_seed, goal_seed)
                         result.connected = is_connected
                         result.touches_start = True
                         result.touches_goal = True
                         if is_connected:
                            result.message = "Red path successfully connects start to goal (Start/End overlap ignored)."
                         else:
                            result.message = "Red path is free of walls but not continuous."
                    else:
                        result.message = "Overlap ignored, but does not touch start/goal."
                        
            except Exception as patch_e:
                print(f"Warning: Failed to apply safe-zone patch for {puzzle_id}: {patch_e}")

        # 3. Debug Visualization (Only if failed or forced)
        # We'll save debug images to a 'debug' folder inside the candidate's directory
        candidate_path = os.path.abspath(candidate_path)
        debug_dir = os.path.join(os.path.dirname(candidate_path), "debug_viz")
        os.makedirs(debug_dir, exist_ok=True)
        
        # Re-load images (if not loaded in patch block) to generate masks
        # To avoid double-loading, check local locals or just reload. Optimization for now: Just structure clearly.
        # For debug logic, let's keep it simple and mostly separate or reuse if variables exist.
        
        # Load Candidate (if not already)
        if 'candidate_pixels' not in locals():
            with Image.open(candidate_path) as cand_image:
                candidate_image_rgb = cand_image.convert("RGB")
            candidate_pixels = np.array(candidate_image_rgb)
        
        if 'red_mask' not in locals():
             red_mask = self._red_mask(candidate_pixels)
             
        if 'wall_mask' not in locals():
            # Need puzzle pixels
            if 'puzzle_pixels' not in locals():
                record = self.get_record(puzzle_id)
                source_path = self.resolve_path(record["image"])
                with Image.open(source_path) as src_image:
                    puzzle_image = src_image.convert("RGB")
                if candidate_pixels.shape[:2] != (puzzle_image.height, puzzle_image.width): # Numpy is H,W
                     # PIL size is W,H
                     scaled_size = (candidate_pixels.shape[1], candidate_pixels.shape[0])
                     puzzle_image = puzzle_image.resize(scaled_size, Image.NEAREST)
                puzzle_pixels = np.array(puzzle_image)
            wall_mask = self._wall_mask(puzzle_pixels)
        
        # Create Visualization
        h, w = red_mask.shape
        debug_img = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Background: Dimmed Original
        debug_img = (candidate_pixels * 0.3).astype(np.uint8)
        
        # Highlight Red Detection (Green for visibility)
        debug_img[red_mask] = [0, 255, 0] 
        
        # Highlight Walls (Blue)
        debug_img[wall_mask] = [0, 0, 255]
        
        # Highlight Overlap (Red - Danger!)
        overlap = red_mask & wall_mask
        debug_img[overlap] = [255, 0, 0]
        
        # Draw Safe Zones if applied
        if 'safe_mask' in locals():
             # Draw Safe Zone in Yellow (faint)
             debug_img[safe_mask & overlap] = [255, 255, 0] # Overlaps in safe zone -> Yellow

        # Save
        # Recalculate 'valid' status based on potentially modified result
        is_valid = result.connected and not result.overlaps_walls
        save_name = f"{puzzle_id}_status_{'PASS' if is_valid else 'FAIL'}.png"
        Image.fromarray(debug_img).save(os.path.join(debug_dir, save_name))
        
        return result

def init_worker(gpu_id, model_base_path, tokenizer_path, lora_ckpt, metadata_path):
    global pipe, evaluator
    device = f"cuda:{gpu_id}"
    print(f"[Worker {gpu_id}] Initializing on {device}...")

    # ... (Pipeline Init remains same) ...

    # 3. Initialize Evaluator
    if metadata_path and os.path.exists(metadata_path):
        try:
           # Use DebugEvaluator for visualization
           evaluator = DebugMazeEvaluator(metadata_path, base_dir=os.path.dirname(metadata_path))
        except Exception as e:
           print(f"[Worker {gpu_id}] Warning: Failed to init evaluator: {e}")
           evaluator = None

def process_item(args):
    """
    Args:
        puzzle_path: Path to input image
        output_dir: Directory to save results
        prompt: Text prompt
        gpu_id: Assigned GPU
    """
    puzzle_path, output_dir, prompt, width, height, num_frames = args
    global pipe, evaluator
    
    puzzle_id = os.path.basename(puzzle_path).replace("_puzzle.png", "")
    video_filename = f"{puzzle_id}_solution.mp4"
    video_path = os.path.join(output_dir, video_filename)
    frame_path = os.path.join(output_dir, f"{puzzle_id}_last_frame.png")
    
    # 1. Inference
    try:
        input_image = Image.open(puzzle_path).convert("RGB")
        # Use NEAREST to avoid anti-aliasing blurring walls, matching Evaluator logic
        input_image = input_image.resize((width, height), resample=Image.NEAREST)
        
        video = None
        
        # Check if video already exists
        if os.path.exists(video_path):
            print(f"Skipping generation for {puzzle_id}, video exists.")
            # We still need to load it or at least ensure last frame exists for evaluation
            # If we don't load the video, we can't get the last frame easily without loading it.
            # However, if last_frame also exists, we can skip that too.
        else:
            video = pipe(
                prompt=prompt,
                negative_prompt="",
                input_image=input_image,
                num_frames=num_frames,
                height=height,
                width=width,
                seed=42,
                tiled=True
            )
            save_video(video, video_path, fps=15, quality=5)
        
        # 2. Extract Last Frame for Evaluation
        if not os.path.exists(frame_path):
            if video is None:
               # Need to load video to extract last frame? Or just error out?
               # For efficiency, let's just attempt to load the video using generic tool if needed, 
               # OR just assume if video exists, last frame *should* exist from previous run.
               # But if it doesn't, we must extract it.
               # Loading video is slow. Let's try to load just the last frame?
               pass
               # To be safe and simple: if frame missing, we must generate or load video.
               # If video exists but frame missing, we can try to extract frame from video file?
               # That requires OpenCV or MoviePy. 
               # Let's keep it simple: Only skip if BOTH exist.
               pass 

        # Refining Logic:
        # If video_path AND frame_path exist -> Skip Generation AND Extraction -> Go to Eval
        # If video_path exists but frame_path missing -> Hard to extract without video object from pipe (which is list of PIL).
        # Re-reading video file to PIL list is heavy.
        # Decision: Only skip if BOTH video and frame exist.
        
        if os.path.exists(video_path) and os.path.exists(frame_path):
             pass # Skip generation and extraction
        else:
             # Run Pipeline
             if not os.path.exists(video_path):
                 video = pipe(
                    prompt=prompt,
                    negative_prompt="",
                    input_image=input_image,
                    num_frames=num_frames,
                    height=height,
                    width=width,
                    seed=42,
                    tiled=True
                )
                 save_video(video, video_path, fps=15, quality=5)
             
             # If video was just generated, 'video' is valid list of PIL.
             # If video existed but frame didn't, we have a problem unless we load it.
             # For now, let's assume if video exists, we re-generate if frame is missing? 
             # Or better: pipe returns PIL list. 
             # Let's force regeneration if frame is missing to avoid complexity of loading video.
             # So: Condition to skip = Video Exists AND Frame Exists.
             
             if video:
                last_frame = video[-1] 
                last_frame.save(frame_path)
        
        # 3. Evaluate
        result_dict = {}
        if evaluator:
            try:
                # MazeEvaluator.evaluate(record_id, candidate_path)
                result = evaluator.evaluate(puzzle_id, frame_path)
                # result is MazeEvaluationResult object
                result_dict = {
                    'status': 'success',
                    'puzzle_id': puzzle_id,
                    'connected': result.connected,
                    'hit_wall': result.overlaps_walls, # Map to overlaps_walls
                    'is_valid': result.connected and not result.overlaps_walls, # Derived valid status
                    'message': result.message
                }
            except Exception as e:
                result_dict = {'status': 'eval_error', 'message': str(e), 'puzzle_id': puzzle_id}
        else:
             result_dict = {'status': 'no_evaluator', 'puzzle_id': puzzle_id}

        return result_dict

    except Exception as e:
        print(f"Error processing {puzzle_id}: {e}")
        return {'status': 'error', 'message': str(e), 'puzzle_id': puzzle_id}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing _puzzle.png files")
    parser.add_argument("--output_dir", type=str, default="evaluation_results", help="Output directory")
    parser.add_argument("--lora_ckpt", type=str, default=None, help="Path to LoRA checkpoint (Optional)")
    parser.add_argument("--gpu_ids", type=str, default="0,1,2,3,4,5,6,7", help="Comma separated GPU IDs")
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--height", type=int, default=896)
    parser.add_argument("--num_frames", type=int, default=81)
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    setup_logger(args.output_dir)
    
    # Locate Data
    puzzle_files = sorted(glob.glob(os.path.join(args.input_dir, "*_puzzle.png")))
    if not puzzle_files:
        print("No puzzles found!")
        return
        
    print(f"Found {len(puzzle_files)} puzzles.")
    
    # Locate Metadata (Assume data.json is in the parent of input_dir)
    # input_dir: .../mazes/maze_square/puzzles
    # metadata: .../mazes/maze_square/data.json
    parent_dir = os.path.dirname(os.path.normpath(args.input_dir))
    metadata_path = os.path.join(parent_dir, "data.json")
    if not os.path.exists(metadata_path):
        print(f"Warning: Metadata not found at {metadata_path}. Evaluation will be skipped (only generation).")
        metadata_path = None
    else:
        print(f"Using metadata: {metadata_path}")
    
    # Prepare Workers
    gpu_list = [int(x) for x in args.gpu_ids.split(",")]
    num_gpus = len(gpu_list)
    
    # Manual Process spawning for exact GPU control
    ctx = multiprocessing.get_context('spawn')
    
    # Manual Process spawning for exact GPU control
    chunk_size = int(np.ceil(len(puzzle_files) / num_gpus))
    chunks = [puzzle_files[i:i + chunk_size] for i in range(0, len(puzzle_files), chunk_size)]
    
    processes = []
    results_queue = ctx.Queue()
    
    for i, gpu_id in enumerate(gpu_list):
        if i >= len(chunks): break
        chunk = chunks[i]
        p = ctx.Process(
            target=worker_process,
            args=(gpu_id, chunk, args, MODEL_BASE_PATH, TOKENIZER_PATH, metadata_path, results_queue)
        )
        p.start()
        processes.append(p)
        print(f"Started worker for GPU {gpu_id} with {len(chunk)} tasks.")
    
    # Collect results
    results = []
    total_tasks = len(puzzle_files)
    with tqdm(total=total_tasks, desc="Evaluating") as pbar:
        for _ in range(total_tasks):
            res = results_queue.get()
            results.append(res)
            pbar.update(1)
            
            if res.get('status') == 'success':
                # Log detailed reason even if technical status is success (but logical failure)
                is_valid = res.get('is_valid', False)
                msg = f"Puzzle {res['puzzle_id']}: Valid={is_valid}, Connected={res.get('connected')}, HitWall={res.get('hit_wall')}, Msg='{res.get('message')}'"
                if is_valid:
                    logging.info(f"[SUCCESS] {msg}")
                else:
                    logging.warning(f"[FAILURE] {msg}")
            else:
                logging.error(f"[ERROR] Failed {res.get('puzzle_id')}: {res.get('message')}")

    for p in processes:
        p.join()
        
    # Calculate Stats
    process_stats(results)

def worker_process(gpu_id, puzzle_paths, args, model_base, tokenizer_path, metadata_path, result_queue):
    init_worker(gpu_id, model_base, tokenizer_path, args.lora_ckpt, metadata_path)
    
    for puzzle_path in puzzle_paths:
        task_args = (puzzle_path, args.output_dir, args.prompt, args.width, args.height, args.num_frames)
        res = process_item(task_args)
        result_queue.put(res)

def process_stats(results):
    total = len(results)
    success = 0
    connected = 0
    
    for r in results:
        if r.get('status') == 'success':
            success += 1
            if r.get('connected'):
                connected += 1
                
    accuracy = (connected / total) * 100 if total > 0 else 0
    print("\n" + "="*40)
    print(f"Total Puzzles: {total}")
    print(f"Successful Gens: {success}")
    print(f"Connected Paths: {connected}")
    print(f"Accuracy: {accuracy:.2f}%")
    print("="*40)

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    main()
