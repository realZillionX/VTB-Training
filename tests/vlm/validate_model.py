import os
import sys
from pathlib import Path
# Force offline mode
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import json
from swift.llm import get_model_tokenizer, get_template, inference
from swift.utils import seed_everything

from training.vlm.rewards.vlm_rewards import reward_eyeballing, reward_maze


def load_samples(path: Path, max_samples: int) -> list[dict]:
    samples = []
    with path.open('r', encoding='utf-8') as handle:
        for idx, line in enumerate(handle):
            if idx >= max_samples:
                break
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))
    return samples


def extract_query(sample: dict) -> tuple[str, list]:
    if 'messages' in sample:
        messages = sample.get('messages') or []
        query = messages[0].get('content', '') if messages else ''
        images = sample.get('images', [])
        return query, images
    return sample.get('query', ''), sample.get('images', [])


def score_prediction(prediction: str, solution: str) -> float:
    sol = solution.strip()
    if sol.startswith('[') and sol.endswith(']'):
        return reward_maze([prediction], [sol])[0]
    return reward_eyeballing([prediction], [sol])[0]


def main():
    parser = argparse.ArgumentParser(description='Validate Qwen3-VL on prepared JSONL data')
    parser.add_argument('--model_path', type=str, required=True, help='Path to Qwen3-VL model weights')
    parser.add_argument('--data_path', type=str, default='train_sft.jsonl', help='Prepared JSONL path')
    parser.add_argument('--output_dir', type=str, default='output/validate_vlm', help='Output directory')
    parser.add_argument('--num_samples', type=int, default=50, help='Number of samples to validate')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device string for model loading')
    args = parser.parse_args()

    data_path = Path(args.data_path)
    if not data_path.exists():
        raise FileNotFoundError(f'Data file not found: {data_path}')

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = load_samples(data_path, args.num_samples)
    print(f'Loaded {len(samples)} samples from {data_path}')

    print(f'Loading model from {args.model_path}...')
    model, tokenizer = get_model_tokenizer(args.model_path, model_kwargs={'device_map': args.device})
    template = get_template(model.model_meta.template, tokenizer)
    seed_everything(42)

    results_path = output_dir / 'validate.jsonl'
    total = 0
    correct = 0
    scores = []

    with results_path.open('w', encoding='utf-8') as handle:
        for sample in samples:
            query, images = extract_query(sample)
            solution = sample.get('solution', sample.get('response', '')).strip()
            prediction, _ = inference(model, template, query, images=images)
            score = score_prediction(prediction, solution)
            scores.append(score)
            total += 1
            if score >= 0.999:
                correct += 1

            payload = {
                'query': query,
                'images': images,
                'solution': solution,
                'prediction': prediction,
                'score': score,
                'task_type': sample.get('task_type'),
                'task_group': sample.get('task_group'),
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + '\n')

    accuracy = (correct / total) if total else 0.0
    avg_score = sum(scores) / len(scores) if scores else 0.0

    print('=' * 60)
    print(f'Total: {total}')
    print(f'Exact Match: {correct} ({accuracy:.1%})')
    print(f'Avg Score: {avg_score:.3f}')
    print(f'Results saved to: {results_path}')
    print('=' * 60)


if __name__ == '__main__':
    main()
