import argparse
import json
import os
from typing import Dict, List, Optional

import numpy as np

from matpy import add_blind_gaussian_noise, add_gaussian_noise, add_mixture_noise, evaluate_metrics
from matpy.io_utils import load_mat_hsi, normalize_hsi, save_mat_hsi
from matpy.model_eval import evaluate_model_on_dir, infer_on_hsi, load_pretrained_qrnn3d


def iter_mat_files(dataset_dir: str) -> List[str]:
    files: List[str] = []
    for root, _, names in os.walk(dataset_dir):
        for name in sorted(names):
            if name.lower().endswith('.mat'):
                files.append(os.path.join(root, name))
    return files


def generate_noisy_dataset(
    clean_dir: str,
    out_dir: str,
    noise_type: str = 'gaussian',
    sigma: Optional[float] = 25.0,
    seed: int = 0,
) -> List[str]:
    os.makedirs(out_dir, exist_ok=True)
    generated: List[str] = []

    for path in iter_mat_files(clean_dir):
        gt = load_mat_hsi(path, key='rad')
        gt = normalize_hsi(gt)

        if noise_type == 'gaussian':
            noisy = add_gaussian_noise(gt, sigma=sigma or 25.0, seed=seed)
        elif noise_type == 'blind_gaussian':
            noisy = add_blind_gaussian_noise(gt, low=10.0, high=70.0, seed=seed)
        elif noise_type == 'mixture':
            noisy = add_mixture_noise(gt, sigma=sigma or 30.0, seed=seed)
        else:
            raise ValueError(f'Unsupported noise_type: {noise_type}')

        name = os.path.basename(path)
        out_path = os.path.join(out_dir, name)
        save_mat_hsi(out_path, gt, noisy, sigma=sigma or 25.0)
        generated.append(out_path)

    return generated


def predict_dataset(noisy_dir: str, model_path: str, out_dir: str) -> List[str]:
    os.makedirs(out_dir, exist_ok=True)
    model, device = load_pretrained_qrnn3d(model_path)
    generated: List[str] = []

    for path in iter_mat_files(noisy_dir):
        noisy = load_mat_hsi(path, key='input')
        pred = infer_on_hsi(model, noisy, device)

        name = os.path.basename(path)
        out_path = os.path.join(out_dir, name)
        save_mat_hsi(out_path, pred, pred, sigma=0.0)
        generated.append(out_path)

    return generated


def evaluate_dataset(clean_dir: str, pred_dir: str) -> Dict[str, float]:
    results = []
    for clean_path in iter_mat_files(clean_dir):
        name = os.path.basename(clean_path)
        pred_path = os.path.join(pred_dir, name)
        if not os.path.exists(pred_path):
            continue

        gt = load_mat_hsi(clean_path, key='rad')
        pred = load_mat_hsi(pred_path, key='input')
        metrics = evaluate_metrics(gt, pred)
        results.append(metrics)

    if not results:
        raise ValueError('No matching prediction files found for evaluation.')

    summary = {k: float(np.mean([r[k] for r in results])) for k in results[0].keys()}
    return summary


def full_pipeline(
    clean_dir: str,
    out_dir: str,
    model_path: str,
    noise_type: str = 'gaussian',
    sigma: Optional[float] = 25.0,
    seed: int = 0,
    result_json: Optional[str] = None,
) -> Dict[str, float]:
    noisy_dir = os.path.join(out_dir, 'noisy')
    pred_dir = os.path.join(out_dir, 'pred')

    generate_noisy_dataset(clean_dir, noisy_dir, noise_type=noise_type, sigma=sigma, seed=seed)
    predict_dataset(noisy_dir, model_path, pred_dir)
    summary = evaluate_dataset(clean_dir, pred_dir)

    if result_json is not None:
        os.makedirs(os.path.dirname(result_json) or '.', exist_ok=True)
        with open(result_json, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description='Python-only HSI noise generation, inference, and evaluation helper.')
    parser.add_argument('--clean-dir', type=str, required=True)
    parser.add_argument('--out-dir', type=str, required=True)
    parser.add_argument('--model-path', type=str, default=None)
    parser.add_argument('--noise-type', choices=['gaussian', 'blind_gaussian', 'mixture'], default='gaussian')
    parser.add_argument('--sigma', type=float, default=25.0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--result-json', type=str, default=None)
    parser.add_argument('--action', choices=['generate', 'evaluate', 'full'], default='full')
    args = parser.parse_args()

    if args.action == 'generate':
        files = generate_noisy_dataset(args.clean_dir, args.out_dir, args.noise_type, args.sigma, args.seed)
        print(f'Generated {len(files)} noisy .mat files in {args.out_dir}')
        return

    if args.action == 'evaluate':
        if args.model_path is None:
            raise ValueError('Provide --model-path when using evaluate action.')
        pred_dir = os.path.join(args.out_dir, 'pred')
        predict_dataset(os.path.join(args.out_dir, 'noisy'), args.model_path, pred_dir)
        summary = evaluate_dataset(args.clean_dir, pred_dir)
        print(json.dumps(summary, indent=2))
        return

    if args.action == 'full':
        if args.model_path is None:
            raise ValueError('Provide --model-path when using full action.')
        summary = full_pipeline(args.clean_dir, args.out_dir, args.model_path, args.noise_type, args.sigma, args.seed, args.result_json)
        print(json.dumps(summary, indent=2))
        return


if __name__ == '__main__':
    main()
