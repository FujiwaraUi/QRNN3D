import argparse
import json
import os
from typing import Dict, List, Optional

import numpy as np

from matpy import evaluate_metrics
from matpy.generate_noisy_dataset import iter_mat_files, select_mat_files
from matpy.io_utils import load_mat_hsi, save_mat_hsi
from matpy.model_eval import infer_on_hsi, load_pretrained_qrnn3d


def predict_dataset(
    noisy_dir: str,
    model_path: str,
    out_dir: str,
    tile_size: int = 64,
    split_file: Optional[str] = None,
) -> List[str]:
    os.makedirs(out_dir, exist_ok=True)
    model, device = load_pretrained_qrnn3d(model_path)
    generated: List[str] = []

    for path in select_mat_files(noisy_dir, split_file=split_file):
        noisy = load_mat_hsi(path, key='input')
        pred = infer_on_hsi(model, noisy, device, tile_size=tile_size)

        name = os.path.basename(path)
        out_path = os.path.join(out_dir, name)
        save_mat_hsi(out_path, pred, pred, sigma=0.0)
        generated.append(out_path)

    return generated


def evaluate_dataset(clean_dir: str, pred_dir: str, split_file: Optional[str] = None) -> Dict[str, float]:
    results = []
    clean_files = select_mat_files(clean_dir, split_file=split_file)

    for clean_path in clean_files:
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


def main() -> None:
    parser = argparse.ArgumentParser(description='Run HSI inference and evaluate the prediction against clean references.')
    parser.add_argument('--clean-dir', type=str, required=True)
    parser.add_argument('--noisy-dir', type=str, required=True)
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--out-dir', type=str, default=None)
    parser.add_argument('--tile-size', type=int, default=64)
    parser.add_argument('--split-file', type=str, default=None, help='Optional .txt list of filenames to include.')
    parser.add_argument('--result-json', type=str, default=None)
    args = parser.parse_args()

    pred_dir = args.out_dir or os.path.join(args.noisy_dir, 'pred')
    predict_dataset(args.noisy_dir, args.model_path, pred_dir, tile_size=args.tile_size, split_file=args.split_file)
    summary = evaluate_dataset(args.clean_dir, pred_dir, split_file=args.split_file)

    if args.result_json is not None:
        os.makedirs(os.path.dirname(args.result_json) or '.', exist_ok=True)
        with open(args.result_json, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
