import argparse
import json
import os
from typing import Optional

from matpy.evaluate_model import evaluate_dataset, predict_dataset
from matpy.generate_noisy_dataset import generate_noisy_dataset


def run_experiment(
    clean_dir: str,
    out_dir: str,
    model_path: str,
    noise_type: str = 'gaussian',
    sigma: Optional[float] = 25.0,
    seed: int = 0,
    result_json: Optional[str] = None,
    tile_size: int = 64,
    split_file: Optional[str] = None,
):
    noisy_dir = os.path.join(out_dir, 'noisy')
    pred_dir = os.path.join(out_dir, 'pred')

    generate_noisy_dataset(
        clean_dir,
        noisy_dir,
        noise_type=noise_type,
        sigma=sigma,
        seed=seed,
        split_file=split_file,
    )
    predict_dataset(noisy_dir, model_path, pred_dir, tile_size=tile_size, split_file=split_file)
    summary = evaluate_dataset(clean_dir, pred_dir, split_file=split_file)

    if result_json is not None:
        os.makedirs(os.path.dirname(result_json) or '.', exist_ok=True)
        with open(result_json, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description='Generate noisy HSI data, run model inference, and evaluate the result.')
    parser.add_argument('--clean-dir', type=str, required=True)
    parser.add_argument('--out-dir', type=str, required=True)
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--noise-type', choices=['gaussian', 'blind_gaussian', 'mixture'], default='gaussian')
    parser.add_argument('--sigma', type=float, default=25.0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--result-json', type=str, default=None)
    parser.add_argument('--tile-size', type=int, default=64)
    parser.add_argument('--split-file', type=str, default=None, help='Optional .txt list of filenames to include.')
    args = parser.parse_args()

    summary = run_experiment(
        args.clean_dir,
        args.out_dir,
        args.model_path,
        noise_type=args.noise_type,
        sigma=args.sigma,
        seed=args.seed,
        result_json=args.result_json,
        tile_size=args.tile_size,
        split_file=args.split_file,
    )
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
