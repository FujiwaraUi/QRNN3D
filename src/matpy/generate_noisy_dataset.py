import argparse
import os
from typing import List, Optional, Set

from matpy import add_blind_gaussian_noise, add_gaussian_noise, add_mixture_noise
from matpy.io_utils import load_mat_hsi, normalize_hsi, save_mat_hsi


def iter_mat_files(dataset_dir: str) -> List[str]:
    files: List[str] = []
    for root, _, names in os.walk(dataset_dir):
        for name in sorted(names):
            if name.lower().endswith('.mat'):
                files.append(os.path.join(root, name))
    return files


def load_split_names(split_file: Optional[str]) -> Optional[Set[str]]:
    if split_file is None:
        return None

    with open(split_file, 'r', encoding='utf-8') as f:
        names = [line.strip() for line in f if line.strip()]

    normalized = set()
    for name in names:
        normalized.add(os.path.basename(name))
        if os.path.splitext(name)[1]:
            normalized.add(os.path.splitext(os.path.basename(name))[0])
    return normalized


def select_mat_files(dataset_dir: str, split_file: Optional[str] = None) -> List[str]:
    allowed = load_split_names(split_file)
    if allowed is None:
        return iter_mat_files(dataset_dir)

    selected: List[str] = []
    for path in iter_mat_files(dataset_dir):
        base = os.path.basename(path)
        stem = os.path.splitext(base)[0]
        if base in allowed or stem in allowed:
            selected.append(path)
    return selected


def generate_noisy_dataset(
    clean_dir: str,
    out_dir: str,
    noise_type: str = 'gaussian',
    sigma: Optional[float] = 25.0,
    seed: int = 0,
    split_file: Optional[str] = None,
) -> List[str]:
    os.makedirs(out_dir, exist_ok=True)
    generated: List[str] = []

    for path in select_mat_files(clean_dir, split_file=split_file):
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


def main() -> None:
    parser = argparse.ArgumentParser(description='Generate noisy HSI data from clean .mat files.')
    parser.add_argument('--clean-dir', type=str, required=True)
    parser.add_argument('--out-dir', type=str, required=True)
    parser.add_argument('--noise-type', choices=['gaussian', 'blind_gaussian', 'mixture'], default='gaussian')
    parser.add_argument('--sigma', type=float, default=25.0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--split-file', type=str, default=None, help='Optional .txt list of filenames to include.')
    args = parser.parse_args()

    files = generate_noisy_dataset(
        args.clean_dir,
        args.out_dir,
        noise_type=args.noise_type,
        sigma=args.sigma,
        seed=args.seed,
        split_file=args.split_file,
    )
    print(f'Generated {len(files)} noisy .mat files in {args.out_dir}')


if __name__ == '__main__':
    main()
