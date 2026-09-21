import os
from typing import Optional

import numpy as np
from scipy.io import loadmat, savemat


def load_mat_hsi(path: str, key: str = 'rad') -> np.ndarray:
    data = loadmat(path)
    if key not in data:
        raise KeyError(f'Mat file {path} has no key {key!r}')
    arr = np.asarray(data[key], dtype=np.float32)
    if arr.ndim == 2:
        return arr[:, :, None]
    return arr


def save_mat_hsi(path: str, gt: np.ndarray, input_img: np.ndarray, sigma: float, key_gt: str = 'gt', key_input: str = 'input', key_sigma: str = 'sigma') -> None:
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    savemat(path, {
        key_gt: np.asarray(gt, dtype=np.float32),
        key_input: np.asarray(input_img, dtype=np.float32),
        key_sigma: np.asarray(float(sigma), dtype=np.float32),
    })


def normalize_hsi(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    amin = arr.min()
    amax = arr.max()
    if amax == amin:
        return arr
    return (arr - amin) / (amax - amin)


def crop_center(image: np.ndarray, crop_h: int, crop_w: int) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f'Expected 3D HSI, got shape {arr.shape}')
    h, w, c = arr.shape
    y0 = max(0, (h - crop_h) // 2)
    x0 = max(0, (w - crop_w) // 2)
    y1 = min(h, y0 + crop_h)
    x1 = min(w, x0 + crop_w)
    return arr[y0:y1, x0:x1, :].copy()
