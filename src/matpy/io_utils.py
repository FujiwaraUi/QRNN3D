import os
from typing import Optional

import h5py
import numpy as np
from scipy.io import loadmat, savemat


def _extract_hdf5_array(group, key: str):
    if key in group:
        arr = np.asarray(group[key], dtype=np.float32)
        return arr

    for candidate in ('rad', 'gt', 'hsi', 'img', 'input', 'data'):
        if candidate in group:
            arr = np.asarray(group[candidate], dtype=np.float32)
            return arr

    raise KeyError(f'No valid HSI key found. Available keys: {list(group.keys())[:20]}')


def _read_hdf5_mat(path: str, key: str):
    with h5py.File(path, 'r') as f:
        if key in f:
            return _extract_hdf5_array(f, key)
        for candidate in ('rad', 'gt', 'hsi', 'img', 'input', 'data'):
            if candidate in f:
                return _extract_hdf5_array(f, candidate)
        raise KeyError(f'Mat file {path} has no key {key!r} and no fallback key found. Available keys: {list(f.keys())[:20]}')


def load_mat_hsi(path: str, key: str = 'rad') -> np.ndarray:
    try:
        with h5py.File(path, 'r') as f:
            arr = _extract_hdf5_array(f, key)
            if arr.ndim == 2:
                return arr[:, :, None]
            return arr
    except (OSError, ValueError):
        pass

    try:
        data = loadmat(path, struct_as_record=False, squeeze_me=True)
    except Exception:
        data = None

    if data is not None and key in data:
        arr = np.asarray(data[key], dtype=np.float32)
        if arr.ndim == 2:
            return arr[:, :, None]
        return arr

    if data is not None:
        for candidate in (key, 'rad', 'gt', 'hsi', 'img', 'input', 'data'):
            if candidate in data:
                arr = np.asarray(data[candidate], dtype=np.float32)
                if arr.ndim == 2:
                    return arr[:, :, None]
                return arr

    raise KeyError(f'Mat file {path} has no usable key {key!r}')


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
