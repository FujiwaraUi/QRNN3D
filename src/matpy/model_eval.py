import json
import os
from typing import Dict, List, Optional

import numpy as np
import torch

from matpy.io_utils import load_mat_hsi
from matpy.metrics import evaluate_metrics


def _best_hsi_key(mat: Dict[str, np.ndarray]) -> str:
    for key in ('gt', 'rad', 'hsi', 'img', 'input'):
        if key in mat:
            return key
    raise KeyError(f'Could not find a suitable HSI key in .mat file. Available keys: {list(mat.keys())[:10]}')


def _as_hwc(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f'Expected 3D HSI, got shape {arr.shape}')
    if arr.shape[0] > 10 and arr.shape[0] < arr.shape[-1]:
        arr = arr.transpose(1, 2, 0)
    return arr


def _to_chw(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f'Expected 3D HSI, got shape {arr.shape}')
    if arr.shape[0] > 10 and arr.shape[0] < arr.shape[-1]:
        arr = arr.transpose(1, 2, 0)
    return arr.transpose(2, 0, 1)


def load_pretrained_qrnn3d(model_path: str, device: Optional[torch.device] = None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    import importlib
    models_module = importlib.import_module('models')
    model = models_module.qrnn3d()
    model.to(device)
    model.eval()

    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'net' in checkpoint:
        state_dict = checkpoint['net']
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=False)
    return model, device


def _infer_tile(model: torch.nn.Module, tile: np.ndarray, device: torch.device) -> np.ndarray:
    original_shape = tile.shape
    pad_h = 1 if tile.shape[1] % 2 != 0 else 0
    pad_w = 1 if tile.shape[2] % 2 != 0 else 0
    if pad_h or pad_w:
        tile = np.pad(tile, ((0, 0), (0, pad_h), (0, pad_w)), mode='reflect')

    tensor = torch.from_numpy(tile[None, None]).to(device).float()
    with torch.no_grad():
        out = model(tensor)

    if isinstance(out, (tuple, list)):
        out = out[0]

    out = out.detach().cpu().numpy()
    if out.ndim == 5 and out.shape[0] == 1:
        out = out[0]
    if out.ndim == 4 and out.shape[0] == 1:
        out = out[0]
    if out.ndim not in (3, 4):
        raise ValueError(f'Unexpected model output shape {out.shape} for tile inference.')

    if out.ndim == 4 and out.shape[0] == 1:
        out = out[0]

    if pad_h or pad_w:
        out = out[:, :original_shape[1], :original_shape[2]]
    return np.asarray(out, dtype=np.float32)


def infer_on_hsi(
    model: torch.nn.Module,
    hsi: np.ndarray,
    device: torch.device,
    clamp: bool = True,
    tile_size: Optional[int] = 64,
    overlap: int = 16,
) -> np.ndarray:
    arr = np.asarray(hsi, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f'Expected 3D HSI, got shape {arr.shape}')

    # Real project data is stored in CHW layout as (C, H, W). Some helper code may
    # produce HWC arrays, so convert only in that case.
    if arr.shape[0] <= 10 and arr.shape[-1] > 10:
        arr = arr.transpose(2, 0, 1)

    if tile_size is None or tile_size <= 0:
        tile_size = max(arr.shape[1], arr.shape[2])
    overlap = max(0, min(overlap, tile_size // 2))

    bands, height, width = arr.shape
    if height <= tile_size and width <= tile_size:
        pred = _infer_tile(model, arr, device)
        pred = pred.astype(np.float32)
        if clamp:
            pred = np.clip(pred, 0.0, 1.0)
        return pred.transpose(1, 2, 0)

    step_h = max(1, tile_size - overlap)
    step_w = max(1, tile_size - overlap)
    out = np.zeros((bands, height, width), dtype=np.float32)
    weights = np.zeros((height, width), dtype=np.float32)

    for y0 in range(0, height, step_h):
        y1 = min(height, y0 + tile_size)
        for x0 in range(0, width, step_w):
            x1 = min(width, x0 + tile_size)
            tile = arr[:, y0:y1, x0:x1]
            pred = _infer_tile(model, tile, device)
            if pred.shape != tile.shape:
                pred = pred[:, : y1 - y0, : x1 - x0]
            out[:, y0:y1, x0:x1] += pred
            weights[y0:y1, x0:x1] += 1.0

    if device.type == 'cuda':
        torch.cuda.empty_cache()

    out = out / np.maximum(weights, 1.0)[None, :, :]
    out = out.astype(np.float32)
    if clamp:
        out = np.clip(out, 0.0, 1.0)
    return out.transpose(1, 2, 0)


def evaluate_single_pair(clean_path: str, noisy_path: str, model_path: str) -> Dict[str, float]:
    clean_mat = loadmat_object(clean_path)
    noisy_mat = loadmat_object(noisy_path)

    gt_key = _best_hsi_key(clean_mat)
    input_key = _best_hsi_key(noisy_mat)

    gt = _as_hwc(np.asarray(clean_mat[gt_key], dtype=np.float32))
    noisy = _as_hwc(np.asarray(noisy_mat[input_key], dtype=np.float32))

    model, device = load_pretrained_qrnn3d(model_path)
    pred = infer_on_hsi(model, noisy, device)
    pred = _as_hwc(pred)

    gt = np.clip(gt, 0.0, 1.0)
    pred = np.clip(pred, 0.0, 1.0)
    return evaluate_metrics(gt, pred)


def loadmat_object(path: str) -> Dict[str, np.ndarray]:
    import h5py
    try:
        with h5py.File(path, 'r') as f:
            return {k: np.asarray(f[k]) for k in f.keys()}
    except Exception:
        pass

    from scipy.io import loadmat
    data = loadmat(path, squeeze_me=True, struct_as_record=False)
    return {k: v for k, v in data.items() if not k.startswith('__')}


def evaluate_model_on_dir(clean_dir: str, noisy_dir: str, model_path: str, out_json: Optional[str] = None) -> List[Dict[str, float]]:
    results: List[Dict[str, float]] = []
    for file_name in sorted(os.listdir(noisy_dir)):
        if not file_name.lower().endswith('.mat'):
            continue
        clean_path = os.path.join(clean_dir, file_name)
        noisy_path = os.path.join(noisy_dir, file_name)
        if not os.path.exists(clean_path):
            continue

        metrics = evaluate_single_pair(clean_path, noisy_path, model_path)
        metrics['file'] = file_name
        results.append(metrics)

    if out_json is not None:
        with open(out_json, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)

    return results
