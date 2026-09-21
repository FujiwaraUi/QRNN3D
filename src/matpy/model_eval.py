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


def infer_on_hsi(model: torch.nn.Module, hsi: np.ndarray, device: torch.device, clamp: bool = True) -> np.ndarray:
    arr = np.asarray(hsi, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f'Expected 3D HSI, got shape {arr.shape}')
    if arr.shape[0] > 10 and arr.shape[0] < arr.shape[-1]:
        arr = arr.transpose(1, 2, 0)

    tensor = torch.from_numpy(arr.transpose(2, 0, 1)[None]).to(device).float()
    with torch.no_grad():
        out = model(tensor)

    if isinstance(out, (tuple, list)):
        out = out[0]

    out = out.detach().cpu().numpy()
    if out.ndim == 4 and out.shape[0] == 1:
        out = out[0]

    if out.shape[0] > 10 and out.shape[0] < out.shape[-1]:
        out = out.transpose(1, 2, 0)

    out = out.astype(np.float32)
    if clamp:
        out = np.clip(out, 0.0, 1.0)
    return out


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
    from scipy.io import loadmat
    data = loadmat(path, squeeze_me=True)
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
