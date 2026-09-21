import numpy as np
from typing import Dict


def _to_float_image(img: np.ndarray) -> np.ndarray:
    arr = np.asarray(img, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f'Expected 3D HSI, got shape {arr.shape}')
    return arr


def psnr(gt: np.ndarray, pred: np.ndarray, data_range: float = 1.0) -> float:
    gt = _to_float_image(gt)
    pred = _to_float_image(pred)
    mse = np.mean((gt - pred) ** 2)
    if mse <= 0:
        return float('inf')
    return float(10.0 * np.log10((data_range ** 2) / mse))


def ssim(gt: np.ndarray, pred: np.ndarray, data_range: float = 1.0) -> float:
    gt = np.asarray(gt, dtype=np.float32)
    pred = np.asarray(pred, dtype=np.float32)
    if gt.shape != pred.shape:
        raise ValueError(f'gt and pred shapes differ: {gt.shape} vs {pred.shape}')

    if gt.ndim == 3:
        gt_2d = np.mean(gt, axis=2)
        pred_2d = np.mean(pred, axis=2)
    else:
        gt_2d = gt
        pred_2d = pred

    mu_x = gt_2d.mean()
    mu_y = pred_2d.mean()
    sigma_x2 = ((gt_2d - mu_x) ** 2).mean()
    sigma_y2 = ((pred_2d - mu_y) ** 2).mean()
    sigma_xy = ((gt_2d - mu_x) * (pred_2d - mu_y)).mean()

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x2 + sigma_y2 + c2)
    if denominator == 0:
        return 1.0
    return float(numerator / denominator)


def sam(gt: np.ndarray, pred: np.ndarray) -> float:
    gt = _to_float_image(gt)
    pred = _to_float_image(pred)
    eps = 1e-8
    dot = np.sum(gt * pred, axis=2)
    gt_norm = np.linalg.norm(gt, axis=2)
    pred_norm = np.linalg.norm(pred, axis=2)
    cos_ang = dot / (gt_norm * pred_norm + eps)
    ang = np.arccos(np.clip(cos_ang, -1.0, 1.0))
    return float(np.mean(np.rad2deg(ang)))


def ergas(gt: np.ndarray, pred: np.ndarray, ratio: int = 1) -> float:
    gt = _to_float_image(gt)
    pred = _to_float_image(pred)
    h, w, c = gt.shape
    mse = np.mean((gt - pred) ** 2, axis=(0, 1))
    mean_gt = np.mean(gt, axis=(0, 1))
    val = 0.0
    for i in range(c):
        if mean_gt[i] == 0:
            continue
        val += (mse[i] / (mean_gt[i] ** 2)) * (100.0 / ratio) ** 2
    return float(np.sqrt(val / c))


def evaluate_metrics(gt: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    return {
        'psnr': psnr(gt, pred),
        'ssim': ssim(gt, pred),
        'sam': sam(gt, pred),
        'ergas': ergas(gt, pred),
    }


def evaluate_metrics_dict(gt: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    return evaluate_metrics(gt, pred)
