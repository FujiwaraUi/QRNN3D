import numpy as np
from typing import Optional, Tuple


def _as_hsi(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f'Expected HSI shaped (H, W, C), got {arr.shape}')
    return arr


def _ensure_seed(seed: Optional[int]) -> Optional[int]:
    if seed is None:
        return None
    np.random.seed(seed)
    return seed


def _rand_sigma(low: float, high: float, size: Tuple[int, ...], rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(low, high, size=size).astype(np.float32)


def normalize_hsi(image: np.ndarray) -> np.ndarray:
    arr = _as_hsi(image)
    amin = arr.min()
    amax = arr.max()
    if amax == amin:
        return arr.astype(np.float32)
    return ((arr - amin) / (amax - amin)).astype(np.float32)


def crop_center(image: np.ndarray, crop_h: int, crop_w: int) -> np.ndarray:
    arr = _as_hsi(image)
    h, w, c = arr.shape
    y0 = max(0, (h - crop_h) // 2)
    x0 = max(0, (w - crop_w) // 2)
    y1 = y0 + min(crop_h, h)
    x1 = x0 + min(crop_w, w)
    return arr[y0:y1, x0:x1, :].copy()


def add_gaussian_noise(image: np.ndarray, sigma: float = 25.0, seed: Optional[int] = None) -> np.ndarray:
    arr = _as_hsi(image)
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, sigma / 255.0, size=arr.shape).astype(np.float32)
    return np.clip(arr + noise, 0.0, 1.0).astype(np.float32)


def add_blind_gaussian_noise(image: np.ndarray, low: float = 10.0, high: float = 70.0, seed: Optional[int] = None) -> np.ndarray:
    arr = _as_hsi(image)
    rng = np.random.default_rng(seed)
    sigma = float(rng.uniform(low, high))
    noise = rng.normal(0.0, sigma / 255.0, size=arr.shape).astype(np.float32)
    return np.clip(arr + noise, 0.0, 1.0).astype(np.float32)


def add_stripe_noise(image: np.ndarray, min_amount: float = 0.05, max_amount: float = 0.15, seed: Optional[int] = None) -> np.ndarray:
    arr = _as_hsi(image)
    rng = np.random.default_rng(seed)
    h, w, c = arr.shape
    out = arr.copy()
    band_count = max(1, c // 3)
    bands = rng.choice(c, size=band_count, replace=False)

    for band in bands:
        strip_count = int(rng.integers(int(min_amount * w), int(max_amount * w) + 1))
        loc = rng.choice(w, size=strip_count, replace=False)
        stripe = (rng.random(size=strip_count).astype(np.float32) * 0.5 - 0.25)
        out[:, loc, band] = np.clip(out[:, loc, band] - stripe.reshape(1, -1), 0.0, 1.0)
    return out.astype(np.float32)


def add_deadline_noise(image: np.ndarray, min_amount: float = 0.05, max_amount: float = 0.15, seed: Optional[int] = None) -> np.ndarray:
    arr = _as_hsi(image)
    rng = np.random.default_rng(seed)
    h, w, c = arr.shape
    out = arr.copy()
    band_count = max(1, c // 3)
    bands = rng.choice(c, size=band_count, replace=False)

    for band in bands:
        deadline_count = int(rng.integers(int(min_amount * w), int(max_amount * w) + 1))
        loc = rng.choice(w, size=deadline_count, replace=False)
        out[:, loc, band] = 0.0
    return out.astype(np.float32)


def add_impulse_noise(image: np.ndarray, ratios=(0.1, 0.3, 0.5, 0.7), seed: Optional[int] = None) -> np.ndarray:
    arr = _as_hsi(image)
    rng = np.random.default_rng(seed)
    out = arr.copy()
    h, w, c = arr.shape
    band_count = min(10, c)
    bands = rng.choice(c, size=band_count, replace=False)

    for band in bands:
        ratio = float(rng.choice(ratios))
        mask = rng.random((h, w)) < ratio
        out[:, :, band] = np.where(mask, rng.choice([0.0, 1.0], size=(h, w)), out[:, :, band])
    return out.astype(np.float32)


def add_mixture_noise(
    image: np.ndarray,
    sigma: float = 30.0,
    min_amount: float = 0.05,
    max_amount: float = 0.15,
    seed: Optional[int] = None,
) -> np.ndarray:
    arr = _as_hsi(image)
    rng = np.random.default_rng(seed)
    out = arr.copy()

    gauss = rng.normal(0.0, sigma / 255.0, size=arr.shape).astype(np.float32)
    out = np.clip(out + gauss, 0.0, 1.0)

    h, w, c = out.shape
    bands = rng.choice(c, size=max(1, c // 3), replace=False)
    for band in bands:
        strip_count = int(rng.integers(int(min_amount * w), int(max_amount * w) + 1))
        loc = rng.choice(w, size=strip_count, replace=False)
        stripe = (rng.random(size=strip_count).astype(np.float32) * 0.5 - 0.25)
        out[:, loc, band] = np.clip(out[:, loc, band] - stripe.reshape(1, -1), 0.0, 1.0)

    dead_bands = rng.choice(c, size=max(1, c // 3), replace=False)
    for band in dead_bands:
        deadline_count = int(rng.integers(int(min_amount * w), int(max_amount * w) + 1))
        loc = rng.choice(w, size=deadline_count, replace=False)
        out[:, loc, band] = 0.0

    impulse_bands = rng.choice(c, size=max(1, c // 3), replace=False)
    for band in impulse_bands:
        ratio = float(rng.choice((0.1, 0.3, 0.5, 0.7)))
        mask = rng.random((h, w)) < ratio
        out[:, :, band] = np.where(mask, rng.choice([0.0, 1.0], size=(h, w)), out[:, :, band])

    return out.astype(np.float32)
