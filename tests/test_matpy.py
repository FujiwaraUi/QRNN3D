import numpy as np

from matpy import add_gaussian_noise, add_mixture_noise, evaluate_metrics


def test_gaussian_noise_shape_and_finite():
    clean = np.random.rand(10, 12, 31).astype(np.float32)
    noisy = add_gaussian_noise(clean, sigma=25.0, seed=0)

    assert noisy.shape == clean.shape
    assert np.isfinite(noisy).all()
    assert np.mean(np.abs(noisy - clean)) > 0


def test_mixture_noise_shape_and_finite():
    clean = np.random.rand(12, 14, 31).astype(np.float32)
    noisy = add_mixture_noise(clean, sigma=30.0, seed=0)

    assert noisy.shape == clean.shape
    assert np.isfinite(noisy).all()
    assert np.mean(np.abs(noisy - clean)) > 0


def test_metrics_are_finite():
    gt = np.random.rand(8, 10, 5).astype(np.float32)
    pred = gt.copy()
    pred += 0.01 * np.random.randn(*pred.shape)

    metrics = evaluate_metrics(gt, pred)

    assert np.isfinite(metrics['psnr'])
    assert np.isfinite(metrics['ssim'])
    assert np.isfinite(metrics['sam'])
    assert np.isfinite(metrics['ergas'])
