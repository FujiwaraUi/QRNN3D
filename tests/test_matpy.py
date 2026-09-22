import os
import tempfile

import h5py
import numpy as np
import torch

from matpy import add_gaussian_noise, add_mixture_noise, evaluate_metrics
from matpy.io_utils import load_mat_hsi
from matpy.model_eval import infer_on_hsi


def test_load_hdf5_v73_mat_file():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'sample.mat')
        data = np.random.rand(3, 10, 12).astype(np.float32)
        with h5py.File(path, 'w') as f:
            f.create_dataset('rad', data=data)

        arr = load_mat_hsi(path, key='rad')
        assert arr.shape == data.shape
        assert np.allclose(arr, data)


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


def test_infer_on_hsi_tiles_large_images_without_changing_shape():
    class IdentityModel(torch.nn.Module):
        def forward(self, x):
            return x

    hsi = np.random.rand(31, 260, 260).astype(np.float32)
    out = infer_on_hsi(IdentityModel(), hsi, torch.device('cpu'), tile_size=64)

    assert out.shape == hsi.transpose(1, 2, 0).shape
    assert np.isfinite(out).all()
