# QRNN3D Python Noise Generation / Evaluation Helper

This repository adds a Python-only evaluation helper for the QRNN3D workflow.
The original MATLAB benchmark code generated noisy hyperspectral images and then evaluated the restoration quality. This utility reproduces the core parts in Python so it can be used without MATLAB.

## What is included

- HSI loading from .mat files
- HSI normalization and center-crop preprocessing
- Gaussian noise generation
- Blind Gaussian noise generation
- Mixture noise generation
- Standard HSI quality metrics:
  - PSNR
  - SSIM
  - SAM
  - ERGAS
- Utility functions for saving noisy datasets in a MATLAB-like .mat format

## Directory layout

```text
src/
  matpy/
    __init__.py
    noise.py
    metrics.py
    io_utils.py
    runner.py
```

## Quick start

```bash
cd src
python -m matpy.runner --clean-dir /path/to/clean_dataset --out-dir /path/to/noisy_dataset --noise-type gaussian --sigma 25 --seed 0
```

## Notes

- This is a Python approximation of the MATLAB benchmark flow.
- It focuses on the parts that are needed to evaluate a pretrained model on noisy hyperspectral data.
- It does not re-implement the full MATLAB `lib` benchmark library, which contains external methods such as BM4D, LLRT, ITSReg, FastHyDe, and similar packages.

## Example

```python
import numpy as np
from matpy import add_gaussian_noise, evaluate_metrics

clean = np.random.rand(64, 64, 31).astype(np.float32)
noisy = add_gaussian_noise(clean, sigma=25.0, seed=0)
metrics = evaluate_metrics(clean, noisy)
print(metrics)
```
