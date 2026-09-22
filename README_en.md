# QRNN3D Python Noise Generation / Evaluation Helper

This repository now keeps the active workflow in a small Python utility set that focuses on:

- noisy dataset generation
- model inference
- evaluation against the clean target
- split-file based selection to reduce leakage risk between train and evaluation data

Legacy training and benchmark scripts were archived under `tests/PRE_TRASH` to keep the active project focused on the current Python evaluation pipeline.

## Active modules

```text
src/
  matpy/
    __init__.py
    noise.py
    metrics.py
    io_utils.py
    generate_noisy_dataset.py
    evaluate_model.py
    run_experiment.py
    model_eval.py
```

## Data split recommendation

Use the existing split files instead of processing every file in a folder:

- `src/ICVL_train.txt`
- `src/ICVL_test_gauss.txt`
- `src/ICVL_test_complex.txt`

This keeps evaluation data out of the generation/inference pipeline when split-based evaluation is intended.

## Quick start

### Generate noisy files

```bash
PYTHONPATH=src conda run -n qrnn3d python -m matpy.generate_noisy_dataset \
  --clean-dir /mnt/data/User/HSID/Dataset/ICVL-BGU/ICVL_HS_2016/mat \
  --out-dir /mnt/data/User/HSID/Dataset/ICVL-BGU/ICVL_HS_2016/nois \
  --noise-type gaussian \
  --sigma 25.0 \
  --seed 0 \
  --split-file /home/fy/Mnt/reearch/Methods/06_QRNN3D/QRNN3D/src/ICVL_test_gauss.txt
```

### Evaluate a pretrained model

```bash
PYTHONPATH=src conda run -n qrnn3d python -m matpy.evaluate_model \
  --clean-dir /mnt/data/User/HSID/Dataset/ICVL-BGU/ICVL_HS_2016/mat \
  --noisy-dir /mnt/data/User/HSID/Dataset/ICVL-BGU/ICVL_HS_2016/nois \
  --model-path /mnt/data/User/HSID/Pre_Model/qrnn3d/gauss/model_epoch_50_118454.pth \
  --out-dir /mnt/data/User/HSID/Dataset/ICVL-BGU/ICVL_HS_2016/pred \
  --tile-size 32 \
  --split-file /home/fy/Mnt/reearch/Methods/06_QRNN3D/QRNN3D/src/ICVL_test_gauss.txt
```

### Run the full pipeline

```bash
PYTHONPATH=src conda run -n qrnn3d python -m matpy.run_experiment \
  --clean-dir /mnt/data/User/HSID/Dataset/ICVL-BGU/ICVL_HS_2016/mat \
  --out-dir /mnt/data/User/HSID/Dataset/ICVL-BGU/ICVL_HS_2016/nois \
  --model-path /mnt/data/User/HSID/Pre_Model/qrnn3d/gauss/model_epoch_50_118454.pth \
  --noise-type gaussian \
  --sigma 25.0 \
  --tile-size 32 \
  --split-file /home/fy/Mnt/reearch/Methods/06_QRNN3D/QRNN3D/src/ICVL_test_gauss.txt
```

## Notes

- This is a Python approximation of the MATLAB benchmark flow.
- The current workflow is deliberately split into generation, evaluation, and experiment orchestration for clarity and safer experimental practice.
- Older benchmark or training scripts were archived to `tests/PRE_TRASH` rather than kept in the active path.

## Example

```python
import numpy as np
from matpy import add_gaussian_noise, evaluate_metrics

clean = np.random.rand(64, 64, 31).astype(np.float32)
noisy = add_gaussian_noise(clean, sigma=25.0, seed=0)
metrics = evaluate_metrics(clean, noisy)
print(metrics)
```
