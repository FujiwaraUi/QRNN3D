# QRNN3D: shell commands for validation and evaluation

This document collects the shell commands needed to verify the Python helper pipeline and execute the actual evaluation flow.

## 1. Enter the project root

```bash
cd /home/fy/Mnt/reearch/Methods/06_QRNN3D/QRNN3D
pwd
ls
```

## 2. Check which Python interpreter is active

```bash
which python
python --version
python3 --version
/home/fy/.local/bin/python3.12 - <<'PY'
import sys
print(sys.executable)
try:
    import numpy, scipy, torch
    print('ok')
except Exception as e:
    print(type(e).__name__, e)
PY
```

If the project environment is not the right one, do not rely on the system Python. The project was previously failing because the active interpreter did not have `numpy`/`torch` installed.

## 3. Recommended environment setup

### Option A: use a local venv

```bash
cd /home/fy/Mnt/reearch/Methods/06_QRNN3D/QRNN3D
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy scipy matplotlib h5py opencv-python-headless pytest torch torchvision
```

### Option B: use conda from environment.yml

```bash
cd /home/fy/Mnt/reearch/Methods/06_QRNN3D/QRNN3D
conda env create -f environment.yml
conda activate QRNN3D
python -m pip install torch torchvision
```

If the environment name differs, replace `QRNN3D` with the actual environment name from your conda setup.

## 4. Run the unit tests

```bash
cd /home/fy/Mnt/reearch/Methods/06_QRNN3D/QRNN3D
export PYTHONPATH=src
pytest tests/test_matpy.py -q
```

Or, if you want the same validation without pytest:

```bash
cd /home/fy/Mnt/reearch/Methods/06_QRNN3D/QRNN3D
export PYTHONPATH=src
python - <<'PY'
import numpy as np
from matpy import add_gaussian_noise, add_mixture_noise, evaluate_metrics

clean = np.random.rand(10, 12, 31).astype(np.float32)
noisy = add_gaussian_noise(clean, sigma=25.0, seed=0)
assert noisy.shape == clean.shape
assert np.isfinite(noisy).all()

clean2 = np.random.rand(12, 14, 31).astype(np.float32)
noisy2 = add_mixture_noise(clean2, sigma=30.0, seed=0)
assert noisy2.shape == clean2.shape
assert np.isfinite(noisy2).all()

gt = np.random.rand(8, 10, 5).astype(np.float32)
pred = gt.copy() + 0.01 * np.random.randn(*gt.shape)
metrics = evaluate_metrics(gt, pred)
print(metrics)
print('ok')
PY
```

## 5. Generate noisy data from a clean directory

```bash
cd /home/fy/Mnt/reearch/Methods/06_QRNN3D/QRNN3D
export PYTHONPATH=src
python -m matpy.runner \
  --clean-dir /path/to/clean_mat_dir \
  --out-dir /path/to/noisy_output \
  --noise-type gaussian \
  --sigma 25.0 \
  --seed 0 \
  --action generate
```

Example:

```bash
python -m matpy.runner \
  --clean-dir /mnt/data/User/HSID/Dataset/ICVL-BGU/ICVL_HS_2016/mat \
  --out-dir /tmp/qrnn_noisy \
  --noise-type gaussian \
  --sigma 25.0 \
  --seed 0 \
  --action generate
```

## 6. Run inference with a pretrained model

```bash
cd /home/fy/Mnt/reearch/Methods/06_QRNN3D/QRNN3D
export PYTHONPATH=src
python -m matpy.runner \
  --clean-dir /path/to/clean_mat_dir \
  --out-dir /path/to/workdir \
  --model-path /path/to/model.pth \
  --noise-type gaussian \
  --sigma 25.0 \
  --seed 0 \
  --action evaluate
```

This workflow expects:

- clean data directory with reference `.mat` files
- noisy data directory placed under `workdir/noisy`
- output predictions produced under `workdir/pred`

## 7. Full end-to-end pipeline

```bash
cd /home/fy/Mnt/reearch/Methods/06_QRNN3D/QRNN3D
export PYTHONPATH=src
python -m matpy.runner \
  --clean-dir /path/to/clean_mat_dir \
  --out-dir /path/to/workdir \
  --model-path /path/to/model.pth \
  --noise-type gaussian \
  --sigma 25.0 \
  --seed 0 \
  --result-json /path/to/results.json \
  --action full
```

This does the following internally:

1. generate noisy `.mat` files under `workdir/noisy`
2. run model inference into `workdir/pred`
3. calculate metrics against the clean reference
4. save the averaged results as JSON

## 8. Example with an actual checkpoint path

```bash
cd /home/fy/Mnt/reearch/Methods/06_QRNN3D/QRNN3D
export PYTHONPATH=src
python -m matpy.runner \
  --clean-dir /mnt/data/User/HSID/Dataset/ICVL-BGU/ICVL_HS_2016/mat \
  --out-dir /tmp/qrnn_eval \
  --model-path /mnt/data/User/HSID/Pre_Model/qrnn3d/gauss/model_epoch_50_118454.pth \
  --noise-type gaussian \
  --sigma 25.0 \
  --seed 0 \
  --result-json /tmp/qrnn_eval/results.json \
  --action full
```

## 9. Quick smoke test for the model wrapper

```bash
cd /home/fy/Mnt/reearch/Methods/06_QRNN3D/QRNN3D
export PYTHONPATH=src
python - <<'PY'
import os, tempfile
import numpy as np
from scipy.io import savemat
from matpy.model_eval import load_pretrained_qrnn3d, infer_on_hsi

# A synthetic 3D HSI candidate
x = np.random.rand(8, 10, 5).astype(np.float32)
print('shape', x.shape)
# If a real checkpoint exists, replace the model path below.
# model, device = load_pretrained_qrnn3d('/path/to/model.pth')
# pred = infer_on_hsi(model, x, device)
# print(pred.shape)
PY
```

## 10. Summary of the minimal workflow

For a real evaluation, the minimal command chain is:

```bash
cd /home/fy/Mnt/reearch/Methods/06_QRNN3D/QRNN3D
export PYTHONPATH=src
python -m pip install numpy scipy matplotlib h5py opencv-python-headless pytest torch torchvision
pytest tests/test_matpy.py -q
python -m matpy.runner --clean-dir /path/to/clean --out-dir /tmp/qrnn_eval --model-path /path/to/model.pth --action full
```

This is the practical path from “code present” to “actual denoising evaluation” in this project.
