# QRNN3D

The implementation of TNNLS 2020 paper ["3D Quasi-Recurrent Neural Network for Hyperspectral Image Denoising"](https://arxiv.org/abs/2003.04547).

This repository now keeps the active experiment flow in a small Python utility set that separately handles:

- noisy data generation
- model inference
- evaluation against clean references
- split-file based selection to reduce train/test leakage risk

Legacy training and benchmark scripts that were originally scattered around the repo were archived to `tests/PRE_TRASH` to keep the active workflow lean and explicit.

## Active workflow

The current experiment path is:

1. generate noisy HSI data
2. run inference with a trained QRNN3D checkpoint
3. evaluate against the clean target using PSNR / SSIM / SAM / ERGAS

The active modules are:

- `src/matpy/generate_noisy_dataset.py`
- `src/matpy/evaluate_model.py`
- `src/matpy/run_experiment.py`
- `src/matpy/model_eval.py`
- `src/matpy/noise.py`
- `src/matpy/metrics.py`
- `src/matpy/io_utils.py`

## Recommended data split

Use the split files already provided in the repo instead of processing the whole directory blindly:

- `src/ICVL_train.txt`
- `src/ICVL_test_gauss.txt`
- `src/ICVL_test_complex.txt`

This keeps evaluation samples out of the generation/inference path when the experiment is meant to be split-based.

## Quick start

### 1. Generate noisy files

```bash
PYTHONPATH=src conda run -n qrnn3d python -m matpy.generate_noisy_dataset \
  --clean-dir /mnt/data/User/HSID/Dataset/ICVL-BGU/ICVL_HS_2016/mat \
  --out-dir /mnt/data/User/HSID/Dataset/ICVL-BGU/ICVL_HS_2016/nois \
  --noise-type gaussian \
  --sigma 25.0 \
  --seed 0 \
  --split-file /home/fy/Mnt/reearch/Methods/06_QRNN3D/QRNN3D/src/ICVL_test_gauss.txt
```

### 2. Evaluate a pretrained model

```bash
PYTHONPATH=src conda run -n qrnn3d python -m matpy.evaluate_model \
  --clean-dir /mnt/data/User/HSID/Dataset/ICVL-BGU/ICVL_HS_2016/mat \
  --noisy-dir /mnt/data/User/HSID/Dataset/ICVL-BGU/ICVL_HS_2016/nois \
  --model-path /mnt/data/User/HSID/Pre_Model/qrnn3d/gauss/model_epoch_50_118454.pth \
  --out-dir /mnt/data/User/HSID/Dataset/ICVL-BGU/ICVL_HS_2016/pred \
  --tile-size 32 \
  --split-file /home/fy/Mnt/reearch/Methods/06_QRNN3D/QRNN3D/src/ICVL_test_gauss.txt
```

### 3. Run the full experiment flow

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

## Legacy files

Older benchmark and training scripts have been archived into `tests/PRE_TRASH` in order to keep the active project focused on the current Python evaluation pipeline.

These legacy files are kept only for reference history and are not the preferred execution path for current experiments.

## Citation

If you find this work useful for your research, please cite:

```bibtex
@article{wei2020QRNN3D,
  title={3-D Quasi-Recurrent Neural Network for Hyperspectral Image Denoising},
  author={Wei, Kaixuan and Fu, Ying and Huang, Hua},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  year={2020},
  publisher={IEEE}
}
```

and follow-up works

```bibtex
@article{lai2022dphsir,
    title = {Deep plug-and-play prior for hyperspectral image restoration},
    journal = {Neurocomputing},
    volume = {481},
    pages = {281-293},
    year = {2022},
    issn = {0925-2312},
    doi = {https://doi.org/10.1016/j.neucom.2022.01.057},
    author = {Zeqiang Lai and Kaixuan Wei and Ying Fu},
}

@inproceedings{lai2023hsdt,
  author = {Lai, Zeqiang and Chenggang, Yan and Fu, Ying},
  title = {Hybrid Spectral Denoising Transformer with Guided Attention},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  year = {2023},
}

@article{lai2023mixed,
  title={Mixed Attention Network for Hyperspectral Image Denoising},
  author={Lai, Zeqiang and Fu, Ying},
  journal={arXiv preprint arXiv:2301.11525},
  year={2023}
}
```

## Contact

Please contact the original authors for questions about the method itself.
