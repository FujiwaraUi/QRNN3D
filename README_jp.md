# QRNN3D Python ノイズ生成 / 評価補助

このリポジトリでは、現在の実験フローを以下の 3 つに分離して運用しています。

- ノイズ生成
- 推論実行
- 評価

また、train / test のリークを避けるため、ディレクトリ全件をまとめて処理するのではなく、既存の split ファイルを使う運用を推奨しています。

古い学習用・評価用スクリプトは、今後の整理のために `tests/PRE_TRASH` に保管しています。

## 現在のアクティブなモジュール

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

## スプリットの推奨方法

以下のファイルを使って評価対象を限定してください。

- `src/ICVL_train.txt`
- `src/ICVL_test_gauss.txt`
- `src/ICVL_test_complex.txt`

これにより、評価セットが生成や推論の対象に混ざるリスクを下げられます。

## 使い方

### 1. ノイズ生成

```bash
PYTHONPATH=src conda run -n qrnn3d python -m matpy.generate_noisy_dataset \
  --clean-dir /mnt/data/User/HSID/Dataset/ICVL-BGU/ICVL_HS_2016/mat \
  --out-dir /mnt/data/User/HSID/Dataset/ICVL-BGU/ICVL_HS_2016/nois \
  --noise-type gaussian \
  --sigma 25.0 \
  --seed 0 \
  --split-file /home/fy/Mnt/reearch/Methods/06_QRNN3D/QRNN3D/src/ICVL_test_gauss.txt
```

### 2. モデル評価

```bash
PYTHONPATH=src conda run -n qrnn3d python -m matpy.evaluate_model \
  --clean-dir /mnt/data/User/HSID/Dataset/ICVL-BGU/ICVL_HS_2016/mat \
  --noisy-dir /mnt/data/User/HSID/Dataset/ICVL-BGU/ICVL_HS_2016/nois \
  --model-path /mnt/data/User/HSID/Pre_Model/qrnn3d/gauss/model_epoch_50_118454.pth \
  --out-dir /mnt/data/User/HSID/Dataset/ICVL-BGU/ICVL_HS_2016/pred \
  --tile-size 32 \
  --split-file /home/fy/Mnt/reearch/Methods/06_QRNN3D/QRNN3D/src/ICVL_test_gauss.txt
```

### 3. 全体実行

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

## 補足

- 今回の実験フローは、生成・推論・評価を分けて管理する形に整理しています。
- これは、論文や実験の再現性を保ち、train/test の混在を避けやすくするためです。
- 旧来の実験・訓練用スクリプトは `tests/PRE_TRASH` に移動済みです。

## 例

```python
import numpy as np
from matpy import add_gaussian_noise, evaluate_metrics

clean = np.random.rand(64, 64, 31).astype(np.float32)
noisy = add_gaussian_noise(clean, sigma=25.0, seed=0)
metrics = evaluate_metrics(clean, noisy)
print(metrics)
```
