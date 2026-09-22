# QRNN3D Python ノイズ生成 / 評価補助

このリポジトリには、QRNN3D の評価フローを MATLAB なしで動かすための Python 補助機能を追加しています。
元の MATLAB ベンチマークコードでは、ノイズ付きハイパースペクトル画像を生成し、その後に復元結果の評価を行っていました。ここでは、その中核部分を Python で再現しています。

## 含まれている機能

- .mat 形式の HSI 読み込み
- HSI の正規化と中心クロップ
- Gaussian ノイズ生成
- Blind Gaussian ノイズ生成
- Mixture ノイズ生成
- 主要評価指標
  - PSNR
  - SSIM
  - SAM
  - ERGAS
- MATLAB 互換の .mat 保存機能

## ディレクトリ構成

```text
src/
  matpy/
    __init__.py
    noise.py
    metrics.py
    io_utils.py
    runner.py
```

## 使い方

```bash
cd src
python -m matpy.runner --clean-dir /path/to/clean_dataset --out-dir /path/to/noisy_dataset --noise-type gaussian --sigma 25 --seed 0

python -m matpy.runner   --clean-dir /mnt/data/User/HSID/Dataset/ICVL-BGU/ICVL_HS_2016/mat   --out-dir /mnt/data/User/HSID/Dataset/ICVL-BGU/ICVL_HS_2016/nois   --model-path /mnt/data/User/HSID/Pre_Model/qrnn3d/gauss/model_epoch_50_118454.pth   --noise-type gaussian   --sigma 25.0   --seed 0   --result-json /mnt/data/User/HSID/Dataset/ICVL-BGU/ICVL_HS_2016/nois/results.json   --action full
```

## 補足

- これは MATLAB のベンチマークフローを Python に近い形で再現したものです。
- 事前学習済みモデルをノイズ付き HSI に対して評価するために必要な部分を中心にしています。
- MATLAB の `lib` 全体を完全再現するものではなく、BM4D や LLRT、ITSReg、FastHyDe など外部手法群の実装までは含めていません。

## 例

```python
import numpy as np
from matpy import add_gaussian_noise, evaluate_metrics

clean = np.random.rand(64, 64, 31).astype(np.float32)
noisy = add_gaussian_noise(clean, sigma=25.0, seed=0)
metrics = evaluate_metrics(clean, noisy)
print(metrics)
```
