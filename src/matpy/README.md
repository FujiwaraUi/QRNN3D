# matpy

Python helper utilities for QRNN3D evaluation.

## Included components

- noise generation
- quality metrics
- .mat I/O utilities
- model evaluation entry point

## Example

```python
import numpy as np
from matpy.model_eval import evaluate_single_pair

metrics = evaluate_single_pair(
    clean_path='/path/to/clean.mat',
    noisy_path='/path/to/noisy.mat',
    model_path='/path/to/model.pth',
)
print(metrics)
```
