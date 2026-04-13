# som-olp

Python implementation of **Self-Organizing Maps with Optimized Latent Positions (SOM-OLP)**.

## Overview

SOM-OLP is an objective-based topographic mapping method that introduces a continuous latent position for each data point. This repository provides a minimal implementation of the method together with simple experiment scripts.

## Repository structure

```text
som-olp/
├── somolp.py
├── requirements.txt
├── CITATION.cff
├── LICENSE
└── experiments/
    ├── example.py
    ├── optuna_hyperparameter_search.py
    └── results/
```

- `somolp.py`: minimal implementation of the `SOMOLP` class
- `experiments/example.py`: example script for running SOM-OLP on several benchmark datasets
- `experiments/optuna_hyperparameter_search.py`: Optuna-based hyperparameter search script

## Installation

```bash
git clone https://github.com/subukata/som-olp.git
cd som-olp
pip install -r requirements.txt
```

If you want to run the hyperparameter search script, install Optuna as well:

```bash
pip install optuna
```

## Quick start

A minimal example is shown below.

```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

from somolp import SOMOLP

X, _ = load_iris(return_X_y=True)
X = StandardScaler().fit_transform(X)

m_side = 16
t = np.linspace(-1, 1, m_side)
gx, gy = np.meshgrid(t, t)
R = np.column_stack([gx.ravel(), gy.ravel()])

model = SOMOLP(
    R,
    gamma=67.26,
    lam=1.165,
    max_iters=1000,
    tol=1e-4,
    pca_scale=1.86,
).fit(X)

print("n_iter =", model.n_iter)
print("final objective =", model.history[-1])
print("latent positions shape =", model.V.shape)
print("reference vectors shape =", model.W.shape)
```

## Example scripts

Run the example script:

```bash
python experiments/example.py
```

This script runs SOM-OLP on several datasets and saves output figures to:

```text
experiments/results/
```

Run the Optuna-based hyperparameter search:

```bash
python experiments/optuna_hyperparameter_search.py
```

## Citation

If you use this software, please cite the Zenodo record:

- https://doi.org/10.5281/zenodo.19547951

Citation information is also available in `CITATION.cff`.

## License

This project is released under the MIT License. See `LICENSE` for details.