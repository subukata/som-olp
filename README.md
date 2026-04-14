# som-olp

Python implementation of **Self-Organizing Maps with Optimized Latent Positions (SOM-OLP)**.

## Overview

SOM-OLP learns prototypes and continuous latent positions for individual data points within a single objective-based framework. This repository provides a minimal implementation of the method together with simple experiment scripts.

### Key features

- Based on a single objective function optimization problem
- Monotonic objective decrease under closed-form cyclic block updates
- Per-iteration computational complexity is O(NM) without explicit node-node interactions
- Relates to entropy-regularized fuzzy c-means and k-means in limiting or simplified settings

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

Optuna is optional and required only for hyperparameter search:

```bash
pip install optuna
```

## Quick start

A minimal example is shown below. The hyperparameters `gamma` and `lam` are dataset-dependent; the values here were tuned for Iris via Optuna (see `experiments/optuna_hyperparameter_search.py`).

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
    gamma=51.94,
    lam=1.32,
    max_iters=1000,
    tol=1e-4,
).fit(X)

print("n_iter =", model.n_iter)
print("final objective =", model.history[-1])
print("V (latent positions, first 5) =\n", model.V[:5])
print("W (reference vectors, first 5) =\n", model.W[:5])
```

## Example scripts

`experiments/example.py` runs SOM-OLP on Iris, Wine, Breast Cancer, and Digits (all from scikit-learn) using Optuna-tuned hyperparameters, and saves latent-space visualizations as PNG files to `experiments/results/`.

```bash
python experiments/example.py
```

`experiments/optuna_hyperparameter_search.py` searches for the best `gamma` and `lam` per dataset by maximizing the average of trustworthiness and continuity.

```bash
python experiments/optuna_hyperparameter_search.py
```

## Example results

The following figures show the learned latent representations produced by `experiments/example.py`.

Each figure shows the learned latent space on a 16×16 grid (`M=256`). Gray dots and lines represent the predefined latent-node grid, while colored points represent the learned continuous latent positions of the data points.

### Iris
<img src="experiments/results/som-olp_iris.png" width="400">

### Wine
<img src="experiments/results/som-olp_wine.png" width="400">

### Breast Cancer
<img src="experiments/results/som-olp_breast_cancer.png" width="400">

### Digits
<img src="experiments/results/som-olp_digits.png" width="400">

## Citation

If you use this software, please cite the Zenodo record:

- https://doi.org/10.5281/zenodo.19547951

```bibtex
@software{ubukata2026somolp,
  author    = {Ubukata, Seiki},
  title     = {som-olp},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.19547951},
  url       = {https://github.com/subukata/som-olp}
}
```

Citation information is also available in `CITATION.cff`.

## License

This project is released under the MIT License. See `LICENSE` for details.
