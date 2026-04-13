import numpy as np
import optuna
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_digits
from sklearn.manifold import trustworthiness
from sklearn.preprocessing import StandardScaler

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from somolp import SOMOLP

DATASETS = [
    ("iris", load_iris),
    ("wine", load_wine),
    ("breast_cancer", load_breast_cancer),
    ("digits", load_digits),
]

M_SIDE = 16
t = np.linspace(-1, 1, M_SIDE)
gx, gy = np.meshgrid(t, t)
R = np.column_stack([gx.ravel(), gy.ravel()])


def objective(trial, X):
    gamma = trial.suggest_float("gamma", 0.001, 1000.0, log=True)
    lam = trial.suggest_float("lam", 0.001, 1000.0, log=True)
    pca_scale = trial.suggest_float("pca_scale", 0.1, 5.0, log=True)

    model = SOMOLP(R, gamma=gamma, lam=lam, max_iters=1000, tol=1e-4, pca_scale=pca_scale).fit(X)
    tw = trustworthiness(X, model.V, n_neighbors=5)
    cn = trustworthiness(model.V, X, n_neighbors=5)
    return (tw + cn) / 2


def main():
    for name, loader in DATASETS:
        X, _ = loader(return_X_y=True)
        X = StandardScaler().fit_transform(X)
        study = optuna.create_study(
            direction="maximize", study_name=name,
            sampler=optuna.samplers.TPESampler(seed=42, multivariate=True),
        )
        study.optimize(lambda trial: objective(trial, X), n_trials=100)
        b = study.best_trial
        print(f"[{name}] (TW+CN)/2={b.value:.4f}  {b.params}")


if __name__ == "__main__":
    main()
