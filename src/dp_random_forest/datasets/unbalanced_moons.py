from sklearn.datasets import make_moons, make_blobs
import numpy as np

from .splits import train_val_test_split as _train_val_test_split


def get_unbalanced_moons(cfg):
    seed = 10000
    Xmoons, ymoons = make_moons(n_samples=(cfg.dataset.n_samples * 55 // 100, cfg.dataset.n_samples * 40 // 100), noise=cfg.dataset.noise, random_state=seed)
    Xmoons += np.array([1.75, 1.25]) # Offset moons to keep all bounds in the positive quadrant for simplicity

    center = [(4, 3), (49, 39), (4999, 3999)][cfg.dataset.center] # NOTE: center argument to dataset is an index 0, 1, 2 corresponds to standard, outliers, and extreme
    Xblob, yblob = make_blobs(n_samples=[(cfg.dataset.n_samples // 20)], centers=[center], cluster_std=0.175, random_state=seed)

    X = np.concatenate([Xmoons, Xblob])
    y = np.concatenate([ymoons, [2] * len(yblob)])

    sp = _train_val_test_split(
        X,
        y,
        test_size=getattr(cfg.dataset, "test_size", 0.1),
        val_size=getattr(cfg.dataset, "val_size", 0.1),
        random_state=getattr(cfg.dataset, "random_state", 42),
        stratify=True,
    )

    data = {
        "X_train": sp["X_train"],
        "X_val": sp["X_val"],
        "X_test": sp["X_test"],
        "y_train": sp["y_train"],
        "y_val": sp["y_val"],
        "y_test": sp["y_test"],
        "bounds" : ([0,0],[center[0] + 1,center[1] + 1]), 
        "features" : ['Feature 1', 'Feature 2'],
        "categorical" : []
    }

    return data
