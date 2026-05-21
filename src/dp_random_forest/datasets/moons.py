from sklearn.datasets import make_moons

from .splits import train_val_test_split as _train_val_test_split


def get_moons(cfg):
    X, y = make_moons(n_samples=cfg.dataset.n_samples, noise=cfg.dataset.noise, random_state=42)

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
        "bounds" : ([-2,-2],[3,3]), 
        "features" : ['Feature 1', 'Feature 2'],
        "categorical" : []
    }

    return data
