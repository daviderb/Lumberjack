import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.sparse import issparse
from sklearn.preprocessing import OrdinalEncoder

from .splits import train_val_test_split as _train_val_test_split


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve_uci_cache_dir(dataset_id, dataset_name=None, cache_dir=None):
    if cache_dir is None:
        base_dir = _repo_root() / "data" / "uci"
    else:
        base_dir = Path(cache_dir)

    candidates = []
    if dataset_name:
        candidates.append(base_dir / str(dataset_name))
    candidates.append(base_dir / str(dataset_id))
    candidates.append(base_dir / f"id_{dataset_id}")

    for candidate in candidates:
        if (candidate / "features.csv").exists() and (candidate / "targets.csv").exists():
            return candidate
    return candidates[0]


def _load_cached_uci_frames(cache_dir):
    features_path = cache_dir / "features.csv"
    targets_path = cache_dir / "targets.csv"
    metadata_path = cache_dir / "metadata.json"

    X_raw = pd.read_csv(features_path)
    y_raw_df = pd.read_csv(targets_path)
    if y_raw_df.shape[1] == 1:
        y_raw = y_raw_df.iloc[:, 0]
    else:
        y_raw = y_raw_df.iloc[:, 0]

    metadata = {"source": "local_cache", "cache_dir": str(cache_dir)}
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)

    dataset = SimpleNamespace(metadata=metadata)
    return dataset, X_raw, y_raw


def _write_uci_cache(dataset, X_raw, y_raw, cache_dir):
    cache_dir.mkdir(parents=True, exist_ok=True)
    X_raw.to_csv(cache_dir / "features.csv", index=False)
    pd.DataFrame(y_raw).to_csv(cache_dir / "targets.csv", index=False)
    with (cache_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(dataset.metadata, handle, indent=2)


def _fetch_openml_fallback(dataset_name):
    if not dataset_name:
        raise ValueError("dataset_name is required for OpenML fallback.")

    openml_name_map = {
        "adult": "adult",
        "banknote": "banknote-authentication",
        "mushroom": "mushroom",
        "nursery": "nursery",
        "vehicle": "vehicle",
        "wbc": "breast-w",
    }
    if dataset_name not in openml_name_map:
        raise ValueError(
            f"No OpenML fallback is configured for dataset_name='{dataset_name}'."
        )

    from sklearn.datasets import fetch_openml

    openml_name = openml_name_map[dataset_name]
    dataset = fetch_openml(
        name=openml_name,
        version=1,
        as_frame=True,
        parser="auto",
    )
    X_raw = dataset.data
    y_raw = dataset.target
    if not isinstance(X_raw, pd.DataFrame):
        X_raw = pd.DataFrame(X_raw)
    if isinstance(y_raw, pd.DataFrame):
        y_raw = y_raw.iloc[:, 0]
    else:
        y_raw = pd.Series(y_raw)

    metadata = {
        "source": "openml_fallback",
        "openml_name": openml_name,
    }
    return SimpleNamespace(metadata=metadata), X_raw, y_raw


def fetch_uci_frames(dataset_id, dataset_name=None, cache_dir=None, allow_download=True):
    resolved_cache_dir = _resolve_uci_cache_dir(
        dataset_id,
        dataset_name=dataset_name,
        cache_dir=cache_dir,
    )
    if (resolved_cache_dir / "features.csv").exists() and (
        resolved_cache_dir / "targets.csv"
    ).exists():
        return _load_cached_uci_frames(resolved_cache_dir)

    if not allow_download:
        raise FileNotFoundError(
            f"Cached UCI dataset not found for id={dataset_id}. "
            f"Expected files in '{resolved_cache_dir}'. "
            "Run scripts/download_all_datasets.py first or set dataset.download=true."
        )

    from ucimlrepo import fetch_ucirepo

    try:
        dataset = fetch_ucirepo(id=dataset_id)
        X_raw = dataset.data.features.copy()
        y_raw = dataset.data.targets.copy()
    except Exception:
        dataset, X_raw, y_raw = _fetch_openml_fallback(dataset_name)

    if not isinstance(X_raw, pd.DataFrame):
        X_raw = pd.DataFrame(X_raw)
    if isinstance(y_raw, pd.DataFrame):
        y_raw = y_raw.iloc[:, 0]
    else:
        y_raw = pd.Series(y_raw)

    _write_uci_cache(dataset, X_raw, y_raw, resolved_cache_dir)
    return dataset, X_raw, y_raw


def encode_target_series(y_raw, replacements=None):
    y = y_raw.astype(str).str.strip()
    if replacements:
        y = y.replace(replacements)

    labels = sorted(y.dropna().unique())
    label_mapping = {label: idx for idx, label in enumerate(labels)}
    y_encoded = y.map(label_mapping)

    if y_encoded.isna().any():
        raise ValueError("Dataset target contains unexpected labels.")

    return y_encoded.astype(int), label_mapping


def preprocess_uci_features(X_raw):
    X_raw = X_raw.replace("?", np.nan)

    categorical_columns = list(
        X_raw.select_dtypes(
            include=["object", "string", "category", "bool"]
        ).columns
    )
    continuous_columns = [
        column for column in X_raw.columns if column not in categorical_columns
    ]

    X_cont = X_raw[continuous_columns].apply(pd.to_numeric, errors="coerce")
    if continuous_columns:
        X_cont = X_cont.fillna(X_cont.median())
    else:
        X_cont = pd.DataFrame(index=X_raw.index)

    X_cat = X_raw[categorical_columns].copy()
    if categorical_columns:
        modes = X_cat.mode(dropna=True)
        fill_values = (
            modes.iloc[0].to_dict()
            if not modes.empty
            else {col: "missing" for col in categorical_columns}
        )
        X_cat = X_cat.fillna(fill_values)
        encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            encoded_missing_value=-1,
        )
        X_cat_encoded = encoder.fit_transform(X_cat)
        X_cat_encoded = pd.DataFrame(
            X_cat_encoded,
            columns=categorical_columns,
            index=X_cat.index,
        )
    else:
        X_cat_encoded = pd.DataFrame(index=X_raw.index)

    feature_names = continuous_columns + categorical_columns
    X_processed = pd.concat([X_cont, X_cat_encoded], axis=1)[feature_names]

    return X_processed, feature_names, categorical_columns


def make_bad_bounds(X, scale=1.0, floor_padding=10.0, add_padding=False):
    if issparse(X):
        X = X.astype(float, copy=False)
        lower = X.min(axis=0).toarray().ravel()
        upper = X.max(axis=0).toarray().ravel()
    else:
        X = np.asarray(X, dtype=float)
        lower = np.min(X, axis=0)
        upper = np.max(X, axis=0)
    span = upper - lower
    padding = np.where(span > 0, scale * span + 1.0, floor_padding)
    if add_padding:
        return lower - padding, upper + padding
    else:
        return lower, upper


def get_uci_dataset(cfg, dataset_id, target_replacements=None, dataset_name=None):
    dataset, X_raw, y_raw = fetch_uci_frames(
        dataset_id,
        dataset_name=dataset_name,
        cache_dir=getattr(cfg.dataset, "cache_dir", None),
        allow_download=bool(getattr(cfg.dataset, "download", False)),
    )

    X_processed, feature_names, categorical_columns = preprocess_uci_features(X_raw)
    y, label_mapping = encode_target_series(
        y_raw,
        replacements=target_replacements,
    )

    sp = _train_val_test_split(
        X_processed.to_numpy(dtype=float),
        y.to_numpy(dtype=int),
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
        "bounds": make_bad_bounds(X_processed.to_numpy(dtype=float)),
        "features": feature_names,
        "categorical": categorical_columns,
        "metadata": dataset.metadata,
        "label_mapping": label_mapping,
    }

    return data
