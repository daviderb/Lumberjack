import numpy as np
import pandas as pd
from pathlib import Path

from scipy.sparse import csr_matrix

from .splits import train_val_test_split as _train_val_test_split
from .uci import make_bad_bounds


def _resolve_task(task_name, folktables_module):
    task_map = {
        "income": folktables_module.ACSIncome,
        "employment": folktables_module.ACSEmployment,
        "public_coverage": folktables_module.ACSPublicCoverage,
        "mobility": folktables_module.ACSMobility,
        "travel_time": folktables_module.ACSTravelTime,
    }

    normalized = str(task_name).strip().lower()
    if normalized not in task_map:
        valid = ", ".join(sorted(task_map))
        raise ValueError(
            f"Unsupported folktables task '{task_name}'. Expected one of: {valid}."
        )
    return normalized, task_map[normalized]


def _resolve_states(raw_states):
    if isinstance(raw_states, str):
        return [raw_states.upper()]
    return [str(state).upper() for state in raw_states]


def _infer_categorical_features(acs_frame, feature_names, mode, max_unique):
    normalized_mode = str(mode).strip().lower()
    if normalized_mode == "none":
        return []
    if normalized_mode == "all":
        return list(feature_names)
    if normalized_mode != "auto":
        raise ValueError(
            "dataset.categorical_mode must be one of: auto, none, all."
        )

    categorical = []
    for feature in feature_names:
        if feature not in acs_frame.columns:
            continue
        series = acs_frame[feature]
        if pd.api.types.is_bool_dtype(series):
            categorical.append(feature)
            continue
        if pd.api.types.is_object_dtype(series) or pd.api.types.is_categorical_dtype(series):
            categorical.append(feature)
            continue
        if pd.api.types.is_integer_dtype(series) and series.nunique(dropna=True) <= max_unique:
            categorical.append(feature)
    return categorical


def _encode_folktables_categoricals(
    X_frame,
    feature_names,
    categorical_features,
    one_hot_encode=False,
    sparse_dtype=True,
):
    categorical_features = list(categorical_features)
    categorical_set = set(categorical_features)
    continuous_features = [
        feature for feature in feature_names if feature not in categorical_set
    ]

    X_cont = X_frame[continuous_features].apply(pd.to_numeric, errors="coerce")
    if continuous_features:
        X_cont = X_cont.fillna(X_cont.median())
    else:
        X_cont = pd.DataFrame(index=X_frame.index)

    if not categorical_features:
        return X_cont, continuous_features, [], None

    X_cat = X_frame[categorical_features].copy()
    modes = X_cat.mode(dropna=True)
    fill_values = (
        modes.iloc[0].to_dict()
        if not modes.empty
        else {col: "missing" for col in categorical_features}
    )
    X_cat = X_cat.fillna(fill_values)

    categorical_sizes = {}
    X_cat_encoded = pd.DataFrame(index=X_frame.index)
    X_cat_one_hot = pd.DataFrame(index=X_frame.index)
    one_hot_feature_names = []

    for feature_name in categorical_features:
        encoded_values, unique_values = pd.factorize(
            X_cat[feature_name], sort=True
        )
        if np.any(encoded_values < 0):
            raise ValueError(
                f"Folktables categorical feature '{feature_name}' contains unexpected unknown values after preprocessing."
            )

        encoded_values = encoded_values.astype(int)
        X_cat_encoded[feature_name] = encoded_values.astype(float)

        if len(unique_values) >= 2:
            categorical_sizes[feature_name] = int(len(unique_values))

        if not one_hot_encode:
            continue

        if sparse_dtype:
            for category in range(1, len(unique_values)):
                column_name = f"{feature_name}_{category}"
                # Store only the 1s, leaving 0s unallocated in memory
                dense_array = (encoded_values == category).astype(float)
                X_cat_one_hot[column_name] = pd.arrays.SparseArray(dense_array, fill_value=0.0)
                one_hot_feature_names.append(column_name)
        else:
            for category in range(1, len(unique_values)):
                column_name = f"{feature_name}_{category}"
                X_cat_one_hot[column_name] = (encoded_values == category).astype(float)
                one_hot_feature_names.append(column_name)

    if not one_hot_encode:
        X_processed = pd.concat([X_cont, X_cat_encoded], axis=1)
        processed_feature_names = continuous_features + categorical_features
        X_processed = X_processed[processed_feature_names]
        return X_processed, processed_feature_names, categorical_features, None

    X_processed = pd.concat([X_cont, X_cat_one_hot], axis=1)
    processed_feature_names = continuous_features + one_hot_feature_names
    X_processed = X_processed[processed_feature_names]
    return X_processed, processed_feature_names, one_hot_feature_names, categorical_sizes


def get_folktables_dataset(cfg):
    try:
        import folktables
    except ImportError as exc:
        raise ImportError(
            "folktables is required for dataset=folktables. "
            "Install it with `pip install folktables`."
        ) from exc

    repo_root = Path(__file__).resolve().parents[3]
    data_root = Path(getattr(cfg.dataset, "cache_dir", repo_root / "data"))

    task_name = getattr(cfg.dataset, "task", "income")
    resolved_task_name, task = _resolve_task(task_name, folktables)
    states = _resolve_states(getattr(cfg.dataset, "states", ["CA"]))

    data_source = folktables.ACSDataSource(
        survey_year=str(getattr(cfg.dataset, "survey_year", 2018)),
        horizon=getattr(cfg.dataset, "horizon", "1-Year"),
        survey=getattr(cfg.dataset, "survey", "person"),
        root_dir=str(data_root),
    )
    download = bool(getattr(cfg.dataset, "download", False))
    try:
        acs_data = data_source.get_data(states=states, download=download)
    except Exception as exc:
        if not download:
            raise RuntimeError(
                "Folktables cache not found locally. "
                "Run scripts/download_all_datasets.py first or set dataset.download=true."
            ) from exc
        raise

    X, y, group = task.df_to_numpy(acs_data)

    X = np.asarray(X, dtype=float)
    y = np.asarray(y).astype(int)
    group = np.asarray(group)
    print("Dataset length: ", len(X))
    feature_names = list(getattr(task, "features", []))
    if not feature_names:
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]

    categorical_features = _infer_categorical_features(
        acs_data,
        feature_names,
        mode=getattr(cfg.dataset, "categorical_mode", "auto"),
        max_unique=int(getattr(cfg.dataset, "categorical_max_unique", 32)),
    )
    one_hot_encode = bool(getattr(cfg.dataset, "one_hot_encode", False))
    sparse_dtype = bool(getattr(cfg.dataset, "sparse_dtype", True))

    if one_hot_encode:
        X_processed, processed_feature_names, categorical_feature_names, categorical_sizes = _encode_folktables_categoricals(
            pd.DataFrame(X, columns=feature_names),
            feature_names,
            categorical_features,
            one_hot_encode=True,
            sparse_dtype=sparse_dtype,
        )
        if sparse_dtype:
            X_final = csr_matrix(X_processed)
        else:
            X_final = X_processed.to_numpy(dtype=float)

        if sparse_dtype:
            # Size of the dense array before encoding
            print(f"Size before one-hot encoding: {X.nbytes / (1024 ** 2):.2f} MB")
            
            # Size of the sparse matrix components combined
            sparse_bytes = X_final.data.nbytes + X_final.indptr.nbytes + X_final.indices.nbytes
            print(f"Size after sparse one-hot encoding: {sparse_bytes / (1024 ** 2):.2f} MB")
    else:
        X_final = X
        processed_feature_names = feature_names
        categorical_feature_names = list(categorical_features)
        categorical_sizes = None

    rs = int(getattr(cfg.dataset, "random_state", 42))
    sp = _train_val_test_split(
        X_final,
        y,
        group,
        test_size=float(getattr(cfg.dataset, "test_size", 0.1)),
        val_size=float(getattr(cfg.dataset, "val_size", 0.1)),
        random_state=rs,
        stratify=True,
    )

    data = {
        "X_train": sp["X_train"],
        "X_val": sp["X_val"],
        "X_test": sp["X_test"],
        "y_train": sp["y_train"],
        "y_val": sp["y_val"],
        "y_test": sp["y_test"],
        "bounds": make_bad_bounds(X_final),
        "features": processed_feature_names,
        "feature_names": processed_feature_names,
        "categorical": list(categorical_features),
        "categorical_features": categorical_feature_names,
        "group_train": sp["group_train"],
        "group_val": sp["group_val"],
        "group_test": sp["group_test"],
        "metadata": {
            "source": "folktables",
            "task": resolved_task_name,
            "states": states,
            "survey_year": str(getattr(cfg.dataset, "survey_year", 2018)),
            "horizon": getattr(cfg.dataset, "horizon", "1-Year"),
            "survey": getattr(cfg.dataset, "survey", "person"),
            "one_hot_encode": one_hot_encode,
            "sparse_dtype": sparse_dtype,
        },
    }
    if categorical_sizes is not None:
        data["categorical_sizes"] = categorical_sizes
    return data
