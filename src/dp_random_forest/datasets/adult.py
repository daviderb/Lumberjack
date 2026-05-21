import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

from .splits import train_val_test_split as _train_val_test_split
from .uci import fetch_uci_frames, make_bad_bounds


def _encode_adult_categoricals(X_cat, categoricals_dict, one_hot_encode=False):
    categorical_columns = list(categoricals_dict.keys())
    if not categorical_columns:
        return pd.DataFrame(index=X_cat.index), [], None

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

    if not one_hot_encode:
        return X_cat_encoded, categorical_columns, None

    X_cat_one_hot = pd.DataFrame(index=X_cat.index)
    one_hot_feature_names = []
    for feature_name, size in categoricals_dict.items():
        encoded_values = X_cat_encoded[feature_name].to_numpy()
        if np.any(encoded_values < 0):
            raise ValueError(
                f"Adult categorical feature '{feature_name}' contains unexpected unknown values after preprocessing."
            )
        if np.any(encoded_values >= size):
            raise ValueError(
                f"Adult categorical feature '{feature_name}' exceeds configured size {size}."
            )

        encoded_values = encoded_values.astype(int)
        for category in range(1, size):
            column_name = f"{feature_name}_{category}"
            X_cat_one_hot[column_name] = (encoded_values == category).astype(float)
            one_hot_feature_names.append(column_name)

    return X_cat_one_hot, one_hot_feature_names, dict(categoricals_dict)


def get_adult_dataset(cfg):
    adult, X_raw, y_raw = fetch_uci_frames(
        dataset_id=2,
        dataset_name="adult",
        cache_dir=getattr(cfg.dataset, "cache_dir", None),
        allow_download=bool(getattr(cfg.dataset, "download", False)),
    )

    if not isinstance(X_raw, pd.DataFrame):
        X_raw = pd.DataFrame(X_raw)
    if isinstance(y_raw, pd.DataFrame):
        y_raw = y_raw.iloc[:, 0]
    else:
        y_raw = pd.Series(y_raw)

    X_raw = X_raw.replace("?", np.nan)

    #categorical_columns = list(
    #    X_raw.select_dtypes(include=["object", "category"]).columns
    #)
    categoricals_dict = { # the number indicates the size of the domain: x means that the categorical takes values in {0, 1, ..., x - 1}
        'workclass': 8, 
        'education': 16, 
        'marital-status': 7, 
        'occupation': 14, 
        'relationship': 6, 
        'race': 5, 
        'sex': 2,
        'native-country': 41,
    }
    categorical_columns = list(categoricals_dict.keys())
    one_hot_encode = bool(getattr(cfg.dataset, "one_hot_encode", False))


    continuous_columns = [
        column for column in X_raw.columns if column not in categorical_columns
    ]

    X_cont = X_raw[continuous_columns].apply(pd.to_numeric, errors="coerce")
    X_cont = X_cont.fillna(X_cont.median())

    X_cat = X_raw[categorical_columns].copy()
    X_cat_processed, categorical_feature_names, categorical_sizes = _encode_adult_categoricals(
        X_cat,
        categoricals_dict,
        one_hot_encode=one_hot_encode,
    )

    X_processed = pd.concat([X_cont, X_cat_processed], axis=1)
    feature_names = continuous_columns + categorical_feature_names
    X_processed = X_processed[feature_names]

    y = (
        y_raw.astype(str)
        .str.strip()
        .replace({"<=50K.": "<=50K", ">50K.": ">50K"})
        .map({"<=50K": 0, ">50K": 1})
    )
    if y.isna().any():
        raise ValueError("Adult dataset target contains unexpected labels.")

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
        "feature_names": feature_names,
        "categorical": categorical_columns,
        "categorical_features": categorical_feature_names,
        "metadata": adult.metadata,
    }
    if categorical_sizes is not None:
        data["categorical_sizes"] = categorical_sizes

    return data
