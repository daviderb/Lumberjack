from typing import Any

import numpy as np


from dp_random_forest.algorithms.diprimeflip_forest import DiPriMeFlipForestClassifier

from dp_random_forest.algorithms.notebook_dprf import NotebookDPRFClassifier
from dp_random_forest.algorithms.smooth_sensitivity_dp_rf import (
    SmoothSensitivityDPRandomForestClassifier,
)
from dp_random_forest.algorithms.snr_dp_forest import SNRDPForestClassifier

from dp_random_forest.algorithms.dp_sparse_forest import DPSparseExtraTreesClassifier




def _cfg_get(config: Any, key: str, default: Any = None) -> Any:
    getter = getattr(config, "get", None)
    if callable(getter):
        return getter(key, default)
    return getattr(config, key, default)


def get_model(cfg: Any, data: dict) -> Any:
    model_cfg = cfg.model
    model_name = model_cfg.name
    bounds = data["bounds"]
    splitbudget_cfg = _cfg_get(model_cfg, "splitbudget")
    leafbudget_cfg = _cfg_get(model_cfg, "leafbudget")
    epsilon = _cfg_get(model_cfg, "epsilon")
    delta = _cfg_get(model_cfg, "delta")
    splitting_fraction = _cfg_get(model_cfg, "splitting_fraction")
    
    if model_name in ["dp_sparse_random_forest"]:
        feature_names = data.get("feature_names") or data.get("features")
        categorical_features = (
            data.get("categorical_sizes")
            or data.get("categorical_features")
            or data.get("categorical")
            or []
        )
        return DPSparseExtraTreesClassifier(
            n_estimators=model_cfg.n_estimators,
            leafbudget=leafbudget_cfg,
            epsilon=model_cfg.epsilon,
            delta=model_cfg.delta,
            splitting_fraction=model_cfg.splitting_fraction,
            split_threshold=model_cfg.split_threshold,
            leaf_algorithm=model_cfg.leaf_algorithm,
            max_depth=model_cfg.max_depth,
            random_state=model_cfg.random_state,
            n_jobs=model_cfg.n_jobs,
            verbose=model_cfg.verbose,
            bounds=bounds,
            feature_names=feature_names,
            categorical_features=categorical_features,
            split_weights=data.get("split_weights") or data.get("feature_split_weights"),
        )
    if model_name == "sklearn_random_forest":
        from sklearn.ensemble import RandomForestClassifier as sklearnRF

        return sklearnRF(
            n_estimators=model_cfg.get("n_estimators", 100),
            max_depth=model_cfg.get("max_depth"),
            min_samples_split=model_cfg.get("min_samples_split", 2),
            min_samples_leaf=model_cfg.get("min_samples_leaf", 1),
            max_features=model_cfg.get("max_features", "sqrt"),
            bootstrap=model_cfg.get("bootstrap", True),
            criterion=model_cfg.get("criterion", "gini"),
            n_jobs=model_cfg.get("n_jobs", -1),
            random_state=model_cfg.get("random_state", 42),
            verbose=model_cfg.get("verbose", 0),
        )
    if model_name == "sklearn_extra_trees":
        from sklearn.ensemble import ExtraTreesClassifier

        return ExtraTreesClassifier(
            n_estimators=model_cfg.get("n_estimators", 100),
            max_depth=model_cfg.get("max_depth"),
            min_samples_split=model_cfg.get("min_samples_split", 2),
            min_samples_leaf=model_cfg.get("min_samples_leaf", 1),
            max_features=model_cfg.get("max_features", "sqrt"),
            bootstrap=model_cfg.get("bootstrap", False), 
            criterion=model_cfg.get("criterion", "gini"),
            n_jobs=model_cfg.get("n_jobs", -1),
            random_state=model_cfg.get("random_state", 42),
            verbose=model_cfg.get("verbose", 0),
        )

    if model_name == "sklearn_decision_tree":
        from sklearn.tree import DecisionTreeClassifier

        return DecisionTreeClassifier(
            max_depth=model_cfg.get("max_depth"),
            min_samples_split=model_cfg.get("min_samples_split", 2),
            min_samples_leaf=model_cfg.get("min_samples_leaf", 1),
            max_features=model_cfg.get("max_features", None),
            criterion=model_cfg.get("criterion", "gini"),
            random_state=model_cfg.get("random_state", 42),
        )

    raise ValueError(f"Unsupported model.name '{model_name}'")
