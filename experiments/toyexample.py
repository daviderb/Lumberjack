from pathlib import Path
import sys
import numpy as np

# Ensure local src/ is importable even if .pth processing is skipped.
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "src"))

from dp_random_forest.algorithms import get_model
from dp_random_forest.datasets import get_dataset
from dp_random_forest.datasets.splits import concat_train_val
from dp_random_forest.utils.cache_results import cache_results
from dp_random_forest.utils.plotting import (
    evaluate_fitted_forest,
    plot_moons_unbalanced_decision_regions,
    plot_leaf_size_distribution,
)
from dp_random_forest.utils.tree_stats import write_tree_stats_report
import hydra
from omegaconf import DictConfig


TREE_STATS_MODELS = {
    "dp_sparse_random_forest",
}

from dp_random_forest.algorithms import DPSparseExtraTreesClassifier

def getSparseForest(bounds, feature_names, categorical_features):
    return DPSparseExtraTreesClassifier(
            n_estimators=25,
            epsilon=2,
            delta=10**-6,
            splitting_fraction=0.75,
            leaf_algorithm="exponential_mechanism",
            max_depth=100,
            random_state=123,
            bounds=bounds,
            feature_names=feature_names,
            categorical_features=categorical_features,
        )

def getFullyRandomForest(bounds, feature_names, categorical_features):
    return DPExtraTreesClassifier(
            n_estimators=25,
            splitbudget=None,
            leafbudget=None,
            splitting_algorithm="full_tree",
            leaf_algorithm="exponential_mechanism",
            max_depth=10,
            random_state=123,
            bounds=bounds,
            feature_names=feature_names,
            categorical_features=categorical_features,
        )

@hydra.main(version_base=None, config_path="conf", config_name="random_forest.yaml")
def main(cfg: DictConfig):
    cfg.dataset.name = "moonsunbalanced"
    cfg.dataset.n_samples = 10000
    cfg.dataset.noise = 0.125
    cfg.dataset.random_state = 10000
    
    for center in [0,1,2]:
        cfg.dataset.center = center
        data = get_dataset(cfg)
        X, y = concat_train_val(data)
        X_full = np.vstack((X, data["X_test"]))
        y_full = np.concatenate((y, data["y_test"]))
        
        forest = getFullyRandomForest(data["bounds"], data.get("feature_names"), data.get("categorical"))
        forest.fit(X_full, y_full)

        plot_moons_unbalanced_decision_regions(
                forest,
                X_full,
                y_full,
                show=True,
                filename = f'moons_center{center}_them.png'
            )
        
        forest = getSparseForest(data["bounds"], data.get("feature_names"), data.get("categorical"))
        forest.fit(X_full, y_full)

        plot_moons_unbalanced_decision_regions(
                forest,
                X_full,
                y_full,
                show=True,
                filename = f'moons_center{center}_us.png'
            )

    cache_results(f"forest", forest)


if __name__ == "__main__":
    main()
