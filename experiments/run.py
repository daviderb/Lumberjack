from pathlib import Path
import sys

# Ensure local src/ is importable even if .pth processing is skipped.
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "src"))
# The diffprivlib baseline is vendored under src/differential-privacy-library.
sys.path.insert(0, str(repo_root / "src" / "differential-privacy-library"))

from dp_random_forest.algorithms import get_model
from dp_random_forest.datasets import get_dataset
from dp_random_forest.datasets.splits import concat_train_val
from dp_random_forest.utils.cache_results import cache_results
from dp_random_forest.utils.plotting import (
    evaluate_fitted_forest,
    plot_leaf_size_distribution,
)
from dp_random_forest.utils.tree_stats import write_tree_stats_report
import hydra
from omegaconf import DictConfig

TREE_STATS_MODELS = {
    "dp_sparse_random_forest",
}


@hydra.main(version_base=None, config_path="conf", config_name="random_forest.yaml")
def main(cfg: DictConfig):
    data = get_dataset(cfg)
    X, y = concat_train_val(data)
    forest = get_model(cfg, data)
    print("fitting the model")
    forest.fit(X, y)

    train_acc = forest.score(X, y)
    test_acc = forest.score(data["X_test"], data["y_test"])
    print(f"Training Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")

    # if cfg.model.name in TREE_STATS_MODELS:
    #     tree_stats_path = write_tree_stats_report(forest, X, y=y)
    #     print(f"Saved tree stats: {tree_stats_path}")
    #     leaf_size_plot_path = plot_leaf_size_distribution(forest, X)
    #     if leaf_size_plot_path is not None:
    #         print(f"Saved leaf size distribution: {leaf_size_plot_path}")
    # else:
    #     print(f"Skipping tree stats for model: {cfg.model.name}")

    # plotting_cfg = cfg.get("plotting")
    # plotting_enabled = plotting_cfg is None or plotting_cfg.get("enabled", True)
    # show_plots = plotting_cfg is not None and plotting_cfg.get("show", False)
    # if plotting_enabled:
    #     plot_outputs = evaluate_fitted_forest(
    #         forest,
    #         data=data,
    #         dataset_name=cfg.dataset.name,
    #         show=show_plots,
    #     )
    #     print("Saved evaluation plots:")
    #     for key, value in plot_outputs.items():
    #         if key.endswith("_plot"):
    #             print(f"  {key}: {value}")


    cache_results(f"forest", forest)


if __name__ == "__main__":
    main()
