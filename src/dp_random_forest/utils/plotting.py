import copy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import hydra
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.tree import _tree, plot_tree
from mpl_toolkits.axes_grid1.inset_locator import inset_axes



def _resolve_output_dir(outdir: Optional[Union[str, Path]] = None) -> Path:
    if outdir is not None:
        output_dir = Path(outdir)
    else:
        try:
            hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
            output_dir = Path(hydra_cfg["runtime"]["output_dir"])
        except ValueError:
            output_dir = Path.cwd()

    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_plot(
    fig: Any,
    filename: str,
    show: bool = False,
    outdir: Optional[Union[str, Path]] = None,
    dpi: int = 200,
) -> Path:
    output_dir = _resolve_output_dir(outdir)
    path = output_dir / filename
    fig.savefig(path, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()

    plt.close(fig)
    return path


def _get_tree_estimator(model: Any) -> Any:
    if hasattr(model, "estimators_") and len(model.estimators_) > 0:
        return model.estimators_[0]
    return model


def _make_plot_tree_copy(estimator: Any) -> Any:
    estimator_copy = copy.deepcopy(estimator)
    values = np.asarray(estimator_copy.tree_.value, dtype=float)

    # sklearn's filled tree plot expects classification values to behave like
    # per-class proportions. Our DP tree can store noisy counts, including
    # negative values, so normalize a plotting-only copy into valid fractions.
    clipped = np.clip(values, 0.0, None)
    totals = clipped.sum(axis=2, keepdims=True)
    valid = totals.squeeze(axis=2) > 0

    if np.any(valid):
        clipped[valid] /= totals[valid]

    invalid_rows = ~valid[:, 0]
    if np.any(invalid_rows):
        original = values[invalid_rows, 0, :]
        winners = np.argmax(original, axis=1)
        fallback = np.zeros_like(original)
        fallback[np.arange(len(winners)), winners] = 1.0
        clipped[invalid_rows, 0, :] = fallback

    estimator_copy.tree_.value[:] = clipped
    return estimator_copy


def plot_accuracy_summary(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    show: bool = False,
    outdir: Optional[Union[str, Path]] = None,
) -> Tuple[float, float, Path]:
    train_acc = float(model.score(X_train, y_train))
    test_acc = float(model.score(X_test, y_test))

    fig, ax = plt.subplots(figsize=(6, 4))
    names = ["Train", "Test"]
    values = [train_acc, test_acc]
    bars = ax.bar(names, values, color=["#3A86FF", "#FF5D8F"])

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            value + 0.02,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Accuracy")
    ax.set_title("Forest Accuracy")

    path = save_plot(fig, "accuracy-summary.png", show=show, outdir=outdir)
    return train_acc, test_acc, path


def plot_confusion_matrix(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    show: bool = False,
    outdir: Optional[Union[str, Path]] = None,
) -> Path:
    y_pred = np.asarray(model.predict(X_test))
    labels = np.unique(np.concatenate((np.asarray(y_test), y_pred)))
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Test Confusion Matrix")
    path = save_plot(fig, "confusion-matrix.png", show=show, outdir=outdir)
    return path


def plot_first_tree(
    model: Any,
    feature_names: Optional[List[str]] = None,
    show: bool = False,
    outdir: Optional[Union[str, Path]] = None,
) -> Optional[Path]:
    estimator = _get_tree_estimator(model)
    if not hasattr(estimator, "tree_"):
        return None

    estimator = _make_plot_tree_copy(estimator)

    fig, ax = plt.subplots(figsize=(14, 8))
    plot_tree(estimator, feature_names=feature_names, class_names=None, filled=True, ax=ax)
    ax.set_title("First Tree in Forest")
    path = save_plot(fig, "first-tree.png", show=show, outdir=outdir)
    return path


def plot_feature_importance(
    model: Any,
    feature_names: Optional[List[str]] = None,
    show: bool = False,
    outdir: Optional[Union[str, Path]] = None,
) -> Optional[Path]:
    if not hasattr(model, "feature_importances_"):
        return None

    importances = np.asarray(model.feature_importances_)
    if importances.ndim != 1 or importances.size == 0:
        return None

    if feature_names and len(feature_names) == importances.size:
        names = feature_names
    else:
        names = [f"feature_{i}" for i in range(importances.size)]

    order = np.argsort(importances)[::-1]
    ordered_importances = importances[order]
    ordered_names = [names[i] for i in order]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(range(len(ordered_names)), ordered_importances, color="#00A896")
    ax.set_xticks(range(len(ordered_names)))
    ax.set_xticklabels(ordered_names, rotation=40, ha="right")
    ax.set_ylabel("Importance")
    ax.set_title("Feature Importances")
    path = save_plot(fig, "feature-importances.png", show=show, outdir=outdir)
    return path


def _collect_nonempty_leaf_sizes(model: Any, X: np.ndarray) -> np.ndarray:
    estimators = getattr(model, "estimators_", None)
    if estimators is None:
        estimators = [model]

    leaf_sizes = []
    for estimator in estimators:
        if not hasattr(estimator, "tree_") or not hasattr(estimator, "apply"):
            continue

        tree = estimator.tree_
        leaf_nodes = np.flatnonzero(tree.children_left == _tree.TREE_LEAF)
        assigned_leaves = np.asarray(estimator.apply(X), dtype=int).reshape(-1)
        leaf_sizes_by_node = np.bincount(
            assigned_leaves,
            minlength=tree.node_count,
        )
        nonempty_leaf_sizes = leaf_sizes_by_node[leaf_nodes]
        nonempty_leaf_sizes = nonempty_leaf_sizes[nonempty_leaf_sizes > 0]
        if nonempty_leaf_sizes.size:
            leaf_sizes.append(nonempty_leaf_sizes)

    if not leaf_sizes:
        return np.array([], dtype=int)

    return np.concatenate(leaf_sizes)


def plot_leaf_size_distribution(
    model: Any,
    X: np.ndarray,
    show: bool = False,
    outdir: Optional[Union[str, Path]] = None,
) -> Optional[Path]:
    leaf_sizes = _collect_nonempty_leaf_sizes(model, X)
    if leaf_sizes.size == 0:
        return None

    max_size = int(np.max(leaf_sizes))
    if max_size <= 50:
        bins = np.arange(1, max_size + 2) - 0.5
        use_log_x = False
    else:
        bins = np.logspace(0, np.log10(max_size + 1), num=40)
        use_log_x = True

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(leaf_sizes, bins=bins, color="#3A86FF", edgecolor="white")
    ax.set_title("Nonempty Leaf Size Distribution")
    ax.set_xlabel("Leaf size (training samples)")
    ax.set_ylabel("Number of leaves")
    if use_log_x:
        ax.set_xscale("log")
        ax.set_xlim(1, max_size + 1)
    else:
        ax.set_xlim(0.5, max_size + 0.5)
    ax.grid(axis="y", alpha=0.25)

    path = save_plot(fig, "leaf-size-distribution.png", show=show, outdir=outdir)
    return path


def plot_moons_decision_regions(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    show: bool = False,
    outdir: Optional[Union[str, Path]] = None,
) -> Path:
    xx, yy = _make_moons_mesh(X)
    grid = np.c_[xx.ravel(), yy.ravel()]
    pred = np.asarray(model.predict(grid)).reshape(xx.shape)

    # 3 classes now: red, blue, green
    region_cmap = ListedColormap(["#F8C8C8", "#CFE3FF", "#C8F8C8"])
    point_cmap = ListedColormap(["#C61A1A", "#1555B5", "#1A9C3A"])

    fig, ax = plt.subplots(figsize=(8, 6))

    # levels for classes 0, 1, 2
    levels = [-0.5, 0.5, 1.5, 2.5]

    ax.contourf(xx, yy, pred, levels=levels, cmap=region_cmap, alpha=0.85)
    ax.contour(xx, yy, pred, levels=[0.5, 1.5], colors="#1F1F1F", linewidths=1.0)

    ax.scatter(
        X[:, 0], X[:, 1],
        c=y,
        cmap=point_cmap,
        s=22,
        edgecolors="white",
        linewidths=0.25
    )

    ax.set_title("Moons: Decision Regions (3 Classes)")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")

    path = save_plot(fig, "moons-decision-regions.png", show=show, outdir=outdir)
    return path

def plot_moons_unbalanced_decision_regions(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    show: bool = False,
    outdir: Optional[Union[str, Path]] = None,
) -> Path:
    fig, ax = plt.subplots(figsize=(8, 6))

    # ---- MAIN REGION (classes 0 and 1 only) ----
    x_min, x_max = 0, 5
    y_min, y_max = 0, 4

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 400),
        np.linspace(y_min, y_max, 400)
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    pred = np.asarray(model.predict(grid)).reshape(xx.shape)

    region_cmap = ListedColormap(["#F8C8C8", "#CFE3FF", "#C8F8C8"])
    point_cmap = ListedColormap(["#C61A1A", "#1555B5", "#1A9C3A"])

    levels = [-0.5, 0.5, 1.5, 2.5]

    ax.contourf(xx, yy, pred, levels=levels, cmap=region_cmap, alpha=0.85)
    ax.contour(xx, yy, pred, levels=[0.5, 1.5], colors="#1F1F1F", linewidths=1.0)

    # Plot only class 0 and 1 points in main view
    ax.scatter(
        X[:, 0], X[:, 1],
        c=y[:],
        cmap=point_cmap,
        s=22,
        edgecolors="white",
        linewidths=0.25
    )

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_yticks([0,1,2,3,4])

    ax_acc_box = inset_axes(
        ax,
        width="7.5%",
        height="20%",
        loc="upper left",
        borderpad=0.5 
    )
    ax_acc_box.set_xticks([])
    ax_acc_box.set_yticks([])

    ax_acc_box.text(
        0.03, 0.975,
        f"Acc",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top'
    )
    ax_acc_box.plot(
        [0.0275, 0.07],
        [0.945, 0.945],  # small offset downward
        transform=ax.transAxes,
        linewidth=1,
        color="black"
    )

    y_pred = model.predict(X)

    y_pos = 0.925
    colors = ["#C61A1A", "#1555B5", "#1A9C3A"]
    for i, acc in enumerate([98, 56, 100]):
        mask = (y == i)
        acc = int((y_pred[mask] == i).mean() * 100)
        ax_acc_box.text(
            0.08, y_pos,
            f"{acc:}%",
            transform=ax.transAxes,
            fontsize=10,
            color=colors[i],
            verticalalignment='top',
            horizontalalignment='right'
        )
        y_pos -= 0.05

    ax_acc_box.patch.set_edgecolor("black")
    ax_acc_box.patch.set_linewidth(1.5)

    # ---- INSET FOR CLASS 2 (outlier region) ----
    if max(X[:, 0]) > 10: # Check if class 2 is significant outlier
        ax_ins_behind = inset_axes(
            ax,
            width="41.5%" if max(X[:, 0]) < 1000 else "43.5%",
            height="43%",
            loc="upper right",
            borderpad=0 
        )
        ax_ins_behind.set_xticks([])
        ax_ins_behind.set_yticks([])

        ax_inset = inset_axes(
            ax,
            width="35%",
            height="35%",
            loc="upper right",
            borderpad=1 
        )

        # NOTE: Hardcoded to the centers (40,30) and (4000, 3000)
        # xmid = 40 if max(X[:, 0]) < 1000 else 4000
        # ymid = 30 if max(X[:, 0]) < 1000 else 3000
        xmid = 49 if max(X[:, 0]) < 1000 else 4999
        ymid = 39 if max(X[:, 0]) < 1000 else 3999

        x_min_o, x_max_o = xmid - 1, xmid + 1
        y_min_o, y_max_o = ymid - 1, ymid + 1

        xx_o, yy_o = np.meshgrid(
            np.linspace(x_min_o, x_max_o, 200),
            np.linspace(y_min_o, y_max_o, 200)
        )
        grid_o = np.c_[xx_o.ravel(), yy_o.ravel()]
        pred_o = np.asarray(model.predict(grid_o)).reshape(xx_o.shape)

        ax_inset.contourf(xx_o, yy_o, pred_o, levels=levels, cmap=region_cmap, alpha=0.85)
        ax_inset.contour(xx_o, yy_o, pred_o, levels=[0.5, 1.5], colors="#1F1F1F", linewidths=0.8)

        # Plot only class 2 points
        ax_inset.scatter(
            X[:, 0], X[:, 1],
            c=y[:],
            cmap=point_cmap,
            s=22,
            edgecolors="white",
            linewidths=0.25
        )

        # Make border more visible
        # for spine in ax_inset.spines.values():
        #     spine.set_linewidth(1.5)
        #     spine.set_color("black")

        ax_inset.set_xlim(x_min_o, x_max_o)
        ax_inset.set_ylim(y_min_o, y_max_o)
        ax_inset.set_xticks([xmid - 1, xmid, xmid + 1])
        ax_inset.set_yticks([ymid - 1, ymid, ymid + 1])
        ax_inset.tick_params(labelsize=7)
        #ax_inset.set_title("Class 2 (outlier)", fontsize=9)
        # ax_inset.patch.set_edgecolor("black")
        # ax_inset.patch.set_linewidth(1.5)    

    # ---- FINAL TOUCHES ----
    # ax.set_title("Decision Regions (0/1 + inset for class 2)")
    # ax.set_xlabel("Feature 1")
    # ax.set_ylabel("Feature 2")

    path = save_plot(fig, "moons-decision-regions.png", show=show, outdir=outdir)
    return path


def plot_moons_input_distribution(
    X: np.ndarray,
    y: np.ndarray,
    show: bool = False,
    outdir: Optional[Union[str, Path]] = None,
) -> Path:
    point_cmap = ListedColormap(["#C61A1A", "#1555B5", "#1A9C3A"])

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=point_cmap, s=22, edgecolors="white", linewidths=0.25)
    ax.set_title("Moons: Input Distribution")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    path = save_plot(fig, "moons-input.pdf", show=show, outdir=outdir)
    return path


def _make_moons_mesh(
    X: np.ndarray,
    padding: float = 0.6,
    points: int = 350,
) -> Tuple[np.ndarray, np.ndarray]:
    x_min, x_max = X[:, 0].min() - padding, X[:, 0].max() + padding
    y_min, y_max = X[:, 1].min() - padding, X[:, 1].max() + padding
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, points),
        np.linspace(y_min, y_max, points),
    )
    return xx, yy


def _get_positive_class_proba(model: Any, X: np.ndarray) -> Optional[np.ndarray]:
    if not hasattr(model, "predict_proba"):
        return None

    proba = np.asarray(model.predict_proba(X))
    if proba.ndim != 2 or proba.shape[1] < 2:
        return None

    positive_index = proba.shape[1] - 1
    classes = getattr(model, "classes_", None)
    if classes is not None:
        classes = np.asarray(classes)
        if classes.ndim == 1:
            positive_match = np.flatnonzero(classes == 1)
            if positive_match.size > 0:
                positive_index = int(positive_match[0])

    return proba[:, positive_index]


def plot_moons_decision_boundary_confidence(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    show: bool = False,
    outdir: Optional[Union[str, Path]] = None,
) -> Optional[Path]:
    xx, yy = _make_moons_mesh(X)
    grid = np.c_[xx.ravel(), yy.ravel()]
    positive_proba = _get_positive_class_proba(model, grid)
    if positive_proba is None:
        return None

    positive_proba = positive_proba.reshape(xx.shape)
    confidence = np.maximum(positive_proba, 1.0 - positive_proba)
    point_cmap = ListedColormap(["#C61A1A", "#1555B5"])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    ax_proba, ax_conf = axes

    proba_im = ax_proba.contourf(
        xx,
        yy,
        positive_proba,
        levels=np.linspace(0.0, 1.0, 21),
        cmap="bwr",
        alpha=0.75,
    )
    ax_proba.contour(xx, yy, positive_proba, levels=[0.5], colors="black", linewidths=1.0)
    ax_proba.scatter(
        X[:, 0],
        X[:, 1],
        c=y,
        cmap=point_cmap,
        s=20,
        edgecolors="white",
        linewidths=0.25,
    )
    ax_proba.set_title("P(y=1 | x)")
    ax_proba.set_xlabel("Feature 1")
    ax_proba.set_ylabel("Feature 2")
    fig.colorbar(proba_im, ax=ax_proba, fraction=0.046, pad=0.04)

    conf_im = ax_conf.contourf(
        xx,
        yy,
        confidence,
        levels=np.linspace(0.5, 1.0, 21),
        cmap="viridis",
        alpha=0.8,
    )
    ax_conf.contour(xx, yy, positive_proba, levels=[0.5], colors="white", linewidths=1.0)
    ax_conf.scatter(
        X[:, 0],
        X[:, 1],
        c=y,
        cmap=point_cmap,
        s=20,
        edgecolors="white",
        linewidths=0.25,
    )
    ax_conf.set_title("Classification Confidence")
    ax_conf.set_xlabel("Feature 1")
    ax_conf.set_ylabel("Feature 2")
    fig.colorbar(conf_im, ax=ax_conf, fraction=0.046, pad=0.04)

    return save_plot(fig, "moons-decision-boundary-confidence.png", show=show, outdir=outdir)


def evaluate_fitted_forest(
    model: Any,
    data: dict,
    dataset_name: str,
    show: bool = False,
    outdir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    from dp_random_forest.datasets.splits import concat_train_val

    X_train_fit, y_train_fit = concat_train_val(data)
    X_test = np.asarray(data["X_test"])
    y_test = np.asarray(data["y_test"])
    feature_names = data.get("features")

    train_acc, test_acc, acc_path = plot_accuracy_summary(
        model,
        X_train_fit,
        y_train_fit,
        X_test,
        y_test,
        show=show,
        outdir=outdir,
    )

    cm_path = plot_confusion_matrix(model, X_test, y_test, show=show, outdir=outdir)
    tree_path = plot_first_tree(
        model,
        feature_names=feature_names,
        show=show,
        outdir=outdir,
    )
    importance_path = plot_feature_importance(
        model,
        feature_names=feature_names,
        show=show,
        outdir=outdir,
    )

    paths = {
        "accuracy_plot": str(acc_path),
        "confusion_matrix_plot": str(cm_path),
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
    }

    if tree_path is not None:
        paths["first_tree_plot"] = str(tree_path)

    if importance_path is not None:
        paths["feature_importance_plot"] = str(importance_path)

    if (dataset_name.lower() == "moons" or dataset_name.lower() == "moonsunbalanced") and X_train_fit.shape[1] == 2:
        X_full = np.vstack((X_train_fit, X_test))
        y_full = np.concatenate((y_train_fit, y_test))
        moons_input_path = plot_moons_input_distribution(
            X_full,
            y_full,
            show=show,
            outdir=outdir,
        )
        if dataset_name.lower() == "moons":
            moons_path = plot_moons_decision_regions(
                model,
                X_full,
                y_full,
                show=show,
                outdir=outdir,
            )
        else:
            moons_path = plot_moons_unbalanced_decision_regions(
                model,
                X_full,
                y_full,
                show=show,
                outdir=outdir,
            )
        boundary_confidence_path = plot_moons_decision_boundary_confidence(
            model,
            X_full,
            y_full,
            show=show,
            outdir=outdir,
        )
        paths["moons_input_plot"] = str(moons_input_path)
        paths["moons_decision_regions_plot"] = str(moons_path)
        if boundary_confidence_path is not None:
            paths["moons_boundary_confidence_plot"] = str(boundary_confidence_path)

    return paths
