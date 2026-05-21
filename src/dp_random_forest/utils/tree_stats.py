from __future__ import annotations

from collections import Counter
from pathlib import Path

import hydra
import numpy as np
from sklearn.tree import _tree


def _resolve_output_dir(outdir: str | Path | None = None) -> Path:
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


def _compute_node_depths(tree) -> np.ndarray:
    depths = np.zeros(tree.node_count, dtype=int)
    stack = [(0, 0)]

    while stack:
        node_id, depth = stack.pop()
        depths[node_id] = depth

        left_child = tree.children_left[node_id]
        right_child = tree.children_right[node_id]
        if left_child != _tree.TREE_LEAF:
            stack.append((left_child, depth + 1))
            stack.append((right_child, depth + 1))

    return depths


def _ratio(count: int, total: int) -> float:
    return 100.0 * count / total if total else 0.0


def _format_count_pct(count: int, total: int) -> str:
    return f"{count} ({_ratio(count, total):.2f}%)"


def _format_stat(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return f"{float(value):.2f}"


def _summarize_values(values: np.ndarray) -> dict[str, float | int | None]:
    values = np.asarray(values)

    if values.size == 0:
        return {
            "min": None,
            "mean": None,
            "median": None,
            "p90": None,
            "max": None,
        }

    is_integral = np.issubdtype(values.dtype, np.integer)

    return {
        "min": int(np.min(values)) if is_integral else float(np.min(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "max": int(np.max(values)) if is_integral else float(np.max(values)),
    }


def _format_depth_breakdown(depth_counts: Counter[int]) -> str:
    if not depth_counts:
        return "none"
    return ", ".join(
        f"depth {depth}: {depth_counts[depth]}"
        for depth in sorted(depth_counts)
    )


def _encode_targets(y, classes) -> np.ndarray:
    y_array = np.asarray(y)
    if y_array.ndim == 2:
        if y_array.shape[1] != 1:
            raise ValueError("tree_stats only supports single-output classification targets.")
        y_array = y_array[:, 0]
    else:
        y_array = y_array.reshape(-1)

    class_to_index = {
        cls: idx for idx, cls in enumerate(np.asarray(classes).tolist())
    }
    try:
        return np.asarray([class_to_index[label] for label in y_array.tolist()], dtype=int)
    except KeyError as exc:
        raise ValueError(
            "tree_stats received labels that are not present in estimator.classes_."
        ) from exc


def _compute_positive_totals_and_gini(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.clip(np.asarray(values, dtype=float), 0.0, None)
    if values.ndim == 1:
        values = values.reshape(1, -1)

    totals = np.sum(values, axis=1)
    gini = np.zeros_like(totals, dtype=float)
    nonzero_mask = totals > 0.0
    if np.any(nonzero_mask):
        sum_squared = np.sum(values[nonzero_mask] * values[nonzero_mask], axis=1)
        gini[nonzero_mask] = 1.0 - sum_squared / np.square(totals[nonzero_mask])

    return totals, gini


def _compute_tree_leaf_stats(
    estimator,
    X,
    y,
    split_mass_threshold: float,
    split_tau: float,
    large_leaf_threshold: float,
) -> dict:
    tree = estimator.tree_
    leaf_mask = tree.children_left == _tree.TREE_LEAF
    leaf_nodes = np.flatnonzero(leaf_mask)
    node_depths = _compute_node_depths(tree)
    leaf_depths = node_depths[leaf_nodes]

    assigned_leaves = np.asarray(estimator.apply(X), dtype=int).reshape(-1)
    leaf_sizes_by_node = np.bincount(assigned_leaves, minlength=tree.node_count)
    leaf_sizes = leaf_sizes_by_node[leaf_nodes]
    max_leaf_depth = int(np.max(leaf_depths)) if leaf_depths.size else 0
    max_depth_leaf_mask = leaf_depths == max_leaf_depth if leaf_depths.size else np.array([], dtype=bool)
    largest_leaf_size_at_max_leaf_depth = (
        int(np.max(leaf_sizes[max_depth_leaf_mask]))
        if leaf_depths.size
        else 0
    )

    empty_mask = leaf_sizes == 0
    nonempty_mask = leaf_sizes > 0
    large_mask = leaf_sizes >= large_leaf_threshold
    max_depth = estimator.max_depth
    splitting_algorithm = getattr(estimator, "splitting_algorithm", "").lower()

    if y is not None:
        encoded_targets = _encode_targets(y, estimator.classes_)
        if encoded_targets.shape[0] != assigned_leaves.shape[0]:
            raise ValueError("X and y must contain the same number of samples.")

        class_counts_by_node = np.zeros(
            (tree.node_count, len(np.asarray(estimator.classes_))),
            dtype=int,
        )
        np.add.at(class_counts_by_node, (assigned_leaves, encoded_targets), 1)
        leaf_true_class_counts = class_counts_by_node[leaf_nodes]
        true_leaf_totals, true_leaf_gini = _compute_positive_totals_and_gini(
            leaf_true_class_counts
        )
    else:
        true_leaf_totals = np.array([], dtype=float)
        true_leaf_gini = np.array([], dtype=float)

    has_split_rule_proxy = (
        y is not None and max_depth is not None and splitting_algorithm == "sparse"
    )
    split_rule_mask = (
        (true_leaf_totals >= split_mass_threshold)
        & (true_leaf_gini > split_tau)
        & (leaf_depths < max_depth)
        if has_split_rule_proxy
        else np.zeros_like(leaf_sizes, dtype=bool)
    )
    split_rule_mass_excess = (
        true_leaf_totals[split_rule_mask] - split_mass_threshold
        if has_split_rule_proxy
        else np.array([], dtype=float)
    )
    split_rule_gini = (
        true_leaf_gini[split_rule_mask]
        if has_split_rule_proxy
        else np.array([], dtype=float)
    )

    leaf_algorithm = getattr(estimator, "leaf_algorithm", "").lower()
    has_use_estimates_stats = leaf_algorithm == "useestimates"
    if has_use_estimates_stats:
        leaf_estimates = np.asarray(tree.value[leaf_nodes, 0, :], dtype=float)
        estimate_totals, estimate_gini = _compute_positive_totals_and_gini(leaf_estimates)
        zero_estimate_total_mask = estimate_totals <= 0.0
        zeroed_nonempty_mask = nonempty_mask & zero_estimate_total_mask
        below_split_mass_nonempty_mask = nonempty_mask & (
            estimate_totals < split_mass_threshold
        )
        estimate_rule_pass_nonempty_mask = nonempty_mask & (
            (estimate_totals >= split_mass_threshold) & (estimate_gini > split_tau)
        )
        nonempty_estimate_totals = estimate_totals[nonempty_mask]
        nonempty_estimate_gini = estimate_gini[nonempty_mask]
        nonempty_active_classes = np.sum(leaf_estimates[nonempty_mask] > 0.0, axis=1)
        nonempty_estimate_gap = nonempty_estimate_totals - leaf_sizes[nonempty_mask]
    else:
        zero_estimate_total_mask = np.zeros_like(leaf_sizes, dtype=bool)
        zeroed_nonempty_mask = np.zeros_like(leaf_sizes, dtype=bool)
        below_split_mass_nonempty_mask = np.zeros_like(leaf_sizes, dtype=bool)
        estimate_rule_pass_nonempty_mask = np.zeros_like(leaf_sizes, dtype=bool)
        nonempty_estimate_totals = np.array([], dtype=float)
        nonempty_estimate_gini = np.array([], dtype=float)
        nonempty_active_classes = np.array([], dtype=int)
        nonempty_estimate_gap = np.array([], dtype=float)

    return {
        "total_leaves": int(leaf_nodes.size),
        "empty_count": int(np.sum(empty_mask)),
        "nonempty_count": int(np.sum(nonempty_mask)),
        "large_count": int(np.sum(large_mask)),
        "leaf_sizes": leaf_sizes,
        "max_leaf_depth": max_leaf_depth,
        "largest_leaf_size_at_max_leaf_depth": largest_leaf_size_at_max_leaf_depth,
        "has_true_leaf_gini": y is not None,
        "true_leaf_gini": true_leaf_gini,
        "has_split_rule_proxy": has_split_rule_proxy,
        "split_rule_count": int(np.sum(split_rule_mask)),
        "split_rule_mass_excess": split_rule_mass_excess,
        "split_rule_gini": split_rule_gini,
        "split_rule_depth_counts": Counter(
            int(depth) for depth in leaf_depths[split_rule_mask]
        ),
        "has_use_estimates_stats": has_use_estimates_stats,
        "zero_estimate_total_count": int(np.sum(zero_estimate_total_mask)),
        "zeroed_nonempty_count": int(np.sum(zeroed_nonempty_mask)),
        "below_split_mass_nonempty_count": int(np.sum(below_split_mass_nonempty_mask)),
        "estimate_rule_pass_nonempty_count": int(
            np.sum(estimate_rule_pass_nonempty_mask)
        ),
        "zeroed_nonempty_leaf_sizes": leaf_sizes[zeroed_nonempty_mask],
        "below_split_mass_nonempty_leaf_sizes": leaf_sizes[below_split_mass_nonempty_mask],
        "zeroed_nonempty_depth_counts": Counter(
            int(depth) for depth in leaf_depths[zeroed_nonempty_mask]
        ),
        "nonempty_estimate_totals": nonempty_estimate_totals,
        "nonempty_estimate_gini": nonempty_estimate_gini,
        "nonempty_active_classes": nonempty_active_classes,
        "nonempty_estimate_gap": nonempty_estimate_gap,
    }


def build_tree_stats_report(
    forest,
    X,
    y=None,
    large_leaf_threshold: float | int | None = None,
) -> str:
    split_threshold = float(forest.split_threshold)
    split_alpha = 1.0
    split_tau = 0.25
    split_mass_threshold = split_alpha * split_threshold
    large_leaf_threshold = (
        split_mass_threshold
        if large_leaf_threshold is None
        else float(large_leaf_threshold)
    )

    per_tree_stats = [
        _compute_tree_leaf_stats(
            estimator,
            X,
            y,
            split_mass_threshold,
            split_tau,
            large_leaf_threshold,
        )
        for estimator in forest.estimators_
    ]

    all_leaf_sizes = np.concatenate([stats["leaf_sizes"] for stats in per_tree_stats])
    forest_total_leaves = int(sum(stats["total_leaves"] for stats in per_tree_stats))
    forest_empty_count = int(sum(stats["empty_count"] for stats in per_tree_stats))
    forest_nonempty_count = int(sum(stats["nonempty_count"] for stats in per_tree_stats))
    forest_large_count = int(sum(stats["large_count"] for stats in per_tree_stats))
    forest_leaf_summary = _summarize_values(all_leaf_sizes)

    true_leaf_gini_available = all(stats["has_true_leaf_gini"] for stats in per_tree_stats)
    if true_leaf_gini_available:
        all_true_leaf_gini = np.concatenate([stats["true_leaf_gini"] for stats in per_tree_stats])
        forest_true_leaf_gini_summary = _summarize_values(all_true_leaf_gini)

    split_rule_available = all(stats["has_split_rule_proxy"] for stats in per_tree_stats)
    if split_rule_available:
        all_split_rule_mass_excess = np.concatenate(
            [stats["split_rule_mass_excess"] for stats in per_tree_stats]
        )
        all_split_rule_gini = np.concatenate(
            [stats["split_rule_gini"] for stats in per_tree_stats]
        )
        forest_split_rule_count = int(
            sum(stats["split_rule_count"] for stats in per_tree_stats)
        )
        forest_split_rule_mass_excess_summary = _summarize_values(
            all_split_rule_mass_excess
        )
        forest_split_rule_gini_summary = _summarize_values(all_split_rule_gini)
        forest_split_rule_depth_counts = Counter()
        for stats in per_tree_stats:
            forest_split_rule_depth_counts.update(stats["split_rule_depth_counts"])

    use_estimates_available = all(stats["has_use_estimates_stats"] for stats in per_tree_stats)
    if use_estimates_available:
        forest_zero_estimate_total_count = int(
            sum(stats["zero_estimate_total_count"] for stats in per_tree_stats)
        )
        forest_zeroed_nonempty_count = int(
            sum(stats["zeroed_nonempty_count"] for stats in per_tree_stats)
        )
        forest_below_split_mass_nonempty_count = int(
            sum(stats["below_split_mass_nonempty_count"] for stats in per_tree_stats)
        )
        forest_estimate_rule_pass_nonempty_count = int(
            sum(stats["estimate_rule_pass_nonempty_count"] for stats in per_tree_stats)
        )
        all_zeroed_nonempty_leaf_sizes = np.concatenate(
            [stats["zeroed_nonempty_leaf_sizes"] for stats in per_tree_stats]
        )
        all_below_split_mass_nonempty_leaf_sizes = np.concatenate(
            [stats["below_split_mass_nonempty_leaf_sizes"] for stats in per_tree_stats]
        )
        all_nonempty_estimate_totals = np.concatenate(
            [stats["nonempty_estimate_totals"] for stats in per_tree_stats]
        )
        all_nonempty_estimate_gini = np.concatenate(
            [stats["nonempty_estimate_gini"] for stats in per_tree_stats]
        )
        all_nonempty_active_classes = np.concatenate(
            [stats["nonempty_active_classes"] for stats in per_tree_stats]
        )
        all_nonempty_estimate_gap = np.concatenate(
            [stats["nonempty_estimate_gap"] for stats in per_tree_stats]
        )
        forest_zeroed_nonempty_leaf_summary = _summarize_values(all_zeroed_nonempty_leaf_sizes)
        forest_below_split_mass_nonempty_leaf_summary = _summarize_values(
            all_below_split_mass_nonempty_leaf_sizes
        )
        forest_nonempty_estimate_total_summary = _summarize_values(all_nonempty_estimate_totals)
        forest_nonempty_estimate_gini_summary = _summarize_values(all_nonempty_estimate_gini)
        forest_nonempty_active_classes_summary = _summarize_values(all_nonempty_active_classes)
        forest_nonempty_estimate_gap_summary = _summarize_values(all_nonempty_estimate_gap)
        forest_zeroed_nonempty_depth_counts = Counter()
        for stats in per_tree_stats:
            forest_zeroed_nonempty_depth_counts.update(stats["zeroed_nonempty_depth_counts"])

    lines = [
        "config:",
        f"  leaf_algorithm = {forest.leaf_algorithm}",
        f"  n_estimators = {forest.n_estimators}",
        f"  max_depth = {forest.max_depth}",
        f"  split_threshold = {_format_stat(split_threshold)}",
        f"  split_mass_threshold = {_format_stat(split_mass_threshold)}",
        f"  split_sigma = {getattr(forest, 'split_sigma', 'n/a')}",
        f"  epsilon = {getattr(forest, 'epsilon', None)}",
        f"  delta = {getattr(forest, 'delta', None)}",
        f"  splitting_fraction = {getattr(forest, 'splitting_fraction', None)}",
        f"  splitbudget = {forest.splitbudget}",
        f"  leafbudget = {forest.leafbudget}",
        f"  splitbudget_tree = {forest.splitbudget_tree}",
        f"  leafbudget_tree = {forest.leafbudget_tree}",
        f"  large_leaf_threshold = {_format_stat(large_leaf_threshold)}",
        "",
        "forest:",
        f"  total_leaves = {forest_total_leaves}",
        f"  empty_leaves = {_format_count_pct(forest_empty_count, forest_total_leaves)}",
        f"  nonempty_leaves = {_format_count_pct(forest_nonempty_count, forest_total_leaves)}",
        "  leaves_ge_large_leaf_threshold = "
        f"{_format_count_pct(forest_large_count, forest_total_leaves)}",
        "  leaf_size = "
        f"min {_format_stat(forest_leaf_summary['min'])}, "
        f"mean {_format_stat(forest_leaf_summary['mean'])}, "
        f"median {_format_stat(forest_leaf_summary['median'])}, "
        f"p90 {_format_stat(forest_leaf_summary['p90'])}, "
        f"max {_format_stat(forest_leaf_summary['max'])}",
        f"  max_leaf_depth = {max(stats['max_leaf_depth'] for stats in per_tree_stats)}",
    ]

    if true_leaf_gini_available:
        lines.extend(
            [
                "  true_leaf_gini = "
                f"min {_format_stat(forest_true_leaf_gini_summary['min'])}, "
                f"mean {_format_stat(forest_true_leaf_gini_summary['mean'])}, "
                f"median {_format_stat(forest_true_leaf_gini_summary['median'])}, "
                f"p90 {_format_stat(forest_true_leaf_gini_summary['p90'])}, "
                f"max {_format_stat(forest_true_leaf_gini_summary['max'])}",
            ]
        )

    if split_rule_available:
        lines.extend(
            [
                f"  split_rule_proxy = {_format_count_pct(forest_split_rule_count, forest_total_leaves)}",
                "  split_rule_mass_excess = "
                f"mean {_format_stat(forest_split_rule_mass_excess_summary['mean'])}, "
                f"median {_format_stat(forest_split_rule_mass_excess_summary['median'])}, "
                f"max {_format_stat(forest_split_rule_mass_excess_summary['max'])}",
                "  split_rule_true_leaf_gini = "
                f"mean {_format_stat(forest_split_rule_gini_summary['mean'])}, "
                f"median {_format_stat(forest_split_rule_gini_summary['median'])}, "
                f"max {_format_stat(forest_split_rule_gini_summary['max'])}",
                f"  split_rule_depth_breakdown = {_format_depth_breakdown(forest_split_rule_depth_counts)}",
            ]
        )
    elif getattr(forest, "splitting_algorithm", "").lower() == "sparse":
        lines.append("  split_rule_proxy = unavailable (y is not provided)")

    if use_estimates_available:
        lines.extend(
            [
                f"  zero_estimate_total_leaves = {_format_count_pct(forest_zero_estimate_total_count, forest_total_leaves)}",
                "  nonempty_leaves_zeroed_by_threshold = "
                f"{_format_count_pct(forest_zeroed_nonempty_count, forest_nonempty_count)}",
                "  nonempty_leaves_below_split_mass_threshold = "
                f"{_format_count_pct(forest_below_split_mass_nonempty_count, forest_nonempty_count)}",
                "  nonempty_leaves_passing_split_rule_on_estimates = "
                f"{_format_count_pct(forest_estimate_rule_pass_nonempty_count, forest_nonempty_count)}",
                f"  samples_in_zeroed_nonempty_leaves = {int(np.sum(all_zeroed_nonempty_leaf_sizes))}",
                "  samples_in_below_split_mass_nonempty_leaves = "
                f"{int(np.sum(all_below_split_mass_nonempty_leaf_sizes))}",
                "  zeroed_nonempty_leaf_size = "
                f"mean {_format_stat(forest_zeroed_nonempty_leaf_summary['mean'])}, "
                f"median {_format_stat(forest_zeroed_nonempty_leaf_summary['median'])}, "
                f"max {_format_stat(forest_zeroed_nonempty_leaf_summary['max'])}",
                "  below_split_mass_nonempty_leaf_size = "
                f"mean {_format_stat(forest_below_split_mass_nonempty_leaf_summary['mean'])}, "
                f"median {_format_stat(forest_below_split_mass_nonempty_leaf_summary['median'])}, "
                f"max {_format_stat(forest_below_split_mass_nonempty_leaf_summary['max'])}",
                "  nonempty_estimate_total = "
                f"min {_format_stat(forest_nonempty_estimate_total_summary['min'])}, "
                f"mean {_format_stat(forest_nonempty_estimate_total_summary['mean'])}, "
                f"median {_format_stat(forest_nonempty_estimate_total_summary['median'])}, "
                f"p90 {_format_stat(forest_nonempty_estimate_total_summary['p90'])}, "
                f"max {_format_stat(forest_nonempty_estimate_total_summary['max'])}",
                "  nonempty_estimate_gini = "
                f"min {_format_stat(forest_nonempty_estimate_gini_summary['min'])}, "
                f"mean {_format_stat(forest_nonempty_estimate_gini_summary['mean'])}, "
                f"median {_format_stat(forest_nonempty_estimate_gini_summary['median'])}, "
                f"p90 {_format_stat(forest_nonempty_estimate_gini_summary['p90'])}, "
                f"max {_format_stat(forest_nonempty_estimate_gini_summary['max'])}",
                "  nonempty_active_estimated_classes = "
                f"min {_format_stat(forest_nonempty_active_classes_summary['min'])}, "
                f"mean {_format_stat(forest_nonempty_active_classes_summary['mean'])}, "
                f"median {_format_stat(forest_nonempty_active_classes_summary['median'])}, "
                f"p90 {_format_stat(forest_nonempty_active_classes_summary['p90'])}, "
                f"max {_format_stat(forest_nonempty_active_classes_summary['max'])}",
                "  nonempty_estimate_minus_true_leaf_size = "
                f"min {_format_stat(forest_nonempty_estimate_gap_summary['min'])}, "
                f"mean {_format_stat(forest_nonempty_estimate_gap_summary['mean'])}, "
                f"median {_format_stat(forest_nonempty_estimate_gap_summary['median'])}, "
                f"max {_format_stat(forest_nonempty_estimate_gap_summary['max'])}",
                f"  zeroed_nonempty_depth_breakdown = {_format_depth_breakdown(forest_zeroed_nonempty_depth_counts)}",
            ]
        )

    lines.extend(["", "per_tree:"])
    for tree_idx, stats in enumerate(per_tree_stats):
        line = (
            f"  tree {tree_idx}: leaves {stats['total_leaves']}, "
            f"empty {_format_count_pct(stats['empty_count'], stats['total_leaves'])}, "
            f"ge_large_leaf_threshold {_format_count_pct(stats['large_count'], stats['total_leaves'])}, "
            f"max_leaf_depth {stats['max_leaf_depth']}, "
            f"largest_leaf_at_max_leaf_depth {stats['largest_leaf_size_at_max_leaf_depth']}"
        )
        if stats["has_split_rule_proxy"]:
            line += (
                f", split_rule_proxy {_format_count_pct(stats['split_rule_count'], stats['total_leaves'])}"
            )
        if stats["has_use_estimates_stats"]:
            line += (
                f", zero_estimate_total {_format_count_pct(stats['zero_estimate_total_count'], stats['total_leaves'])}"
                f", zeroed_nonempty {_format_count_pct(stats['zeroed_nonempty_count'], stats['nonempty_count'])}"
                f", below_split_mass {_format_count_pct(stats['below_split_mass_nonempty_count'], stats['nonempty_count'])}"
            )
        lines.append(line)

    lines.extend(
        [
            "",
            "notes:",
            "  leaf sizes are computed by applying the training data to the fitted trees.",
            "  split_rule_proxy counts sparse leaves at depth < max_depth whose true leaf counts satisfy total >= split_alpha * split_threshold and gini > split_tau.",
            "  true_leaf_gini and split_rule_proxy require the training labels y.",
            "  zero_estimate_total/useEstimates diagnostics compare actual leaf occupancy with stored leaf estimate vectors.",
            "  nonempty_leaves_zeroed_by_threshold counts leaves with training samples whose reused estimate vector sums to 0.",
        ]
    )

    return "\n".join(lines) + "\n"


def write_tree_stats_report(
    forest,
    X,
    y=None,
    outdir: str | Path | None = None,
    filename: str = "tree_stats.txt",
    large_leaf_threshold: float | int | None = None,
) -> Path:
    output_dir = _resolve_output_dir(outdir)
    report_path = output_dir / filename
    report = build_tree_stats_report(
        forest,
        X,
        y=y,
        large_leaf_threshold=large_leaf_threshold,
    )
    report_path.write_text(report, encoding="utf-8")
    return report_path
