# NOTE: This is the alternative implementation technique for sparse trees that more closely follows the paper
# The two techniques should give reoughly the same outputs and utility. 
# Although the individual trees will look different since splits can be performed in a different order which changes the pseudo-random splits.
# The code is meant to be identical except for the treebuilder class


from math import ceil
import numbers
import copy
import threading
import numpy as np
from collections import namedtuple
import random
from collections import deque
import math 

from sklearn.tree import DecisionTreeClassifier

from ..mechanisms.privacy_mechanisms import zCDPGaussianMech as GaussianMechanism, LaplaceMechanism, addGaussianNoise
from ..mechanisms.privacy_mechanisms import checkAboveThreshold

from sklearn.utils import (
    check_random_state,
    compute_sample_weight,
)
from sklearn.tree._criterion import Criterion
from sklearn.utils.multiclass import check_classification_targets
from scipy.sparse import issparse
from scipy.stats import norm as Normal
from sklearn.tree import _criterion, _tree
from sklearn.tree._tree import NODE_DTYPE, Tree
from sklearn.utils.validation import (
    _check_sample_weight,
    validate_data,
)
from sklearn.tree._tree import (
    BestFirstTreeBuilder,
)
from sklearn.base import (
    is_classifier,
)
from pathlib import Path
import sys

# Ensure local src/ is importable even if .pth processing is skipped.
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "src"))

import opendp.prelude as dp

DTYPE = _tree.DTYPE
DOUBLE = _tree.DOUBLE

CRITERIA_CLF = {
    "gini": _criterion.Gini,
    "log_loss": _criterion.Entropy,
    "entropy": _criterion.Entropy,
}
CRITERIA_REG = {
    "squared_error": _criterion.MSE,
    "absolute_error": _criterion.MAE,
    "poisson": _criterion.Poisson,
}

DETERMINISTIC_IGNORE_THRESHOLD = 2

StackNodeExpander = namedtuple("StackNodeExpander", ["parent", "is_left", "depth", "bounds", "X"])

# Enable the 'contrib' feature flag, which is often required for these measurements
dp.enable_features("contrib")

def get_seeded_noisy_max(target_rho: float, sensitivity: float = 1.0, seed: int = None):
    # Calculate the exact calibrated scale used by OpenDP for monotonic zCDP Noisy Max
    scale = sensitivity / math.sqrt(8 * target_rho)
    
    # Isolate the random state so it doesn't interfere with global np.random calls
    rng = np.random.default_rng(seed)
    
    def measurement(scores):
        # OpenDP's Noisy Max under zCDP adds Gumbel noise to each score
        noise = rng.gumbel(loc=0.0, scale=scale, size=len(scores))
        noisy_scores = np.asarray(scores) + noise
        
        # Return the index of the maximum noisy score
        return int(np.argmax(noisy_scores))
        
    return measurement

def get_calibrated_noisy_max(target_rho: float):
    def make_measurement(scale):
        return dp.m.make_noisy_max(
            input_domain=dp.vector_domain(dp.atom_domain(T=float, nan=False)),
            input_metric=dp.linf_distance(T=float, monotonic=True),
            output_measure=dp.zero_concentrated_divergence(),
            scale=scale
        )
    
    # 2. Ask OpenDP to find the exact scale that satisfies your budget
    # d_in = your sensitivity (how much scores can change for one user)
    # d_out = your privacy budget (rho)
    sensitivity = 1
    calibrated_scale = dp.binary_search_param(
        make_measurement,
        d_in=sensitivity,
        d_out=target_rho
    )
    
    debug = False
    if debug:
        print(f"OpenDP automatically found the scale: {calibrated_scale}")
    
    return make_measurement(calibrated_scale)

class DPAlternativeSparseExtraTreeClassifier(DecisionTreeClassifier):
    """
    Adapted from https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/tree/_classes.py#L1466
    
    TODO: We store some values on nodes that are not supposed to be access as it would violate DP.
    A proper implementation must clear those unless a debug flag is used.

    An extremely randomized tree classifier.

    Extra-trees differ from classic decision trees in the way they are built.
    When looking for the best split to separate the samples of a node into two
    groups, random splits are drawn for each of the `max_features` randomly
    selected features and the best split among those is chosen. When
    `max_features` is set 1, this amounts to building a totally random
    decision tree. 

    NOTE: The prior DP work use the variant with max_features = 1.
    But the problem with that variant could be that most samples fall in
    very few leaves in many cases. 

    Warning: Extra-trees should only be used within ensemble methods.

    Parameters
    ----------
    splitbudget_tree : float, default=None
        Total privacy budget used by each tree for splitting nodes.
        None is used for infinite privacy budget.
        NOTE: We use zCDP for now, care is needed if we want to change to RDP, GDP etc.
        We could change budget to a tuple, e.g. (1, "zCDP").

    splitting_algorithm : string
        DFS_noisy_threshold - expands the tree in DFS order. Split privacy budget based on max height.
                        Performs a noisy threshold check at each node.
        full_tree - expands all nodes untill the max_depth is reached. This corresponds to prior work and consumed no privacybudget.
        SVT - TODO
        noisy-binary-search - TODO

    leafbudget_tree : float, default=None
        Total privacy budget used by each tree to privatize the leaf nodes.
        None is used for infinite privacy budget.
        NOTE: We use zCDP for now, care is needed if we want to change to RDP etc.

    splitting_algorithm : string
        noisy_majority - Trees report a one-hot leaf label after adding Gaussian noise
            to the per-class counts.
        noisy_count - Trees store the noisy per-class counts directly in the leaf
            value vector. This works with ``predict`` because sklearn takes an
            ``argmax`` over the stored values, but ``predict_proba`` should no longer
            be interpreted as calibrated probabilities.
        noisy_fraction - Trees store noisy per-class counts normalized by their
            noisy total, clipped to the ``[0, 1]`` range. This is closer to a
            probability vector, but clipping can still make the entries sum to
            something other than 1.
        useEstimates - Trees reuse precomputed per-class count estimates stored
            on sparse-build leaves.

    split_threshold : int, default=1
        Threshold for deciding to split a node

    split_alpha : float, default=1.0
        Minimum surviving estimated mass, expressed as a multiple of
        ``split_threshold``, required by the sparse split rule.

    split_tau : float, default=0.25
        Minimum Gini impurity required by the sparse split rule.

    bounds : (list<float>, list<float>), default=None
        Minimum and maximum bounds for all attributes.
        NOTE: If None, we compute bounds from the data, breaking DP best practices

    criterion : {"gini", "entropy", "log_loss"}, default="gini"
        Used only for debug purposes, might be used for splitting in extented version

    max_depth : int, default=20
        The maximum depth of the tree. Non-DP variants sometimes set this
        to none and expand to a certain size. For DP we need a stopping 
        condition that does not depend deterministically on the node size.

    random_state : int, RandomState instance or None, default=None
        Controls 3 sources of randomness:

        - the bootstrapping of the samples used when building trees
          (if ``bootstrap=True``)
        - the sampling of the features to consider when looking for the best
          split at each node (if ``max_features < n_features``)
        - the draw of the splits for each of the `max_features`

        TODO: Make sure we don't mess this up when adding privacy randomness.
        We have to use fixed random seeds as well.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,) or list of ndarray
        The classes labels (single output problem),
        or a list of arrays of class labels (multi-output problem).

    max_features_ : int
        The inferred value of max_features.

    n_classes_ : int or list of int
        The number of classes (for single output problems),
        or a list containing the number of classes for each
        output (for multi-output problems).

    feature_importances_ : ndarray of shape (n_features,)
        The impurity-based feature importances.
        The higher, the more important the feature.
        The importance of a feature is computed as the (normalized)
        total reduction of the criterion brought by that feature.  It is also
        known as the Gini importance.

        Warning: impurity-based feature importances can be misleading for
        high cardinality features (many unique values). See
        :func:`sklearn.inspection.permutation_importance` as an alternative.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

    n_outputs_ : int
        The number of outputs when ``fit`` is performed.

    tree_ : DPTree instance
        The underlying DPTree object. Please refer to
        ``help(sklearn.tree._tree.Tree)`` for attributes of Tree object and
        :ref:`sphx_glr_auto_examples_tree_plot_unveil_tree_structure.py`
        for basic usage of these attributes.

    References
    ----------

    .. [1] P. Geurts, D. Ernst., and L. Wehenkel, "Extremely randomized trees",
           Machine Learning, 63(1), 3-42, 2006.
    """

    def __init__(
        self,
        splitbudget_tree=None,
        leafbudget_tree=None,
        leaf_algorithm="noisy_majority",
        split_sigma=None,
        split_threshold=100,
        *,
        criterion="gini",
        #splitter="random",
        max_depth=None,
        #max_features="sqrt",
        random_state=None,
        verbose=0,
        #max_leaf_nodes=None,
        #class_weight=None,
        #min_samples_leaf=0,
        bounds=None,
        feature_names=None,
        categorical_features=None,
        split_weights=None,
    ):
        super().__init__(
            criterion=criterion,
            splitter=None,
            max_depth=max_depth,
            max_features=1,
            max_leaf_nodes=None,
            class_weight=None,
            random_state=random_state,
            ccp_alpha=0.0, # Parameter used for pruning. We don't expect to prune. If we prune, we will make a custom function
            min_samples_split=0,
            #min_samples_split=0, # The parameters below are fixed for privacy purposes
            min_samples_leaf=0,
            min_weight_fraction_leaf=0,
            min_impurity_decrease=-1, # Children are allowed to have worse impurity
            monotonic_cst=None,
        )

        self.splitbudget_tree = splitbudget_tree
        self.leafbudget_tree = leafbudget_tree
        self.leaf_algorithm = leaf_algorithm
        self.verbose = verbose
        self.split_threshold = split_threshold
        self.bounds = bounds
        self.feature_names = feature_names
        self.categorical_features = categorical_features
        self.split_weights = split_weights
        self.split_sigma = split_sigma



    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        # XXX: nan is only supported for dense arrays, but we set this for the
        # common test to pass, specifically: check_estimators_nan_inf
        allow_nan = False #self.splitter == "random" and self.criterion in {
        #     "gini",
        #     "log_loss",
        #     "entropy",
        # }
        tags.classifier_tags.multi_label = False # NOTE: Differs from scikit-learn. We don't support multilabels. 
        tags.input_tags.allow_nan = allow_nan
        return tags
    

    # Code adapted from the _fit function of BaseDecisionTree
    # Note that the scikitlearn forest algorithms calls the _fit function instead of the fit function
    def _fit(
        self,
        X,
        y,
        sample_weight=None,
        check_input=True,
        missing_values_in_feature_mask=None,
    ):
        random_state = check_random_state(self.random_state)

        if check_input:
            # _compute_missing_values_in_feature_mask will check for finite values and
            # compute the missing mask if the tree supports missing values
            check_X_params = dict(
                dtype=DTYPE, accept_sparse=("csr", "csc"), ensure_all_finite=False
            )
            check_y_params = dict(ensure_2d=False, dtype=None)
            X, y = validate_data(
                self, X, y, validate_separately=(check_X_params, check_y_params)
            )

            missing_values_in_feature_mask = (
                self._compute_missing_values_in_feature_mask(X)
            )
            if issparse(X):
                X = X.tocsr(copy=False)

            if self.criterion == "poisson":
                if np.any(y < 0):
                    raise ValueError(
                        "Some value(s) of y are negative which is"
                        " not allowed for Poisson regression."
                    )
                if np.sum(y) <= 0:
                    raise ValueError(
                        "Sum of y is not positive which is "
                        "necessary for Poisson regression."
                    )

        # Determine output settings
        n_samples, self.n_features_in_ = X.shape
        is_classification = is_classifier(self)

        y = np.atleast_1d(y)
        expanded_class_weight = None

        if y.ndim == 1:
            # reshape is necessary to preserve the data contiguity against vs
            # [:, np.newaxis] that does not.
            y = np.reshape(y, (-1, 1))

        self.n_outputs_ = y.shape[1]

        if is_classification:
            check_classification_targets(y)
            y = np.copy(y)

            self.classes_ = []
            self.n_classes_ = []

            if self.class_weight is not None:
                y_original = np.copy(y)

            y_encoded = np.zeros(y.shape, dtype=int)
            for k in range(self.n_outputs_):
                classes_k, y_encoded[:, k] = np.unique(y[:, k], return_inverse=True)
                self.classes_.append(classes_k)
                self.n_classes_.append(classes_k.shape[0])
            y = y_encoded

            if self.class_weight is not None:
                expanded_class_weight = compute_sample_weight(
                    self.class_weight, y_original
                )

            self.n_classes_ = np.array(self.n_classes_, dtype=np.intp)

        if getattr(y, "dtype", None) != DOUBLE or not y.flags.contiguous:
            y = np.ascontiguousarray(y, dtype=DOUBLE)

        max_depth = np.iinfo(np.int32).max if self.max_depth is None else self.max_depth


        # TODO: Parameters that should not be used deterministically by a DP variant
        # We keep them for now for possible debug purposes and we can override by DP variants when we implement stopping conditions.
        if isinstance(self.min_samples_leaf, numbers.Integral):
            min_samples_leaf = self.min_samples_leaf
        else:  # float
            min_samples_leaf = ceil(self.min_samples_leaf * n_samples)
        if isinstance(self.min_samples_split, numbers.Integral):
            min_samples_split = self.min_samples_split
        else:  # float
            min_samples_split = ceil(self.min_samples_split * n_samples)
            #min_samples_split = max(2, min_samples_split)
            min_samples_split = max(0, min_samples_split) # NOTE: Changed for privacy purposes

        min_samples_split = max(min_samples_split, 2 * min_samples_leaf)

        # Determine number of features to sample
        if isinstance(self.max_features, str):
            if self.max_features == "sqrt":
                max_features = max(1, int(np.sqrt(self.n_features_in_)))
            elif self.max_features == "log2":
                max_features = max(1, int(np.log2(self.n_features_in_)))
        elif self.max_features is None:
            max_features = self.n_features_in_
        elif isinstance(self.max_features, numbers.Integral):
            max_features = self.max_features
        else:  # float
            if self.max_features > 0.0:
                max_features = max(1, int(self.max_features * self.n_features_in_))
            else:
                max_features = 0
        self.max_features_ = max_features

        if len(y) != n_samples:
            raise ValueError(
                "Number of labels=%d does not match number of samples=%d"
                % (len(y), n_samples)
            )

        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X, dtype=DOUBLE)

        if expanded_class_weight is not None:
            if sample_weight is not None:
                sample_weight = sample_weight * expanded_class_weight
            else:
                sample_weight = expanded_class_weight

        # Set min_weight_leaf from min_weight_fraction_leaf
        if sample_weight is None:
            min_weight_leaf = self.min_weight_fraction_leaf * n_samples
        else:
            min_weight_leaf = self.min_weight_fraction_leaf * np.sum(sample_weight)

        # Build tree
        criterion = self.criterion
        if not isinstance(criterion, Criterion):
            if is_classification:
                criterion = CRITERIA_CLF[self.criterion](
                    self.n_outputs_, self.n_classes_
                )
            else:
                criterion = CRITERIA_REG[self.criterion](self.n_outputs_, n_samples)
        else:
            # Make a deepcopy in case the criterion has mutable attributes that
            # might be shared and modified concurrently during parallel fitting
            criterion = copy.deepcopy(criterion)

        if not is_classifier(self):
            raise ValueError(
                "Non classifier trees not supported yet."
            )

        if self.bounds is None:
            if issparse(X):
                self.bounds = (
                    X.min(axis=0).toarray().ravel(),
                    X.max(axis=0).toarray().ravel(),
                )
            else:
                self.bounds = (np.min(X, axis=0), np.max(X, axis=0))
        #self.bounds = self._check_bounds(self.bounds, shape=X.shape[1])

        feature_names = self._resolve_feature_names(X)
        self.feature_metadata_ = self._resolve_feature_metadata(feature_names)

        if not is_classifier(self):
            print("ERROR: Non-classifier not supported yet!")
            exit()

        if self.n_outputs_ == 1 and is_classifier(self):
            self.n_classes_ = self.n_classes_[0]
            self.classes_ = self.classes_[0]

            
        #self.tree_ = DPTree(self.n_features_in_, self.n_classes_, self.n_outputs_)
        builder = _TreeBuilder(
            self.max_depth,
            self.n_features_in_,
            self.classes_,
            self.bounds,
            self.feature_metadata_,
            random_state,
            self.split_threshold,
            self.splitbudget_tree,
            self.leafbudget_tree,
            self.leaf_algorithm,
            self.split_sigma,
        )
        
        debug = False
        if debug:
            self._debug_print_sparse_configuration()

        builder.fit(X, y)

        # Load params from _FittingTree into sklearn.Tree
        d = builder.__getstate__()
        tree = Tree(self.n_features_in_, np.array([self.n_classes_]), self.n_outputs_)
        tree.__setstate__(d)
        self.tree_ = tree

        return self

    def _debug_print_sparse_configuration(self):
        print("[DPExtraTreeClassifier][verbose] sparse tree hyperparameters:")
        print(
            "  "
            f"tree_id={hex(id(self))}, "
            f"splitting_algorithm={self.splitting_algorithm}, "
            f"leaf_algorithm={self.leaf_algorithm}, "
            f"max_depth={self.max_depth}, "
            f"random_state={self.random_state}, "
            f"verbose={self.verbose}"
        )
        print("[DPExtraTreeClassifier][verbose] sparse tree derived privacy values:")
        print(
            "  "
            f"splitbudget_tree={self.splitbudget_tree}, "
            f"leaf_rho_per_tree={self.leafbudget_tree}, "
            f"split_sigma={self.split_sigma}, "
            f"split_threshold={self.split_threshold}, "
            f"split_alpha={self.split_alpha}, "
            f"split_tau={self.split_tau}"
        )

    def _resolve_feature_names(self, X):
        if self.feature_names is not None:
            if len(self.feature_names) != self.n_features_in_:
                raise ValueError(
                    "feature_names must match the number of columns in X."
                )
            return list(self.feature_names)

        if hasattr(X, "columns"):
            columns = list(X.columns)
            if len(columns) == self.n_features_in_:
                return columns

        return [f"feature_{i}" for i in range(self.n_features_in_)]

    def _resolve_categorical_feature_sizes(self):
        if not isinstance(self.categorical_features, dict):
            return {}

        categorical_sizes = {}
        for name, size in self.categorical_features.items():
            if not isinstance(name, str):
                raise TypeError(
                    "categorical_features dict keys must be feature names."
                )
            if not isinstance(size, numbers.Integral) or int(size) < 2:
                raise ValueError(
                    "categorical_features dict values must be integers >= 2."
                )
            categorical_sizes[name] = int(size)

        return categorical_sizes

    def _resolve_feature_metadata(self, feature_names):
        metadata = [
            {
                "name": feature_name,
                "kind": "continuous",
                "group_name": feature_name,
                "split_weight": 1.0,
                "fixed_threshold": None,
                "block_reuse": False,
                "max_group_splits": None,
            }
            for feature_name in feature_names
        ]

        name_to_index = {name: idx for idx, name in enumerate(feature_names)}

        if self.categorical_features and not isinstance(self.categorical_features, dict):
            for feature in self.categorical_features:
                if isinstance(feature, str):
                    if feature not in name_to_index:
                        raise ValueError(
                            f"Unknown categorical feature name: {feature}."
                        )
                    feature_idx = name_to_index[feature]
                elif isinstance(feature, numbers.Integral):
                    if feature < 0 or feature >= self.n_features_in_:
                        raise ValueError(
                            f"Categorical feature index {feature} is out of bounds."
                        )
                    feature_idx = int(feature)
                else:
                    raise TypeError(
                        "categorical_features entries must be feature names, indices, or a dict of feature sizes."
                    )

                metadata[feature_idx]["kind"] = "categorical"

        for feature_name, size in self._resolve_categorical_feature_sizes().items():
            encoded_indices = []
            for category in range(1, size):
                encoded_name = f"{feature_name}_{category}"
                encoded_idx = name_to_index.get(encoded_name)
                if encoded_idx is not None:
                    encoded_indices.append(encoded_idx)

            if encoded_indices:
                if len(encoded_indices) != size - 1:
                    raise ValueError(
                        f"Expected {size - 1} one-hot columns for categorical feature '{feature_name}', got {len(encoded_indices)}."
                    )
                for encoded_idx in encoded_indices:
                    metadata[encoded_idx].update(
                        kind="categorical",
                        group_name=feature_name,
                        fixed_threshold=0.5,
                        block_reuse=True,
                        max_group_splits=size - 1,
                    )
                continue

            feature_idx = name_to_index.get(feature_name)
            if feature_idx is None:
                raise ValueError(
                    f"Categorical feature '{feature_name}' was not found in feature_names."
                )

            metadata[feature_idx].update(
                kind="categorical",
                group_name=feature_name,
                max_group_splits=size - 1,
            )

        self.split_weights_ = self._resolve_split_weights(feature_names, metadata)
        for feature_idx, split_weight in enumerate(self.split_weights_):
            metadata[feature_idx]["split_weight"] = float(split_weight)

        return metadata

    def _resolve_split_weights(self, feature_names, metadata):
        weights = np.ones(self.n_features_in_, dtype=float)
        group_to_indices = {}
        for feature_idx, feature_meta in enumerate(metadata):
            group_to_indices.setdefault(feature_meta["group_name"], []).append(
                feature_idx
            )

        for group_indices in group_to_indices.values():
            if (
                len(group_indices) > 1
                and all(metadata[idx]["block_reuse"] for idx in group_indices)
            ):
                group_weight = 1.0 / len(group_indices)
                #group_weight = 1.0 / 2.0  # TODO
                for idx in group_indices:
                    weights[idx] = group_weight

        if self.split_weights is None:
            return weights

        if isinstance(self.split_weights, dict):
            name_to_index = {name: idx for idx, name in enumerate(feature_names)}
            for feature, weight in self.split_weights.items():
                if not isinstance(weight, numbers.Real):
                    raise TypeError("split_weights values must be numeric.")
                if weight < 0:
                    raise ValueError("split_weights values must be non-negative.")

                if isinstance(feature, str):
                    if feature in name_to_index:
                        weights[name_to_index[feature]] = float(weight)
                    elif feature in group_to_indices:
                        group_indices = group_to_indices[feature]
                        distributed_weight = float(weight) / len(group_indices)
                        for idx in group_indices:
                            weights[idx] = distributed_weight
                    else:
                        raise ValueError(
                            f"Unknown split_weights feature name: {feature}."
                        )
                elif isinstance(feature, numbers.Integral):
                    if feature < 0 or feature >= self.n_features_in_:
                        raise ValueError(
                            f"split_weights feature index {feature} is out of bounds."
                        )
                    weights[int(feature)] = float(weight)
                else:
                    raise TypeError(
                        "split_weights keys must be feature names, group names, or indices."
                    )
        else:
            weights = np.asarray(self.split_weights, dtype=float)
            if weights.shape != (self.n_features_in_,):
                raise ValueError(
                    "split_weights must have one entry per feature column."
                )
            if np.any(weights < 0):
                raise ValueError("split_weights values must be non-negative.")

        if not np.any(weights > 0):
            raise ValueError("split_weights must leave at least one feature splittable.")

        return weights


class _TreeBuilder():
    r"""Array-based representation of a classification tree, trained with differential privacy.

    This tree mimics the architecture of the corresponding Tree from sklearn.tree.tree_, but without many methods given
    in Tree. The purpose of _FittingTree is to fit the parameters of the model, and have those parameters passed to
    Tree (using _FittingTree.__getstate__() and Tree.__setstate__()), to be used for prediction.

    Parameters
    ----------
    max_depth : int
        The maximum depth of the tree.

    n_features : int
        The number of features of the training dataset.

    classes : array-like of shape (n_classes,)
        The classes of the training dataset.

    bounds : tuple
        Bounds of the data, provided as a tuple of the form (min, max).  `min` and `max` can either be scalars, covering
        the min/max of the entire data.

    random_state : RandomState
        Controls the randomness of the building and training process: the feature to split at each node, the threshold
        to split at and the randomisation of the label at each leaf.
    """
    _TREE_LEAF = -1
    _TREE_UNDEFINED = -2

    def __init__(self, max_depth, n_features, classes, bounds, feature_metadata, random_state, split_threshold, splitbudget, leafbudget, leaf_algorithm, split_sigma):
        self.node_count = 0
        self.nodes = []
        self.max_depth = max_depth
        self.n_features = n_features
        self.classes = classes
        self.bounds = bounds
        self.feature_metadata = feature_metadata
        self.random_state = random_state
        self.split_threshold = split_threshold
        self.splitbudget = splitbudget
        self.leafbudget = leafbudget
        self.leaf_algorithm = leaf_algorithm
        self.split_sigma = split_sigma
        

    def __getstate__(self):
        """Get state of _FittingTree to feed into __setstate__ of sklearn.Tree"""
        d = {"max_depth": self.max_depth,
             "node_count": self.node_count,
             "nodes": np.array([tuple(node) for node in self.nodes], dtype=NODE_DTYPE),
             "values": self.values_}
        return d
    
    def _check_bounds_dpl(self, bounds, shape=0, min_separation=0.0, dtype=float):
        if not isinstance(bounds, tuple):
            raise TypeError(f"Bounds must be specified as a tuple of (min, max), got {type(bounds)}.")
        if not isinstance(shape, numbers.Integral):
            raise TypeError(f"shape parameter must be integer-valued, got {type(shape)}.")

        lower, upper = bounds

        if np.asarray(lower).size == 1 or np.asarray(upper).size == 1:
            lower = np.ravel(lower).astype(dtype)
            upper = np.ravel(upper).astype(dtype)
        else:
            lower = np.asarray(lower, dtype=dtype)
            upper = np.asarray(upper, dtype=dtype)

        if lower.shape != upper.shape:
            raise ValueError("lower and upper bounds must be the same shape array")
        if lower.ndim > 1:
            raise ValueError("lower and upper bounds must be scalar or a 1-dimensional array")
        if lower.size not in (1, shape):
            raise ValueError(f"lower and upper bounds must have {shape or 1} element(s), got {lower.size}.")

        n_bounds = lower.shape[0]

        for i in range(n_bounds):
            _lower = lower[i]
            _upper = upper[i]

            if not isinstance(_lower, numbers.Real) or not isinstance(_upper, numbers.Real):
                raise TypeError(f"Each bound must be numeric, got {_lower} ({type(_lower)}) and {_upper} ({type(_upper)}).")

            if _lower > _upper:
                raise ValueError(f"For each bound, lower bound must be smaller than upper bound, got {lower}, {upper})")

            if _upper - _lower < min_separation:
                mid = (_upper + _lower) / 2
                lower[i] = mid - min_separation / 2
                upper[i] = mid + min_separation / 2

        if shape == 0:
            return lower.item(), upper.item()

        if n_bounds == 1:
            lower = np.ones(shape, dtype=dtype) * lower.item()
            upper = np.ones(shape, dtype=dtype) * upper.item()

        return lower, upper


    def _is_categorical_feature(self, feature_idx):
        return self.feature_metadata[feature_idx]["kind"] == "categorical"

    def _updated_used_features(self, used_features, feature):
        next_used_features = set(used_features or ())
        if self.feature_metadata[feature]["block_reuse"]:
            next_used_features.add(feature)
        return frozenset(next_used_features)

    def _available_features(self, used_features):
        used_features = used_features or ()
        return np.asarray(
            [
                feature_idx
                for feature_idx, feature_meta in enumerate(self.feature_metadata)
                if feature_meta["split_weight"] > 0
                and not (
                    feature_meta["block_reuse"] and feature_idx in used_features
                )
            ],
            dtype=int,
        )

    def _sample_split(self, bounds_lower, bounds_upper, used_features=None):
        if not used_features:
            available_features = self._cached_features
            weights = self._cached_weights
        else:
            available_features = self._available_features(used_features)
            if available_features.size == 0:
                return None, None
            weights = np.asarray(
                [
                    self.feature_metadata[feature_idx]["split_weight"]
                    for feature_idx in available_features
                ],
                dtype=float,
            )
            weights /= weights.sum()

        if available_features.size == 0:
            return None, None

        feature = int(self.random_state.choice(available_features, p=weights))

        threshold = self.feature_metadata[feature]["fixed_threshold"]
        if threshold is None:
            threshold = self.random_state.uniform(
                bounds_lower[feature], bounds_upper[feature]
            )

        return feature, float(threshold)


    def _total_count(self, X_by_class):
        return sum(class_samples.shape[0] for class_samples in X_by_class)

    def _class_counts(self, X_by_class):
        return [class_samples.shape[0] for class_samples in X_by_class]

    def _group_samples_by_class(self, X, y):
        indices = np.arange(X.shape[0])
        return [indices[y[:, 0] == class_idx] for class_idx in range(len(self.classes))]

    def _split_samples_by_threshold(self, X_by_class, feature, threshold):
        left = []
        right = []
        
        is_sp = issparse(self.X)
            
        for class_indices in X_by_class:
            if len(class_indices) == 0:
                left.append(class_indices)
                right.append(class_indices)
                continue
            
            if is_sp:
                feature_values = self.X[class_indices, feature].toarray().ravel()
            else:
                feature_values = self.X[class_indices, feature]
                
            left_mask = feature_values <= threshold
            left.append(class_indices[left_mask])
            right.append(class_indices[~left_mask])
            
        return left, right
    

    def get_tree_expander(self, X_by_class, bounds, root, depth, expanded):
        if depth == 0:
            raise ValueError("This variant of the algorithm show never be called with depth zero. Only used for h>1.")
        elif expanded:
            return self._NodeIterator(root, depth)
        else:
            if not X_by_class or not bounds:
                raise ValueError("X_by_class and bounds must be set if tree is not expanded.")
            return self._NodeExpander(self, X_by_class, bounds, root, depth)
    
    
    def _make_split(self, bounds, X_by_class, used_features=None):
        bounds_lower, bounds_upper = bounds
        
        feature, threshold = self._sample_split(
            bounds_lower,
            bounds_upper,
            used_features=None,
        )
        if feature is None:
            return None
            
        left_bounds_upper = bounds_upper.copy()
        left_bounds_upper[feature] = threshold
        right_bounds_lower = bounds_lower.copy()
        right_bounds_lower[feature] = threshold
        
        left_by_class, right_by_class = self._split_samples_by_threshold(
            X_by_class,
            feature,
            threshold,
        )
        return feature, threshold, bounds_lower, left_bounds_upper, right_bounds_lower, bounds_upper, left_by_class, right_by_class
    

    def sparseHistogramQuery(self, value):
        if value <= DETERMINISTIC_IGNORE_THRESHOLD:
            return 0
        noisy_value = addGaussianNoise(
            value,
            sigma=self.split_sigma,
            random_state=self.random_state,
        )
        return noisy_value if noisy_value > self.split_threshold else 0


    def sparseAboveThresholdQuery(self, n_node_samples):
        return self.sparseHistogramQuery(n_node_samples) >= self.split_threshold


    def expandNode(self, X_by_class, bounds, root, depth, tree):
        #print(f"Expanding {root.index} by {depth}")
        if depth < 1 or root.left_child:
            raise ValueError(f"expandNode called with depth {depth} and left_child {root.left_child}")

        limit_depth = root.depth + depth

        

        feature, threshold, bounds_lower, left_bounds_upper, right_bounds_lower, bounds_upper, left_by_class, right_by_class  = self._make_split(bounds, X_by_class)

        root.feature = feature
        root.threshold = threshold
        
        # For garbage collection
        root.X_by_class = None
        root.bounds = None
        stack = [
            StackNodeExpander(parent=root, is_left=False, depth=root.depth + 1,
                                    bounds=(right_bounds_lower, bounds_upper), X=right_by_class),
            StackNodeExpander(parent=root, is_left=True, depth=root.depth + 1,
                                    bounds=(bounds_lower, left_bounds_upper), X=left_by_class),
        ]
        
        while len(stack):
            parent, is_left, depth, bounds, X_by_class = stack.pop()
            node_sample_count = self._total_count(X_by_class)
            idx = 2 * parent.index if is_left else 2 * parent.index + 1

            node = _BuildNode(_TreeBuilder._TREE_UNDEFINED, _TreeBuilder._TREE_UNDEFINED, depth=depth, index=idx, n_node_samples=node_sample_count, parent = parent, left_child=None, right_child=None, values=self._class_counts(X_by_class))
            tree[depth].append(node)

            if is_left:
                parent.left_child = node
            else:
                parent.right_child = node

            if depth == limit_depth:
                # we reached the desired depth -> no more splitting
                # store values needed if we split later
                node.X_by_class = X_by_class
                node.bounds = bounds
                continue
            
            if node_sample_count <= DETERMINISTIC_IGNORE_THRESHOLD:
                # We do not split empty nodes
                continue
            
            # make split
            feature, threshold, bounds_lower, left_bounds_upper, right_bounds_lower, bounds_upper, left_by_class, right_by_class = self._make_split(bounds, X_by_class)
            node.feature = feature
            node.threshold = threshold

            stack.append(StackNodeExpander(parent=node, is_left=False, depth=depth+1,
                                        bounds=(right_bounds_lower, bounds_upper), X=right_by_class))
            stack.append(StackNodeExpander(parent=node, is_left=True, depth=depth+1,
                                        bounds=(bounds_lower, left_bounds_upper), X=left_by_class))
        

    def recursiveSparseBuilder(self, height, X_by_class, bounds, root, expanded, tree):
        if height == 1:
            if root.mark == -1:
                raise ValueError("Queried light node again in recursive call. This should never happen")
            if root.mark == 0:
                root.mark = 1 if self.sparseAboveThresholdQuery(root.n_node_samples) else -1
            return

        l_mid = height // 2
        
        if not expanded:
            self.expandNode(root.X_by_class, root.bounds, root, l_mid, tree)
        
        midqueue = tree[root.depth + l_mid]
        midfirstidx = root.index * (2**l_mid)
        midafteridx = (root.index + 1) * (2**l_mid)

        # Remove nodes that were never directly from a past subtree
        while len(midqueue) and midqueue[0].index < midfirstidx:
            midqueue.popleft() 

        # All nodes on layer l_mid in this subtree has index [root.index * (2**l_mid), (root.index + 1) * (2**l_mid) - 1] 
        # We use queues since many of the nodes might not exist if they have zero nodes as parents   
        while len(midqueue) and midqueue[0].index < midafteridx:
            node = midqueue.popleft()
            if node.mark == -1:
                raise ValueError("We should never revisit LIGHT nodes")

            if node.mark == 1 or self.sparseAboveThresholdQuery(node.n_node_samples):
                cur = node
                for _ in range(l_mid + 1): # Mark ancestors HEAVY up potentially including root
                    if cur.mark:
                        break
                    cur.mark = 1 
                    cur = cur.parent

                if height <= 2: # "mid" layer is bottom layer
                    continue

                if expanded:
                    self.recursiveSparseBuilder(height - l_mid - 1, None, None, node.left_child, expanded=expanded, tree=tree)
                    self.recursiveSparseBuilder(height - l_mid - 1, None, None, node.right_child, expanded=expanded, tree=tree)
                else:
                    if node.left_child:
                        raise ValueError("Expanded set to false for node with a child")
                    
                    # This current node is the furthest we ever expanded. Split it and make a recursive call for each child.
                    # Nodes at the bottom layer must have bounds and X_by_class stored. 
                    # We remove them for middle nodes to allow for garbage collection.

                    
                    feature, threshold, bounds_lower, left_bounds_upper, right_bounds_lower, bounds_upper, left_by_class, right_by_class  = self._make_split(node.bounds, node.X_by_class)

                    node.bounds = None
                    node.X_by_class = None

                    node.feature = feature
                    node.threshold = threshold

                    left_bounds = (bounds_lower, left_bounds_upper)
                    right_bounds = (right_bounds_lower, bounds_upper)

                    left_node = _BuildNode(_TreeBuilder._TREE_UNDEFINED, _TreeBuilder._TREE_UNDEFINED, depth= node.depth + 1, index=node.index * 2, n_node_samples=self._total_count(left_by_class), parent=node, left_child=None, right_child=None, values=self._class_counts(left_by_class), bounds=left_bounds, X_by_class=left_by_class)
                    right_node = _BuildNode(_TreeBuilder._TREE_UNDEFINED, _TreeBuilder._TREE_UNDEFINED, depth= node.depth + 1, index=node.index * 2 + 1, n_node_samples=self._total_count(right_by_class), parent=node, left_child=None, right_child=None, values=self._class_counts(right_by_class), bounds=right_bounds, X_by_class=right_by_class)
                    tree[node.depth + 1].append(left_node)
                    tree[node.depth + 1].append(right_node)

                    node.left_child = left_node
                    node.right_child = right_node

                    self.recursiveSparseBuilder(height - l_mid - 1, left_by_class, left_bounds, left_node, expanded=expanded, tree=tree)
                    self.recursiveSparseBuilder(height - l_mid - 1, right_by_class, right_bounds, right_node, expanded=expanded, tree=tree)
            else:
                # Set as LIGHT. In the conceptual algorithm we mark all decendants. We don't need to do actively do that since we just avoid exploring that part of the tree. 
                # The subtree cannot contain any HEAVY nodes since this node was initially unmarked
                node.mark = -1 

        # Recursive call up the tree
        self.recursiveSparseBuilder(height // 2, X_by_class, bounds, root, expanded=True, tree=tree)
        

    def _dense_nodes_from_build_tree(self, root, max_depth):
        dense_nodes = []
        dense_ids = {id(root): 0}
        queue = deque([root])

        while queue:
            build_node = queue.popleft()
            node_id = dense_ids[id(build_node)]

            if not build_node.mark:
                raise ValueError("BUG: _dense_nodes_from_build_tree queried an unmarked node")

            if build_node.mark == -1:
                feature = self._TREE_UNDEFINED
                threshold = self._TREE_UNDEFINED
                left_child = self._TREE_LEAF
                right_child = self._TREE_LEAF
            else:
                if build_node.depth == max_depth - 1:
                    if build_node.left_child is not None or build_node.right_child is not None:
                        raise ValueError("Final leaf already split")

                    # The final layer was a HEAVY node. Still needs to be split
                    # This builder differs slighlty from the other
                    # Here we query layers [0,max_depth-1] for heavy hitters. We don't include layer max_depth because they can never be split. We therefore have to split here
                    feature, threshold, bounds_lower, left_bounds_upper, right_bounds_lower, bounds_upper, left_by_class, right_by_class  = self._make_split(build_node.bounds, build_node.X_by_class)

                    build_node.bounds = None
                    build_node.X_by_class = None

                    build_node.feature = feature
                    build_node.threshold = threshold

                    left_bounds = (bounds_lower, left_bounds_upper)
                    right_bounds = (right_bounds_lower, bounds_upper)

                    left_node = _BuildNode(_TreeBuilder._TREE_UNDEFINED, _TreeBuilder._TREE_UNDEFINED, depth= root.depth + 1, index=root.index * 2, n_node_samples=self._total_count(left_by_class), parent=build_node, left_child=None, right_child=None, values=self._class_counts(left_by_class), bounds=left_bounds, X_by_class=left_by_class)
                    right_node = _BuildNode(_TreeBuilder._TREE_UNDEFINED, _TreeBuilder._TREE_UNDEFINED, depth= root.depth + 1, index=root.index * 2 + 1, n_node_samples=self._total_count(right_by_class), parent=build_node, left_child=None, right_child=None, values=self._class_counts(right_by_class), bounds=right_bounds, X_by_class=right_by_class)

                    build_node.left_child = left_node
                    build_node.right_child = right_node

                    left_node.mark = -1
                    right_node.mark = -1

                if build_node.left_child is None or build_node.right_child is None:
                    raise ValueError("Sparse build tree contains a non-leaf node without both children.")

                left_key = id(build_node.left_child)
                if left_key not in dense_ids:
                    dense_ids[left_key] = len(dense_ids)
                    queue.append(build_node.left_child)
                right_key = id(build_node.right_child)
                if right_key not in dense_ids:
                    dense_ids[right_key] = len(dense_ids)
                    queue.append(build_node.right_child)

                feature = build_node.feature
                threshold = build_node.threshold
                left_child = dense_ids[left_key]
                right_child = dense_ids[right_key]

            dense_nodes.append(
                _Node(
                    node_id,
                    feature,
                    threshold,
                    n_node_samples=build_node.n_node_samples,
                    left_child=left_child,
                    right_child=right_child,
                    values=list(build_node.values) if build_node.values is not None else None,
                )
            )

        return dense_nodes

    
    def sparseBuilder(self, max_depth, X, y):
        print("building tree.")
        self.X = X
        
        self.bounds = self._check_bounds_dpl(self.bounds, shape=self.n_features)
        
        self._cached_features = self._available_features(frozenset())
        if self._cached_features.size > 0:
            weights = np.asarray([
                self.feature_metadata[f_idx]["split_weight"] 
                for f_idx in self._cached_features
            ], dtype=float)
            self._cached_weights = weights / weights.sum()
        else:
            self._cached_weights = np.array([])
            
        X_by_class = self._group_samples_by_class(X, y)
        root = _BuildNode(self._TREE_UNDEFINED, self._TREE_UNDEFINED, depth=0, index=1, n_node_samples=self._total_count(X_by_class), left_child=None, right_child=None, values=self._class_counts(X_by_class), X_by_class=X_by_class, bounds=self.bounds)

        # Stores the nodes in the order we encounter them for each layer.
        # This allows for fast access to the middle layer of the current tree.
        # Note that we exploit that each layer is ALWAYS traversed in left-to-right order both when creating nodes and querying counts.
        tree = [deque([root])] + [deque() for _ in range(max_depth + 1)] # Maybe +1 is not needed but it doesn't matter

        self.recursiveSparseBuilder(max_depth, X_by_class, self.bounds, root, expanded=False, tree=tree)
        self.nodes = self._dense_nodes_from_build_tree(root, max_depth)
        self.node_count = len(self.nodes)
        
        self.X = None

        return self

    def fit(self, X, y):
        """Fit the tree to the given training data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and n_features is the number of features.

        y : array-like, shape (n_samples,)
            Target vector relative to X.

        """
        self.sparseBuilder(self.max_depth, X, y)   

        counts = np.zeros(shape=(self.node_count, 1, len(self.classes)))
        for node in self.nodes:
            if node.left_child == -1:
                counts[node.node_id, 0, : len(node.values)] = node.values

        self.values_ = self._assign_leaf_values(counts)
        
        return self

    def _assign_leaf_values(self, counts):
        """Assign leaf values based on the configured leaf algorithm."""
        values = np.zeros(shape=(self.node_count, 1, len(self.classes)))
        leaf_algorithm = self.leaf_algorithm.lower()
        if leaf_algorithm != "useestimates" and counts is None:
            raise ValueError("counts are required unless leaf_algorithm is useEstimates")
        
        debug = False
        if debug:
            print(f"For this tree, rho={self.leafbudget} is used.")

        if leaf_algorithm == "exponential_mechanism" and self.leafbudget:
            #calibratedNoisyMax = get_calibrated_noisy_max(self.leafbudget)
            calibratedNoisyMax = get_seeded_noisy_max(target_rho=self.leafbudget, seed=self.random_state)

        for node in self.nodes:
            if node.left_child != self._TREE_LEAF:
                if node.values and node.n_node_samples: # NOTE: Only for debugging and plotting
                    class_distribution = np.asarray(node.values, dtype=float) / node.n_node_samples
                    values[node.node_id, 0, : len(class_distribution)] = class_distribution
                continue

            if leaf_algorithm == "useestimates":
                raise RuntimeError(
                    "leaf_algorithm='useEstimates' requires count_estimates on every leaf."
                )
                estimates = np.asarray(node.count_estimates, dtype=float)
                values[node.node_id, 0, : len(estimates)] = estimates
                continue

            noisycounts = np.asarray(
                GaussianMechanism(
                    counts[node.node_id, 0, :].tolist(),
                    self.leafbudget + node.extra_budget
                    if node.extra_budget and self.leafbudget
                    else self.leafbudget,
                    random_state=self.random_state,
                ),
                dtype=float,
            )

            if leaf_algorithm == "noisy_majority":  # TODO: Refactor to privacy_mechanism class?
                majority_class = int(np.argmax(noisycounts))
                values[node.node_id, 0, majority_class] = 1
            elif leaf_algorithm == "exponential_mechanism":
                if self.leafbudget:
                    majority_class = calibratedNoisyMax(counts[node.node_id, 0, :])
                else:
                    leaf_counts = counts[node.node_id, 0, :]
                    candidates = np.flatnonzero(leaf_counts == leaf_counts.max())
                    majority_class = int(self.random_state.choice(candidates))
                values[node.node_id, 0, majority_class] = 1
            elif leaf_algorithm == "noisy_count":
                values[node.node_id, 0, : len(noisycounts)] = noisycounts
            elif leaf_algorithm == "noisy_fraction":
                total_noisy_count = float(np.sum(noisycounts))
                if total_noisy_count != 0:
                    noisyfractions = noisycounts / total_noisy_count
                else:
                    noisyfractions = np.zeros_like(noisycounts)
                values[node.node_id, 0, : len(noisyfractions)] = np.clip(
                    noisyfractions, 0.0, 1.0
                )
            else:
                raise ValueError(
                    f"Unknown leaf algorithm {self.leaf_algorithm}"
                )

        return values

    def apply(self, X):
        """Finds the terminal region (=leaf node) for each sample in X."""
        if issparse(X):
            X = X.tocsr(copy=False)
        n_samples = X.shape[0]
        out = np.zeros((n_samples,), dtype=int)

        for i in range(n_samples):
            node = self.nodes[0]

            while node.left_child != self._TREE_LEAF:
                if X[i, node.feature] <= node.threshold:
                    node = self.nodes[node.left_child]
                else:
                    node = self.nodes[node.right_child]

            out[i] = node.node_id

        return out


class _BuildNode():
    """Base nodes for the binary search trimming approach."""
    def __init__(self, feature, threshold, depth, index, n_node_samples = 0, parent = None, left_child = None, right_child = None, values = None, is_leaf = False, bounds = None, X_by_class = None):
        self.parent = parent
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.n_node_samples = n_node_samples # TODO: Only for debug
        self.values = values # TODO: Only for debug
        self.is_leaf = is_leaf
        self.depth = depth
        self.mark = 0 # -1: light, 0: unmarked, 1: heavy. Most light nodes are never actually marked since we ignore querying that part instead
        self.index = index # From 1 to 2^h - 1. Parent is self.index//2. Children are self.index * 2 and (self.index * 2) + 1
        self.bounds = bounds
        self.X_by_class = X_by_class

    def __iter__(self):
        """Defines parameters needed to populate NODE_DTYPE for Tree.__setstate__ using tuple(_Node)."""
        yield self.left_child
        yield self.right_child
        yield self.feature
        yield self.threshold
        yield 0.0  # Impurity
        yield self.n_node_samples  # n_node_samples
        yield self.n_node_samples  # weighted_n_node_samples

        # remove branch when scikit-learn v1.3 is min requirement
        if len(NODE_DTYPE) > 7:
            yield False

class _Node:
    """Base storage structure for the nodes in a _FittingTree object."""
    def __init__(self, node_id, feature, threshold, n_node_samples = 0, left_child = -1, right_child = -1, values = None, extra_budget = 0):
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.node_id = node_id
        self.n_node_samples = n_node_samples # TODO: Only for debug
        self.values = values # TODO: Only for debug
        self.extra_budget = extra_budget
    

    def __iter__(self):
        """Defines parameters needed to populate NODE_DTYPE for Tree.__setstate__ using tuple(_Node)."""
        yield self.left_child
        yield self.right_child
        yield self.feature
        yield self.threshold
        yield 0.0  # Impurity
        yield self.n_node_samples  # n_node_samples
        yield self.n_node_samples  # weighted_n_node_samples

        # remove branch when scikit-learn v1.3 is min requirement
        if len(NODE_DTYPE) > 7:
            yield False