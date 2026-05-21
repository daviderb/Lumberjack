from sklearn.ensemble._forest import ForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils._param_validation import StrOptions, Interval
from numbers import Real
from .dp_alternative_sparse import DPAlternativeSparseExtraTreeClassifier
from ..utils.threshold_calculator import compute_parameters_min_threshold
import math
import numpy as np


def _rho_epsilon_to_delta(rho: float, epsilon: float) -> float:
    """Helper function translated from JS rhoEpsilonToDelta."""
    # Prevent division by zero if binary search pushes rho too close to 0
    if rho <= 0:
        raise ValueError("rho must be strictly positive.")
        
    term = (epsilon - rho) / (2 * rho)
    
    # Calculate denominator components
    sqrt_component = math.sqrt((1 + term)**2 + 4 / (math.pi * rho))
    delta = 2 * math.exp(-term * term * rho) / (1 + term + sqrt_component)
    
    if delta == 0:
        raise ValueError("Resulting delta is too small (approaches 0).")
        
    return delta

def get_zcdp_rho_from_epsilon_delta(epsilon: float, delta: float, tolerance: float = 1e-9) -> float:
    """
    Calculates rho given an epsilon and delta using a binary search algorithm.
    
    :param epsilon: The target epsilon value.
    :param delta: The target delta value.
    :param tolerance: Float comparison tolerance (replaces JS 'closeEnough').
    :return: The calculated rho value.
    """
    if not (epsilon or delta):
        return None

    low = 0.0
    # The conversion formula only holds for ε ≥ ρ, so we use it as our upper bound.
    high = float(epsilon)
    
    # First, we check that we have a chance to achieve our target δ.
    min_delta = _rho_epsilon_to_delta(epsilon, epsilon)
    if min_delta < delta:
        raise ValueError(
            f"Impossible to find an appropriate value for rho. "
            f"Minimal delta for epsilon={epsilon} is {min_delta}"
        )
        
    while True:
        mid = (low + high) / 2.0
        
        try:
            del_val = _rho_epsilon_to_delta(mid, epsilon)
        except ValueError:
            raise ValueError("Binary search failed: rho approached 0.")
            
        # Check if we are close enough to the target delta
        if math.isclose(del_val, delta, rel_tol=tolerance, abs_tol=1e-12):
            return mid
            
        # Check if binary search is stuck (floating point limits)
        if mid == low or mid == high:
            raise ValueError(
                f"Binary search failed to converge. "
                f"low={low}, high={high}, mid={mid}, del={del_val}"
            )
            
        # Adjust bounds
        if del_val > delta:
            high = mid
        else:
            low = mid


class DPSparseExtraTreesClassifier(ForestClassifier):
    """
    Inspired by the scikit learn implementation https://github.com/scikit-learn/scikit-learn/blob/d3898d9d5/sklearn/ensemble/_forest.py#L1944
    We use some of the same parameters while others have been removed. Parameters that would break DP are removed, such as min splitting size.
    
    Parameters
    ----------
    splitbudget : float, default=None
        Total privacy budget used by each tree for splitting nodes.
        None is used for infinite privacy budget.
        NOTE: We use zCDP for now, care is needed if we want to change to RDP, GDP etc.
        We could change budget to a tuple, e.g. (1, "zCDP").

    leafbudget : float, default=None
        Total privacy budget used by each tree to privatize the leaf nodes.
        None is used for infinite privacy budget.
        NOTE: We use zCDP for now, care is needed if we want to change to RDP etc.

    epsilon : float, default=None
        Total epsilon budget used by sparse and sparseBinSearch forests.

    delta : float, default=None
        Total delta budget used by sparse and sparseBinSearch forests.

    splitting_fraction : float, default=None
        Fraction of the total ``(epsilon, delta)`` budget allocated to sparse
        tree construction. The remaining budget is converted to zCDP and used
        for the leaf mechanism.

    split_threshold : int, default=1
        Threshold for deciding to split a node

    n_estimators : int, default=100
        The number of trees in the forest.

    bounds : (list<float>, list<float>), default=None
        Minimum and maximum bounds for all attributes.
        NOTE: If None, we compute bounds from the data, breaking DP best practices

    max_depth : int, default=20
        The maximum depth of the tree. Non-DP variants sometimes set this
        to none and expand to a certain size. For DP we need a stopping 
        condition that does not depend deterministically on the node size.

    n_jobs : int, default=None
        The number of jobs to run in parallel. :meth:`fit`, :meth:`predict`,
        :meth:`decision_path` and :meth:`apply` are all parallelized over the
        trees. ``None`` means 1 unless in a :obj:`joblib.parallel_backend`
        context. ``-1`` means using all processors. See :term:`Glossary
        <n_jobs>` for more details.

    random_state : int, RandomState instance or None, default=None
        Controls 3 sources of randomness:

        - the bootstrapping of the samples used when building trees
          (if ``bootstrap=True``)
        - the sampling of the features to consider when looking for the best
          split at each node (if ``max_features < n_features``)
        - the draw of the splits for each of the `max_features`

        TODO: Make sure we don't mess this up when adding privacy randomness.
        We have to use fixed random seeds as well.

    verbose : int, default=0
        Controls the verbosity when fitting and predicting.

    Attributes
    ----------
    NOTE: Copied from sklearn. Might not be up to date

    estimator_ : :class:`~DPExtraTreeClassifier`
        The child estimator template used to create the collection of fitted
        sub-estimators.

    estimators_ : list of DecisionTreeClassifier
        The collection of fitted sub-estimators.

    classes_ : ndarray of shape (n_classes,) or a list of such arrays
        The classes labels (single output problem), or a list of arrays of
        class labels (multi-output problem).

    n_classes_ : int or list
        The number of classes (single output problem), or a list containing the
        number of classes for each output (multi-output problem).

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

    oob_score_ : float
        Score of the training dataset obtained using an out-of-bag estimate.
        This attribute exists only when ``oob_score`` is True.

    oob_decision_function_ : ndarray of shape (n_samples, n_classes) or \
            (n_samples, n_classes, n_outputs)
        Decision function computed with out-of-bag estimate on the training
        set. If n_estimators is small it might be possible that a data point
        was never left out during the bootstrap. In this case,
        `oob_decision_function_` might contain NaN. This attribute exists
        only when ``oob_score`` is True.

    estimators_samples_ : list of arrays
        The subset of drawn samples (i.e., the in-bag samples) for each base
        estimator. Each subset is defined by an array of the indices selected.

    References
    ----------
    .. [1] P. Geurts, D. Ernst., and L. Wehenkel, "Extremely randomized
           trees", Machine Learning, 63(1), 3-42, 2006.
    """

    _parameter_constraints: dict = {
        **ForestClassifier._parameter_constraints,
        **DecisionTreeClassifier._parameter_constraints,
        "class_weight": [
            StrOptions({"balanced_subsample", "balanced"}),
            dict,
            list,
            None,
        ],
        "epsilon": [Interval(Real, 0.0, None, closed="right"), None],
        "delta": [Interval(Real, 0.0, 1.0, closed="right"), None],
    }
    _parameter_constraints.pop("splitter")

    _parameter_constraints.pop("min_samples_split") # these values must be non-zero in the scikitlearn reference, but they can be zero in the DP variant.
    _parameter_constraints.pop("min_samples_leaf")
    _parameter_constraints.pop("min_weight_fraction_leaf")
    _parameter_constraints.pop("min_impurity_decrease")
    

    def __init__(
        self,
        splitbudget=None,
        leafbudget=None,
        leaf_algorithm="noisy_majority",
        epsilon=None,
        delta=None,
        splitting_fraction=None,
        split_threshold=None, # DP variant of min_samples_split
        n_estimators=100,
        *,
        bounds=None,
        feature_names=None,
        categorical_features=None,
        split_weights=None,
        max_depth=32,
        #max_features="sqrt", #1
        n_jobs=None,
        random_state=None,
        verbose=0,
        #min_samples_leaf=1,
        #bootstrap=False,
        #max_samples=None,
        #class_weight=None,
        #warm_start=False,
        #oob_score=False,
        #max_leaf_nodes=None,
    ):
        estimator_params = (
            "splitbudget_tree",
            "leafbudget_tree",
            "leaf_algorithm",
            "split_sigma",
            "max_depth",
            #"max_features",
            "random_state",
            "split_threshold",
            "bounds",
            "feature_names",
            "categorical_features",
            "verbose",
            "split_weights",
        )

        super().__init__(
            estimator=DPAlternativeSparseExtraTreeClassifier(),
            n_estimators=n_estimators,
            estimator_params=estimator_params,
            bootstrap=False,
            oob_score=False,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=False,
            class_weight=None,
            max_samples=None,
        )

        # sklearn's ForestClassifier tag machinery still instantiates the base
        # estimator with criterion=self.criterion, so keep a fixed internal
        # value even though criterion is not part of the public API for now.
        self.criterion = "gini"
        self.max_depth = max_depth
        self.max_features = 1
        self.max_leaf_nodes = None
        
        # Values forced to zero as we don't prune or enforce monotonicity constraints
        self.ccp_alpha = 0.0
        self.monotonic_cst = None

        self.splitbudget = splitbudget
        self.leafbudget = leafbudget
        self.leaf_algorithm = leaf_algorithm
        self.epsilon = epsilon
        self.delta = delta
        self.splitting_fraction = splitting_fraction
        self.split_k = None

        self.epsilon = float(epsilon) if epsilon else epsilon
        self.delta = float(delta) if delta else delta
        self.splitting_fraction = float(splitting_fraction) if splitting_fraction else splitting_fraction
        if self.splitting_fraction and not 0.0 < self.splitting_fraction < 1.0:
            raise ValueError(
                "splitting_fraction must lie strictly between 0 and 1."
            )

        split_epsilon = self.epsilon * self.splitting_fraction if self.epsilon else None
        split_delta = self.delta * self.splitting_fraction if self.delta else None

        leaf_epsilon = self.epsilon - split_epsilon if self.epsilon else None
        leaf_delta = self.delta - split_delta if self.delta else None


        self.splitbudget = {"epsilon": split_epsilon, "delta": split_delta}
        self.leafbudget = get_zcdp_rho_from_epsilon_delta(
            epsilon=leaf_epsilon,
            delta=leaf_delta,
        )
        self.splitbudget_tree = (split_epsilon, split_delta)
        self.leafbudget_tree = self.leafbudget / n_estimators if self.leafbudget else None


        self.split_k = n_estimators * (1 + math.floor(np.log2(max_depth + 1)))
        sigma, threshold = compute_parameters_min_threshold(
            epsilon=self.splitbudget_tree[0],
            delta=self.splitbudget_tree[1],
            k=self.split_k,
        )
        if threshold:
            self.split_threshold = threshold
        else:
            self.split_threshold = split_threshold
        self.split_sigma = sigma

        self.bounds = bounds
        self.feature_names = feature_names
        self.categorical_features = categorical_features
        self.split_weights = split_weights

        # These values are set to zero because deterministic stops
        # would break DP. We can use private variants to stop 
        # early.
        self.min_samples_split = 0
        self.min_samples_leaf = 0
        self.min_weight_fraction_leaf = 0
        self.min_impurity_decrease = -1 # Children are allowed to have worse impurity

        debug = False
        if debug:
            self._debug_print_sparse_configuration()

    def _debug_print_sparse_configuration(self):
        print("[DPExtraTreesClassifier][verbose] sparse forest hyperparameters:")
        print(
            "  "
            f"splitting_algorithm={self.splitting_algorithm}, "
            f"leaf_algorithm={self.leaf_algorithm}, "
            f"n_estimators={self.n_estimators}, "
            f"epsilon={self.epsilon}, "
            f"delta={self.delta}, "
            f"splitting_fraction={self.splitting_fraction}, "
            f"max_depth={self.max_depth}, "
            f"random_state={self.random_state}, "
            f"n_jobs={self.n_jobs}, "
            f"verbose={self.verbose}"
        )
        print("[DPExtraTreesClassifier][verbose] sparse forest derived privacy values:")
        print(
            "  "
            f"splitbudget={self.splitbudget}, "
            f"splitbudget_tree={self.splitbudget_tree}, "
            f"leaf_rho_total={self.leafbudget}, "
            f"leaf_rho_per_tree={self.leafbudget_tree}, "
            f"split_k={self.split_k}, "
            f"split_sigma={self.split_sigma}, "
            f"split_threshold={self.split_threshold}"
        )
