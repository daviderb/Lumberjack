import warnings

import numpy as np

# NOTE: For now we are using zCDP. It works well both for Gaussian noise and exponential mechanisms.
# Once we settle on an implementation, we can perform tight accounting e.g. with GDP

_WARNED_RANDOM_STATE_FALLBACKS = set()


def _warn_random_state_fallback(reason):
    if reason in _WARNED_RANDOM_STATE_FALLBACKS:
        return
    _WARNED_RANDOM_STATE_FALLBACKS.add(reason)
    warnings.warn(
        "Falling back to NumPy's global RNG in privacy_mechanisms.py; "
        f"runs may not be deterministic. Reason: {reason}",
        RuntimeWarning,
        stacklevel=3,
    )


def _check_random_state(random_state=None):
    if random_state is None:
        _warn_random_state_fallback("random_state is None")
        return np.random
    if random_state is np.random:
        _warn_random_state_fallback("random_state is NumPy's global RNG")
        return np.random
    if hasattr(random_state, "normal"):
        return random_state
    try:
        return np.random.RandomState(random_state)
    except (TypeError, ValueError):
        _warn_random_state_fallback(f"unsupported random_state={random_state!r}")
        return np.random


# Assumes sensitivity 1 - works for add/remove or zero-out adjacency. Sensitivity is root(2) if we use replacement.
def zCDPGaussianMech(counts, rho=None, random_state=None):
    if not rho:
        return counts
    counts = list(counts)
    rng = _check_random_state(random_state)
    scale = (1 / (2 * rho) ** 0.5)
    return [
        count + noise
        for count, noise in zip(counts, rng.normal(scale=scale, size=len(counts)))
    ]

def addGaussianNoise(val, sigma=None, random_state=None):
    if not sigma:
        return val
    return val + _check_random_state(random_state).normal(scale=sigma)


def LaplaceMechanism(x, sensitivity, epsilon):
    return x + np.random.laplace(loc=0, scale=sensitivity/epsilon)


# TODO: Implement variants of this. Here we simply adds noise to the threshold
def checkAboveThreshold(class0, class1, threshold, rho=None, random_state=None):
    if not rho:
        return class0 + class1 >= threshold
    noise = _check_random_state(random_state).normal(scale=(1 / (2 * rho) ** 0.5))
    return class0 + class1 >= threshold + noise
