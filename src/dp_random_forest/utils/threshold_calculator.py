

import math
import logging
logger = logging.getLogger(__name__)

import numpy as np
from scipy.stats import norm

# This file contains code for calculating noise magnitude and threshold required by the sparse Gaussian mechanism.

def analytic_gaussian(epsilon, mu): return norm.cdf(mu / 2 - epsilon / mu) - np.exp(epsilon) * norm.cdf(-mu / 2 - epsilon / mu)
def delta_bw(epsilon, mu): return  norm.cdf(mu / 2 - epsilon / mu) - np.exp(epsilon) * norm.cdf(-mu / 2 - epsilon / mu)
def tau_diff_func(delta, c_u, sigma): return norm.ppf((1 - delta) ** (1 / c_u)) * sigma

def gshm_exact(k, sigma, tau, epsilon):
    """
    Does the same as gshm_exact_detailed, but magnitures faster.
    :return:
    """
    epsilon2 = lambda a_eq: epsilon  - a_eq * np.log(norm.cdf(tau / sigma))
    epsilon3 = lambda a_eq : epsilon + a_eq * np.log(norm.cdf(tau / sigma))

    case_one = 1 - norm.cdf(tau / sigma) ** k
    case_two = max([1 - norm.cdf(tau / sigma) ** (i) + norm.cdf(tau / sigma)**(i) * analytic_gaussian(epsilon2(i), np.sqrt(k - i) / sigma) for i in range(1, k+1)])
    case_three = max([analytic_gaussian(epsilon3(i), np.sqrt(k - i) / sigma) for i in range(1, k+1)])

    return max(case_one, case_two, case_three)

def gshm_exact_detailed(k, sigma, tau, epsilon):
    """
    Implements the exact analysis due to [Wilkins et al.],
    Returns a list of exact values of delta.
    :param k:
    :param sigma:
    :param tau:
    :param epsilon:
    :return:
    """
    part_one, part_two, part_three, maximum = np.zeros(k), np.zeros(k), np.zeros(k), np.zeros(k)

    for i in range(1, k + 1):
        a_eq = i - 1
        mu = np.sqrt(k - a_eq) / sigma # p. 12
        epsilon2 = epsilon - a_eq * np.log(norm.cdf(tau / sigma))
        epsilon3 = epsilon + a_eq * np.log(norm.cdf(tau / sigma))

        part_two[i-1] = 1 - norm.cdf(tau / sigma) ** a_eq + norm.cdf(tau / sigma) ** a_eq * analytic_gaussian(epsilon2, mu)
        part_three[i - 1] = analytic_gaussian(epsilon3, mu)
        part_one[i-1] = 1 - norm.cdf(tau / sigma) ** k

        maximum[i-1] = np.max([part_one[i-1], part_two[i-1], part_three[i-1]])
    return [
            part_one,
            part_two,
            part_three,
            maximum,
    ]

def check_validity(k, sigma, tau, epsilon, delta, PRECISION=10**-10):
    """
    Checks whether the given parameters satisfy (eps, delta)-dp guarantees for GSHM
    :param k:
    :param sigma:
    :param tau:
    :param epsilon:
    :param delta:
    :return:
    """
    delta_upper = gshm_exact(k, sigma, tau, epsilon)
    logging.debug(f'We have: {delta_upper} vs <{delta}')
    return delta_upper <= delta + PRECISION

def threshold_add_the_delta(total_delta_budget, epsilon, k, sigma):
    """
    Directly returns the threshold for the add_the_delta approach
    Computes the add-the-Deltas as in the paper
    "EXACT PRIVACY ANALYSIS OF THE GAUSSIAN SPARSE HISTOGRAM MECHANISM"
    :param total_delta_budget:
    :param epsilon:
    :param k:
    :param sigma:
    :return:
    """
    mu = math.sqrt(k) / sigma
    return sigma * norm.ppf(
        (1 - total_delta_budget + analytic_gaussian(epsilon, mu)) ** (1 / k)
    )

def compute_threshold_exact(k, delta, sigma):
    """
    Returns the treshold for the infinite privacy loss event part.
    We skip the mixed case here as it should not make any difference.
    :param k:
    :param delta:
    :param sigma:
    :return:
    """
    return norm.ppf((1-delta)**(1/k)) * sigma

def minimum_amount_of_noise(candidate_mu, epsilon, delta, number_iterations=100):
    """
    As the previous work, we use newtons method to find a minimum amount of noise required such that the part of the Gaussian mechanism fulfills our privacy guarantees
    :param k: number of counts to be changed
    :param epsilon:
    :param delta: The delta we would like to get.
    :return: the minimum amount of noise required to satisfy the (eps, delta) guarantee.
    """
    mu = candidate_mu
    for i in range(number_iterations):
        f = norm.cdf(mu/2 - epsilon/mu) - np.e**epsilon * norm.cdf(-mu/2 - epsilon/mu) - delta
        f_derivative = ((norm.pdf(mu/2 - epsilon/mu)) * (1/2 + epsilon/(mu**2))
                        - np.exp(epsilon) * norm.pdf(-mu/2 - epsilon/mu) * (-1/2 + epsilon / (mu**2)))
        mu = mu - f/f_derivative
    return mu


def compute_threshold_curve_tighter(epsilon, delta, k, max_sigma_factor = 2, datapoints=10):
    """
    This return a list of 
    :param total_delta_budget: 
    :param epsilon: 
    :param k: 
    :return: 
    """""
    # Minimum amount of noise required to gain (eps, delta)-DP
    mu = minimum_amount_of_noise(math.sqrt(epsilon), epsilon, delta)
    min_sigma = math.sqrt(k)/mu
    max_sigma = min_sigma * max_sigma_factor
    sigmas = np.linspace(min_sigma, max_sigma, datapoints)
    thresholds = []
    for i, sig in enumerate(sigmas):
        logger.debug(f'Working on {i+1}/{len(sigmas)}')
        tau = compute_threshold_exact(k, delta, sig)
        if not check_validity(k, sig, tau, epsilon, delta):
            logging.debug("The combination of the given paramters does not satisfy the required privacy guarantees")
            thresholds.append(-1)
        else:
            logging.info(f'adding: {tau}')
            thresholds.append(tau)
    return  sigmas, thresholds, min_sigma


# Returns the parameters for the smallest threshold among a set of candidates
# For many parameters the smallest threshold also has the smallest magnitude of noise
# But it could be possible to achieve a smaller threshold by increasing the magnitude of noise slightly
def compute_parameters_min_threshold(epsilon, delta, k, max_sigma_factor=2, datapoints=20):
    if not (epsilon or delta):
        return None, None
    sigmas, thresholds, _ = compute_threshold_curve_tighter(epsilon, delta, k, max_sigma_factor, datapoints)

    return sigmas[np.argmin(thresholds)], thresholds[np.argmin(thresholds)]

if __name__ == "__main__":
    print(compute_parameters_min_threshold(epsilon=1.1, delta=5.5e-07, k = 2 * (1 + math.floor(np.log2(100 + 1)))))