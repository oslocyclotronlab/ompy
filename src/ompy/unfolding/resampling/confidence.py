from typing import Callable

import numpy as np
from scipy.stats import norm, poisson

from .stubs import CI_Method

from ...numbalib import njit

import jax.numpy as jnp
import jax


def resolve_ci_method(
    method: CI_Method,
) -> Callable[[np.ndarray, float], tuple[np.ndarray, np.ndarray]]:
    match method:
        case "standard" | "percentile":
            return standard_ci
        case "poisson":
            return poisson_ci
        case "bca":
            return bca_2
        case "supremum":
            return supremum_ci
        case "studentized supremum":
            return studentized_supremum_ci
        case "bonferroni percentile":
            return bonferroni_percentile_ci
        case "bonferroni bca":
            return bonferroni_bca_ci
        case "hotelling T2":
            return hotelling_T2_ci
        case _:
            raise ValueError(
                f"Unknown CI method: {method}. "
                f"Must be one of: {', '.join(CI_Method.__args__)}"
            )


def make_ci(
    data: np.ndarray,
    original: np.ndarray,
    alpha: float = 0.05,
    method: CI_Method = "standard",
    **kwargs,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate confidence intervals for bootstrap data.

    The CI are not necessarily probablistic, and *they do not describe the
    uncertainty about the central estimate*. They are simply the lower and
    upper bounds of the data.

    Args:
        data: Bootstrap samples array
        original: Original estimate of the statistic
        alpha: Significance level for confidence intervals (default: 0.05)
        method: Method to calculate confidence intervals (default: 'standard')
            One of: 'standard', 'poisson', 'bca'

    Returns:
        Tuple containing:
        - Lower confidence bound array
        - Upper confidence bound array
    """
    match method:
        case "bca":
            # q = bca_2(original, data, alpha, backend='jax')[0]
            q = bca_2(original, data, alpha, **kwargs)[0]
            return q[:, 0], q[:, 1]
        case _:
            lower, upper = resolve_ci_method(method)(data, original, alpha, **kwargs)
    return lower, upper

@jax.jit
def standard_ci(
    data: np.ndarray, original: np.ndarray = None, alpha=0.05
) -> tuple[np.ndarray, np.ndarray]:
    #lower = np.empty_like(original)
    #upper = np.empty_like(original)
    #for i in range(len(original)):
    #    lower[i] = np.percentile(data[i], 100*alpha/2)
    #    upper[i] = np.percentile(data[i], 100*(1-alpha/2))
    lower = jnp.percentile(data, 100 * alpha / 2, axis=0)
    upper = jnp.percentile(data, 100 * (1 - alpha / 2), axis=0)
    return lower, upper


def poisson_ci(lambdas, original, alpha=0.05):
    lower_bounds = poisson.ppf(alpha / 2, lambdas)
    upper_bounds = poisson.ppf(1 - alpha / 2, lambdas)
    return lower_bounds, upper_bounds

@jax.jit
def supremum_ci(data: np.ndarray, original: np.ndarray, alpha=0.05, clip: bool = True):
    f_hat = original
    delta_b = data - f_hat
    supremum = jnp.max(jnp.abs(delta_b), axis=1)
    c_alpha = jnp.quantile(supremum, 1 - alpha)
    bound = c_alpha 

    lower = f_hat - bound if not clip else jnp.clip(f_hat - bound, 0, None)
    return lower, f_hat + bound

@jax.jit
def studentized_supremum_ci(data: np.ndarray, original: np.ndarray, alpha=0.05):
    f_hat = original
    delta_b = data - f_hat
    std = jnp.std(data, axis=0)
    supremum = jnp.max(jnp.abs(delta_b / std), axis=0)
    c_alpha = jnp.quantile(supremum, 1 - alpha)
    bound = c_alpha * std
    return f_hat - bound, f_hat + bound

@jax.jit
def bonferroni_percentile_ci(data: np.ndarray, original: np.ndarray, alpha=0.05):
    N = data.shape[1]
    return standard_ci(data, alpha=alpha / N)


def bonferroni_bca_ci(data: np.ndarray, original: np.ndarray, alpha=0.05):
    N = data.shape[1]
    q = bca_2(original, data, alpha / N)[0]
    return q[:, 0], q[:, 1]


def hotelling_T2_ci(data: np.ndarray, original: np.ndarray, alpha=0.05):
    # Suppose g_boot has shape (M, N), each row is a bootstrap replicate of g.
    M, N = data.shape

    # 1) Compute mean (the "center") of the bootstrap replicates
    mean = data.mean(axis=0)  # shape (N,)

    # 2) Compute the sample covariance
    # dev is shape (M, N), each row: g*_b - average
    dev = data - mean
    Sigma = (dev.T @ dev) / (M - 1)  # shape (N, N)

    # 3) Invert or pseudo-invert the covariance
    # If Sigma is ill-conditioned, np.linalg.inv might fail; so use np.linalg.pinv
    Sigma_inv = np.linalg.pinv(Sigma, rcond=1e-12)

    # 4) For each bootstrap replicate, compute T^2 distance from g_mean
    # T^2_b = dev[b,:]^T Sigma_inv dev[b,:]
    # We'll store them in T_sq array of length M
    T_sq = np.einsum("ij,jk,ik->i", dev, Sigma_inv, dev)  # shape (M,)

    # T_sq is the Mahalanobis distance from g_mean for each replicate
    # Now we can look at the empirical distribution of T_sq
    T_sq_sorted = np.sort(T_sq)

    cutoff_alpha = T_sq_sorted[int(np.floor((1 - alpha) * M))]

    T_sq_cut = np.quantile(T_sq, 1.0 - alpha)
    lower, upper = elliptical_subset_bounds(data, mean, Sigma_inv, T_sq_cut)
    return lower, upper


def elliptical_subset_bounds(boot, mean, sigma_inv, T_sq_cut):
    """
    Returns min/max of each dimension among all bootstrap points whose T^2 <= T_sq_cut.
    This yields a bounding box for the elliptical region, not necessarily a minimal bounding box.
    """
    dev = boot - mean
    # compute T^2 for each replicate
    T_sq = np.einsum("ij,jk,ik->i", dev, sigma_inv, dev)
    # keep those within the cutoff
    mask = T_sq <= T_sq_cut
    inside_points = boot[mask, :]  # shape (K, N), K ~ 0.95*M

    # bounding box for the ellipse
    lower_bounds = inside_points.min(axis=0)
    upper_bounds = inside_points.max(axis=0)

    return lower_bounds, upper_bounds


def bca_2(original_estimate: np.ndarray, bootstrap_samples: np.ndarray, alpha=0.05):
    """
    Compute the Bias-Corrected and Accelerated (BCa) confidence intervals for each variable.

    :param bootstrap_samples: NxM numpy array of N bootstrap samples of M variables.
    :param original_estimate: M-dimensional vector of original estimates.
    :param alpha: Significance level for confidence intervals.
    :return: Mx2 numpy array of BCa confidence intervals for each variable.
    """
    N, M = bootstrap_samples.shape
    assert original_estimate.shape == (M,)
    conf_intervals = np.zeros((M, 2))

    bias = np.zeros(M)
    bias_z0 = np.zeros(M)
    accelerations = np.zeros(M)

    for i in range(M):
        try:
            ci, p, z0, a = bca_var(
                original_estimate[i], bootstrap_samples[:, i], alpha=alpha
            )
        except Exception as e:
            print(f"Index {i} failed")
            print(original_estimate[i])
            print(bootstrap_samples[:, i])
            print(original_estimate[i - 1])
            print(bootstrap_samples[:, i - 1])
            raise e

        # BCa confidence intervals
        conf_intervals[i, 0] = ci[0]
        conf_intervals[i, 1] = ci[1]

        bias[i] = p
        bias_z0[i] = z0
        accelerations[i] = a

    return conf_intervals, bias, bias_z0, accelerations


def bca_var(
    theta_hat: float, theta_star: np.ndarray, alpha: float = 0.05
) -> tuple[np.ndarray, float, float, float]:
    """BCa for a single variable given bootstrap samples
    :param theta_hat: Original estimate for this variable
    :param theta_star: Bootstrap estimates for this variable
    :param alpha: Significance level for confidence intervals.
    :return: BCa confidence intervals for this variable, bias, bias_z0, acceleration
    """
    # theta_hat = np.mean(theta_star)

    # Bias correction z0
    p = np.mean(theta_star < theta_hat)
    p = np.clip(p, 1e-5, 1 - 1e-5)
    z0 = norm.ppf(p)

    # Acceleration by jackknife
    # assuming mean as the statistic
    # Can't allocate as the array is (N-1, N-1)

    theta_hat_jacks = np.zeros_like(theta_star)
    for i in range(len(theta_star)):
        theta_star_jack = np.delete(theta_star, i)
        theta_hat_jack = np.mean(theta_star_jack)
        theta_hat_jacks[i] = theta_hat_jack
    a = np.sum((np.mean(theta_hat_jacks) - theta_hat_jacks) ** 3) / (
        6 * np.sum((np.mean(theta_hat_jacks) - theta_hat_jacks) ** 2) ** 1.5
    )

    # Adjusted percentiles
    z_alpha = norm.ppf(alpha / 2)
    z_1_alpha = norm.ppf(1 - alpha / 2)
    adjusted_lower = z0 + (z0 + z_alpha) / (1 - a * (z0 + z_alpha))
    adjusted_upper = z0 + (z0 + z_1_alpha) / (1 - a * (z0 + z_1_alpha))
    lower_percentile = 100 * norm.cdf(adjusted_lower)
    upper_percentile = 100 * norm.cdf(adjusted_upper)
    # lower_percentile = 100 * norm.cdf(2 * z0 + z_alpha)
    # upper_percentile = 100 * norm.cdf(2 * z0 + z_1_alpha)

    # BCa confidence intervals

    try:
        conf_intervals = np.percentile(theta_star, [lower_percentile, upper_percentile])
    except Exception as e:
        print(theta_hat)
        print(theta_star)
        print(z_alpha)
        print(adjusted_lower, adjusted_upper)
        print(lower_percentile, upper_percentile)
        raise e

    return conf_intervals, p, z0, a
