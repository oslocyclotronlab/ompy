from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import jax
import jax.numpy as jnp
import numpy as np

if TYPE_CHECKING:
    from .contaminant1d import Contaminant1D


try:
    from jax_tqdm import loop_tqdm
except ImportError:

    def loop_tqdm(iterable=None, **kwargs):
        def decorator(func):
            return func

        return decorator


def sigmoid(x, start, stop, midpoint, sigma):
    """Compute a sigmoid function with customizable start, stop, midpoint and width.

    This function implements a sigmoid that transitions smoothly from 'start' to 'stop'
    value, centered at 'midpoint' with transition width controlled by 'sigma'.

    Args:
        x: Input array or value
        start: Lower asymptotic value
        stop: Upper asymptotic value
        midpoint: x-value of the sigmoid's center point
        sigma: Parameter controlling the width of the transition

    Returns:
        Array or value containing the sigmoid evaluated at input x
    """
    return start + (stop - start) / (1 + jnp.exp(-(x - midpoint) / sigma))


def gaussian(x, A, mu, sigma):
    """Compute a Gaussian function with specified amplitude, mean and standard deviation.

    This function evaluates a Gaussian (normal) distribution at the given x values.
    The function has the form: A * exp(-(x - mu)^2 / (2*sigma^2))

    Args:
        x: Input array or value where to evaluate the Gaussian
        A: Amplitude (height) of the Gaussian peak
        mu: Mean (center) of the Gaussian
        sigma: Standard deviation controlling the width of the Gaussian

    Returns:
        Array or value containing the Gaussian function evaluated at input x
    """
    y = jnp.exp(-((x - mu) ** 2) / (2 * sigma**2))
    return A * y


def richardson_rate(R: np.ndarray) -> float:
    """Calculate the optimal learning rate for Richardson's method.

    This function computes the optimal learning rate for gradient descent using
    Richardson's method, which is based on the singular values of the response matrix.
    The rate is chosen as 2/(s_max + s_min) where s_max and s_min are the largest
    and smallest singular values respectively. Based on Chebyshev iteration.

    Args:
        R: Response matrix as a numpy array

    Returns:
        float: The optimal learning rate
    """
    # get the largest and smallest singular values
    s = np.linalg.svd(R, compute_uv=False)
    s_max = s.max()
    s_min = s.min()
    return 2 / (s_max + s_min)


def closure_unpack(
    contaminants: list[Contaminant1D], raw: jnp.ndarray | int
) -> Callable[[jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]]:
    """Create a closure function to unpack optimization parameters into mu and contaminant vectors.

    This function creates and returns a closure that can unpack a flattened array of optimization
    parameters into the unfolded spectrum (mu) and contaminant component vectors. The closure
    handles reshaping the flattened array based on the number of contaminants and spectrum length.

    Args:
        contaminants: List of Contaminant1D objects representing the contaminant components
        raw: Either the raw spectrum array or an integer specifying the spectrum length

    Returns:
        A jitted function that takes a flattened parameter array and returns a tuple containing:
            - mu: The unfolded spectrum vector
            - contaminants: Array of contaminant component vectors reshaped to (n_contaminants, n_bins)
    """

    if isinstance(raw, int):
        N = raw
    else:
        N = len(raw)
    M = len(contaminants)

    @jax.jit
    def func(x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        # returns a tuple of (mu vector, contaminant vectors)
        mu = x[:N]
        contaminants = x[N:].reshape(M, N)
        return mu, contaminants

    return func
