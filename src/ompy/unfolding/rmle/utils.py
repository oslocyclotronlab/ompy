from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import jax
import jax.numpy as jnp
import numpy as np
from ... import Matrix

if TYPE_CHECKING:
    from .contaminant1d import Contaminant1D


try:
    from jax_tqdm import loop_tqdm
except ImportError:

    def loop_tqdm(iterable=None, **kwargs):
        def decorator(func):
            return func

        return decorator

        
from dataclasses import dataclass, fields
from jax import tree_util

def pytree_dataclass(_cls=None, *, frozen=True):
    """
    Combines @dataclass with JAX PyTree registration.
    Usage:
        @pytree_dataclass
        class My:
            a: float
            b: jnp.ndarray
    """
    def wrap(cls):
        # 1) apply dataclass
        cls = dataclass(frozen=frozen)(cls)

        # 2) implement the PyTree flatten/unflatten methods
        def tree_flatten(self):
            # all fields become children; no static aux
            vals = tuple(getattr(self, f.name) for f in fields(self))
            return vals, None

        @classmethod
        def tree_unflatten(cls_, aux, children):
            return cls_(*children)

        cls.tree_flatten = tree_flatten
        cls.tree_unflatten = tree_unflatten

        # 3) register with JAX
        return tree_util.register_pytree_node_class(cls)

    # support both with and without parentheses
    if _cls is None:
        return wrap
    else:
        return wrap(_cls)



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


def relaxed_one_hot[T: jnp.ndarray](logits: T, temperature: float = 0.01) -> T:
    """Compute a continuous relaxation of a one-hot vector using softmax.

    This function takes logits and returns a "soft" one-hot vector by applying
    the softmax function with a temperature parameter. As temperature approaches 0,
    the output approaches a discrete one-hot vector.

    Args:
        logits: Input logits tensor to be converted to probabilities
        temperature: Temperature parameter controlling the sharpness of the distribution.
                    Lower values make the output more discrete. Default is 0.01.

    Returns:
        A tensor of the same shape as logits containing probabilities that sum to 1.
    """
    return jax.nn.softmax(logits / temperature)

    

def var_penalty(param: float, lower: float, upper: float) -> float:
    """Calculate a quadratic penalty for values outside a specified interval.

    This function computes a quadratic penalty that grows as the parameter moves
    outside the specified bounds. Inside the bounds, the penalty is zero.

    Args:
        param: The parameter value to check
        lower: Lower bound of the allowed interval
        upper: Upper bound of the allowed interval

    Returns:
        float: The total penalty, which is the sum of penalties for violating
              the lower and upper bounds. Returns 0.0 if param is within bounds.
    """
    # Quadratic penalty outside the [lower, upper] interval.
    lower_penalty = jnp.where(param < lower, (param - lower) ** 2, 0.0)
    upper_penalty = jnp.where(param > upper, (param - upper) ** 2, 0.0)
    return lower_penalty + upper_penalty

    

def into_array(x: Matrix | np.ndarray | jnp.ndarray) -> jnp.ndarray:
    if hasattr(x, "values"):
        x = x.values
    return jnp.asarray(x)

    
def bounded_param(x, lower, upper):
    """Map an unbounded variable to a bounded interval using sigmoid.

    This function maps a variable u from (-∞, ∞) to the interval [a, b] using
    the sigmoid function. This is useful for constrained optimization where we
    want to optimize an unconstrained variable while ensuring the result lies
    within specified bounds.

    Args:
        u: Input variable to be bounded (can be any real number)
        a: Lower bound of the target interval
        b: Upper bound of the target interval (must be > a)

    Returns:
        The input mapped to the interval [a, b]. As u approaches -∞, the output
        approaches a. As u approaches ∞, the output approaches b.
    """
    return lower + (upper - lower) * jax.nn.sigmoid(x)