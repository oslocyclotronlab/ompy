from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from typing_extensions import override
from functools import partial
from jax_tqdm import scan_tqdm

from ... import JAX_WORKING
from ...stubs import array1D

if JAX_WORKING:
    import jax
    import jax.numpy as jnp
else:
    jax = lambda x: x
    jax.jit = lambda x: x


def unfold_vector(
    R: jnp.ndarray, raw: jnp.ndarray, initial: jnp.ndarray, iterations: int, lr: float
):
    """JAX version of unfold_vector for 1D unfolding"""
    
    mask = raw > 0
    @scan_tqdm(iterations)
    def body_fn(state, i):
        u, f, cost, kl_cost, fluctuations = state
        
        # Update u
        u = u + lr * (raw - f)
        
        # Update f
        f = u @ R
        
        # Calculate costs
        cost_i = chi2_safe_1d(raw, f, mask)
        fluctuations_i = fluctuation_cost(u, 20.0, mask)
        kl_cost_i = kl(f, raw).sum()
        
        # Update arrays
        cost = cost.at[i].set(cost_i)
        kl_cost = kl_cost.at[i].set(kl_cost_i)
        fluctuations = fluctuations.at[i].set(fluctuations_i)
        
        return (u, f, cost, kl_cost, fluctuations), u
    
    # Initialize
    u = initial
    f = u @ R
    cost = jnp.zeros(iterations, dtype=jnp.float32)
    kl_cost = jnp.zeros(iterations, dtype=jnp.float32)
    fluctuations = jnp.zeros(iterations, dtype=jnp.float32)
    
    # Run iterations
    state = (u, f, cost, kl_cost, fluctuations)
    (final_u, final_f, cost, kl_cost, fluctuations), u_all = jax.lax.scan(
        body_fn, state, jnp.arange(iterations)
    )
    
    return u_all, cost, fluctuations, kl_cost


def unfold_vectors(
    R: jnp.ndarray, raw: jnp.ndarray, initial: jnp.ndarray, iterations: int, lr: float
):
    """JAX version of unfold_vectors for batch processing using vmap"""

    def fn(r, i):
        body_fn = jax.jit(unfold_vector, static_argnames=["iterations", "lr"])
        return body_fn(R, r, i, iterations, lr)
    
    # Use vmap to vectorize the unfold_vector function over the batch dimension
    batched_unfold = jax.vmap(
        fn,
        in_axes=(0, 0),  # Vectorize over first argument (raw) and second argument (initial)
        out_axes=(0, 0, 0, 0)  # All outputs are batched
    )
    
    return batched_unfold(raw, initial)


def chi2(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """JAX version of chi2 function"""
    return jnp.sum((a - b) ** 2 / (a + 1e-10))


def chi2_safe_1d(a: jnp.ndarray, b: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    """JAX version of chi2_safe_1d function"""
    diff = (a - b) ** 2 / (a + 1e-10)
    return jnp.sum(diff * mask)


def chi2_safe(a: jnp.ndarray, b: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    """JAX version of chi2_safe function for 2D arrays"""
    diff = (a - b) ** 2 / (a + 1e-10)
    return jnp.sum(diff * mask)


def kl(nu: jnp.ndarray, n: jnp.ndarray) -> jnp.ndarray:
    """JAX version of KL divergence function"""
    return nu - n + n * jnp.log(n / (nu + 1e-10) + 1e-10)


def fluctuation_cost(x: jnp.ndarray, sigma: float, mask: jnp.ndarray) -> jnp.ndarray:
    """JAX version of fluctuation_cost function"""
    smoothed = gaussian_smooth(x, sigma, 4)
    diff = jnp.abs((smoothed - x) / (smoothed + 1e-10))
    return jnp.sum(diff * mask)



@partial(jax.jit, static_argnames=['axis', 'sigma', 'radius'])
def gaussian_smooth(
    x: jnp.ndarray,
    sigma: float,
    radius: int,
    axis: int = -1
) -> jnp.ndarray:
    """
    Smooth an array with a Gaussian kernel via direct convolution.

    Args:
        x: Input array (e.g., spectrum) to smooth.
        sigma: Standard deviation of the Gaussian kernel in sample units.
        radius: Kernel radius (number of samples on either side of center).
        axis: Axis along which to apply the smoothing.

    Returns:
        Smoothed array of the same shape as x.
    """
    # Create window indices (static radius ensures fixed kernel length)
    window = jnp.arange(-radius, radius + 1, dtype=x.dtype)

    # Compute Gaussian kernel and normalize
    kernel = jnp.exp(-0.5 * (window / sigma) ** 2)
    kernel = kernel / jnp.sum(kernel)

    # Move target axis to last for convolution
    x_t = jnp.moveaxis(x, axis, -1)

    # Apply 1D convolution along last axis (mode='same' preserves length)
    smoothed = jnp.apply_along_axis(
        lambda row: jnp.convolve(row, kernel, mode='same'),
        axis=-1,
        arr=x_t
    )

    # Restore original axis order
    return jnp.moveaxis(smoothed, -1, axis)


# Convenience function to convert numpy arrays to JAX and back
def unfold_vector_wrapper(
    R: array1D, raw: array1D, initial: array1D, iterations: int, lr: float
):
    """Wrapper function that converts numpy arrays to JAX, runs the JAX function, and converts back"""
    if not JAX_WORKING:
        raise RuntimeError("JAX is not available or not working")
    
    # Convert to JAX arrays
    R = jnp.array(R)
    raw = jnp.array(raw)
    initial = jnp.array(initial)
    
    # Run JAX function
    u_all, cost, fluctuations, kl_cost = unfold_vector(
        R, raw, initial, iterations, lr
    )
    
    # Convert back to numpy
    return (
        np.array(u_all),
        np.array(cost),
        np.array(fluctuations),
        np.array(kl_cost)
    )


def unfold_vectors_wrapper(
    R: array1D, raw_list: list[array1D], initial_list: list[array1D], iterations: int, lr: float
):
    """Wrapper function that handles lists of raw and initial arrays using vmap for true vectorization"""
    if not JAX_WORKING:
        raise RuntimeError("JAX is not available or not working")
    
    # Convert lists to JAX arrays
    R = jnp.array(R)
    raw_batch = jnp.array([np.array(r) for r in raw_list])
    initial_batch = jnp.array([np.array(i) for i in initial_list])
    
    # Run JAX batch function using vmap
    u_all, cost, fluctuations, kl_cost = unfold_vectors(
        R, raw_batch, initial_batch, iterations, lr
    )
    
    # Convert back to numpy and split into individual results
    u_all_np = np.array(u_all)
    cost_np = np.array(cost)
    fluctuations_np = np.array(fluctuations)
    kl_cost_np = np.array(kl_cost)
    
    # Split results into individual arrays
    results = []
    for i in range(len(raw_list)):
        result = (
            u_all_np[i],        # u_all for this array
            cost_np[i],         # cost for this array
            fluctuations_np[i], # fluctuations for this array
            kl_cost_np[i]       # kl_cost for this array
        )
        results.append(result)
    
    return results
