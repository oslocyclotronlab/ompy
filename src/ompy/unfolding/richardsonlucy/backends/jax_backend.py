from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ....accel import jax_available, jax_working, optax_available, configure_jax_dtype

_JAX_OK = jax_available()
_OPTAX_OK = optax_available()
_JAX_WORKING = jax_working(require_accel=False) if _JAX_OK else False
AVAILABLE = bool(_JAX_OK and _OPTAX_OK and _JAX_WORKING)

# Default dtype for JAX operations (32-bit by default, can be configured)
_JAX_DTYPE = None
_NP_DTYPE = np.float32  # fallback if JAX not available

if AVAILABLE:
    try:  # configure precision using centralized configuration (defaults to 32-bit)
        configure_jax_dtype()  # Use default precision (32-bit)
    except Exception:  # pragma: no cover - only triggers if configuration fails
        pass

    import jax
    import jax.numpy as jnp
    import optax
    
    # Determine the actual dtype being used by JAX
    _JAX_DTYPE = jnp.array(0.0).dtype  # Get default float dtype from JAX config
    _NP_DTYPE = np.float32 if _JAX_DTYPE == jnp.float32 else np.float64

try:
    from jax_tqdm import scan_tqdm
except Exception:  # pragma: no cover - optional dependency
    scan_tqdm = None


def _ensure_available() -> None:
    if not AVAILABLE:  # pragma: no cover - guarded by availability checks
        missing = []
        if not _JAX_OK:
            missing.append("jax")
        if _JAX_OK and not _JAX_WORKING:
            missing.append("jax (no devices)")
        if not _OPTAX_OK:
            missing.append("optax")
        detail = ", ".join(missing) if missing else "jax/optax stack"
        raise RuntimeError(
            f"JAX/Optax backend requested but unavailable: {detail}."
        )


if AVAILABLE:

    def _poisson_loglikelihood(observed: jnp.ndarray, expected: jnp.ndarray, epsilon: float):
        r"""
        Compute the Poisson log-likelihood (up to an additive constant).

        .. math::
           \log \mathcal{L}(\lambda \mid k)
           = \sum_i \Big[k_i \log(\lambda_i + \varepsilon) - \lambda_i\Big]

        Parameters
        ----------
        observed:
            Measured counts :math:`k`.
        expected:
            Model prediction :math:`\lambda`.
        epsilon:
            Floor that keeps the logarithm and division numerically stable.
        """
        expected_safe = jnp.clip(expected, epsilon, None)
        return jnp.sum(observed * jnp.log(expected_safe) - expected_safe)

    def _build_vector_scan_fn(
        response: jnp.ndarray,
        denominator: jnp.ndarray,
        data: jnp.ndarray,
        mask: jnp.ndarray,
        background: jnp.ndarray | None,
        epsilon: float,
    ):
        r"""
        Build the JAX scan for the vector-valued Richardson-Lucy iteration.

        The update matches the classical one-dimensional Richardson-Lucy formula

        .. math::
           x^{(n+1)} = x^{(n)} \odot
           \frac{R^\top \left(\frac{y}{R x^{(n)} + b}\right)}
                {R^\top \mathbf{1}}

        where :math:`R` is ``response``, :math:`y` is ``data``, :math:`b` is the
        optional ``background``, and the product is taken element-wise. The
        pre-computed ``denominator`` corresponds to :math:`R^\top \mathbf{1}`.

        Returns a tuple ``(step_fn, init_fn)`` compatible with :func:`jax.lax.scan`.
        """
        optimizer = optax.identity()

        def step(carry, _):
            x, prev_x, opt_state = carry
            forward = jnp.matmul(x, response)
            if background is not None:
                forward = forward + background
            forward = jnp.clip(forward, epsilon)
            # Ratio between measured and predicted counts.
            ratio = data / forward
            correction = jnp.matmul(ratio, response.T)
            update_factor = correction / denominator
            # Multiplicative RL update written in additive form to re-use optax.
            updates = x * (update_factor - 1.0)
            updates, opt_state = optimizer.update(updates, opt_state, params=x)
            x_next = optax.apply_updates(x, updates)
            x_next = jnp.clip(x_next, epsilon, None)
            x_next = jnp.where(mask, x_next, 0.0)
            forward_next = jnp.matmul(x_next, response)
            if background is not None:
                forward_next = forward_next + background
            forward_next = jnp.clip(forward_next, epsilon, None)
            return (x_next, x, opt_state), (x_next, forward_next)

        def init_state(initial):
            opt_state = optimizer.init(initial)
            return (initial, initial, opt_state)

        return step, init_state

    def _build_matrix_scan_fn(
        response: jnp.ndarray,
        data: jnp.ndarray,
        mask: jnp.ndarray,
        epsilon: float,
        G_ex: jnp.ndarray | None,
        background: jnp.ndarray | None,
    ):
        r"""
        Build the scan primitive for matrix-valued Richardson-Lucy unfolding.

        The update generalises the vector case to two dimensions. For an iterate
        :math:`X` and detector response :math:`R`, optional efficiency matrix
        :math:`G_{\mathrm{ex}}`, and background :math:`B`, we apply

        .. math::
           X^{(n+1)} = X^{(n)} \odot
           \frac{\mathcal{B}\left(\frac{Y}{\mathcal{F}(X^{(n)}) + B}\right)}
                {\mathcal{B}(\mathbf{1})}

        with forward projector :math:`\mathcal{F}(X) = G_{\mathrm{ex}}\, X R`
        (dropping :math:`G_{\mathrm{ex}}` if absent) and back projector
        :math:`\mathcal{B}(Z) = G_{\mathrm{ex}}^\top Z R^\top`. The denominator
        again corresponds to the back projection of ones.

        Returns
        -------
        step:
            Callable compatible with :func:`jax.lax.scan` that carries the iterate,
            previous iterate, and optax optimiser state.
        init_state:
            Callable that initialises the scan state from ``initial``.
        denominator:
            The stabilising back projection :math:`\mathcal{B}(\mathbf{1})`.
        """
        optimizer = optax.identity()
        response_T = response.T

        if G_ex is not None:
            G_ex_T = G_ex.T

            def forward_project(arr: jnp.ndarray) -> jnp.ndarray:
                return jnp.matmul(G_ex, jnp.matmul(arr, response))

            def back_project(arr: jnp.ndarray) -> jnp.ndarray:
                return jnp.matmul(G_ex_T, jnp.matmul(arr, response_T))

        else:

            def forward_project(arr: jnp.ndarray) -> jnp.ndarray:
                return jnp.matmul(arr, response)

            def back_project(arr: jnp.ndarray) -> jnp.ndarray:
                return jnp.matmul(arr, response_T)

        # Equivalent of the standard RL denominator: response^T @ 1.
        denominator = back_project(jnp.ones_like(data))
        denominator = jnp.clip(denominator, epsilon, None)

        def step(carry, _):
            x, prev_x, opt_state = carry
            # Forward project to predicted detector counts.
            forward = forward_project(x)
            if background is not None:
                forward = forward + background
            forward = jnp.clip(forward, epsilon)
            # Standard Richardson-Lucy ratio between measured and predicted counts.
            ratio = data / forward
            correction = back_project(ratio)
            update_factor = correction / denominator
            updates = x * (update_factor - 1.0)
            updates, opt_state = optimizer.update(updates, opt_state, params=x)
            x_next = optax.apply_updates(x, updates)
            x_next = jnp.clip(x_next, epsilon, None)
            x_next = jnp.where(mask, x_next, 0.0)
            forward_next = forward_project(x_next)
            if background is not None:
                forward_next = forward_next + background
            forward_next = jnp.clip(forward_next, epsilon, None)
            # Equivalent to accumulating log P(data | forward_next) for the loss history.
            loglike_next = _poisson_loglikelihood(data, forward_next, epsilon)
            return (x_next, x, opt_state), loglike_next

        def init_state(initial):
            opt_state = optimizer.init(initial)
            return (initial, initial, opt_state)

        return step, init_state, denominator


def run_vector(
    response: NDArray[np.float64],
    data: NDArray[np.float64],
    initial: NDArray[np.float64],
    mask: NDArray[np.bool_],
    background: NDArray[np.float64] | None,
    *,
    iterations: int,
    tolerance: float | None,
    epsilon: float,
    store_history: bool,
    show_progress: bool = False,
    leave: bool = True,
    description: str | None = None,
):
    r"""
    Run the Richardson-Lucy unfolding on vector inputs using JAX.

    The routine performs ``iterations`` multiplicative updates of the form

    .. math::
       x^{(n+1)} = x^{(n)} \odot
       \frac{R^\top \left(\frac{y}{R x^{(n)} + b}\right)}
            {R^\top \mathbf{1}}

    with optional background vector :math:`b`. When ``store_history`` is true,
    the entire sequence :math:`\{x^{(n)}\}` is materialised on the host.

    Parameters
    ----------
    response:
        Response matrix :math:`R`.
    data:
        Observed counts :math:`y`.
    initial:
        Starting iterate :math:`x^{(0)}`.
    mask:
        Boolean mask enforcing fixed zero bins in the solution.
    background:
        Optional additive background :math:`b`.
    iterations:
        Maximum number of Richardson-Lucy steps to execute.
    tolerance:
        Relative convergence threshold on :math:`\lVert x^{(n)} - x^{(n-1)}\rVert`.
    epsilon:
        Numerical floor inserted into divisions and logarithms.
    store_history:
        Whether to retain the full state trace on the host.
    show_progress, leave, description:
        Parameters passed to the optional :mod:`jax_tqdm` progress bar.

    Returns
    -------
    final:
        Final estimate :math:`x^{(\mathrm{end})}`.
    iterations_performed:
        Number of iterations executed (``<= iterations``).
    converged:
        Flag indicating whether the relative tolerance was met.
    history:
        Optional numpy copy of the iterate history.
    loglike:
        Per-iteration Poisson log-likelihood values.
    """
    _ensure_available()

    response_j = jnp.asarray(response, dtype=_JAX_DTYPE)
    data_j = jnp.asarray(data, dtype=_JAX_DTYPE)
    initial_j = jnp.asarray(initial, dtype=_JAX_DTYPE)
    mask_j = jnp.asarray(mask, dtype=bool)
    background_j = None if background is None else jnp.asarray(background, dtype=_JAX_DTYPE)

    denominator = jnp.clip(jnp.sum(response_j, axis=1), epsilon, None)
    step, init_state = _build_vector_scan_fn(
        response_j, denominator, data_j, mask_j, background_j, epsilon
    )

    if scan_tqdm is not None and show_progress and iterations > 0:
        decorated_step = scan_tqdm(iterations, desc=description, leave=leave)(step)
    else:
        decorated_step = step

    state = init_state(initial_j)

    if iterations > 0:
        (final_x, prev_x, _), (history, forwards) = jax.lax.scan(
            decorated_step, state, None, length=iterations
        )
    else:
        final_x, prev_x, _ = state
        history = jnp.empty((0, initial_j.shape[0]), dtype=initial_j.dtype)
        forwards = jnp.empty((0, data_j.shape[0]), dtype=data_j.dtype)

    iterations_performed = iterations
    if iterations == 0:
        forward_initial = jnp.matmul(initial_j, response_j)
        if background_j is not None:
            forward_initial = forward_initial + background_j
        forward_initial = jnp.clip(forward_initial, epsilon, None)
        loglike = jnp.array([_poisson_loglikelihood(data_j, forward_initial, epsilon)])
    else:
        loglike = jax.vmap(
            lambda f: _poisson_loglikelihood(data_j, f, epsilon)
        )(forwards)

    if tolerance is not None and iterations_performed > 0:
        diff = jnp.linalg.norm(final_x - prev_x)
        norm_prev = jnp.maximum(jnp.linalg.norm(prev_x), epsilon)
        converged = bool(diff / norm_prev < tolerance)
    else:
        converged = False

    history_np: NDArray[np.float64] | None
    if store_history and iterations_performed > 0:
        history_np = np.asarray(history, dtype=_NP_DTYPE)
    else:
        history_np = None

    return (
        np.asarray(final_x, dtype=_NP_DTYPE),
        iterations_performed,
        converged,
        history_np,
        np.asarray(loglike, dtype=_NP_DTYPE),
    )


def run_matrix(
    response: NDArray[np.float64],
    data: NDArray[np.float64],
    initial: NDArray[np.float64],
    mask: NDArray[np.bool_],
    background: NDArray[np.float64] | None,
    *,
    iterations: int,
    tolerance: float | None,
    epsilon: float,
    G_ex: NDArray[np.float64] | None,
    store_history: bool = False,
    show_progress: bool = False,
    leave: bool = True,
    description: str | None = None,
):
    r"""
    Run the Richardson-Lucy update for matrix-valued unfolding problems.

    For a two-dimensional spectrum :math:`X`, detector response :math:`R`, and
    optional efficiency matrix :math:`G_{\mathrm{ex}}`, the multiplicative update
    follows

    .. math::
       X^{(n+1)} = X^{(n)} \odot
       \frac{\mathcal{B}\left(\frac{Y}{\mathcal{F}(X^{(n)}) + B}\right)}
            {\mathcal{B}(\mathbf{1})},

    with forward projector :math:`\mathcal{F}(X) = G_{\mathrm{ex}}\, X R` (falling
    back to :math:`X R` if ``G_ex`` is ``None``) and back projector
    :math:`\mathcal{B}(Z) = G_{\mathrm{ex}}^\top Z R^\top`. Only the Poisson
    log-likelihood trace is materialised; the full iterate history is intentionally
    omitted to avoid host-memory blow-up.

    Parameters mirror :func:`run_vector`, with ``G_ex`` enabling acceptance losses.

    Returns
    -------
    final:
        Final unfolded matrix :math:`X^{(\mathrm{end})}`.
    iterations_performed:
        Number of iterations executed (``<= iterations``).
    converged:
        Flag indicating whether the relative tolerance was met.
    history:
        Always ``None`` for this backend (matrix histories are intentionally skipped).
    loglike:
        Array of Poisson log-likelihood values, one per completed iteration.
    """
    _ensure_available()

    response_j = jnp.asarray(response, dtype=_JAX_DTYPE)
    data_j = jnp.asarray(data, dtype=_JAX_DTYPE)
    initial_j = jnp.asarray(initial, dtype=_JAX_DTYPE)
    mask_j = jnp.asarray(mask, dtype=bool)
    G_ex_j = None if G_ex is None else jnp.asarray(G_ex, dtype=_JAX_DTYPE)
    background_j = None if background is None else jnp.asarray(background, dtype=_JAX_DTYPE)

    step, init_state, _ = _build_matrix_scan_fn(
        response_j, data_j, mask_j, epsilon, G_ex_j, background_j
    )

    if scan_tqdm is not None and show_progress:
        decorated_step = scan_tqdm(iterations, desc=description, leave=leave)(step)
    else:
        decorated_step = step

    state = init_state(initial_j)

    xs = jnp.arange(iterations)
    (final_x, prev_x, _), loglikes = jax.lax.scan(
        decorated_step, state, xs, length=iterations
    )

    if tolerance is not None:
        diff = jnp.linalg.norm((final_x - prev_x).reshape(-1))
        norm_prev = jnp.maximum(jnp.linalg.norm(prev_x.reshape(-1)), epsilon)
        converged = bool(diff / norm_prev < tolerance)
    else:
        converged = False

    return (
        np.asarray(final_x, dtype=_NP_DTYPE),
        iterations,
        converged,
        None,
        np.asarray(loglikes, dtype=_NP_DTYPE),
    )


__all__ = ["AVAILABLE", "run_vector", "run_matrix"]
