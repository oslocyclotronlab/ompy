from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ....accel import numba_available

AVAILABLE = numba_available()

if AVAILABLE:
    from ....numbalib import njit

    @njit
    def _poisson_loglikelihood_vector(
        observed: NDArray[np.float64],
        expected: NDArray[np.float64],
        epsilon: float,
    ) -> float:
        total = 0.0
        for i in range(observed.shape[0]):
            exp_val = expected[i]
            if exp_val < epsilon:
                exp_val = epsilon
            total += observed[i] * np.log(exp_val) - exp_val
        return total

    @njit
    def _poisson_loglikelihood_matrix(
        observed: NDArray[np.float64],
        expected: NDArray[np.float64],
        epsilon: float,
    ) -> float:
        total = 0.0
        rows, cols = observed.shape
        for i in range(rows):
            for j in range(cols):
                exp_val = expected[i, j]
                if exp_val < epsilon:
                    exp_val = epsilon
                total += observed[i, j] * np.log(exp_val) - exp_val
        return total

    @njit
    def _norm(arr: NDArray[np.float64]) -> float:
        return np.sqrt(np.sum(arr * arr))

    @njit
    def _vector_iteration(
        response: NDArray[np.float64],
        data: NDArray[np.float64],
        initial: NDArray[np.float64],
        mask: NDArray[np.float64],
        background: NDArray[np.float64],
        use_background: bool,
        iterations: int,
        tolerance: float,
        epsilon: float,
        store_history: bool,
    ):
        x = initial.copy()
        resp_T = response.T

        denominator = response.sum(axis=1)
        for i in range(denominator.shape[0]):
            if denominator[i] < epsilon:
                denominator[i] = epsilon

        hist_len = iterations if store_history else 0
        history = np.empty((hist_len, x.shape[0]), dtype=np.float64)
        loglike = np.empty(max(iterations, 1), dtype=np.float64)

        converged = False
        prev = x.copy()
        iterations_done = 0

        for it in range(iterations):
            forward = x @ response
            if use_background:
                forward = forward + background
            for j in range(forward.shape[0]):
                if forward[j] < epsilon:
                    forward[j] = epsilon
            ratio = data / forward
            correction = ratio @ resp_T

            x = x * (correction / denominator)
            for j in range(x.shape[0]):
                if x[j] < epsilon:
                    x[j] = epsilon
                x[j] = x[j] * mask[j]

            iterations_done = it + 1
            if store_history:
                history[it, :] = x

            forward_updated = x @ response
            if use_background:
                forward_updated = forward_updated + background
            loglike[it] = _poisson_loglikelihood_vector(data, forward_updated, epsilon)

            if tolerance >= 0.0:
                diff = _norm(x - prev)
                norm_prev = _norm(prev)
                if norm_prev < epsilon:
                    norm_prev = epsilon
                if diff / norm_prev < tolerance:
                    converged = True
                    break
                prev = x.copy()

        if iterations_done == 0:
            forward_initial = initial @ response
            if use_background:
                forward_initial = forward_initial + background
            loglike[0] = _poisson_loglikelihood_vector(data, forward_initial, epsilon)

        return x, iterations_done, converged, history, loglike

    @njit(parallel=True)
    def _matrix_iteration(
        response: NDArray[np.float64],
        data: NDArray[np.float64],
        initial: NDArray[np.float64],
        mask: NDArray[np.float64],
        background: NDArray[np.float64],
        use_background: bool,
        iterations: int,
        tolerance: float,
        epsilon: float,
        store_history: bool,
        use_gex: bool,
        G_ex: NDArray[np.float64],
        G_ex_T: NDArray[np.float64],
    ):
        x = initial.copy()
        resp_T = response.T

        ones = np.ones_like(data)
        if use_gex:
            tmp2 = ones @ resp_T
            denominator = G_ex_T @ tmp2
        else:
            denominator = ones @ resp_T

        for i in range(denominator.shape[0]):
            for j in range(denominator.shape[1]):
                if denominator[i, j] < epsilon:
                    denominator[i, j] = epsilon

        hist_len = iterations if store_history else 0
        history = np.empty((hist_len, x.shape[0], x.shape[1]), dtype=np.float64)
        loglike = np.empty(max(iterations, 1), dtype=np.float64)

        converged = False
        prev = x.copy()
        iterations_done = 0

        for it in range(iterations):
            if use_gex:
                tmp = x @ response
                forward = G_ex @ tmp
            else:
                forward = x @ response
            if use_background:
                forward = forward + background
            rows, cols = forward.shape
            for i in range(rows):
                for j in range(cols):
                    if forward[i, j] < epsilon:
                        forward[i, j] = epsilon
            ratio = data / forward

            if use_gex:
                back = ratio @ resp_T
                back = G_ex_T @ back
            else:
                back = ratio @ resp_T

            x = x * (back / denominator)
            rows_x, cols_x = x.shape
            for i in range(rows_x):
                for j in range(cols_x):
                    if x[i, j] < epsilon:
                        x[i, j] = epsilon
                    x[i, j] = x[i, j] * mask[i, j]

            iterations_done = it + 1
            if store_history:
                history[it, :, :] = x

            if use_gex:
                tmp = x @ response
                forward_updated = G_ex @ tmp
            else:
                forward_updated = x @ response
            if use_background:
                forward_updated = forward_updated + background
            loglike[it] = _poisson_loglikelihood_matrix(data, forward_updated, epsilon)

            if tolerance >= 0.0:
                diff = _norm((x - prev).ravel())
                norm_prev = _norm(prev.ravel())
                if norm_prev < epsilon:
                    norm_prev = epsilon
                if diff / norm_prev < tolerance:
                    converged = True
                    break
                prev = x.copy()

        if iterations_done == 0:
            if use_gex:
                tmp_initial = initial @ response
                forward_initial = G_ex @ tmp_initial
            else:
                forward_initial = initial @ response
            if use_background:
                forward_initial = forward_initial + background
            loglike[0] = _poisson_loglikelihood_matrix(
                data, forward_initial, epsilon
            )

        return x, iterations_done, converged, history, loglike


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
    if not AVAILABLE:
        raise RuntimeError("Numba backend requested but numba is not available.")

    tol_value = tolerance if tolerance is not None else -1.0
    mask_float = mask.astype(np.float64)
    if background is not None:
        background_arr = background.astype(np.float64)
        use_background = True
    else:
        background_arr = np.zeros_like(data, dtype=np.float64)
        use_background = False
    result = _vector_iteration(
        response,
        data,
        initial,
        mask_float,
        background_arr,
        use_background,
        iterations,
        tol_value,
        epsilon,
        store_history,
    )
    x, iterations_done, converged, history, loglike = result
    history_out: NDArray[np.float64] | None
    if store_history:
        history_out = history[:iterations_done]
    else:
        history_out = None
    return (
        x,
        iterations_done,
        converged,
        history_out,
        loglike[: max(iterations_done, 1)],
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
    store_history: bool,
    G_ex: NDArray[np.float64] | None,
    show_progress: bool = False,
    leave: bool = True,
    description: str | None = None,
):
    if not AVAILABLE:
        raise RuntimeError("Numba backend requested but numba is not available.")

    tol_value = tolerance if tolerance is not None else -1.0
    mask_float = mask.astype(np.float64)
    if background is not None:
        background_arr = background.astype(np.float64)
    else:
        background_arr = np.zeros_like(data, dtype=np.float64)
    use_gex = G_ex is not None
    if use_gex:
        G_ex_values = G_ex
        G_ex_T = G_ex.T
    else:
        G_ex_values = np.empty((0, 0), dtype=np.float64)
        G_ex_T = np.empty((0, 0), dtype=np.float64)

    result = _matrix_iteration(
        response,
        data,
        initial,
        mask_float,
        background_arr,
        background is not None,
        iterations,
        tol_value,
        epsilon,
        store_history,
        use_gex,
        G_ex_values,
        G_ex_T,
    )
    x, iterations_done, converged, history, loglike = result
    history_out: NDArray[np.float64] | None
    if store_history:
        history_out = history[:iterations_done]
    else:
        history_out = None
    return (
        x,
        iterations_done,
        converged,
        history_out,
        loglike[: max(iterations_done, 1)],
    )


__all__ = ["AVAILABLE", "run_vector", "run_matrix"]
