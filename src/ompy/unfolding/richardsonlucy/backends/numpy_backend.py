from __future__ import annotations

from typing import Tuple

import numpy as np
from numpy.typing import NDArray

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

def _poisson_loglikelihood(
    observed: NDArray[np.float64],
    expected: NDArray[np.float64],
    epsilon: float,
) -> float:
    expected_safe = np.clip(expected, epsilon, None)
    return float(np.sum(observed * np.log(expected_safe) - expected_safe))


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
) -> Tuple[
    NDArray[np.float64],
    int,
    bool,
    NDArray[np.float64] | None,
    NDArray[np.float64],
]:
    """Pure NumPy Richardson-Lucy iteration for 1D spectra."""
    x = np.clip(initial.astype(np.float64, copy=True), epsilon, None)
    response = response.astype(np.float64, copy=False)
    data = np.clip(data.astype(np.float64, copy=False), 0.0, None)
    if background is not None:
        background = background.astype(np.float64, copy=False)

    history = [] if store_history else None
    loglike = np.empty(iterations if iterations > 0 else 1, dtype=np.float64)

    denominator = response.sum(axis=1)
    denominator = np.clip(denominator, epsilon, None)

    if show_progress and iterations > 0:
        try:
            from tqdm.auto import tqdm
        except Exception:  # pragma: no cover - optional dependency
            iterator = range(iterations)
        else:
            iterator = tqdm(
                range(iterations),
                leave=leave,
                desc=description,
                total=iterations,
            )
    else:
        iterator = range(iterations)

    converged = False
    prev = x.copy()
    iterations_performed = 0
    for it in iterator:
        forward = x @ response
        if background is not None:
            forward = forward + background
        forward = np.clip(forward, epsilon, None)
        ratio = data / forward
        correction = ratio @ response.T

        x *= correction / denominator
        x = np.clip(x, epsilon, None)
        x = np.where(mask, x, 0.0)

        iterations_performed = it + 1
        forward_updated = x @ response
        if background is not None:
            forward_updated = forward_updated + background

        if history is not None:
            history.append(x.copy())
        loglike[it] = _poisson_loglikelihood(data, forward_updated, epsilon)

        if tolerance is not None:
            diff = np.linalg.norm(x - prev)
            norm_prev = max(np.linalg.norm(prev), epsilon)
            if diff / norm_prev < tolerance:
                converged = True
                break
            prev = x.copy()

    if iterations_performed == 0:
        forward_final = initial @ response
        if background is not None:
            forward_final = forward_final + background
        forward_final = np.clip(forward_final, epsilon, None)
        loglike[0] = _poisson_loglikelihood(data, forward_final, epsilon)

    history_arr = np.stack(history, axis=0) if history else None
    return x, iterations_performed, converged, history_arr, loglike[:max(iterations_performed, 1)]


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
    show_progress: bool = True,
    leave: bool = True,
    description: str | None = None,
) -> Tuple[
    NDArray[np.float64],
    int,
    bool,
    NDArray[np.float64] | None,
    NDArray[np.float64],
]:
    """Pure NumPy Richardson-Lucy iteration for 2D spectra."""
    x = np.clip(initial.astype(np.float64, copy=True), epsilon, None)
    response = response.astype(np.float64, copy=False)
    data = np.clip(data.astype(np.float64, copy=False), 0.0, None)
    if background is not None:
        background = background.astype(np.float64, copy=False)

    history = [] if store_history else None
    loglike = np.empty(iterations if iterations > 0 else 1, dtype=np.float64)

    response_T = response.T

    G_ex = None
    if G_ex is not None:
        G_ex = G_ex.astype(np.float64, copy=False)
        G_ex_T = G_ex.T

        def forward_project(arr: NDArray[np.float64]) -> NDArray[np.float64]:
            return G_ex @ (arr @ response)

        def back_project(arr: NDArray[np.float64]) -> NDArray[np.float64]:
            return G_ex_T @ (arr @ response_T)

    else:

        def forward_project(arr: NDArray[np.float64]) -> NDArray[np.float64]:
            return arr @ response

        def back_project(arr: NDArray[np.float64]) -> NDArray[np.float64]:
            return arr @ response_T

    ones = np.ones_like(data, dtype=np.float64)
    denominator = back_project(ones)
    denominator = np.clip(denominator, epsilon, None)

    converged = False
    prev = x.copy()
    iterations_performed = 0

    if show_progress and iterations > 0:
        try:
            from tqdm.auto import tqdm
        except Exception:  # pragma: no cover - optional dependency
            iterator = range(iterations)
        else:
            iterator = tqdm(
                range(iterations),
                leave=leave,
                desc=description,
                total=iterations,
            )
    else:
        iterator = range(iterations)

    for it in iterator:
        forward = forward_project(x)
        if background is not None:
            forward = forward + background
        forward = np.clip(forward, epsilon, None)
        ratio = data / forward
        x *= back_project(ratio) / denominator
        x = np.clip(x, epsilon, None)
        x = np.where(mask, x, 0.0)


        iterations_performed = it + 1

        if history is not None:
            history.append(x.copy())
        forward_updated = forward_project(x)
        if background is not None:
            forward_updated = forward_updated + background
        loglike[it] = _poisson_loglikelihood(data, forward_updated, epsilon)

        if tolerance is not None:
            diff = np.linalg.norm(x - prev)
            norm_prev = max(np.linalg.norm(prev), epsilon)
            if diff / norm_prev < tolerance:
                converged = True
                break
            prev = x.copy()


    if iterations_performed == 0:
        forward_final = forward_project(initial)
        if background is not None:
            forward_final = forward_final + background
        loglike[0] = _poisson_loglikelihood(data, forward_final, epsilon)

    history_arr = np.stack(history, axis=0) if history else None
    return x, iterations_performed, converged, history_arr, loglike[:max(iterations_performed, 1)]
