from __future__ import annotations
import numpy as np
from ..array import Index, Matrix, Vector, to_index
from typing import TYPE_CHECKING, Literal
from dataclasses import dataclass

if TYPE_CHECKING:
    from .detector import Detector

def _gaussian_kernel(delta: np.ndarray, sigma: float) -> np.ndarray:
    """Unnormalized Gaussian sampled at delta; returns row-stochastic kernel."""
    # Guard against zero or NaN sigma
    if not np.isfinite(sigma) or sigma <= 0:
        k = np.zeros_like(delta, dtype=float)
        # Put a delta at center if possible
        mid = int(np.argmin(np.abs(delta)))
        k[mid] = 1.0
        return k
    k = np.exp(-0.5 * (delta / sigma) ** 2, dtype=float)
    s = k.sum()
    if s <= 0 or not np.isfinite(s):
        # Fallback to delta at center
        k[:] = 0.0
        k[int(np.argmin(np.abs(delta)))] = 1.0
        return k
    return k / s

@dataclass(frozen=True)
class SmearPlan:
    Ef: np.ndarray
    center_value: float
    kernel: np.ndarray                    # the stationary impulse response k(delta_ef)
    sigma_ef: np.ndarray    # per-eg array (mixture) or scalar (single)

    def build_matrix(self) -> np.ndarray:
        """Build the Toeplitz-like smearing matrix lazily."""
        return _make_rowstochastic_conv_matrix(self.Ef, self.kernel, center_value=self.center_value)



def _gaussian_kernel(delta: np.ndarray, sigma: float) -> np.ndarray:
    """Unnormalized Gaussian sampled at delta; returns row-stochastic kernel."""
    # Guard against zero or NaN sigma
    if not np.isfinite(sigma) or sigma <= 0:
        k = np.zeros_like(delta, dtype=float)
        # Put a delta at center if possible
        mid = int(np.argmin(np.abs(delta)))
        k[mid] = 1.0
        return k
    k = np.exp(-0.5 * (delta / sigma) ** 2, dtype=float)
    s = k.sum()
    if s <= 0 or not np.isfinite(s):
        # Fallback to delta at center
        k[:] = 0.0
        k[int(np.argmin(np.abs(delta)))] = 1.0
        return k
    return k / s

def _make_rowstochastic_conv_matrix(grid: np.ndarray, kernel: np.ndarray, center_value: float = 0.0) -> np.ndarray:
    """
    Build a row-stochastic Toeplitz-like matrix implementing 1D convolution
    with `kernel` on the 1D `grid` (monotone). Each row centers the kernel at
    that row's grid value, truncates to the grid, and re-normalizes to sum to 1.
    `center_value` selects which grid value is considered delta=0 for the kernel.
    """
    grid = np.asarray(grid, dtype=float)
    kernel = np.asarray(kernel, dtype=float)
    n = grid.size
    M = np.zeros((n, n), dtype=float)

    # Find index in grid closest to center_value; align kernel's "zero" there.
    zero_idx = int(np.argmin(np.abs(grid - center_value)))
    nK = kernel.size

    for i in range(n):
        # place kernel centered at row i: j index shift so that j=i aligns with zero_idx in kernel frame
        j_start = max(0, i - zero_idx)
        j_end = min(n, i - zero_idx + nK)
        k_start = max(0, zero_idx - i)
        k_end = k_start + (j_end - j_start)

        row = kernel[k_start:k_end]
        s = row.sum()
        if s > 0 and np.isfinite(s):
            M[i, j_start:j_end] = row / s
        else:
            # fallback to identity at i
            M[i, i] = 1.0

    # Small numeric cleanup to ensure each row sums exactly to 1 (within fp error)
    row_sums = M.sum(axis=1, keepdims=True)
    bad = ~np.isfinite(row_sums) | (row_sums <= 0)
    if np.any(bad):
        M[bad[:, 0], :] = 0.0
        M[bad[:, 0], np.where(bad[:, 0])[0]] = 1.0
        row_sums = M.sum(axis=1, keepdims=True)
    M /= row_sums
    return M

def plan_gef_mixture(
    Ef: np.ndarray,
    Eg: np.ndarray,
    sigma_ex: float,
    sigma_eg: np.ndarray,
    weights: np.ndarray | None = None,
    center_value: float = 0.0,
) -> SmearPlan:
    """
    Prepare a mixture-based Ef smearing *plan* without constructing the matrix.
    sigma_ef(eg) = sqrt(sigma_ex^2 + sigma_eg(eg)^2).

    Returns:
        SmearPlan with:
          - kernel: mixed stationary impulse response k_mixed
          - sigma_ef: array (neg,) of per-eg sigmas
        Use `.build_matrix()` to construct G_ef later.
    """
    Ef = np.asarray(Ef, dtype=float)
    Eg = np.asarray(Eg, dtype=float)
    sigma_eg = np.asarray(sigma_eg, dtype=float)

    # coerce sigma_ex to scalar
    if np.ndim(sigma_ex) != 0:
        sigma_ex = float(np.mean(np.asarray(sigma_ex, dtype=float)))

    neg = Eg.size
    if weights is None:
        w = np.ones(neg, dtype=float) / neg
    else:
        w = np.asarray(weights, dtype=float)
        if w.shape != (neg,):
            raise ValueError(f"weights shape {w.shape} does not match Eg length {neg}")
        s = w.sum()
        if s <= 0 or not np.isfinite(s):
            raise ValueError("weights must have positive finite sum")
        w = w / s

    # centered delta grid on Ef
    zero_idx = int(np.argmin(np.abs(Ef - center_value)))
    delta_ef = Ef - Ef[zero_idx]

    # per-eg sigmas and kernel bank
    sigma_ef = np.sqrt(sigma_ex**2 + sigma_eg**2)  # (neg,)
    K_bank = np.stack([_gaussian_kernel(delta_ef, s) for s in sigma_ef], axis=0)  # (neg, nef)

    # mix kernels
    k_mixed = (w[:, None] * K_bank).sum(axis=0)
    k_mixed /= k_mixed.sum()

    return SmearPlan(Ef=Ef, center_value=center_value, kernel=k_mixed, sigma_ef=sigma_ef)


def plan_gef_single(
    Ef: np.ndarray,
    Eg: np.ndarray,
    sigma_ex: float,
    sigma_eg: np.ndarray,
    weights: np.ndarray | None = None,
    center_value: float = 0.0,
) -> SmearPlan:
    """
    Prepare a single-Gaussian Ef smearing *plan* with
    sigma_eff^2 = sigma_ex^2 + E_w[sigma_eg(eg)^2].

    Returns:
        SmearPlan with:
          - kernel: single Gaussian kernel k_single
          - sigma_ef: scalar sigma_eff
        Use `.build_matrix()` to construct G_ef later.
    """
    Ef = np.asarray(Ef, dtype=float)
    Eg = np.asarray(Eg, dtype=float)
    sigma_eg = np.asarray(sigma_eg, dtype=float)

    if np.ndim(sigma_ex) != 0:
        sigma_ex = float(np.mean(np.asarray(sigma_ex, dtype=float)))

    neg = Eg.size
    if weights is None:
        w = np.ones(neg, dtype=float) / neg
    else:
        w = np.asarray(weights, dtype=float)
        if w.shape != (neg,):
            raise ValueError(f"weights shape {w.shape} does not match Eg length {neg}")
        s = w.sum()
        if s <= 0 or not np.isfinite(s):
            raise ValueError("weights must have positive finite sum")
        w = w / s

    zero_idx = int(np.argmin(np.abs(Ef - center_value)))
    delta_ef = Ef - Ef[zero_idx]

    sigma2_eff = float(np.sum(w * (sigma_eg ** 2)))
    sigma_eff = float(np.sqrt(sigma_ex**2 + sigma2_eff))
    k_single = _gaussian_kernel(delta_ef, sigma_eff)
    k_single /= k_single.sum()

    return SmearPlan(Ef=Ef, center_value=center_value, kernel=k_single, sigma_ef=sigma_eff)

def _to_index(x) -> Index | None:
    match x:
        case Index():
            return x
        case Vector():
            return x.X_index
        case None:
            return None
        case _:
            return to_index(np.asarray(x, float))


def make_plan(Ef: Index, Eg: Index, detector: Detector, Ex: Index | None = None, method: Literal['mixture', 'single'] = 'single') -> SmearPlan:
    match method:
        case 'mixture':
            fn = plan_gef_mixture
        case 'single':
            fn = plan_gef_single
        case _:
            raise ValueError(f"Invalid method: {method}. Must be 'mixture' or 'single'.")
    # We assume here sigma_ex is constant. If it werent, nothing here would work.
    E = Ex if Ex is not None else Ef
    sigma_ex = detector.ex_detector.sigma(E)
    sigma_eg = detector.eg_detector.sigma(Eg)
    return fn(Ef, Eg, sigma_ex, sigma_eg)


def mixture_sigma(Ef: Index, Eg: Index, detector: Detector, Ex: Index | None = None, method: Literal['mixture', 'single'] = 'single') -> Vector:
    Ef = _to_index(Ef)
    Ex = _to_index(Ex)
    Eg = _to_index(Eg)
    plan = make_plan(Ef, Eg, detector, Ex, method)
    sigma_ef =np.atleast_1d(plan.sigma_ef)
    if len(sigma_ef) == 1:
        sigma_ef = np.full_like(Ef, sigma_ef[0])
    vector = Vector(E=Ef, values=sigma_ef, name='Mixture resolution for $E_f$', vlabel=r'$\sigma_{ef}$')
    return vector

    
def mixture_response(Ef: Index, Eg: Index, detector: Detector, Ex: Index | None = None, method: Literal['mixture', 'single'] = 'single') -> Matrix:
    Ef = _to_index(Ef)
    Ex = _to_index(Ex)
    Eg = _to_index(Eg)
    plan = make_plan(Ef, Eg, detector, Ex, method)
    matrix = plan.build_matrix()
    return Matrix(true=Ef, measured=Ef, values=matrix, name='Mixture response for $E_f$',
                  xlabel=r'True $E_f$', ylabel=r'Measured $E_f$')