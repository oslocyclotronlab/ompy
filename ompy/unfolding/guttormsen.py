from __future__ import annotations
import numpy as np
from ..numbalib import njit, prange
from .. import Vector, Matrix, Response, zeros_like
from .unfolder import Unfolder, UnfoldedResult1DAll, ResultMeta, mask_511
from dataclasses import dataclass
import time

#TODO
# -[] Make compton sub work on unfolded result


class Guttormsen(Unfolder):
    """ Unfolding algorithm from Guttormsen et al. 1998

    This algorithm is only valid for 1D histograms with uniform binning.
    The algorithm is described in the paper:

    Guttormsen, K. A., Kjeldsen, H. K., & Nielsen, J. B. (1998).
    Unfolding of multidimensional histograms.
    Nuclear Instruments and Methods in Physics Research Section A: Accelerators, Spectrometers, Detectors and Associated Equipment, 400(1), 1–8. https://doi.org/10.1016/S0168-9002(97)00459-6

    Parameters
    ----------
    R: Matrix
        The unsmoothed response matrix
    G: Matrix
        The gaussian smoothing matrix
    iterations: int
        The number of iterations to perform
    """

    def __init__(self, R: Matrix, G: Matrix, iterations: int = 10, weight: float = 1e-3):
        super().__init__(R, G)
        self.iterations = iterations
        self.weight = weight

    def _unfold_vector(self, R: Matrix, data: Vector, background: Vector | None, initial: Vector,
                       mask: Vector, **kwargs) -> GuttormsenResult1D:
        iterations = kwargs.pop("iterations", self.iterations)
        weight = kwargs.pop("weight", self.weight)
        mask_ = mask.values

        start = time.time()  # Njit does not support time.time()
        uall, cost, fluctuations = _unfold_vector(R.values, data.values, initial.values,
                                                  iterations, mask_)
        elapsed = time.time() - start
        return GuttormsenResult1D(meta=ResultMeta(time=elapsed),
                                  R=R, raw=data, background=background,
                                  initial=initial, uall=uall, cost=cost,
                                  fluctuations=fluctuations,
                                  mask=mask)

    def supports_background() -> bool:
        return False



@njit
def _unfold_vector(R: np.ndarray, raw: np.ndarray, initial: np.ndarray, iterations: int,
                   mask: np.ndarray):
    u = initial
    u_all = np.empty((iterations, len(u)))
    cost = np.empty(iterations)
    fluctuations = np.empty(iterations)
    f = R@u
    for i in range(iterations):
        u += raw - f
        f = R@u
        u_all[i] = u
        cost[i] = chi2(f[mask], raw[mask])
        fluctuations[i] = fluctuation_cost(u, 20, mask)
    return u_all, cost, fluctuations


@njit
def chi2(a, b):
    return np.sum((a - b)**2 / a)


@njit
def fluctuation_cost(x, sigma: float, mask: np.ndarray):
    smoothed = gaussian_filter_1d(x, sigma)
    diff = np.abs(((smoothed - x) / smoothed)[mask])
    return diff.sum()


def compton_subtraction(unfolded: Vector, raw: Vector, response: Response):
    G = response.gaussian_like(unfolded).T
    E = unfolded.X
    fe = response.interpolation.FE(E)
    se = response.interpolation.SE(E)
    de = response.interpolation.DE(E)
    ap = response.interpolation.AP(E)
    eff = response.interpolation.Eff(E)

    ufe = unfolded * fe
    ufe = G@ufe

    use = unfolded * se
    use = G@shift(use, 511)

    ude = unfolded * de
    ude = G@shift(ude, 2*511)

    uap = zeros_like(unfolded)
    uap.vloc[511] = sum(unfolded * ap)
    # Unclear whether 511 should be smoothed

    w = use + ude + uap
    v = ufe + w
    compton = raw - v
    compton = G@compton
    u = (raw - compton - w) / fe
    unf = u / eff

    return ufe, use, ude, uap, compton, u, unf


def shift(x: Vector, shift: float) -> Vector:
    return shift_integer(x, shift)


def shift_integer(v: Vector, shift: float) -> Vector:
    # Calculate the index shift corresponding to the specified shift value
    i = v.index(abs(shift)) + 1

    # Create a new array for the shifted values
    shifted = np.zeros_like(v.X)

    # Shift the values
    shifted[:len(v) - i] = v[i:]

    u = v.clone(values=shifted)
    return u


@njit
def gaussian_filter_1d(x, sigma):
    """
    1D Gaussian filter with standard deviation sigma.
    """
    k = int(4.0 * sigma + 0.5)
    w = np.zeros(2 * k + 1)
    for i in range(-k, k + 1):
        w[i + k] = np.exp(-0.5 * i**2 / sigma**2)
    w /= np.sum(w)

    # Handle edge cases of input signal
    y = np.zeros_like(x)
    for i in range(len(x)):
        for j in range(-k, k + 1):
            if i + j >= 0 and i + j < len(x):
                y[i] += x[i + j] * w[j + k]
    return y



@dataclass(kw_only=True)#(frozen=True, slots=True)
class GuttormsenResult1D(UnfoldedResult1DAll):
    fluctuations: np.ndarray

    def best(self, w: float = 0.2, min: int = 0) -> Vector:
        cost = (1 - w)*self.cost + w * self.fluctuations
        i = max(min, np.argmin(cost))
        return self.unfolded(i)
