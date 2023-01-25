from __future__ import annotations
from . import ResponseData, DiscreteInterpolation, interpolate_compton
from .. import Vector, Matrix, USE_GPU
import warnings
from .numbalib import njit, prange
import numpy as np
if USE_GPU:
    from . import interpolate_gpu
from collections import OrderedDict

# TODO Always make a high resolution compton to save, rebin and reuse?
# Is rebinning correct? We want to preserve probability, not counts.
# The R(e) gives the energy spectrum *at* e, but an experimental vector
# has a bin [e, e+de]. Should R^(e) be a weighted average over R(e) ... R(e+de)?


class Response:
    def __init__(self, data: ResponseData,
                 interpolation: DiscreteInterpolation):
        self.data: ResponseData = data
        self.interpolation: DiscreteInterpolation = interpolation

    @classmethod
    def from_data(cls, data: ResponseData) -> Response:
        intp = DiscreteInterpolation.from_data(data.normalize())
        return cls(data, intp)

    def interpolate(self, E: np.ndarray, normalize: float = True, **kwargs) -> Matrix:
        compton = self.interpolate_compton(E, **kwargs)
        FE, SE, DE, AP = self.interpolation.structures()
        emin = compton.Eg.min()
        j511 = compton.index_Eg(511)
        for i, e in enumerate(E):
            if e > emin:
                j = compton.index_Eg(e)
                compton[i, j] += FE(e)
            if e - 511 > emin:
                j = compton.index_Eg(e - 511)
                compton[i, j] += SE(e)
            if e - 2*511 > emin:
                j = compton.index_Eg(e - 511*2)
                compton[i, j] += DE(e)
            if 511 > emin:
                compton[i, j511] += AP(e)
        if normalize:
            compton.values = compton.values / compton.values.sum(axis=1)[:, None]

        return compton

    def interpolate_compton(self, E: np.ndarray, GPU: bool = True,
                            sigma: float = 6) -> Matrix:
        if not self.interpolation.is_fwhm_normalized:
            raise RuntimeError("FWHM must be normalized before compton interpolation.")
        sigmafn = self.interpolation.sigma
        if USE_GPU and GPU:
            print("GPU")
            compton: Matrix = interpolate_gpu(self.data, E, sigmafn, sigma)
        else:
            print("CPU")
            compton: Matrix = interpolate_compton(self.data, E, sigmafn, sigma)
        return compton

    def gaussian_matrix(self, E: np.ndarray) -> Matrix:
        return gaussian_matrix(E, self.interpolation.sigma)

    def clone(self, data: ResponseData | None = None, interpolation: DiscreteInterpolation | None = None) -> Response:
        return Response(data=data or self.data, interpolation=interpolation or self.interpolation)


def gaussian_matrix(E: np.ndarray, sigmafn) -> Matrix:
    sigma = sigmafn(E)
    values = _gaussian_matrix(E, sigma)
    values = values / values.sum(axis=1)[:, None]
    return Matrix(Eg=E, Ex=E, values=values, xlabel='E Observed', ylabel='E True')


@njit(parallel=True)
def _gaussian_matrix(E, sigma):
    n = len(E)
    m = np.zeros((n, n))
    for i in prange(n):
        mu = E[i]
        sigma_ = sigma[i]
        for j in prange(n):
            m[i, j] = gaussian(E[j], mu, sigma_)
    return m

@njit
def gaussian(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma)**2) / (sigma * np.sqrt(2 * np.pi))