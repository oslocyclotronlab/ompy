from . import Matrix, Vector
from .stubs import ArrayBool, Axes

from scipy.optimize import curve_fit
import numpy as np
from numba import njit
from dataclasses import dataclass
import matplotlib.pyplot as plt


# @njit
def gaussian(E, A, mu, sigma):
    return A * np.exp(-(E - mu) ** 2 / (2 * sigma ** 2))


@dataclass
class GaussFit:
    counts: float
    mu: float
    sigma: float
    fwhm: float
    cov: np.ndarray
    E: np.ndarray

    def __call__(self, E: np.ndarray) -> np.ndarray:
        return gaussian(E, self.counts, self.mu, self.sigma)

    def as_vector(self, x: None | np.ndarray | Vector = None) -> Vector:
        match x:
            case None:
                return Vector(E=self.E, values=self(self.E))
            case np.ndarray:
                return Vector(E=x, values=self(x))
            case _:
                # Autoreload messes with Vector()
                return x.clone(E=x.E, values=self(x.E))

    def plot(self, ax: None | Axes = None, **kwargs) -> Axes:
        return self.as_vector().plot(ax=ax, **kwargs)


def fit_gauss(vec: Vector, mask: ArrayBool) -> GaussFit:
    E = vec.E[mask]
    region = vec[mask]

    p0 = [np.max(region), E[len(E) // 2], 30.0]
    p, cov = curve_fit(gaussian, E, region, p0=p0)
    fwhm = 2 * np.sqrt(2 * np.log(2)) * p[2]
    return GaussFit(*p, fwhm, cov, E)
