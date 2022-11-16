from . import Matrix, Vector
from .stubs import ArrayBool

from scipy.optimize import curve_fit
import numpy as np
from numba import njit
from dataclasses import dataclass


@njit
def gaussian(E, A, mu, sigma):
    return A * np.exp(-(E-mu)**2/(2*sigma**2))


@dataclass
class FWHMFit:
    counts: float
    mu: float
    sigma: float
    fwhm: float
    cov: np.ndarray


def get_fwhm(vec: Vector, mask: ArrayBool):
    E = vec.E[mask]
    region = vec[mask]

    p0 = [np.max(region), E[len(E)//2], 30.0]
    p, cov = curve_fit(gaussian, E, region, p0=p0)
    fwhm = 2*np.sqrt(2*np.log(2))*p[2]
    return FWHMFit(*p, fwhm, cov)
