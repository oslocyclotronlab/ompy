import warnings

from . import Matrix, Vector
from .stubs import ArrayBool, Axes

from scipy.optimize import curve_fit
import numpy as np
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
                return Vector(E=self.E, values=self(self.E), edge='mid')
            case np.ndarray:
                return Vector(E=x, values=self(x), edge='mid')
            case _:
                # Autoreload messes with Vector()
                warnings.warn("Edge should be mid!")
                return x.clone(E=x.E, values=self(x.E))

    def plot(self, ax: None | Axes = None, **kwargs) -> Axes:
        return self.as_vector().plot(ax=ax, **kwargs)

    def __str__(self):
        unc = np.sqrt(np.diagonal(self.cov))
        # unicode for plus minus sign: \u00B1
        params = [f"counts: {self.counts:.2f} \u00B1 {unc[0]:.2f}",
                  f"mu:     {self.mu:.2f} \u00B1 {unc[1]:.2f}",
                  f"sigma:  {self.sigma:.2f} \u00B1 {unc[2]:.2f}",
                  f"fwhm:   {self.fwhm:.2f}"]
        param_str = "\n".join(params)

        return f"{param_str}"


def fit_gauss(vec: Vector, mask: ArrayBool | None = None) -> GaussFit:
    if mask is None:
        mask = np.ones(len(vec), dtype=bool)
    E = vec.to_mid().X[mask]
    region = vec[mask]

    p0 = [np.max(region), E[len(E) // 2], 30.0]
    p, cov = curve_fit(gaussian, E, region, p0=p0)
    fwhm = 2 * np.sqrt(2 * np.log(2)) * p[2]
    return GaussFit(*p, fwhm, cov, E)
