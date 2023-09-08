from __future__ import annotations
from abc import ABC, abstractmethod
from .stubs import Unitlike, Axes, array
from .library import from_unit, into_unit
from . import u, Vector, Matrix, empty
from .response import DiscreteInterpolation, Response
import numpy as np
import matplotlib.pyplot as plt
from typing import overload, Literal, Callable
import warnings
from functools import partial

FWHM_TO_SIGMA = 1 / (2 * np.sqrt(2 * np.log(2)))

"""
TODO:
    - [ ] Make point-and-click calibrator
    - [ ] Fix the _, __ naming
"""


class Detector(ABC):
    def resolution_sigma(self, energy: Unitlike) -> Unitlike:
        fwhm = self.FWHM(energy)
        return fwhm * FWHM_TO_SIGMA

    def _resolution_sigma(self, energy: float) -> float:
        fwhm = self._FWHM(energy)
        return fwhm * FWHM_TO_SIGMA

    def __resolution_sigma(self, energy: float) -> float:
        fwhm = self.__FWHM(energy)
        return fwhm * FWHM_TO_SIGMA

    def sigma(self, energy: Unitlike) -> Unitlike:
        return self.resolution_sigma(energy)

    def _sigma(self, energy: float) -> float:
        return self._resolution_sigma(energy)

    def FWHM(self, energy: Unitlike) -> Unitlike:
        e = from_unit(energy, 'keV')
        fwhm = self._FWHM(e)
        return fwhm*u('keV')

    def __FWHM(self, energy: np.ndarray) -> np.ndarray:
        return np.asarray([self._FWHM(e) for e in energy])

    @abstractmethod
    def _FWHM(self, energy: float) -> float:
        ...

    def resolution_e(self, e: Unitlike, sigma: float = 2) -> Unitlike:
        """ Find the index of the diagonal + resolution of sigma"""
        res = self.resolution_sigma(e)
        Ex = into_unit(e, 'keV') + sigma*res
        return Ex

    def resolution_gauss(self, E: array | Vector, mu: Unitlike, as_array=False) -> array | Vector:
        if isinstance(E, Vector):
            E = E.to('keV').E_true
        Eg = from_unit(mu, 'keV')
        sigma = self._resolution_sigma(Eg)
        gauss = ngaussian(E, Eg, sigma)
        normalized = gauss / gauss.sum()
        if as_array:
            return normalized
        return Vector(E=E, values=normalized)

    @abstractmethod
    def resolution_matrix(self, mat: Matrix, *, as_array=False) -> Matrix | array:
        ...

    def plot_FWHM(self, ax: Axes | None = None, start=0, stop='10MeV', n=100, **kwargs) -> Axes:
        if ax is None:
            ax = plt.subplots()[1]
        assert ax is not None
        start = from_unit(start, 'keV')
        stop = from_unit(stop, 'keV')
        e = np.linspace(start, stop, n)
        ax.plot(e, self.__FWHM(e), **kwargs)
        ax.set_xlabel('Gamma energy [keV]')
        ax.set_ylabel('FWHM [keV]')
        ax.set_title('FWHM of {}'.format(self.__class__.__name__))
        ax2 = ax.twinx()
        ax2.set_ylabel(r'$\sigma$ [keV]')
        ax2.plot(e, self.__resolution_sigma(e), **kwargs)

        return ax


class EgDetector(Detector):
    def resolution_matrix(self, mat: Matrix, *, as_array=False) -> Matrix | array:
        """ Return a matrix with the resolution of sigma

        R := resolution
        U := unfolded
        F = U@R
        """
        E = mat.Y
        R: Matrix = empty(E, E)
        for (i, e) in enumerate(E):
            R[i, :] = self.resolution_gauss(E, e, as_array=True)
        if as_array:
            return R.values
        R.xlabel = r"Measured $E_\gamma$"
        R.ylabel = r"True $E_\gamma$"
        return R


class ExDetector(Detector):
    def resolution_matrix(self, mat: Matrix, *, as_array=False) -> Matrix | array:
        """ Return a matrix with the resolution of sigma

        R := resolution
        U := unfolded
        F = R@U
        """
        warnings.warn("Written while sleepy. Might be buggy")
        E = mat.X
        R: Matrix = empty(E, E)
        for (i, e) in enumerate(E):
            R[:, i] = self.resolution_gauss(E, e, as_array=True)
        if as_array:
            return R.values
        R.ylabel = r"Measured $E_x$"
        R.xlabel = r"True $E_x$"
        return R


class LambdaEgDetector(EgDetector):
    def __init__(self, func: Callable[[float], float]):
        self.func = func

    def _FWHM(self, e: float) -> float:
        return self.func(e)


class OSCAR(EgDetector):
    # From https://doi.org/10.1016/j.nima.2020.164678
    a0 = 60.6473
    a1 = 0.45802
    a2 = 2.655517e-4

    def _FWHM(self, e: float) -> float:
        return np.sqrt(self.a0 + self.a1 * e + self.a2 * e ** 2)


class SiRi(ExDetector):
    # From https://doi.org/10.1016/j.nima.2011.05.055
    # says ~= 100 keV
    # TODO Must be wrong? Way too smeared matrices
    # EX gauss matrix is wrong
    #a0 = 100.0
    a0 = 100.0

    def _FWHM(self, e: float) -> float:
        return self.a0


class CompoundDetector:
    def __init__(self, eg_detector: EgDetector, ex_detector: ExDetector):
        self.egdetector = eg_detector
        self.exdetector = ex_detector

    @overload
    def cut_at_resolution(self, mat: Matrix, *, eg_sigma: ...,
                          ex_sigma: ..., inplace: Literal[False]) -> Matrix: ...

    @overload
    def cut_at_resolution(self, mat: Matrix, *, eg_sigma: ...,
                          ex_sigma: ..., inplace: Literal[True]) -> None: ...

    def cut_at_resolution(self, mat: Matrix, *, eg_sigma: float = 3,
                          ex_sigma: float = 3,
                          inplace: bool = False) -> Matrix | None:
        """ Return a matrix with the resolution of sigma"""
        mask = np.zeros_like(mat.values, dtype=bool)
        for i in range(mat.shape[0]):
            e_diagonal = mat.Ex[i] * mat.Ex_index.unit
            ex = self.exdetector.resolution_e(e_diagonal, sigma=-ex_sigma)
            eg = self.egdetector.resolution_e(e_diagonal, sigma=eg_sigma)
            if not mat.Ex_index.is_inbounds(ex) or not mat.Eg_index.is_inbounds(eg):
                continue
            ex_i = mat.index_Ex(ex)
            eg_i = mat.index_Eg(eg)
            mask[ex_i, eg_i:] = True

        if inplace:
            mat.values[mask] = 0.0
        else:
            matrix = mat.clone()
            matrix.values[mask] = 0.0
            return matrix


class Oslo(CompoundDetector):
    def __init__(self, eg_detector: EgDetector = OSCAR(),
                 ex_detector: ExDetector = SiRi()):
        super().__init__(eg_detector=eg_detector, ex_detector=ex_detector)

    @classmethod
    def from_response(cls, response: DiscreteInterpolation | Response) -> Oslo:
        if isinstance(response, Response):
            fwhm = response.interpolation.FWHM
        else:
            fwhm = response.FWHM
        return Oslo(eg_detector=LambdaEgDetector(fwhm))


# @njit
def ngaussian(x: array, mu: float, sigma: float):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sigma, 2.)))
