from abc import ABC, abstractmethod
from .stubs import Unitlike, Axes, array
from .library import from_unit, into_unit
from . import u, Vector, Matrix, empty_like, empty
import numpy as np
import matplotlib.pyplot as plt
from typing import overload, Literal
import warnings

FWHM_TO_SIGMA = 1 / (2 * np.sqrt(2 * np.log(2)))


class Detector(ABC):
    @classmethod
    def resolution_sigma(cls, energy: Unitlike) -> Unitlike:
        fwhm = cls.FWHM(energy)
        return fwhm * FWHM_TO_SIGMA

    @classmethod
    def _resolution_sigma(cls, energy: float) -> float:
        fwhm = cls.__FWHM(energy)
        return fwhm * FWHM_TO_SIGMA

    @classmethod
    def FWHM(cls, energy: Unitlike) -> Unitlike:
        e = from_unit(energy, 'keV')
        fwhm = cls._FWHM(e)
        return fwhm*u('keV')

    @classmethod
    @abstractmethod
    def _FWHM(cls, energy: float) -> float:
        ...

    @classmethod
    @np.vectorize
    def __FWHM(cls, energy: float) -> float:
        return cls._FWHM(energy)

    @classmethod
    def resolution_e(cls, e: Unitlike, sigma: float = 2) -> Unitlike:
        """ Find the index of the diagonal + resolution of sigma"""
        res = cls.resolution_sigma(e)
        Ex = into_unit(e, 'keV') + sigma*res
        return Ex

    @classmethod
    def resolution_gauss(cls, E: array | Vector, mu: Unitlike, as_array=False) -> array | Vector:
        if isinstance(E, Vector):
            E = E.to('keV').E_true
        Eg = from_unit(mu, 'keV')
        sigma = cls._resolution_sigma(Eg)
        gauss = ngaussian(E, Eg, sigma)
        normalized = gauss / gauss.sum()
        if as_array:
            return normalized
        return Vector(E=E, values=normalized)

    @classmethod
    @abstractmethod
    def resolution_matrix(cls, mat: Matrix, *, as_array=False) -> Matrix | array:
        ...


    @classmethod
    def plot_FWHM(cls, ax: Axes | None = None, start=0, stop='10MeV', n=100, **kwargs) -> Axes:
        if ax is None:
            ax = plt.subplots()[1]
        assert ax is not None
        start = from_unit(start, 'keV')
        stop = from_unit(stop, 'keV')
        e = np.linspace(start, stop, n)
        ax.plot(e, cls.__FWHM(e), **kwargs)
        ax.set_xlabel('Gamma energy [keV]')
        ax.set_ylabel('FWHM [keV]')
        ax.set_title('FWHM of {}'.format(cls.__name__))
        ax2 = ax.twinx()
        ax2.set_ylabel(r'$\sigma$ [keV]')
        ax2.plot(e, cls._resolution_sigma(e), **kwargs)

        return ax


class EgDetector(Detector):
    @classmethod
    def resolution_matrix(cls, mat: Matrix, *, as_array=False) -> Matrix | array:
        """ Return a matrix with the resolution of sigma

        R := resolution
        U := unfolded
        F = U@R
        """
        E = mat.Eg
        R: Matrix = empty(E, E)
        for (i, e) in enumerate(E):
            R[i, :] = cls.resolution_gauss(E, e, as_array=True)
        if as_array:
            return R.values
        R.xlabel = r"Measured $E_\gamma$"
        R.ylabel = r"True $E_\gamma$"
        return R

class ExDetector(Detector):
    @classmethod
    def resolution_matrix(cls, mat: Matrix, *, as_array=False) -> Matrix | array:
        """ Return a matrix with the resolution of sigma

        R := resolution
        U := unfolded
        F = R@U
        """
        warnings.warn("Written while sleepy. Might be buggy")
        E = mat.Ex
        R: Matrix = empty(E, E)
        for (i, e) in enumerate(E):
            R[:, i] = cls.resolution_gauss(E, e, as_array=True)
        if as_array:
            return R.values
        R.ylabel = r"Measured $E_x$"
        R.xlabel = r"True $E_x$"
        return R

class OSCAR(EgDetector):
    # From https://doi.org/10.1016/j.nima.2020.164678
    a0 = 60.6473
    a1 = 0.45802
    a2 = 2.655517e-4

    @classmethod
    def _FWHM(cls, e: float) -> float:
        return np.sqrt(cls.a0 + cls.a1 * e + cls.a2 * e ** 2)


class SiRi(ExDetector):
    # From https://doi.org/10.1016/j.nima.2011.05.055
    # says ~= 100 keV
    a0 = 100.0

    @classmethod
    def _FWHM(cls, e: float) -> float:
        return cls.a0


class CompoundDetector:
    def __init__(self, eg_detector: EgDetector, ex_detector: ExDetector):
        self.egdetector = eg_detector
        self.exdetector = ex_detector

    @overload
    def cut_at_resolution(self, mat: Matrix, *, eg_sigma: ..., ex_sigma: ..., inplace: Literal[False]) -> Matrix: ...

    @overload
    def cut_at_resolution(self, mat: Matrix, *, eg_sigma: ..., ex_sigma: ..., inplace: Literal[True]) -> None: ...

    def cut_at_resolution(self, mat: Matrix, *, eg_sigma: float = 3, ex_sigma: float = 3,
                          inplace: bool = False) -> Matrix | None:
        """ Return a matrix with the resolution of sigma"""
        mask = np.zeros_like(mat.values, dtype=bool)
        for i in range(mat.shape[0]):
            e_diagonal = mat.Ex[i]
            ex = self.exdetector.resolution_e(e_diagonal, sigma=-ex_sigma)
            ex_i = mat.index_Ex(ex) if ex >= 0 else 0
            eg = self.egdetector.resolution_e(e_diagonal, sigma=eg_sigma)
            eg_i = mat.index_Eg(eg)
            mask[ex_i, eg_i:] = True

        if inplace:
            mat.values[mask] = 0.0
        else:
            matrix = mat.clone()
            matrix.values[mask] = 0.0
            return matrix

Oslo = CompoundDetector(ex_detector=SiRi(), eg_detector=OSCAR())
    

#@njit
def ngaussian(x: array, mu: float, sigma: float):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sigma, 2.)))
