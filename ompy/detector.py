from abc import ABC, abstractmethod
from .stubs import Unitlike, Axes, array
from .library import from_unit, into_unit
from . import u, Vector, Matrix, empty_like
import numpy as np
import matplotlib.pyplot as plt
from typing import overload, Literal

FWHM_TO_SIGMA = 1 / (2 * np.sqrt(2 * np.log(2)))


class Detector(ABC):
    @classmethod
    def resolution_sigma(cls, energy: Unitlike) -> Unitlike:
        fwhm = cls.FWHM(energy)
        return fwhm * FWHM_TO_SIGMA

    @classmethod
    def _resolution_sigma(cls, energy: float) -> float:
        fwhm = cls._FWHM(energy)
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
    def resolution_e(cls, Eg: Unitlike, sigma: float = 6) -> Unitlike:
        """ Find the index of the diagonal + resolution of sigma"""
        res = cls.resolution_sigma(Eg)
        Ex = into_unit(Eg, 'keV') + sigma*res
        return Ex

    @classmethod
    def resolution_gauss(cls, E: array | Vector, mu: Unitlike, as_array=False) -> array | Vector:
        if isinstance(E, Vector):
            E = E.to('keV').E
        Eg = from_unit(mu, 'keV')
        sigma = cls._resolution_sigma(Eg)
        gauss = ngaussian(E, Eg, sigma)
        normalized = gauss / gauss.sum()
        if as_array:
            return normalized
        return Vector(E=E, values=normalized)

    @classmethod
    def resolution_matrix(cls, mat: Matrix, *, as_array=False) -> Matrix | array:
        """ Return a matrix with the resolution of sigma"""
        R: Matrix = empty_like(mat)
        for ex_i in range(mat.shape[0]):
            e_diagonal = mat.Ex[ex_i]
            R[ex_i, :] = cls.resolution_gauss(mat.Eg, e_diagonal, as_array=True)
        if as_array:
            return R.values
        return R

    @classmethod
    @overload
    def cut_at_resolution(cls, mat: Matrix, *, sigma: float = 6, inplace=Literal[False]) -> Matrix: ...

    @classmethod
    @overload
    def cut_at_resolution(cls, mat: Matrix, *, sigma: float = 6, inplace=Literal[True]) -> None: ...

    @classmethod
    def cut_at_resolution(cls, mat: Matrix, *, sigma: float = 6, inplace=False) -> Matrix | None:
        """ Return a matrix with the resolution of sigma"""
        mask = np.zeros_like(mat.values, dtype=bool)
        for ex_i in range(mat.shape[0]):
            e_diagonal = mat.Ex[ex_i]
            e = cls.resolution_e(Eg=e_diagonal, sigma=sigma)
            i = mat.index_Eg(e)
            mask[ex_i, i:] = True

        if inplace:
            mat.values[mask] = 0.0
        else:
            matrix = mat.clone()
            matrix.values[mask] = 0.0
            return matrix


    @classmethod
    def plot_FWHM(cls, ax: Axes | None = None, start=0, stop='10MeV', n=100, **kwargs) -> Axes:
        if ax is None:
            ax = plt.subplots()[1]
        start = from_unit(start, 'keV')
        stop = from_unit(stop, 'keV')
        eg = np.linspace(start, stop, n)
        ax.plot(eg, cls._FWHM(eg), **kwargs)
        ax.set_xlabel('Gamma energy [keV]')
        ax.set_ylabel('FWHM [keV]')
        ax.set_title('FWHM of {}'.format(cls.__name__))
        ax2 = ax.twinx()
        ax2.set_ylabel(r'$\sigma$ [keV]')
        ax2.plot(eg, cls._resolution_sigma(eg), **kwargs)

        return ax


class OSCAR(Detector):
    # From https://doi.org/10.1016/j.nima.2020.164678
    a0 = 60.6473
    a1 = 0.45802
    a2 = 2.655517e-4

    def __init__(self):
        pass

    @classmethod
    def _FWHM(cls, e: float) -> float:
        return np.sqrt(cls.a0 + cls.a1 * e + cls.a2 * e ** 2)


#@njit
def ngaussian(x: array, mu: float, sigma: float):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sigma, 2.)))