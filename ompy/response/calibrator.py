from __future__ import annotations
from . import Response
from .response import E_compton
from .. import Vector, zeros_like
from ..peakselect import fit_gauss, GaussFit
from ..stubs import Unitlike, ArrayBool, Axes
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from typing import Any
from .responsedata import Components

Mask = ArrayBool



@dataclass
class Peak:
    e: float
    counts: float


class Calibrator:
    """ Calibrate the different components of the response

    The calibration is done by folding a delta peak with the response and
    comparing the output to a experimental spectrum.
    Steps:
        1. Fit FE to determine FWHM.
        2. Fit the FE component.
        3. Compare Compton to the experimental spectrum.
        4. TODO: Fit SE, DE and 511

    Attributes:
        R (Response): The response to calibrate
        spectrum (Vector): The experimental spectrum

    """

    def __init__(self, R: Response, spectrum: Vector):
        self.R = R
        self.spectrum = spectrum
        self._fwhm_fit: GaussFit | None = None

    def calibrate(self):
        """ Calibrate the response

        Returns:
            components: The calibrated components
        """
        pass

    def calibrate_compton(self, fe_region: Mask, ignore: Mask | None = None):
        """ Calibrate the Compton component

        Args:
            fe_region (Mask): The region to fit the FE component
            ignore (Mask): A mask of the spectrum to ignore
        """
        if ignore is None:
            ignore = np.zeros_like(self.spectrum, dtype=bool)

        def loss(p) -> float:
            R = self.R.interpolate(self.spectrum.E, self.fwhm_fit.fwhm, 
                                   fwhm_peak=self.fwhm_fit.mu, compton=p[0]).T
            fe_e, fe_C, fe_fit = self.fit_FE(fe_region, Components(compton=p[0]))
            compton_edge = E_compton(fe_e, np.pi)
            region = ~ignore & (self.spectrum.E < compton_edge)

            delta = zeros_like(self.spectrum)
            delta.loc[fe_e] = fe_C
            folded = R@delta
            mse: float = np.mean((folded - self.spectrum)[region]**2)
            return mse

        p0 = np.asarray([1.0])
        res = minimize(loss, p0, bounds=[(0.0, 10.0)], method='Nelder-Mead')
        print(res)
        components = Components(compton=res.x[0])
        R = self.R.interpolate(self.spectrum.E, self.fwhm_fit.fwhm, fwhm_peak=self.fwhm_fit.mu, **components.to_dict()).T
        R0 = self.R.interpolate(self.spectrum.E, self.fwhm_fit.fwhm, fwhm_peak=self.fwhm_fit.mu).T
        e, C, _ = self.fit_FE(fe_region, components)
        print(C)
        ax=self.spectrum.plot()
        delta = zeros_like(self.spectrum)
        delta.loc[e] = C
        #delta.plot(ax=ax)
        folded = R@delta
        folded.plot(ax=ax, label='R')

        e, C, _ = self.fit_FE(fe_region)
        print(C)
        delta = zeros_like(self.spectrum)
        delta.loc[e] = C
        #delta.plot(ax=ax)
        folded0 = R0@delta
        folded0.plot(ax=ax, label='R0')
        ax.legend()
        #plt.show()
        return components, Components(), C

    def calibrate_FWHM(self, e: Unitlike | None = None,
                       fwhm: float | None = None) -> Response:
        if e is None:
            e = self.fwhm_fit.mu
        if fwhm is None:
            fwhm = self.fwhm_fit.fwhm
        self.R.fwhm_peak = e
        self.R.fwhm = fwhm
        self.R.get_probabilities()
        return self.R

    def fit_FWHM(self, region: Mask) -> GaussFit:
        # TODO Plot and check normality of FE
        self.fwhm_fit = fit_gauss(self.spectrum, region)
        return self.fwhm_fit

    def fit_FE(self, region: Mask, components: Components = Components()) -> tuple[float, float, Any]:
        """ Fit the FE component

        TODO: Allow one to redo with a "n-sigma" region fit using the 
        previous solution as starting point.

        Returns:
            fe_fit: The fitted FE component
        """
        fe: Vector = self.spectrum.iloc[region]
        emin = fe.E[0]
        emax = fe.E[-1]
        fe: np.ndarray = fe.values
        # Cut the response
        R = self.R.interpolate(self.spectrum.E, self.fwhm_fit.fwhm, **components.to_dict()).T

        def loss(p) -> float:
            e, C = p
            if e < emin or e > emax:
                return np.inf
            if C <= 0:
                return np.inf
            delta = zeros_like(self.spectrum)
            delta.loc[e] = C
            folded = R@delta
            #ax = folded.plot()
            #delta.plot(ax=ax)
            #self.spectrum.plot(ax=ax)
            #plt.show()
            #input()
            mse = np.mean((folded[region] - fe)**2)
            #print(e, C, mse)
            return mse

        p0 = [(emin+emax)/2, fe.sum()]
        #print(f"p0: {p0}")
        res = minimize(loss, p0, method='Nelder-Mead')
        e, C = res.x
        # print(res)
        # ax=self.spectrum.plot()
        # delta = zeros_like(self.spectrum)
        # delta.loc[e] = C
        # delta.plot(ax=ax)
        # folded = R@delta
        # folded.plot(ax=ax)
        # plt.show()
        return e, C, res

    def plot(self, ax: Axes | None = None, **kwargs) -> Axes:
        """ Plot the calibration

        Args:
            ax (Axes): The axes to plot on. If None, a new figure is created.
            **kwargs: Passed to the plot method of the response
        """
        if ax is None:
            _, ax = plt.subplots()

        self.spectrum.plot(ax=ax)
        self.fe_fit.plot(ax=ax)
        return ax

    @property
    def fwhm_fit(self) -> GaussFit:
        if self._fwhm_fit is None:
            raise ValueError("fwhm_fit is not set. Run fit_FWHM(...).")
        return self._fwhm_fit

    @fwhm_fit.setter
    def fwhm_fit(self, val: GaussFit) -> None:
        self._fwhm_fit = val
