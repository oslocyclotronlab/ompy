from . import Response
from .. import Vector, zeros_like
from ..peakselect import fit_gauss, GaussFit
from ..stubs import Unitlike, ArrayBool, Axes
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

Mask = ArrayBool


@dataclass
class Components:
    compton: float = 1.0
    FE: float = 1.0
    DE: float = 1.0
    SE: float = 1.0
    c511: float = 1.0

    def to_dict(self) -> dict[str, float]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


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
        response (Response): The response to calibrate
        spectrum (Vector): The experimental spectrum
    """

    def __init__(self, R: Response, spectrum: Vector,
                 ignore: Mask = np.ndarray([])):
        self.R = R
        self.spectrum = spectrum
        self.ignore = ignore
        self._fwhm_fit: GaussFit | None = None

    def calibrate(self):
        """ Calibrate the response

        Returns:
            components: The calibrated components
        """
        pass

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

    def fit_FE(self, region: Mask):
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
        R = self.R.interpolate(self.spectrum.E, self.fwhm_fit.fwhm).T

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
        print(f"p0: {p0}")
        res = minimize(loss, p0, method='Nelder-Mead')
        e, C = res.x
        print(res)
        ax=self.spectrum.plot()
        delta = zeros_like(self.spectrum)
        delta.loc[e] = C
        delta.plot(ax=ax)
        folded = R@delta
        folded.plot(ax=ax)
        plt.show()

    def calibrate_compton(self, ignore: Mask = np.array([])):
        pass

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
