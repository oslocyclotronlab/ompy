from . import Response
from .. import Vector, zeros_like
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


class Calibrator:
    """ Calibrate the different components of the response

    The calibration is done by folding a delta peak with the response and
    comparing the output to a experimental spectrum.
    Steps:
        1. Fit the FE component. This calibrates FWHM.
        2. Compare Compton to the experimental spectrum.
        3. TODO: Fit SE, DE and 511

    Attributes:
        response (Response): The response to calibrate
        spectrum (Vector): The experimental spectrum
    """

    def __init__(self, R: Response, spectrum: Vector, fe_region: Mask,
                 ignore: Mask = np.ndarray([])):
        self.R = R
        self.spectrum = spectrum
        self.ignore = ignore
        self.fe_region = fe_region

    def calibrate(self):
        """ Calibrate the response

        Returns:
            components: The calibrated components
        """
        pass

    def fit_FE(self):
        """ Fit the FE component

        Returns:
            fe_fit: The fitted FE component
        """
        fe: Vector = self.spectrum.iloc[self.fe_region]
        emin = fe.E[0]
        emax = fe.E[-1]

        fe.plot()
        plt.show()

        def loss(p) -> float:
            e, C = p
            if e < emin or e > emax:
                return np.inf
            if C <= 0:
                return np.inf
            delta = zeros_like(self.spectrum)
            delta.loc[e] = C
            folded = self.R@delta
            return np.mean((folded[self.fe_region] - fe)**2)

        p0 = [(emin+emax)/2, fe.sum()]
        res = minimize(loss, p0)
        e, C = res.x


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