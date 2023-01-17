from __future__ import annotations
from dataclasses import dataclass
from . import ResponseData
from .interpolation import Interpolation
from .interpolations import (EscapeInterpolator, EscapeInterpolation,
                             FEInterpolator, FEInterpolation,
                             FWHMInterpolator, FWHMInterpolation,
                             AnnihilationInterpolator, AnnihilationInterpolation,
                             LinearInterpolator, LinearInterpolation)
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from ..stubs import Axes
import numpy as np


@dataclass
class ResponseInterpolation:
    data: ResponseData
    FE: Interpolation
    SE: Interpolation
    DE: Interpolation
    AP: Interpolation
    Eff: Interpolation
    FWHM: Interpolation
    is_normalized: bool = False

    @staticmethod
    def from_data(data: ResponseData, scale: bool = True) -> ResponseInterpolation:
        if not data.is_fwhm_normalized:
            raise ValueError("The response data must have normalized FWHM before interpolations")
        if scale:
            data = data.scale()
        FE: FEInterpolation = FEInterpolator(data.FE).interpolate(order=9)
        SE: EscapeInterpolation = EscapeInterpolator(data.SE).interpolate()
        DE: EscapeInterpolation = EscapeInterpolator(data.DE).interpolate()
        AP: AnnihilationInterpolation = AnnihilationInterpolator(data.AP).interpolate()
        FWHM: FWHMInterpolation = FWHMInterpolator(data.FWHM).interpolate()
        Eff: LinearInterpolation = LinearInterpolator(data.Eff).interpolate()
        return ResponseInterpolation(data, FE, SE, DE, AP, Eff, FWHM)

    @property
    def E(self) -> np.ndarray:
        return self.data.E

    def sigma(self, E: np.ndarray) -> np.ndarray:
        return self.FWHM(E) / 2.355

    def plot(self, ax: Axes | None = None, **kwargs) -> Axes:
        if ax is None:
            _, ax = plt.subplots(3, 2, sharex=True, constrained_layout=True)
        ax = ax.flatten()
        if len(ax) < 5:
            raise ValueError("Need at least 5 axes")
        E = self.E
        self.FE.plot(ax=ax[0], **kwargs)
        self.SE.plot(ax=ax[1], **kwargs)
        self.DE.plot(ax=ax[2], **kwargs)
        self.AP.plot(ax=ax[3], **kwargs)
        self.Eff.plot(ax=ax[4], **kwargs)
        self.FWHM.plot(ax=ax[5], **kwargs)

        ax[0].set_title("FE")
        ax[1].set_title("SE")
        ax[2].set_title("DE")
        ax[3].set_title("AP")
        ax[4].set_title("Eff")
        ax[5].set_title("FWHM")

        for a in ax:
            a.set_xlabel("")
            a.set_ylabel("")

        figure = ax[0].figure
        figure.supxlabel("E [keV]")
        ylabel = 'Probability ' if self.is_normalized else 'Counts'
        figure.supylabel(ylabel)

        return ax

    def plot_residuals(self, ax: Axes | None = None, **kwargs) -> Axes:
        if ax is None:
            _, ax = plt.subplots(3, 2, sharex=True, constrained_layout=True)
        ax = ax.flatten()
        if len(ax) < 5:
            raise ValueError("Need at least 5 axes")
        E = self.E
        self.FE.plot_residuals(ax=ax[0], **kwargs)
        self.SE.plot_residuals(ax=ax[1], **kwargs)
        self.DE.plot_residuals(ax=ax[2], **kwargs)
        self.AP.plot_residuals(ax=ax[3], **kwargs)
        self.Eff.plot_residuals(ax=ax[4], **kwargs)
        self.FWHM.plot_residuals(ax=ax[5], **kwargs)

        for a in ax:
            a.set_xlabel("")
            a.set_ylabel("")

        figure = ax[0].figure
        figure.supxlabel("E [keV]")
        figure.supylabel(r"$\frac{y - \hat{y}}{y}$")


        ax[0].set_title("FE")
        ax[1].set_title("SE")
        ax[2].set_title("DE")
        ax[3].set_title("AP")
        ax[4].set_title("Eff")
        ax[5].set_title("FWHM")
        return ax
