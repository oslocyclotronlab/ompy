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
from ..stubs import Axes, Pathlike, Unitlike
import numpy as np
from pathlib import Path
import json
from typing import Literal, overload


@dataclass
class DiscreteInterpolation:
    FE: Interpolation
    SE: Interpolation
    DE: Interpolation
    AP: Interpolation
    Eff: Interpolation
    FWHM: Interpolation
    is_normalized: bool = False
    is_fwhm_normalized: bool = False

    @staticmethod
    def from_data(data: ResponseData, scale: bool = True) -> DiscreteInterpolation:
        if not data.is_fwhm_normalized:
            raise ValueError("FWHM must be normalized before interpolation, at least until bug is fixed in FWHM creation")
        if scale:
            data = data.scale()
        FE: FEInterpolation = FEInterpolator(data.FE).interpolate(order=9)
        SE: EscapeInterpolation = EscapeInterpolator(data.SE).interpolate()
        DE: EscapeInterpolation = EscapeInterpolator(data.DE).interpolate()
        AP: AnnihilationInterpolation = AnnihilationInterpolator(data.AP).interpolate()
        FWHM: FWHMInterpolation = FWHMInterpolator(data.FWHM).interpolate()
        Eff: LinearInterpolation = LinearInterpolator(data.Eff).interpolate()
        return DiscreteInterpolation(FE, SE, DE, AP, Eff, FWHM, is_fwhm_normalized=True)

    def normalize(self, inplace: bool = False) -> DiscreteInterpolation | None:
        pass

    @overload
    def normalize_FWHM(self, energy: Unitlike, fwhm: Unitlike, inplace: Literal[True] = ...) -> None: ...

    @overload
    def normalize_FWHM(self, energy: Unitlike, fwhm: Unitlike, inplace: Literal[False] = ...) -> ResponseData: ...

    def normalize_FWHM(self, energy: Unitlike, fwhm: Unitlike, inplace: bool = False) -> ResponseData | None:
        raise NotImplementedError("FWHM Must be normalized before interpolation. This is a bug in the initial creation of FWHM")
        old = self.FWHM(energy)
        ratio = fwhm / old  * self.FWHM.E / energy

        if inplace:
            self.FWHM = fwhm
            self.is_fwhm_normalized = True
        else:
            return self.clone(FWHM=fwhm, is_fwhm_normalized=True)

    def save(self, path: Pathlike, exist_ok: bool = True) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=exist_ok)
        meta = {'is_normalized': self.is_normalized}
        with (path / 'meta.json').open('w') as f:
            json.dump(meta, f)
        self.FE.save(path / 'FE')
        self.SE.save(path / 'SE')
        self.DE.save(path / 'DE')
        self.AP.save(path / 'AP')
        self.Eff.save(path / 'Eff')
        self.FWHM.save(path / 'FWHM')

    @classmethod
    def from_path(cls, path: Pathlike) -> DiscreteInterpolation:
        path = Path(path)
        with (path / 'meta.json').open('r') as f:
            meta = json.load(f)
        FE = FEInterpolation.from_path(path / 'FE')
        SE = EscapeInterpolation.from_path(path / 'SE')
        DE = EscapeInterpolation.from_path(path / 'DE')
        AP = AnnihilationInterpolation.from_path(path / 'AP')
        Eff = LinearInterpolation.from_path(path / 'Eff')
        FWHM = FWHMInterpolation.from_path(path / 'FWHM')
        return cls(FE, SE, DE, AP, Eff, FWHM, meta['is_normalized'])

    @property
    def E(self) -> np.ndarray:
        return self.FE.x

    def sigma(self, E: np.ndarray) -> np.ndarray:
        return self.FWHM(E) / 2.355

    def clone(self, FE: Interpolation | None = None,
              SE: Interpolation | None = None,
              DE: Interpolation | None = None,
              AP: Interpolation | None = None,
              Eff: Interpolation | None = None,
              FWHM: Interpolation | None = None,
              is_normalized: bool | None = None,
              is_fwhm_normalized: bool | None = None) -> DiscreteInterpolation:
        return DiscreteInterpolation(
            FE or self.FE,
            SE or self.SE,
            DE or self.DE,
            AP or self.AP,
            Eff or self.Eff,
            FWHM or self.FWHM,
            is_normalized if is_normalized is not None else self.is_normalized,
            is_fwhm_normalized if is_fwhm_normalized is not None else self.is_fwhm_normalized
        )

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

    def __str__(self) -> str:
        s = f"Interpolation of discrete response structures.\n"
        s += f"Normalized: {self.is_normalized}\n"
        s += f"Normalized FWHM: {self.is_fwhm_normalized}\n"
        s += f"FE: {self.FE}\n"
        s += f"SE: {self.SE}\n"
        s += f"DE: {self.DE}\n"
        s += f"AP: {self.AP}\n"
        s += f"Eff: {self.Eff}\n"
        s += f"FWHM: {self.FWHM}\n"
        return s