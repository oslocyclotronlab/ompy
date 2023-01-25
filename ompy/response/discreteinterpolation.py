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
from .. import Vector, __full_version__
import numpy as np
from pathlib import Path
import json
from typing import Literal, overload
import warnings

# TODO Normalization is a bit tricky, as the raw data is noisy, so
# a normalization there will propagate (?) the noise. An initial counts
# interpolation would be better, but that requires C=1.0 in the p0 in GF3
# and introduces scaling errors which scale() may or may not correct.
# In addition, the points must be normalized and interpolated *again*
# after the initial interpolation. The error is small, and is therefore
# ignored in this version.

@dataclass
class DiscreteInterpolation:
    FE: Interpolation
    SE: Interpolation
    DE: Interpolation
    AP: Interpolation
    Eff: Interpolation
    FWHM: Interpolation | None = None  # TODO Replace with fwhm_function
    is_fwhm_normalized: bool = False

    @staticmethod
    def from_data(data: ResponseData) -> DiscreteInterpolation:
        if not data.is_normalized:
            raise ValueError("Data must be normalized before interpolation")
        FE: FEInterpolation = FEInterpolator(data.FE).interpolate(order=9)
        SE: EscapeInterpolation = EscapeInterpolator(data.SE).interpolate()
        DE: EscapeInterpolation = EscapeInterpolator(data.DE).interpolate()
        AP: AnnihilationInterpolation = AnnihilationInterpolator(data.AP).interpolate()
        FWHM = None
        if data.FWHM is not None:
            FWHM = FWHMInterpolator(data.FWHM).interpolate()
        Eff: LinearInterpolation = LinearInterpolator(data.Eff).interpolate()
        return DiscreteInterpolation(FE, SE, DE, AP, Eff, FWHM, is_fwhm_normalized=data.is_fwhm_normalized)


    @overload
    def normalize_FWHM(self, energy: Unitlike, fwhm: Unitlike, inplace: Literal[True] = ...) -> None: ...

    @overload
    def normalize_FWHM(self, energy: Unitlike, fwhm: Unitlike, inplace: Literal[False] = ...) -> DiscreteInterpolation: ...

    def normalize_FWHM(self, energy: Unitlike, fwhm: Unitlike, inplace: bool = False) -> DiscreteInterpolation | None:
        old = self.FWHM(energy)
        ratio = fwhm / old
        if inplace:
            self.FWHM.scale(ratio, inplace=True)
            self.is_fwhm_normalized = True
        else:
            return self.clone(FWHM=self.FWHM.scale(ratio), is_fwhm_normalized=True)

    def save(self, path: Pathlike, exist_ok: bool = True) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=exist_ok)
        meta = {'version': __full_version__}
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
        version = meta.pop('version')
        if version != __full_version__:
            warnings.warn(f"Version mismatch: {version} != {__full_version__}")
        FE = FEInterpolation.from_path(path / 'FE')
        SE = EscapeInterpolation.from_path(path / 'SE')
        DE = EscapeInterpolation.from_path(path / 'DE')
        AP = AnnihilationInterpolation.from_path(path / 'AP')
        Eff = LinearInterpolation.from_path(path / 'Eff')
        FWHM = FWHMInterpolation.from_path(path / 'FWHM')
        return cls(FE, SE, DE, AP, Eff, FWHM, **meta)

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
              is_fwhm_normalized: bool | None = None) -> DiscreteInterpolation:
        return DiscreteInterpolation(
            FE or self.FE,
            SE or self.SE,
            DE or self.DE,
            AP or self.AP,
            Eff or self.Eff,
            FWHM or self.FWHM,
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
        if self.FWHM is not None:
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
        figure.supylabel('Probability')

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

    def structures(self) -> tuple[Interpolation, ...]:
        return self.FE, self.SE, self.DE, self.AP

    def __str__(self) -> str:
        s = f"Interpolation of discrete response structures.\n"
        s += f"Normalized FWHM: {self.is_fwhm_normalized}\n"
        s += f"FE: {self.FE}\n"
        s += f"SE: {self.SE}\n"
        s += f"DE: {self.DE}\n"
        s += f"AP: {self.AP}\n"
        s += f"Eff: {self.Eff}\n"
        s += f"FWHM: {self.FWHM}\n"
        return s