from __future__ import annotations
from dataclasses import dataclass
from .responsedata import ResponseData
from .interpolation import (Interpolation,
                            LinearInterpolator, LinearInterpolation,
                            SplineInterpolator, SplineInterpolation,
                            ISplineInterpolator, ISplineInterpolation,
                            Scalable
                            )
from .interpolations import (EscapeInterpolator, EscapeInterpolation,
                             FEInterpolator, FEInterpolation,
                             FWHMInterpolator, FWHMInterpolation,
                             AnnihilationInterpolator, AnnihilationInterpolation)
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from ..stubs import Axes, Pathlike, Unitlike
from .. import Vector, __full_version__, Index
from .responsepath import ResponseName, get_response_path
import numpy as np
from pathlib import Path
import json
from typing import Literal, overload, TypeAlias, Self
from ..version import warn_version

SIGMA_TO_FWHM = 2 * np.sqrt(2 * np.log(2))

# TODO The energy is mid, but loaded as left. Gives a small error.
# TODO Normalization is a bit tricky, as the raw data is noisy, so
# a normalization there will propagate (?) the noise. An initial counts
# interpolation would be better, but that requires C=1.0 in the p0 in GF3
# and introduces scaling errors which scale() may or may not correct.
# In addition, the points must be normalized and interpolated *again*
# after the initial interpolation. The error is small, and is therefore
# ignored in this version.

InterpolationScheme: TypeAlias = Literal['oscar', 'cactus', 'spline', 'interpolating spline']

@dataclass
class DiscreteInterpolation:
    FE: Interpolation
    SE: Interpolation
    DE: Interpolation
    AP: Interpolation
    Eff: Interpolation
    FWHM: Scalable | None = None  # TODO Replace with fwhm_function
    is_fwhm_normalized: bool = False

    @classmethod
    def from_data(cls, data: ResponseData, method: InterpolationScheme = 'oscar',
                  **kwargs) -> Self:
        """ Interpolates the discrete response data.
        Args:
            data: ResponseData: The discrete response data, obtained from
                loading either ompy or mama formatted data. Must be normalized.
            method: InterpolationScheme: The interpolation method to use.
                'oscar' uses the Oscar interpolation scheme based on radware,
                'cactus' uses the CACTUS interpolation scheme based on
                mama using linear interpolations.
        Returns:
            DiscreteInterpolation: The interpolated discrete response data.
        """
        match method.lower():
            case 'oscar':
                return cls.from_data_oscar(data, **kwargs)
            case 'cactus':
                return cls.from_data_cactus(data, **kwargs)
            case 'spline':
                return cls.from_data_spline(data, **kwargs)
            case 'interpolating spline':
                return cls.from_data_interpolating_spline(data, **kwargs)
            case _:
                raise ValueError(f"Unknown interpolation method: {method}.\n"
                                 f"Valid methods are 'oscar', 'cactus', 'spline'.")

    @classmethod
    def from_data_oscar(cls, data: ResponseData) -> Self:
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
        return cls(FE, SE, DE, AP, Eff, FWHM, is_fwhm_normalized=data.is_fwhm_normalized)

    @classmethod
    def from_data_cactus(cls, data: ResponseData) -> Self:
        if not data.is_normalized:
            raise ValueError("Data must be normalized before interpolation")

        FE: LinearInterpolation = LinearInterpolator(data.FE).interpolate()
        SE: LinearInterpolation = LinearInterpolator(data.SE).interpolate()
        DE: LinearInterpolation = LinearInterpolator(data.DE).interpolate()
        AP: LinearInterpolation = LinearInterpolator(data.AP).interpolate()
        FWHM = None
        if data.FWHM is not None:
            FWHM = LinearInterpolator(data.FWHM).interpolate()
            FWHM = Scalable(FWHM)
        Eff: LinearInterpolation = LinearInterpolator(data.Eff).interpolate()
        return cls(FE, SE, DE, AP, Eff, FWHM, is_fwhm_normalized=data.is_fwhm_normalized)

    @classmethod
    def from_data_spline(cls, data: ResponseData, **kwargs) -> Self:
        if not data.is_normalized:
            raise ValueError("Data must be normalized before interpolation")

        FE: SplineInterpolation = SplineInterpolator(data.FE).interpolate(**kwargs)
        SE: SplineInterpolation = SplineInterpolator(data.SE).interpolate(**kwargs)
        DE: SplineInterpolation = SplineInterpolator(data.DE).interpolate(**kwargs)
        AP: SplineInterpolation = SplineInterpolator(data.AP).interpolate(**kwargs)
        FWHM = None
        if data.FWHM is not None:
            FWHM = SplineInterpolator(data.FWHM).interpolate(**kwargs)
            FWHM = Scalable(FWHM)
        Eff: SplineInterpolation = SplineInterpolator(data.Eff).interpolate(**kwargs)
        return cls(FE, SE, DE, AP, Eff, FWHM, is_fwhm_normalized=data.is_fwhm_normalized)

    @classmethod
    def from_data_interpolating_spline(cls, data: ResponseData, **kwargs) -> Self:
        if not data.is_normalized:
            raise ValueError("Data must be normalized before interpolation")

        FE: ISplineInterpolation = ISplineInterpolator(data.FE).interpolate(**kwargs)
        SE: ISplineInterpolation = ISplineInterpolator(data.SE).interpolate(**kwargs)
        DE: ISplineInterpolation = ISplineInterpolator(data.DE).interpolate(**kwargs)
        AP: ISplineInterpolation = ISplineInterpolator(data.AP).interpolate(**kwargs)
        FWHM = None
        if data.FWHM is not None:
            FWHM = ISplineInterpolator(data.FWHM).interpolate(**kwargs)
            FWHM = Scalable(FWHM)
        Eff: ISplineInterpolation = ISplineInterpolator(data.Eff).interpolate(**kwargs)
        return cls(FE, SE, DE, AP, Eff, FWHM, is_fwhm_normalized=data.is_fwhm_normalized)

    @overload
    def normalize_FWHM(self, energy: Unitlike, fwhm: Unitlike, inplace: Literal[True] = ...) -> None: ...

    @overload
    def normalize_FWHM(self, energy: Unitlike, fwhm: Unitlike, inplace: Literal[False] = ...) -> Self: ...

    def normalize_FWHM(self, energy: Unitlike, fwhm: Unitlike, inplace: bool = False) -> Self | None:
        fwhm: float = self.FWHM.to_same_unit(fwhm)
        energy: float = self.FWHM.to_same_unit(energy)
        old = self.FWHM(energy)
        ratio = fwhm / old
        if inplace:
            self.FWHM.scale(ratio, inplace=True)
            self.is_fwhm_normalized = True
        else:
            return self.clone(FWHM=self.FWHM.scale(ratio), is_fwhm_normalized=True)

    def normalize_sigma(self, energy: Unitlike, sigma: Unitlike, inplace: bool = False) -> Self | None:
        sigma = self.FWHM.to_same_unit(sigma)
        return self.normalize_FWHM(energy, sigma * SIGMA_TO_FWHM, inplace=inplace)

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
    def from_path(cls, path: Pathlike) -> Self:
        path = Path(path)
        with (path / 'meta.json').open('r') as f:
            meta = json.load(f)
        version = meta.pop('version')
        warn_version(version)
        FE = Interpolation.from_path(path / 'FE')
        SE = Interpolation.from_path(path / 'SE')
        DE = Interpolation.from_path(path / 'DE')
        AP = Interpolation.from_path(path / 'AP')
        Eff = Interpolation.from_path(path / 'Eff')
        FWHM = Interpolation.from_path(path / 'FWHM')
        return cls(FE, SE, DE, AP, Eff, FWHM, **meta)

    @classmethod
    def from_response_path(cls, path: Pathlike) -> Self:
        path = Path(path) / 'interpolation'
        return cls.from_path(path)

    @classmethod
    def from_db(cls, name: ResponseName) -> Self:
        return cls.from_response_path(get_response_path(name))

    @property
    def E(self) -> np.ndarray:
        return self.FE.x

    @property
    def E_index(self) -> Index:
        return self.FE.points.X_index

    def sigma(self, E: np.ndarray) -> np.ndarray:
        return self.FWHM(E) / SIGMA_TO_FWHM

    def clone(self, FE: Interpolation | None = None,
              SE: Interpolation | None = None,
              DE: Interpolation | None = None,
              AP: Interpolation | None = None,
              Eff: Interpolation | None = None,
              FWHM: Interpolation | None = None,
              is_fwhm_normalized: bool | None = None) -> Self:
        return type(self)(
            FE or self.FE,
            SE or self.SE,
            DE or self.DE,
            AP or self.AP,
            Eff or self.Eff,
            FWHM or self.FWHM,
            is_fwhm_normalized if is_fwhm_normalized is not None else self.is_fwhm_normalized
        )

    def plot(self, ax: Axes | None = None, **kwargs) -> list[Axes]:
        if ax is None:
            _, ax = plt.subplots(3, 2, sharex=True, constrained_layout=True)
        ax: list[Axes] = ax.flatten()
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

        add_subplot_border(ax[4], 0.5, '#3155cb')
        add_subplot_border(ax[5], 0.5, '#3155cb')

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


def add_subplot_border(ax, width=1, color=None ):

    fig = ax.get_figure()

    # Convert bottom-left and top-right to display coordinates
    x0, y0 = ax.transAxes.transform((0, 0))
    x1, y1 = ax.transAxes.transform((1, 1))

    # Convert back to Axes coordinates
    x0, y0 = ax.transAxes.inverted().transform((x0, y0))
    x1, y1 = ax.transAxes.inverted().transform((x1, y1))

    rect = plt.Rectangle(
        (x0, y0), x1-x0, y1-y0,
        color=color,
        transform=ax.transAxes,
        zorder=-1,
        lw=2*width+1,
        fill=None,
    )
    fig.patches.append(rect)
