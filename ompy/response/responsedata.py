from __future__ import annotations
from ..stubs import Pathlike, Axes, keV, Unitlike
from .. import Vector, Matrix
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from typing import overload, Literal
import warnings
from .io import load

"""
TODO: Remove prefix/suffix and only use a glob pattern.
TODO: FWHM is useless. Make it "optional" for compatibility.
"""


ResponseFunctionName = Literal['Oscar2017', 'Oscar2020']
RESPONSE_FUNCTIONS = {'Oscar2017': Path(__file__).parent.parent.parent / "OCL_response_functions/oscar2017_scale1.15",
                      'Oscar2020': Path(__file__).parent.parent.parent / "OCL_response_functions/oscar2020"}


@dataclass
class Components:
    FE: float = 1.0
    SE: float = 1.0
    DE: float = 1.0
    AP: float = 1.0
    compton: float = 1.0

    def to_dict(self) -> dict[str, float]:
        return {'compton': self.compton, 'FE': self.FE,
                'DE': self.DE, 'SE': self.SE,
                'AP': self.AP}

    def __iter__(self):
        return iter([self.compton, self.FE, self.DE, self.SE, self.AP])


@dataclass
class ResponseData:
    FE: Vector
    SE: Vector
    DE: Vector
    AP: Vector
    compton: list[Vector]
    Eff: Vector
    FWHM: Vector | None = None
    is_normalized: bool = False
    is_fwhm_normalized: bool = False

    @overload
    def normalize_components(self, components: Components, inplace: Literal[True] = ...) -> None: ...
    @overload
    def normalize_components(self, components: Components, inplace: Literal[False] = ...) -> ResponseData: ...

    def normalize_components(self, components: Components, inplace: Literal[True] | Literal[False] = False) -> ResponseData | None:
        return self.normalize(**components.to_dict(), inplace=inplace)

    @overload
    def normalize(self, compton: float = 1.0, FE: float = 1.0, SE: float = 1.0,
                  DE: float = 1.0, AP: float = 1.0, inplace: Literal[True] = ...) -> None: ...

    @overload
    def normalize(self, compton: float = 1.0, FE: float = 1.0, SE: float = 1.0,
                  DE: float = 1.0, AP: float = 1.0, inplace: Literal[False] = ...) -> ResponseData: ...

    def normalize(self, compton: float = 1.0, FE: float = 1.0, SE: float = 1.0,
                  DE: float = 1.0, AP: float = 1.0, inplace: bool = False) -> ResponseData | None:
        """

        The normalization is not obvious, but is straightforward to derive. We require that
        the sum of the probabilities p_i (not counts, c_i) weighted by custom weights w_i is equal to 1.
        In the derivation most normalization factors disappear, and we are left with
                                    p_i = w_i * c_i / sum(w_i * c_i)
        """
        warnings.warn("You should not be normalizing the raw counts. Normalize the interpolations instead.")
        T = compton*self.compton_sum() + FE*self.FE + SE*self.SE + DE*self.DE + AP*self.AP
        if inplace:
            self.FE *= FE / T
            self.SE *= SE / T
            self.DE *= DE / T
            self.AP *= AP / T
            for i, cmp in enumerate(self.compton):
                self.compton[i] *= compton / T
            self.is_normalized = True
        else:
            return self.clone(FE=FE*self.FE / T, SE=SE*self.SE / T, DE=DE*self.DE / T,
                              AP=AP*self.AP / T,
                              compton=[compton*cmp / T[i] for i, cmp in enumerate(self.compton)],
                              is_normalized=True)

    def scale(self, inplace: bool = False) -> ResponseData | None:
        """ Scale the elements to correct for simulations with different number of runs.

        Algorithm:
            Weight each element by the median of the [total weighted by the efficiency]
        Args:
            inplace: If True, the data is scaled in place. If False, a new ResponseData object is returned.

        Returns:
            If inplace is True, None is returned. If inplace is False, a new ResponseData object is returned.

        """
        total = self.FE + self.SE + self.DE + self.AP + self.compton_sum()
        total_eff = total / self.Eff
        # Scale all counts to the median. Not obvious if this is correct,
        # as I would have assumed all elements in total_eff corresponding to the
        # same simulations should be equal. they are not, but close.
        median = np.median(total_eff)
        weight = median / total_eff
        if inplace:
            self.FE *= weight
            self.SE *= weight
            self.DE *= weight
            self.AP *= weight
            for i in range(len(self.compton)):
                self.compton[i] *= weight[i]
        else:
            return self.clone(FE=self.FE * weight, SE=self.SE * weight, DE=self.DE * weight,
                              AP=self.AP * weight,
                              compton=[cmp * weight[i] for i, cmp in enumerate(self.compton)])


    @overload
    def normalize_FWHM(self, energy: Unitlike, fwhm: Unitlike, inplace: Literal[True] = ...) -> None: ...

    @overload
    def normalize_FWHM(self, energy: Unitlike, fwhm: Unitlike, inplace: Literal[False] = ...) -> ResponseData: ...

    def normalize_FWHM(self, energy: Unitlike, fwhm: Unitlike, inplace: bool = False) -> ResponseData | None:
        #warnings.warn("You should not be normalizing the raw counts. Normalize the interpolations instead.")
        if self.FWHM is None:
            raise ValueError("No FWHM data available.")
        old = self.FWHM.loc[energy]
        ratio = fwhm / old  * self.FWHM.E / energy
        if inplace:
            self.FWHM *= ratio
            self.is_fwhm_normalized = True
        else:
            return self.clone(FWHM=self.FWHM * ratio, is_fwhm_normalized=True)

    @staticmethod
    def from_path(path: Pathlike, **kwargs) -> ResponseData:
        (FE, SE, DE, AP, Eff, *FWHM), (compton, _) = load(path, **kwargs)
        return ResponseData(FE, SE, DE, AP, compton, Eff, FWHM)

    @staticmethod
    def from_db(name: ResponseFunctionName) -> ResponseData:
        return ResponseData.from_path(RESPONSE_FUNCTIONS[name])

    def plot(self, ax: Axes | None = None, **kwargs):
        if ax is None:
            _, ax = plt.subplots(4, 2, constrained_layout=True,
                                 sharex=True)
        ax = ax.flatten()
        if len(ax) < 7:
            raise ValueError("Need at least 5 axes")
        self.FE.plot(ax=ax[0], **kwargs)
        self.SE.plot(ax=ax[1], **kwargs)
        self.DE.plot(ax=ax[2], **kwargs)
        self.AP.plot(ax=ax[3], **kwargs)
        self.Eff.plot(ax=ax[4], **kwargs)
        self.FWHM.plot(ax=ax[5], **kwargs)
        titles = ['FE', 'SE', 'DE', 'AP', 'Eff', 'FWHM']
        for i, title in enumerate(titles):
            ax[i].set_title(title)
            ax[i].set_xlabel('')
            ax[i].set_ylabel('')
        compton = self.compton_matrix()
        compton.plot(ax=ax[6], **kwargs)
        ax[7].plot(self.E, self.sum(axis=1))
        fig = ax[0].figure
        fig.supxlabel('E [keV]')
        fig.supylabel('Counts')
        return ax

    @property
    def E(self) -> np.ndarray:
        return self.FE.E

    @property
    def E_observed(self) -> np.ndarray:
        i = np.argmax([len(cmp) for cmp in self.compton])
        return self.compton[i].E

    def compton_matrix(self) -> Matrix:
        max_length = max(len(cmp) for cmp in self.compton)
        mat = np.zeros((len(self.compton), max_length))
        for i, cmp in enumerate(self.compton):
            mat[i, :len(cmp)] = cmp.values
        return Matrix(Ex=self.E, Eg=self.E_observed, values=mat,
                      xlabel=r"Observed $\gamma$", ylabel=r"True $\gamma$")

    def compton_sum(self) -> Vector:
        return Vector(E=self.E, values=np.asarray([cmp.values.sum() for cmp in self.compton]))

    def sum(self, axis: int | None = None) -> float:
        FE = self.FE.values
        DE = self.DE.values
        SE = self.SE.values
        AP = self.AP.values
        cmp = self.compton_sum().values
        if axis is None:
            return sum(FE + DE + SE + AP + cmp)
        else:
            return FE + DE + SE + AP + cmp

    def clone(self, FE=None, DE=None, SE=None, AP=None, compton=None, Eff=None,
              FWHM=None, is_normalized=None, is_fwhm_normalized=None) -> ResponseData:
        return ResponseData(FE=FE if FE is not None else self.FE,
                            DE=DE if DE is not None else self.DE,
                            SE=SE if SE is not None else self.SE,
                            AP=AP if AP is not None else self.AP,
                            compton=compton if compton is not None else self.compton,
                            Eff=Eff if Eff is not None else self.Eff,
                            FWHM=FWHM if FWHM is not None else self.FWHM,
                            is_normalized=is_normalized if is_normalized is not None else self.is_normalized,
                            is_fwhm_normalized=is_fwhm_normalized if is_fwhm_normalized is not None else self.is_fwhm_normalized)

    def __len__(self) -> int:
        return len(self.FE)