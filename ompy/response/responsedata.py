from __future__ import annotations

from ..stubs import Pathlike, Axes, keV, Unitlike
from .. import Vector, Matrix, __full_version__
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from typing import overload, Literal
from .io import load, save

"""
TODO: Remove prefix/suffix and only use a glob pattern.
TODO: FWHM is useless. Make it "optional" for compatibility.
"""


ResponseFunctionName = Literal['Oscar2017', 'Oscar2020']
RESPONSE_FUNCTIONS = {'Oscar2017': Path(__file__).parent.parent.parent / "OCL_response_functions/oscar2017_scale1.15",
                      'Oscar2020': Path(__file__).parent.parent.parent / "OCL_response_functions/oscar2020/mama_export"}


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
    compton: Matrix
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

        T = compton*self.compton.sum(axis='observed') + FE*self.FE + SE*self.SE + DE*self.DE + AP*self.AP
        if inplace:
            self.FE *= FE / T
            self.SE *= SE / T
            self.DE *= DE / T
            self.AP *= AP / T
            self.compton *= compton / T[:, None]
            self.is_normalized = True
        else:
            return self.clone(FE=FE*self.FE / T, SE=SE*self.SE / T, DE=DE*self.DE / T,
                              AP=AP*self.AP / T,
                              compton=self.compton * compton / T[:, None],
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
        total_eff = self.sum() / self.Eff
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
            self.compton.values *= weight[:, None]
        else:
            compton = self.compton.clone()
            compton.values *= weight[:, None]
            return self.clone(FE=self.FE * weight, SE=self.SE * weight, DE=self.DE * weight,
                              AP=self.AP * weight,
                              compton=compton)


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
        (FE, SE, DE, AP, Eff, *FWHM), compton = load(path, **kwargs)
        return ResponseData(FE, SE, DE, AP, compton, Eff, FWHM=FWHM[0] if FWHM else None)

    @staticmethod
    def from_db(name: ResponseFunctionName) -> ResponseData:
        if name not in RESPONSE_FUNCTIONS.keys():
            raise ValueError(f"Response function {name} available. Available functions are {list(RESPONSE_FUNCTIONS.keys())}")
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
        if self.FWHM is not None:
            self.FWHM.plot(ax=ax[5], **kwargs)
        for i in range(6):
            ax[i].set_xlabel('')
            ax[i].set_ylabel('')
        self.compton.plot(ax=ax[6], **kwargs)
        self.sum().plot(ax=ax[7], **kwargs)
        ax[7].set_xlabel('')
        ax[7].set_ylabel('')
        fig = ax[0].figure
        fig.supxlabel('E [keV]')
        label = 'Probability' if self.is_normalized else 'Counts'
        fig.supylabel(label)
        return ax

    @property
    def E_true(self) -> np.ndarray:
        return self.compton.true

    @property
    def E_observed(self) -> np.ndarray:
        return self.compton.observed

    def sum(self, as_vector: bool = False) -> float | Vector:
        FE = self.FE.values
        SE = self.SE.values
        DE = self.DE.values
        AP = self.AP.values
        cmp = self.compton.sum(axis='observed').values
        if as_vector:
            return sum(FE + DE + SE + AP + cmp)
        else:
            return Vector(E=self.E_true, values=FE + SE + DE + AP + cmp, name='Total')

    def clone(self, FE=None, DE=None, SE=None, AP=None, compton=None, Eff=None,
              FWHM=None, is_normalized=None, is_fwhm_normalized=None) -> ResponseData:
        return ResponseData(FE=FE if FE is not None else self.FE,
                            SE=SE if SE is not None else self.SE,
                            DE=DE if DE is not None else self.DE,
                            AP=AP if AP is not None else self.AP,
                            compton=compton if compton is not None else self.compton,
                            Eff=Eff if Eff is not None else self.Eff,
                            FWHM=FWHM if FWHM is not None else self.FWHM,
                            is_normalized=is_normalized if is_normalized is not None else self.is_normalized,
                            is_fwhm_normalized=is_fwhm_normalized if is_fwhm_normalized is not None else self.is_fwhm_normalized)

    def save(self, path: Pathlike, **kwargs) -> None:
        return save(path, self, **kwargs)

    def __len__(self) -> int:
        return len(self.FE)