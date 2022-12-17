from __future__ import annotations
from ..stubs import Pathlike, Axes, keV, Unitlike
from .. import Vector
from pathlib import Path
from dataclasses import dataclass
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import overload, Literal
import warnings

ResponseFunctionName = Literal['Oscar2017', 'Oscar2020']
RESPONSE_FUNCTIONS = {'Oscar2017': Path(__file__).parent.parent.parent / "OCL_response_functions/oscar2017_scale1.15"}


@dataclass
class Components:
    FE: float = 1.0
    DE: float = 1.0
    SE: float = 1.0
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
    FWHM: Vector
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
                              #compton=fix_lengths([compton*cmp / T[i] for i, cmp in enumerate(self.compton)]),
                              compton=[compton*cmp / T[i] for i, cmp in enumerate(self.compton)],
                              is_normalized=True)

    @overload
    def normalize_FWHM(self, energy: Unitlike, fwhm: Unitlike, inplace: Literal[True] = ...) -> None: ...

    @overload
    def normalize_FWHM(self, energy: Unitlike, fwhm: Unitlike, inplace: Literal[False] = ...) -> ResponseData: ...

    def normalize_FWHM(self, energy: Unitlike, fwhm: Unitlike, inplace: bool = False) -> ResponseData | None:
        warnings.warn("Untested and not sanity-checked.")
        ratio = fwhm / energy
        if inplace:
            self.FWHM *= ratio
            self.is_fwhm_normalized = True
        else:
            return self.clone(FWHM=self.FWHM * ratio, is_fwhm_normalized=True)

    @staticmethod
    def from_file(path: Pathlike, name: str = 'resp.dat', prefix: str = 'cmp', suffix: str = '.m') -> ResponseData:
        path = Path(path)
        with open(path / name, 'r') as f:
            lines = f.readlines()

        number_of_lines = 0
        i = 0
        while (i := i+1) < len(lines):
            line = lines[i]
            # Number of lines. Some resps are misspelled
            if line.startswith("# Next: Num"):
                number_of_lines = int(lines[i+1])
                i += 1
                break

        df = pd.DataFrame([line.split() for line in lines[i+2:i+number_of_lines+3]],
                          columns=['E', 'FWHM', 'Eff', 'FE', 'DE', 'SE', 'AP'])
        df = df.astype(float)
        df['E'] = df['E'].astype(int)
        assert len(df) == number_of_lines
        E = df['E'].to_numpy()

        compton, _ = load_compton(path, prefix, suffix, Eg=E)
        FE = Vector(E=E, values=df['FE'].to_numpy())
        DE = Vector(E=E, values=df['DE'].to_numpy())
        SE = Vector(E=E, values=df['SE'].to_numpy())
        AP = Vector(E=E, values=df['AP'].to_numpy())
        Eff = Vector(E=E, values=df['Eff'].to_numpy())
        FWHM = Vector(E=E, values=df['FWHM'].to_numpy())
        return ResponseData(FE, SE, DE, AP, compton, Eff, FWHM)

    @staticmethod
    def from_db(name: ResponseFunctionName) -> ResponseData:
        return ResponseData.from_file(RESPONSE_FUNCTIONS[name])

    def plot(self, ax: Axes | None = None, **kwargs):
        # TODO Make me pretty
        if ax is None:
            _, ax = plt.subplots(4, 2)
        ax = ax.flatten()
        if len(ax) < 7:
            raise ValueError("Need at least 5 axes")
        self.FE.plot(ax=ax[0], **kwargs)
        self.SE.plot(ax=ax[1], **kwargs)
        self.DE.plot(ax=ax[2], **kwargs)
        self.AP.plot(ax=ax[3], **kwargs)
        self.Eff.plot(ax=ax[4], **kwargs)
        self.FWHM.plot(ax=ax[5], **kwargs)
        for i, cmp in enumerate(self.compton):
            cmp.plot(ax=ax[6], **kwargs)
        return ax

    @property
    def E(self) -> np.ndarray:
        return self.FE.E

    @property
    def E_compton(self) -> np.ndarray:
        return self.compton[0].E

    def compton_matrix(self) -> np.ndarray:
        raise NotImplementedError()
        return np.array([cmp.values for cmp in self.compton])

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


def load_compton(path: Pathlike, prefix: str = 'cmp', suffix: str = '.m',
                 Eg: list[int] = None) -> tuple[list[Vector], np.ndarray]:
    """ Load Compton response data from files.

    Parameters
    ----------
    path : Pathlike
        Path to directory containing the files.
    prefix : str, optional
        Prefix of the files, by default 'cmp'
    suffix : str, optional
        Suffix of the files, by default '.m'
    Eg : list[int], optional
        List of Eg values to load, by default None, in which case it loads all matching files.
    """
    path = Path(path)
    compton: list[Vector] = []
    # This energy is the true gamma energy. The compton vectors are indexed by the measured gamma energy
    E: list[int] = []
    for file in path.glob(f'{prefix}*{suffix}'):
        e = int(file.stem[len(prefix):])
        if Eg is not None:
            if e not in Eg:
                continue
        E.append(e)
        compton.append(Vector(path=file))
    if not len(compton):
        raise FileNotFoundError(f'No files found with prefix {prefix} and suffix {suffix}')
    if Eg is not None and set(E) != set(Eg):
        raise FileNotFoundError(f'Not all files found. Missing: {set(Eg) - set(E)}')
    # Sort and check for equal calibration
    E, compton = zip(*sorted(zip(E, compton), key=lambda x: x[0]))
    compton = list(compton)
    for i in range(1, len(compton)):
        assert compton[i].calibration() == compton[i-1].calibration()
    return compton, np.asarray(E)


def fix_lengths(compton: list):
    raise NotImplementedError()
    # Some compton vectors are longer than others. Rebin them to the shortest one
    minlength = min(len(v) for v in compton)
    mask = [len(v) == minlength for v in compton]
    # Find one good vector as a prototype
    prototype = compton[[i for i in range(len(compton)) if mask[i]][0]]
    # Rebin all vectors to the prototype
    wrong_length = [i for i in range(len(compton)) if not mask[i]]
    #prototype.summary()
    for i in wrong_length:
        N = sum(compton[i])
        compton[i] = compton[i].rebin_like(prototype)
        compton[i] = compton[i] * N / sum(compton[i])
