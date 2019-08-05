import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Union, Any, Tuple
from .ensemble import Ensemble
from .fit_rho_T import FitRhoT
from .matrix import Vector


class Extractor:
    def __init__(self, ensemble: Optional[Ensemble] = None):
        self.ensemble = ensemble
        self._path = Path('extraction_ensemble')
        self.num_fits = 10 if ensemble is None else ensemble.size
        self.bin_width = 120
        self.regenerate = False
        self.method = 'Powell'
        self.options = {'disp': True, 'ftol': 1e-3, 'maxfev': None}

    def extract_gsf_nld(self, Ex_min: float, Ex_max: float, Eg_min: float,
                        ensemble: Optional[Ensemble] = None):
        if ensemble is not None:
            self.ensemble = ensemble
        elif self.ensemble is None:
            raise ValueError("ensemble must be given")

        assert self.ensemble.size >= self.num_fits, "Ensemble is too small"

        rhos = []
        gsfs = []
        for i in range(self.num_fits):
            rho_path = self.save_path / f'rho_{i}.tar'
            gsf_path = self.save_path / f'gsf_{i}.tar'
            if rho_path.exists() and gsf_path.exists() and not self.regenerate:
                rhos.append(Vector(path=rho_path))
                gsfs.append(Vector(path=gsf_path))
            else:
                rho, gsf = self.fit(i, Ex_min, Ex_max, Eg_min)
                rho.save(rho_path)
                gsf.save(gsf_path)
                rhos.append(rho)
                gsfs.append(gsf)

        self.rho = rhos
        self.gsf = gsfs

    def fit(self, num: int, Ex_min: float,
            Ex_max: float, Eg_min: float) -> Tuple[Vector, Vector]:
        assert self.ensemble is not None
        matrix = self.ensemble.get_firstgen(num)
        std = self.ensemble.std_firstgen
        fit = FitRhoT(matrix, std, self.bin_width,
                      Ex_min, Ex_max, Eg_min, self.method, self.options)
        fit.fit()
        gsf = fit.T.values / (2*np.pi*(fit.T.E)**3)
        return fit.rho, Vector(gsf, fit.T.E)

    def plot(self, ax: Optional[Any] = None, scale: str = 'log'):
        if ax is None:
            fig, ax = plt.subplots(1, 2)
        for rho, gsf in zip(self.rho, self.gsf):
            ax[0].plot(rho.E, rho.values, color='k', alpha=1/self.num_fits)
            ax[1].plot(gsf.E, gsf.values, color='k', alpha=1/self.num_fits)

        ax[0].errorbar(rho.E, self.rho_mean(), yerr=self.rho_std(), fmt='o', ms=1)
        ax[1].errorbar(gsf.E, self.gsf_mean(), yerr=self.gsf_std(), fmt='o', ms=1)

        ax[0].set_title("Level density")
        ax[1].set_title("γSF")
        if scale == 'log':
            ax[0].set_yscale("log")
            ax[1].set_yscale("log")
        return ax

    def rho_mean(self) -> np.ndarray:
        return np.mean([rho.values for rho in self.rho], axis=0)

    def gsf_mean(self) -> np.ndarray:
        return np.mean([gsf.values for gsf in self.gsf], axis=0)

    def rho_std(self) -> np.ndarray:
        return np.std([rho.values for rho in self.rho], axis=0)

    def gsf_std(self) -> np.ndarray:
        return np.std([gsf.values for gsf in self.gsf], axis=0)

    @property
    def save_path(self) -> Path:
        return self._path

    @save_path.setter
    def path(self, path: Union[str, Path]) -> None:
        if isinstance(path, str):
            self._path = Path(path)
        elif isinstance(path, Path):
            self._path = path
        else:
            raise TypeError(f"path must be str or Path, got {type(path)}")
