import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Union
from .ensemble import Ensemble
from .fit_rho_T import FitRhoT
from .matrix import Vector


class Extractor:
    def __init__(self, ensemble: Optional[Ensemble] = None):
        self.ensemble = ensemble
        self.path = Path('extraction_ensemble')
        self.num_fits = 10
        self.bin_width = 120
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
            rho_path = self.path / f'rho_{i}.tar'
            gsf_path = self.path / f'gsf_{i}.tar'
            if rho_path.exists() and gsf_path.exists():
                rhos.append(Vector(rho_path))
                gsfs.append(Vector(gsf_path))
            else:
                rho, gsf = self.fit(i)
                rho.save(self.path / f'rho_{i}.tar')
                gsf.save(self.path / f'gsf_{i}.tar')
                rhos.append(rho)
                gsf.append(gsf)

        self.rho = rhos
        self.gsf = gsf
        
    def fit(self, num: int):
        matrix = self.ensemble.get_firstgen(num)
        std = self.ensemble.std_firstgen
        fit = FitRhoT(matrix, std, self.bin_width, self.method, self.options)
        fit.fit()
        gsf = fit.T.values / (2*np.pi*(fit.T.E)**3)
        return fit.rho, gsf

    @property
    def path(self) -> Path:
        return self._path

    @path.setter
    def path(self, path: Union[str, Path]) -> None:
        if isinstance(path, str):
            self._path = Path(path)
        elif isinstance(path, Path):
            self._path = path
        else:
            raise TypeError(f"path must be str or Path, got {type(path)}")
