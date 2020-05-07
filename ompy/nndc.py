import pandas as pd
from typing import Any, Optional
from .vector import Vector
import numpy as np
from pathlib import Path
from scipy.ndimage import gaussian_filter1d


class NNDC:
    URL = "https://www.nndc.bnl.gov/nudat2/getdatasetClassic.jsp?nucleus={}&unc=nds"

    def __init__(self, nucleus: Optional[str],
                 url: Optional[str] = None) -> None:

        if nucleus is not None:
            url = self.URL.format(nucleus)
        elif url is None:
            raise ValueError("Provide complete url or the nucleus")

        self.url = url
        self.fetch(url)

    def fetch(self, url: str) -> None:
        meta, references, data, *comments = pd.read_html(url)
        data.columns = data.iloc[0]
        data = data.drop(0)[:-2]
        self.meta = meta
        self.references = references
        self.data = data
        self.comments = comments
        self.Sn = float(meta.iloc[0, 1].split('=')[1].split('keV')[0].strip())

    def levels(self) -> np.ndarray:
        def to_number(x: str) -> float:
            # remove annotations
            return float(x.split(' ')[0])

        E = self.data.iloc[:, 0].apply(to_number).to_numpy()
        return E

    def level_hist(self, energy: np.ndarray,
                   resolution: float = 0.1,
                   smooth: bool = False) -> Vector:
        energies = self.levels()
        energies /= 1e3  # convert to MeV

        binsize = energy[1] - energy[0]
        bin_edges = np.append(energy, energy[-1] + binsize)
        bin_edges -= binsize / 2

        hist, _ = np.histogram(energies, bins=bin_edges)
        hist = hist.astype(float) / binsize  # convert to levels/MeV

        if smooth and resolution > 0:
            resolution /= 2.3548
            hist = gaussian_filter1d(hist, sigma=resolution / binsize)

        vec = Vector(E=energy, values=hist, units='MeV')
        return vec

    def save(self, path: Path) -> None:
        self.levels().save(path)
