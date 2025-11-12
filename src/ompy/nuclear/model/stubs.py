from typing import TypeAlias, Literal, Callable, Any

import numpy as np
import pandas as pd

# TODO: Replace with proper types when xarray is installed, but needs a refactor
Levels: TypeAlias = Any #xr.DataArray
TALYSPopulation: TypeAlias = Any #xr.DataArray
DiscreteLevels: TypeAlias = pd.DataFrame

SamplingType: TypeAlias = Literal['poisson', 'wigner']
SamplingFunction: TypeAlias = Callable[[float | np.ndarray], float | np.ndarray]
DensityFunction: TypeAlias = Callable[[np.ndarray | float, float, int], np.ndarray | float]
VectorizedFunction: TypeAlias = Callable[[np.ndarray | float], np.ndarray | float]
SpinDensityFunction: TypeAlias = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]
ParityDistributionFunction: TypeAlias = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]
GammaFunction: TypeAlias = VectorizedFunction


Parity: TypeAlias = Literal[0, 1]
Energy: TypeAlias = float
Spin: TypeAlias = float