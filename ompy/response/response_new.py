from __future__ import annotations

import json
from pathlib import Path
from typing import TypeAlias, Literal

from . import ResponseData, DiscreteInterpolation, interpolate_compton
from .. import Vector, Matrix, USE_GPU, __full_version__, to_index, Index
import warnings
from .numbalib import njit, prange
import numpy as np

from ..library import handle_rebin_arguments
from ..stubs import Pathlike, Unitlike

if USE_GPU:
    from . import interpolate_gpu
from collections import OrderedDict
import logging

LOG = logging.getLogger(__name__)
logging.captureWarnings(True)

# TODO Always make a high resolution compton to save, rebin and reuse?
# Is rebinning correct? We want to preserve probability, not counts.
# The R(e) gives the energy spectrum *at* e, but an experimental vector
# has a bin [e, e+de]. Should R^(e) be a weighted average over R(e) ... R(e+de)? YES!


class Response:
    def __init__(self, data: ResponseData,
                 interpolation: DiscreteInterpolation,
                 R: Matrix | None = None):
        self.data: ResponseData = data
        self.interpolation: DiscreteInterpolation = interpolation
        self.R: Matrix | None = R

    @classmethod
    def from_data(cls, data: ResponseData) -> Response:
        intp = DiscreteInterpolation.from_data(data.normalize())
        return cls(data, intp)

    def interpolate(self, E: Index | np.ndarray | None = None, normalize: float = True, **kwargs) -> Matrix:
        R: Matrix = self.interpolate_compton(E, **kwargs)
        FE, SE, DE, AP = self.interpolation.structures()
        emin = R.observed.min()
        j511 = R.index_observed(511)
        for i, e in enumerate(E):
            if e > emin:
                R.loc[i, e] += FE(e)
            if e - 511 > emin:
                R.loc[i, e - 511.0] += SE(e)
            if e - 2*511 > emin:
                R.loc[i, e - 511.0*2] += DE(e)
            if 511 > emin:
                R[i, j511] += AP(e)
        if normalize:
            R.normalize(axis='observed', inplace=True)
        self.R = R

        return R

    def best_energy_resolution(self) -> np.ndarray:
        E_true = self.data.compton.true_index
        E_observed = self.data.compton.observed_index
        E0 = max(E_true[0], E_observed[0])
        E1 = min(E_true[-1], E_observed[-1])
        assert E_observed.is_uniform()
        width = E_observed.step(0)
        x = np.arange(E0, E1, width)
        return to_index(x, edge='mid')

    def interpolate_compton(self, E: Index | np.ndarray | None = None, GPU: bool = True,
                            sigma: float = 6) -> Matrix:
        if not self.interpolation.is_fwhm_normalized:
            warnings.warn("Interpolating with non-normalized FWHM. Unclear whether this is reasonable.")
        E = self.best_energy_resolution() if E is None else E
        if not isinstance(E, Index):
            E = to_index(E, edge='mid')
        sigmafn = self.interpolation.sigma
        if USE_GPU and GPU:
            compton: Matrix = interpolate_gpu(self.data, E, sigmafn, sigma)
        else:
            compton: Matrix = interpolate_compton(self.data, E, sigmafn, sigma)
        return compton

    def specialize(self, bins: np.ndarray | None = None, factor: float | None = None,
                   binwidth: float | None = None, numbins: int | None = None, **kwargs) -> Matrix:
        bins = self.R.observed_index.handle_rebin_arguments(bins=bins, factor=factor, numbins=numbins,
                                                            binwidth=binwidth)
        return self.specialize_(bins, **kwargs)

    def specialize_(self, E: np.ndarray, **kwargs) -> Matrix:
        """ Rebins the response matrix to the requested energy grid. """
        if self.R is None:
            print("Interpolating...")
            R = self.interpolate(E, **kwargs)
        else:
            R = self.R
        R = R.rebin('both', bins=E)
        R.normalize(axis='observed', inplace=True)
        return R

    def specialize_like(self, other: Matrix | Vector, **kwargs) -> Matrix:
        match other:
            case Matrix():
                return self.specialize_(other.Y, **kwargs)
            case Vector():
                return self.specialize_(other.X, **kwargs)
            case _:
                raise ValueError(f"Can only specialize to Matrix or Vector, got {type(other)}")

    def gaussian(self, E: np.ndarray) -> Matrix:
        return gaussian_matrix(E, self.interpolation.sigma)

    def gaussian_like(self, other: Matrix | Vector) -> Matrix:
        match other:
            case Matrix():
                return self.gaussian(other.Y)
            case Vector():
                return self.gaussian(other.X)
            case _:
                raise ValueError(f"Expected Matrix or Vector, got {type(other)}")

    def clone(self, data: ResponseData | None = None, interpolation: DiscreteInterpolation | None = None,
              R: Matrix | None = None) -> Response:
        return Response(data=data or self.data, interpolation=interpolation or self.interpolation,
                        R=R or self.R)

    @classmethod
    def from_path(cls, path: Pathlike) -> Response:
        path = Path(path)
        with (path / 'meta.json').open() as f:
            meta = json.load(f)
        if meta['version'] != __full_version__:
            warnings.warn(f"Loading response from version {meta['version']} into version {__full_version__}.")
        data = ResponseData.from_path(path / 'data', format='numpy')
        interpolation = DiscreteInterpolation.from_path(path / 'interpolation')
        if (path / 'R.npz').exists():
            R = Matrix.from_path(path / 'R.npz')
        else:
            R = None
        return cls(data, interpolation, R)

    def save(self, path: Pathlike, exist_ok: bool = False) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=exist_ok)
        meta = {'version': __full_version__}
        with (path / 'meta.json').open('w') as f:
            json.dump(meta, f)
        self.data.save(path / 'data', exist_ok=exist_ok)
        self.interpolation.save(path / 'interpolation', exist_ok=exist_ok)
        if self.R is not None:
            self.R.save(path / 'R.npz', exist_ok=exist_ok)

    def normalize_FWHM(self, energy: Unitlike, fwhm: Unitlike, inplace: bool = False) -> Response | None:
        """ Normalizes the FWHM of the response to the requested value. """
        if inplace:
            self.interpolation.normalize_FWHM(energy, fwhm, inplace=inplace)
        else:
            return self.clone(interpolation=self.interpolation.normalize_FWHM(energy, fwhm, inplace=inplace))

    @classmethod
    def from_db(cls, name: ResponseName) -> Response:
        """ Loads a response from the database. """
        return cls.from_path(get_response_path(name))


ResponseName: TypeAlias = Literal['OSCAR2017', 'OSCAR2020']


def get_response_path(name: ResponseName) -> Path:
    """ Returns the path to the response in the database. """
    name = name.upper()
    if name not in ResponseName:
        raise ValueError(f"Unknown response name {name}. Must be one of {ResponseName}.")
    return Path(__file__).parent / 'data' / 'responses' / name

def gaussian_matrix(E: np.ndarray, sigmafn) -> Matrix:
    sigma = sigmafn(E)
    values = _gaussian_matrix(E, sigma)
    values = values / values.sum(axis=1)[:, None]
    return Matrix(true=E, observed=E, values=values, ylabel='Observed', xlabel='True',
                  edge='mid')


@njit(parallel=True)
def _gaussian_matrix(E, sigma):
    n = len(E)
    m = np.zeros((n, n))
    for i in prange(n):
        mu = E[i]
        sigma_ = sigma[i]
        for j in prange(n):
            m[i, j] = gaussian(E[j], mu, sigma_)
    return m

@njit
def gaussian(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma)**2) / (sigma * np.sqrt(2 * np.pi))