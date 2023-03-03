from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import TypeAlias, Literal

import numpy as np

from . import ResponseData, DiscreteInterpolation, interpolate_compton
from .numbalib import njit, prange
from .. import Vector, Matrix, USE_GPU, __full_version__, to_index, Index
from ..stubs import Pathlike, Unitlike

if USE_GPU:
    from . import interpolate_gpu
import logging

LOG = logging.getLogger(__name__)
logging.captureWarnings(True)


# TODO Always make a high resolution compton to save, rebin and reuse?
# Is rebinning correct? We want to preserve probability, not counts.
# The R(e) gives the energy spectrum *at* e, but an experimental vector
# has a bin [e, e+de]. Should R^(e) be a weighted average over R(e) ... R(e+de)? YES!
# TODO Save only compton, not entire response. Recreate respons upon loading.


class Response:
    def __init__(self, data: ResponseData,
                 interpolation: DiscreteInterpolation,
                 R: Matrix | None = None,
                 compton: Matrix | None = None):
        self.data: ResponseData = data
        self.interpolation: DiscreteInterpolation = interpolation
        self.R: Matrix | None = R
        self.compton: Matrix | None = compton
        self.compton_special: Matrix | None = None

    @classmethod
    def from_data(cls, data: ResponseData) -> Response:
        intp = DiscreteInterpolation.from_data(data.normalize())
        return cls(data, intp)

    def interpolate(self, E: Index | np.ndarray | None = None, normalize: float = True,
                    force: bool = False, **kwargs) -> Matrix:
        if self.compton is None or force:
            self.compton = self.interpolate_compton(E, **kwargs)
        if self.R is None or force:
            self.R = self.add_structures(self.compton, normalize=normalize)
        return self.R

    def add_structures(self, C: Matrix | None = None, normalize: bool = True) -> Matrix:
        if C is None:
            if self.compton is None:
                raise ValueError("Compton matrix must be set or given as argument before adding structures")
            C = self.compton
        R: Matrix = C.clone(copy=True, name='Response')
        FE, SE, DE, AP = self.interpolation.structures()
        emin = R.observed.min()
        j511 = R.index_observed(511)
        for i, e in enumerate(R.true):
            if e > emin:
                R.loc[i, e] += FE(e)
            if e - 511 > emin:
                R.loc[i, e - 511.0] += SE(e)
            if e - 2 * 511 > emin:
                R.loc[i, e - 511.0 * 2] += DE(e)
            if e > 1022:
                R[i, j511] += AP(e)
        if normalize:
            R.normalize(axis='observed', inplace=True)
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
                   width: float | None = None, numbins: int | None = None, **kwargs) -> Matrix:
        bins = self.R.true_index.handle_rebin_arguments(bins=bins, factor=factor, numbins=numbins,
                                                        binwidth=width)
        return self.specialize_(bins, **kwargs)

    def specialize_(self, E: Index, **kwargs) -> Matrix:
        """ Rebins the response matrix to the requested energy grid. """
        if self.R is None:
            raise ValueError("Response matrix not yet interpolated. Use `.interpolate()` first.")
        if self.R.true_index.leftmost > E.leftmost:
            t = self.R.true_index
            raise ValueError(("Requested energy grid is too low. "
                              f"The lowest energy in the response is {t.leftmost:.2f} {t.unit:~}. "
                              f"The requested energy grid starts at {E.leftmost:.2f} {E.unit:~}. "
                              f"The energy grid must be truncated at index {E.index(t.leftmost)+1}."))
        R = self.R.rebin('true', bins=E).to_left()  # TODO Left?? Should the E edge be preserved?
        R = R.rebin('observed', bins=E)

        c = self.compton.rebin('true', bins=E).to_left()
        c = c.rebin('observed', bins=c.true)
        c.values /= R.sum(axis='observed')
        self.compton_special = c

        R.normalize(axis='observed', inplace=True)

        return R

    def specialize_like(self, other: Matrix | Vector, **kwargs) -> Matrix:
        match other:
            case Matrix():
                return self.specialize_(other.Y_index, **kwargs)
            case Vector():
                return self.specialize_(other.X_index, **kwargs)
            case _:
                raise ValueError(f"Can only specialize to Matrix or Vector, got {type(other)}")

    def gaussian(self, E: np.ndarray | Index) -> Matrix:
        """ Returns a matrix with the same shape as `E`.

        If E is an array, the edge is assumed to be mid. If E is an Index, the gaussians
        are evaluated at the midpoints of the bins but transformed back to the edge of
        the index.
        """
        if isinstance(E, Index):
            E_ = E.to_mid().bins
        else:
            E_ = E
        G: Matrix = gaussian_matrix(E_, self.interpolation.sigma)
        if isinstance(E, Index) and E.is_left():
            G = G.to_left()
        G.name = 'Detector resolution'
        return G

    def gaussian_like(self, other: Matrix | Vector) -> Matrix:
        match other:
            case Matrix():
                return self.gaussian(other.Y_index)
            case Vector():
                return self.gaussian(other.X_index)
            case _:
                raise ValueError(f"Expected Matrix or Vector, got {type(other)}")

    def clone(self, data: ResponseData | None = None, interpolation: DiscreteInterpolation | None = None,
              R: Matrix | None = None, compton: Matrix | None = None) -> Response:
        return Response(data=data or self.data, interpolation=interpolation or self.interpolation,
                        R=R or self.R, compton=compton or self.compton)

    @classmethod
    def from_path(cls, path: Pathlike) -> Response:
        path = Path(path)
        with (path / 'meta.json').open() as f:
            meta = json.load(f)
        if meta['version'] != __full_version__:
            warnings.warn(f"Loading response from version {meta['version']} into version {__full_version__}.")
        data = ResponseData.from_path(path / 'data', format='numpy')
        interpolation = DiscreteInterpolation.from_path(path / 'interpolation')
        compton = None
        if (path / 'compton.npz').exists():
            compton = Matrix.from_path(path / 'compton.npz')
        R = None
        if (path / 'R.npz').exists():
            R = Matrix.from_path(path / 'R.npz')
        return cls(data, interpolation, R=R, compton=compton)

    def save(self, path: Pathlike, exist_ok: bool = False, save_R: bool = False) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=exist_ok)
        meta = {'version': __full_version__}
        with (path / 'meta.json').open('w') as f:
            json.dump(meta, f)
        self.data.save(path / 'data', exist_ok=exist_ok)
        self.interpolation.save(path / 'interpolation', exist_ok=exist_ok)
        if self.compton is not None:
            self.compton.save(path / 'compton.npz', exist_ok=exist_ok)
        if save_R:
            if self.R is None:
                raise ValueError("Response matrix not yet interpolated. Use `.interpolate()` first.")
            self.R.save(path / 'R.npz', exist_ok=exist_ok)

    def normalize_FWHM(self, energy: Unitlike, fwhm: Unitlike, inplace: bool = False) -> Response | None:
        """ Normalizes the FWHM of the response to the requested value. """
        if inplace:
            self.interpolation.normalize_FWHM(energy, fwhm, inplace=inplace)
        else:
            return self.clone(interpolation=self.interpolation.normalize_FWHM(energy, fwhm, inplace=inplace))

    def normalize_sigma(self, energy: Unitlike, sigma: Unitlike, inplace: bool = False) -> Response | None:
        """ Normalizes the sigma of the response to the requested value. """
        if inplace:
            self.interpolation.normalize_sigma(energy, sigma, inplace=inplace)
        else:
            return self.clone(interpolation=self.interpolation.normalize_sigma(energy, sigma, inplace=inplace))

    @classmethod
    def from_db(cls, name: ResponseName) -> Response:
        """ Loads a response from the database. """
        obj = cls.from_path(get_response_path(name))
        obj.interpolate()
        return obj


ResponseName: TypeAlias = Literal['OSCAR2017', 'OSCAR2020']


def get_response_path(name: ResponseName) -> Path:
    """ Returns the path to the response in the database. """
    name = name.upper()
    if name not in ResponseName.__args__:
        raise ValueError(f"Unknown response name {name}. Must be one of {ResponseName}.")
    return Path(__file__).parent.parent.parent / 'data' / 'response' / name


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
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
