from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import TypeAlias, Literal

import numpy as np

from . import ResponseData, DiscreteInterpolation, interpolate_compton, Components
from .numbalib import njit, prange
from .. import Vector, Matrix, USE_GPU, __full_version__, to_index, Index, zeros_like
from ..stubs import Pathlike, Unitlike
from .responsepath import ResponseName, get_response_path

if USE_GPU:
    from . import interpolate_gpu
import logging

LOG = logging.getLogger(__name__)
logging.captureWarnings(True)

#TODO
# [x] Save and load components
# [ ] Interpolate compton down to 0
# [x] Is the response specialized correctly? We *know* the components at all E
# [ ] Refactor the specialization
# Note! The gaussian matrix will not be equal to "manual" gaussians, as the mus are taken from
# the midbin-value, ensuring perfect symmetric distributions.


class Response:
    def __init__(self, data: ResponseData,
                 interpolation: DiscreteInterpolation,
                 R: Matrix | None = None,
                 compton: Matrix | None = None,
                 components: Components = Components(),
                 copy: bool = False):
        self.data: ResponseData = data if not copy else data.copy()
        self.interpolation: DiscreteInterpolation = interpolation if not copy else interpolation.copy()
        self.compton: Matrix | None = compton if not copy else compton.copy() if compton is not None else None
        self.compton_special: Matrix | None = None
        self.components = components

    @classmethod
    def from_data(cls, data: ResponseData) -> Response:
        intp = DiscreteInterpolation.from_data(data.normalize())
        return cls(data, intp)

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
        self.compton = compton
        return compton

    def specialize(self, bins: np.ndarray | None = None, factor: float | None = None,
                   width: float | None = None, numbins: int | None = None, **kwargs) -> Matrix:
        bins = self.compton.true_index.handle_rebin_arguments(bins=bins, factor=factor, numbins=numbins,
                                                        binwidth=width)
        return self.specialize_(bins, **kwargs)

    def specialize_(self, *, E: Index | None = None, compton: Matrix | None = None, weights: Components | None = None, normalize: bool = True) -> Matrix:
        """ Rebins the response matrix to the requested energy grid. """
        if compton is None:
            if self.compton is None:
                raise ValueError("Compton matrix must be set or given as argument before adding structures. Use `interpolate_compton` first")
            compton = self.compton
        if self.compton.true_index.leftmost > E.leftmost:
            t = self.compton.true_index
            raise ValueError(("Requested energy grid is too low. "
                              f"The lowest energy in the response is {t.leftmost:.2f} {t.unit:~}. "
                              f"The requested energy grid starts at {E.leftmost:.2f} {E.unit:~}. "
                              f"The energy grid must be truncated at index {E.index(t.leftmost)+1}."))
        if weights is None:
            weights = self.components

        # We preserve area as we want a mean value, not the sum
        R = compton.rebin('true', bins=E, preserve='area').to_left()
        R.rebin('observed', bins=R.true, inplace=True)
        R = R.to_left()
        R *= weights.compton
        R.name = "Response"

        # The functions need to be evaluated over the values within the bin
        # to account for their behaviour across the bin. The resolution
        # is the same as that of the raw structure data (go finer?)
        if E is not None:
            dE_intp = np.max(R.true_index.steps())
            dE = np.min(self.interpolation.E_index.steps())
            N = int(np.ceil(dE_intp / dE))

            def mean(fn, e):
                return np.mean(fn(np.linspace(e, e + dE_intp, N)))
        else:
            def mean(fn, e):
                return fn(e)

        FE, SE, DE, AP = self.interpolation.structures()
        emin = R.observed_index.leftmost
        j511 = R.index_observed(511)
        for i, e in enumerate(R.true):
            #if e > emin:
            R.loc[i, e] += mean(FE, e) * weights.FE
            if e - 511 > emin:
                R.loc[i, e - 511.0] += mean(SE, e) * weights.SE
            if e - 2 * 511 > emin:
                R.loc[i, e - 511.0 * 2] += mean(DE, e) * weights.DE
            if e > 1022:
                R[i, j511] += mean(AP, e) * weights.AP
        if normalize:
            R.normalize(axis='observed', inplace=True)
        return R

    def specialize_like(self, other: Matrix | Vector, **kwargs) -> Matrix:
        match other:
            case Matrix():
                return self.specialize_(E=other.Y_index, **kwargs)
            case Vector():
                return self.specialize_(E=other.X_index, **kwargs)
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
              compton: Matrix | None = None, components: Components | None = None,
              copy: bool = False) -> Response:
        return Response(data=data or self.data, interpolation=interpolation or self.interpolation,
                        compton=compton or self.compton, components=components or self.components,
                        copy=copy)

    def copy(self, **kwargs):
        return self.clone(copy=True, **kwargs)

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
        components = Components()
        if (path / 'components').exists():
            components = Components.from_path(path / 'components')
        return cls(data, interpolation, compton=compton, components=components)

    def save(self, path: Pathlike, exist_ok: bool = False, save_R: bool = False) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=exist_ok)
        meta = {'version': __full_version__}
        with (path / 'meta.json').open('w') as f:
            json.dump(meta, f)
        self.data.save(path / 'data', exist_ok=exist_ok)
        self.interpolation.save(path / 'interpolation', exist_ok=exist_ok)
        self.components.save(path / 'components', exist_ok=exist_ok)
        if self.compton is not None:
            self.compton.save(path / 'compton.npz', exist_ok=exist_ok)

    def component_matrices(self, *, E: Index | None, compton: Matrix | None = None, weights: Components | None = None,
                           normalize: bool = True) -> dict[str, Matrix]:
        if weights is None:
            weights = self.components
        if self.compton is None and compton is None:
            raise ValueError("Compton matrix must be set before adding structures")
        if compton is None:
            compton = self.compton.copy()
        if E is not None:
            if self.compton.true_index.leftmost > E.leftmost:
                t = self.compton.true_index
                raise ValueError(("Requested energy grid is too low. "
                                  f"The lowest energy in the response is {t.leftmost:.2f} {t.unit:~}. "
                                  f"The requested energy grid starts at {E.leftmost:.2f} {E.unit:~}. "
                                  f"The energy grid must be truncated at index {E.index(t.leftmost) + 1}."))

            compton = compton.rebin('true', bins=E, preserve='area').to_left()
            compton.rebin('observed', bins=compton.true, inplace=True)
        compton = compton.to_left()
        compton *= weights.compton
        FE, SE, DE, AP = self.interpolation.structures()
        FEm = zeros_like(compton, name='FE')
        SEm = zeros_like(compton, name='SE')
        DEm = zeros_like(compton, name='DE')
        APm = zeros_like(compton, name='AP')

        # The functions need to be evaluated over the values within the bin
        # to account for their behaviour across the bin. The resolution
        # is the same as that of the raw structure data (go finer?)
        if E is not None:
            dE_intp = np.max(compton.true_index.steps())
            dE = np.min(self.interpolation.E_index.steps())
            N = int(np.ceil(dE_intp/dE))
            def mean(fn, e):
                return np.mean(fn(np.linspace(e, e+dE_intp, N)))
        else:
            def mean(fn, e):
                return fn(e)

        emin = compton.observed_index.leftmost
        j511 = APm.index_observed(511.0)
        for i, e in enumerate(compton.true):
            #if e > emin:
            FEm.loc[i, e] += mean(FE, e) * weights.FE
            if e - 511 > emin:
                SEm.loc[i, e - 511.0] += mean(SE, e) * weights.SE
            if e - 2 * 511 > emin:
                DEm.loc[i, e - 511.0 * 2] += mean(DE, e) * weights.DE
            if e > 1022:
                APm[i, j511] += mean(AP, e) * weights.AP

        total = compton + FEm + SEm + DEm + APm
        total.name = 'total'
        if normalize:
            T = total.sum(axis=1)
            T[T == 0] = 1
            def norm(x):
                x.values /= T[:, np.newaxis]
            [norm(x) for x in [total, compton, FEm, SEm, DEm, APm]]
        return {'total': total, 'compton': compton, 'FE': FEm, 'SE': SEm, 'DE': DEm, 'AP': APm}

    def component_matrices_like(self, other: Matrix | Vector | Index,
                                weights: Components | None = None) -> dict[str, Matrix]:
        match other:
            case Matrix():
                return self.component_matrices(E=other.Y_index, weights=weights)
            case Vector():
                return self.component_matrices(E=other.X_index, weights=weights)
            case _:
                raise ValueError(f"Can only specialize to Matrix or Vector, got {type(other)}")

    def fold_componentwise(self, other: Matrix | Vector,
                           weights: Components | None = None) -> dict[str, Matrix] | dict[str, Vector]:
        components = self.component_matrices_like(other, weights=weights)
        return {k: comp.T @ other for k, comp in components.items()}

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
        return obj


def gaussian_matrix(E: np.ndarray, sigmafn) -> Matrix:
    # TODO HACK Add 2 bins to account for the interpolation error.
    sigma = sigmafn(E + 2*(E[1] - E[0]))
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
