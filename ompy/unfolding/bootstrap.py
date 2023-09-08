from __future__ import annotations

from ompy.array.ufunc import unpack_to_vectors
from .unfolder import Unfolder
from .result import Result, RESULT_CLASSES
from .result1d import UnfoldedResult1D, has_cost
from .result2d import UnfoldedResult2D
from tqdm.autonotebook import tqdm
import numpy as np
from scipy.stats import poisson
from .. import Matrix, Vector
from ..version import FULLVERSION
from dataclasses import dataclass, field
from typing import Any, Literal, TypeAlias, overload, Self, TypeVar, Generic
from pathlib import Path
import warnings
import json
import matplotlib.pyplot as plt
from ..stubs import Axes, Lines, Plot1D, Plots1D, Unitlike
from ..array import AsymmetricVector
from numba import njit
from abc import ABC, abstractmethod
from ..helpers import make_ax

"""
TODO
- [ ] Measure bootstrap convergence
- [ ] Automatic coverage test
- [ ] Vector bootstrap
- [ ] Covariance
"""

VSpace: TypeAlias = Literal['mu', 'eta', 'nu']
MV = TypeVar('MV', bound=Matrix | Vector)


def bootstrap(res: Result, N: int, base: Literal['raw', 'nu'] = 'raw',
              **kwargs) -> Bootstrap:
    if base != 'raw':
        raise NotImplementedError("Only raw bootstrap implemented")
    match res:
        case UnfoldedResult1D():
            return bootstrap_vector(res, N, **kwargs)
        case UnfoldedResult2D():
            return bootstrap_matrix(res, N, **kwargs)
        case _:
            raise ValueError(f"Unknown result type {res.__class__.__name__}")

def bootstrap_vector_(res: UnfoldedResult1D, N: int, **kwargs) -> BootstrapVector:
    A_boots: list[Vector] = []
    unfolded_boot: list[Vector] = []
    best = res.best()
    R = (res.meta.space, res.R.T)
    G = res.G.T
    kwargs = res.meta.kwargs | kwargs
    unfolder = Unfolder.resolve_method(res.meta.method)(R=res.R.T, G=G.T)
    A = res.raw.copy()
    bg: Vector | None = None
    if res.background is not None:
        bg = res.background.copy()
        assert bg is not None
        bg.values = np.where(bg <= 0, 3, bg.values)
    bgs: list[Vector] = []
    # To avoid Poisson(0)
    A.values = np.where(A <= 0, 3, A.values)
    costs: list[np.ndarray] = []
    for i in tqdm(range(N)):
        A_boot: Vector = A.clone(values=np.random.poisson(A.values))
        if bg is not None:
            bg_boot = bg.clone(values=np.random.poisson(bg.values))
            bgs.append(bg_boot)
        else:
            bg_boot = None
        res_ = unfolder.unfold(A_boot, initial=best, R=R, G=G,
                                background=bg_boot,
                                disable_tqdm=True,
                                **kwargs)
        if False and has_cost(res_):
            fig, ax = plt.subplots()
            ax.plot(res_.cost)
            plt.show()
            if res_.cost[-1] > res_.cost[0]:
                raise RuntimeError("Unfolding diverged")
            costs.append(res_.cost)
        unfolded_boot.append(res_.best())
        A_boots.append(A_boot)
    bootstraped = BootstrapVector(base=res, bootstraps=A_boots, unfolded=unfolded_boot,  # type: ignore
                                  backgrounds=bgs, costs=costs,
                                  kwargs=kwargs)
    return bootstraped


def bootstrap_vector(res: UnfoldedResult1D, N: int, **kwargs) -> BootstrapVector:
    A_boots: list[Vector] = []
    unfolded_boot: list[Vector] = []
    best = res.best()
    R = (res.meta.space, res.R.T)
    G = res.G.T
    kwargs = res.meta.kwargs | kwargs
    unfolder = Unfolder.resolve_method(res.meta.method)(R=res.R.T, G=G.T)
    A = res.raw.copy()
    bg: Vector | None = None
    if res.background is not None:
        bg = res.background.copy()
        assert bg is not None
        bg.values = np.where(bg <= 0, 3, bg.values)
    bgs: list[Vector] = []
    # To avoid Poisson(0)
    A.values = np.where(A <= 0, 3, A.values)
    costs: list[np.ndarray] = []
    A_boots = [A.clone(values=np.random.poisson(A)) for _ in range(N)]
    bg_boots = None
    if bg is not None:
        bg_boots = [bg.clone(values=np.random.poisson(bg)) for _ in range(N)]
    res_: UnfoldedResult2D = unfolder.unfold(A_boots, initial=1.1*best, R=R, G=G, background=bg_boots)
    unfolded_boots: list[Vector] = unpack_to_vectors(res_.best())
    bootstraped = BootstrapVector(base=res, bootstraps=A_boots, unfolded=unfolded_boots,  # type: ignore
                                  backgrounds=bgs, costs=costs,
                                  kwargs=kwargs)
    return bootstraped

def sample(A: Matrix, N: int, mask: np.ndarray | None = None) -> list[Matrix]:
    """ Sample N matrices from A

    """
    if mask is None:
        mask = A.last_nonzeros()
    X = np.where(A.values <= 3, 3, A.values)
    X[mask] = 0
    As: list[Matrix] = []
    for i in range(N):
        A_i = A.clone(values=np.random.poisson(X))
        As.append(A_i)
    return As



def bootstrap_matrix(res: UnfoldedResult2D, N: int, **kwargs) -> BootstrapMatrix:
    """ Create Bootstrap ensemble of `A` using `res` method

    """
    best = res.best()
    R = (res.meta.space, res.R.T)
    G = res.G.T
    kwargs = res.meta.kwargs | kwargs
    unfolder = Unfolder.resolve_method(res.meta.method)(R=res.R.T, G=G.T)
    A = res.raw.copy()
    mask = A.last_nonzeros()
    A.values = np.where(A <= 0, 3, A.values)
    A[~mask] = 0
    #TODO Add background
    A_boots: list[Matrix] = []
    unfolded_boot: list[Matrix] = []
    costs: list[np.ndarray] = []
    best = 0.01 * best
    for i in tqdm(range(N)):
        A_boot = A.clone(values=np.random.poisson(A))
        unf = unfolder.unfold(A_boot, initial=best, R=R, G=G,
                              background=res.background,
                              disable_tqdm=True,
                              **kwargs)
        if unf.cost[-1] > unf.cost[0]:
            raise RuntimeError("Unfolding diverged")
        unfolded_boot.append(unf.best())
        A_boots.append(A_boot)
        costs.append(unf.cost)
    bootstraped = BootstrapMatrix(base=res, bootstraps=A_boots, unfolded=unfolded_boot,
                                  kwargs=kwargs, costs=costs)
    return bootstraped

T = TypeVar('T', bound=Matrix | Vector)

@dataclass(kw_only=True)
class Bootstrap(ABC, Generic[T]):
    base: Result[T]
    bootstraps: list[T]
    unfolded: list[T]
    costs: list[np.ndarray]
    backgrounds: list[T] | None = None
    kwargs: dict[str, Any] = field(default_factory=dict)
    ndim: int = field(init=False)
    _ubox: np.ndarray | None = None
    _etabox: np.ndarray | None = None
    _nubox: np.ndarray | None = None

    def save(self, path: str | Path, exist_ok: bool = False):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=exist_ok)
        metadata = dict(version=FULLVERSION, base=self.base.__class__.__name__,
                        ndim=self.ndim)
        with open(path / "metadata.json", "w") as f:
            json.dump(metadata, f)

        self.base.save(path / "base", exist_ok=True)

        for i, (A, unfolded) in enumerate(zip(self.bootstraps, self.unfolded)):
            A.save(path / f"boot_{i}.npz", exist_ok=True)
            unfolded.save(path / f"unfolded_{i}.npz", exist_ok=True)
            if self.backgrounds is not None:
                self.backgrounds[i].save(path / f"background_{i}.npz", exist_ok=True)
            np.save(path / f"cost_{i}.npy", self.costs[i])

        #self.base.save(path / "base", exists_ok=True)
        warnings.warn("Not saving kwargs")

    @overload
    @classmethod
    def _load(cls, path: Path, arraytype: type[Matrix],
               basearray: type[BootstrapMatrix],
               n: int | None = None) -> BootstrapMatrix: ...

    @overload
    @classmethod
    def _load(cls, path: Path, arraytype: type[Vector],
               basearray: type[BootstrapVector],
               n: int | None = None) -> BootstrapVector: ...
    @classmethod
    def _load(cls, path: Path, arraytype: type[Matrix] | type[Vector],
               basearray: type[BootstrapMatrix] | type[BootstrapVector],
               n: int | None = None) -> BootstrapMatrix | BootstrapVector:
        with open(path / "metadata.json", "r") as f:
            metadata = json.load(f)
        if metadata["version"] != FULLVERSION:
            warnings.warn(f"Version mismatch: {metadata['version']} != {FULLVERSION}")
        if metadata['ndim'] != basearray.ndim:
            raise ValueError(f"Wrong ndim: {metadata['ndim']} != {cls.ndim}")
        #TODO Load base
        result_cls: type[Result] = RESULT_CLASSES[metadata["base"]]
        base = result_cls.from_path(path / "base")  # type: ignore
        unfolded = []
        bootstraps = []
        costs = []
        backgrounds = []
        has_cost = True    # HACK: For backwards compatibility, temporary!
        for i in range(len(list(path.glob("boot_*.npz")))):
            unfolded.append(arraytype.from_path(path / f"unfolded_{i}.npz"))
            bootstraps.append(arraytype.from_path(path / f"boot_{i}.npz"))
            if (path / 'cost_i.npy').exists():
                cost = np.load(path / f"cost_{i}.npy")
                costs.append(cost)
            else:
                has_cost = False
            if (path / 'background_i.npz').exists():
                backgrounds.append(arraytype.from_path(path / f"background_{i}.npz"))
            if n is not None and i > n:
                break
        if not has_cost:
            print("No cost found. Probably wrong version or corrupted files")
        return basearray(base=base, bootstraps=bootstraps, unfolded=unfolded, costs=costs,
                         backgrounds=backgrounds if backgrounds else None)

    @classmethod
    @abstractmethod
    def from_path(cls, path: Path | str, n: int | None = None) -> Bootstrap: ...

    @property
    def G(self) -> Matrix:
        return self.base.G

    @property
    def R(self) -> Matrix:
        return self.base.R

    @property
    def raw(self) -> T:
        return self.base.raw

    @property
    def background(self) -> T | None:
        return self.base.background

    @property
    @abstractmethod
    def ubox(self) -> np.ndarray: ...

    @property
    @abstractmethod
    def etabox(self) -> np.ndarray: ...

    @property
    @abstractmethod
    def nubox(self) -> np.ndarray: ...

    def __len__(self) -> int:
        return len(self.unfolded)

@dataclass(kw_only=True)
class BootstrapVector(Bootstrap[Vector]):
    base: UnfoldedResult1D
    bootstraps: list[Vector]
    unfolded: list[Vector]
    kwargs: dict[str, Any] = field(default_factory=dict)
    ndim: Literal[1] = 1

    @classmethod
    def from_path(cls, path: str | Path, n: int | None = None) -> BootstrapVector:
        return cls.__load(Path(path), Vector, n=n)  # type: ignore


    def plot_unfolded(self, ax: Axes | None = None, **kwargs) -> Plots1D:
        ax = make_ax(ax)
        mu: Vector = self.base.best()
        lines: list[Lines] = []
        _, l = mu.plot(ax=ax, label=r'$\hat{\mu}$')
        lines.append(l)
        for i in range(len(self)):
            u: Vector = self.unfolded[i]
            _, l = u.plot(ax=ax, color='k', alpha=1/len(self))
            if i == 0:
                lines.append(l)
        return ax, lines

    def plot_backgrounds(self, ax: Axes | None = None, **kwargs) -> Plots1D:
        ax = make_ax(ax)
        if self.background is None:
            return ax, []
        _, l = self.background.plot(ax=ax, **kwargs)
        lines = [l]
        if self.backgrounds is None:
            return ax, lines
        color = kwargs.pop('color', 'k')
        alpha = kwargs.pop('alpha', 1/len(self))
        for i in range(len(self)):
            _, l = self.backgrounds[i].plot(ax=ax, color=color, alpha=alpha, **kwargs)
            if i == 0:
                lines.append(l)
        return ax, (l, lines[-1])

    def mu(self, alpha=0.05, summary = np.median) -> AsymmetricVector:
        mu = summary(self.ubox, axis=0)
        mu = self.base.raw.clone(values=mu)
        lower = np.percentile(self.ubox, 100*alpha/2, axis=0)
        upper = np.percentile(self.ubox, 100*(1-alpha/2), axis=0)
        mu = AsymmetricVector.from_CI(mu, lower=lower, upper=upper, clip=True)
        return mu

    def eta(self, alpha=0.05, summary = np.median) -> AsymmetricVector:
        eta = summary(self.etabox, axis=0)
        eta = self.base.raw.clone(values=eta)
        lower = np.percentile(self.etabox, 100*alpha/2, axis=0)
        upper = np.percentile(self.etabox, 100*(1-alpha/2), axis=0)
        eta = AsymmetricVector.from_CI(eta, lower=lower, upper=upper, clip=True)
        return eta

    def nu(self, alpha=0.05, summary = np.median) -> AsymmetricVector:
        nubox = self.nubox
        nu = summary(nubox, axis=0)
        nu = self.base.raw.clone(values=nu)
        lower = np.percentile(nubox, 100*alpha/2, axis=0)
        upper = np.percentile(nubox, 100*(1-alpha/2), axis=0)
        nu = AsymmetricVector.from_CI(nu, lower=lower, upper=upper, clip=True)
        return nu

    def bg(self, alpha=0.05, summary = np.median) -> AsymmetricVector | None:
        if self.backgrounds is None:
            return None
        box = np.stack([b.values for b in self.backgrounds])
        b = summary(box, axis=0)
        b = self.raw.clone(values=b)
        lower = np.percentile(box, 100*alpha/2, axis=0)
        upper = np.percentile(box, 100*(1-alpha/2), axis=0)
        bg = AsymmetricVector.from_CI(b, lower=lower, upper=upper, clip=True)
        return bg

    @property
    def ubox(self) -> np.ndarray:
        if self._ubox is None:
            self._ubox = np.stack(self.unfolded)  # type: ignore
        return self._ubox

    @property
    def etabox(self) -> np.ndarray:
        if self._etabox is None:
            self._etabox = self.ubox@self.G.values
        return self._etabox

    @property
    def nubox(self) -> np.ndarray:
        if self._nubox is None:
            self._nubox = (self.R.values@(self.ubox.T)).T
        return self._nubox

@dataclass(kw_only=True)
class BootstrapMatrix(Bootstrap[Matrix]):
    base: UnfoldedResult2D
    bootstraps: list[Matrix]
    unfolded: list[Matrix]
    kwargs: dict[str, Any] = field(default_factory=dict)
    ndim: Literal[2] = 2

    @classmethod
    def from_path(cls, path: str | Path, n: int | None = None) -> BootstrapMatrix:
        return Bootstrap._load(Path(path), Matrix, BootstrapMatrix, n=n)

    def plot_unfolded(self, Ex: float | int, ax: Axes | None = None) -> Plots1D:
        ax = make_ax(ax)
        mu: Vector = self.base.best().loc[Ex, :]
        j = mu.last_nonzero()
        mu = mu.iloc[:j]
        lines: list[Lines] = []
        _, l = mu.plot(ax=ax, label=r'$\hat{\mu}$')
        lines.append(l)
        for i in range(len(self)):
            u: Vector = self.unfolded[i].loc[Ex, :j]
            _, l = u.plot(ax=ax, color='k', alpha=0.01)
            if i == 0:
                lines.append(l)
        return ax, lines

    def plot_folded(self, Ex: float | int, ax: Axes | None = None) -> Plots1D:
        ax = make_ax(ax)
        nu: Vector = self.base.best_folded().loc[Ex, :]
        j = nu.last_nonzero()
        nu = nu.iloc[:j]
        lines: list[Lines] = []
        _, l = nu.plot(ax=ax, label=r'$\hat{\mu}$')
        lines.append(l)
        for i in range(len(self)):
            u: Vector = ((self.base.R@(self.unfolded[i]).T).T).loc[Ex, :j]
            _, l = u.plot(ax=ax, color='k', alpha=0.01)
            if i == 0:
                lines.append(l)
        return ax, lines

    def plot_eta(self, Ex: float | int, ax: Axes | None = None, **kwargs) -> Plots1D:
        ax = make_ax(ax)
        eta: Vector = (self.base.best()@self.G).loc[Ex, :]
        j = eta.last_nonzero()
        eta = eta.iloc[:j]
        lines: list[Lines] = []
        N = len(self)
        alpha = kwargs.pop('alpha', 1/(N*0.5))
        for i in range(N):
            u: Vector = ((self.base.G@(self.unfolded[i]).T).T).loc[Ex, :j]
            _, l = u.plot(ax=ax, color='k', alpha=alpha, label=r'$\hat{\eta}_\mathrm{boot}$')
            if i == 0:
                lines.append(l)
        _, l = eta.plot(ax=ax, label=r'$\hat{\eta}$')
        lines.append(l)
        return ax, lines

    def eta_vec(self, Ex: float | int, alpha=0.05, summary = np.median) -> AsymmetricVector:
        i = self.base.raw.X_index.index_expression(Ex, strict=False)
        j = last_nonzero(self.etabox[:, i, :])
        eta = summary(self.etabox[:, i, :j], axis=0)
        eta = self.base.raw.iloc[i, :j].clone(values=eta)
        lower = np.percentile(self.etabox[:, i, :j], 100*alpha/2, axis=0)
        upper = np.percentile(self.etabox[:, i, :j], 100*(1-alpha/2), axis=0)
        eta = AsymmetricVector.from_CI(eta, lower=lower, upper=upper, clip=True)
        return eta

    def nu_vec(self, Ex: float | int, alpha=0.05, summary = np.median) -> AsymmetricVector:
        i = self.base.raw.X_index.index_expression(Ex, strict=False)
        nubox = self.nubox
        j = last_nonzero(nubox[:, i, :])
        nu = summary(nubox[:, i, :j], axis=0)
        nu = self.base.raw.iloc[i, :j].clone(values=nu)
        lower = np.percentile(nubox[:, i, :j], 100*alpha/2, axis=0)
        upper = np.percentile(nubox[:, i, :j], 100*(1-alpha/2), axis=0)
        nu = AsymmetricVector.from_CI(nu, lower=lower, upper=upper, clip=True)
        return nu

    def eta_mat(self, summary = np.median) -> Matrix:
        eta = summary(self.etabox, axis=0)
        eta = self.base.raw.clone(values=eta)
        return eta

    def nu_mat(self, summary = np.median) -> Matrix:
        nu = summary(self.nubox, axis=0)
        nu = self.base.raw.clone(values=nu)
        return nu

    @property
    def ubox(self) -> np.ndarray:
        if self._ubox is None:
            self._ubox = np.stack(self.unfolded)  # type: ignore
        return self._ubox

    @property
    def etabox(self) -> np.ndarray:
        if self._etabox is None:
            self._etabox = mul(self.ubox, self.G.values)
        return self._etabox

    @property
    def nubox(self) -> np.ndarray:
        if self._nubox is None:
            self._nubox = mul(self.ubox, self.R.values)
        return self._nubox

def poisson_ci(lambdas, alpha=0.05):
    lower_bounds = poisson.ppf(alpha/2, lambdas)
    upper_bounds = poisson.ppf(1 - alpha/2, lambdas)
    return lower_bounds, upper_bounds

def coverage(data_points, lower_bounds, upper_bounds):
    coverage_count = np.sum((data_points >= lower_bounds) & (data_points <= upper_bounds))
    coverage_rate = coverage_count / len(data_points)
    return coverage_rate

def coverage_ci(data, expectation, **kwargs):
    lower, upper = poisson_ci(expectation, **kwargs)
    return coverage(data, lower, upper)

def coverage_of(res: Result, **kwargs):
    return coverage_ci(res.raw.values, res.best_folded().values, **kwargs)

@njit
def last_nonzero(box: np.ndarray) -> int:
    S = np.sum(box, axis=0)
    for i in range(len(S)-1, -1, -1):
        if S[i] > 0:
            return i
    return 0


#@njit
def mul(X, A):
    Y = np.zeros_like(X)
    for i in range(X.shape[0]):
        Y[i, :, :] = (A@(X[i, :, :].T)).T
    return np.ascontiguousarray(Y)

def bootstrap_CI(b: Bootstrap, N: int, alpha=0.05,
                 space: VSpace = 'nu') -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the bootstrap confidence interval for a poisson sample.

    Parameters
    ----------
    b : Bootstrap
        The bootstrap distribution.
    N : int
        The number of bootstrap samples.
    alpha : float, optional
        The confidence level, by default 0.05.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The lower and upper bounds of the confidence interval.
    """
    X = resolve_box_from_space(b, space)
    m = X.shape[0]
    M = N * m
    samples = np.empty((M, *X.shape[1:]))
    for i in tqdm(range(N)):
        for j in range(m):
            samples[i*m + j, :, :] = np.random.poisson(X[j, :, :])
    lower = np.percentile(samples, 100*alpha/2, axis=0)
    upper = np.percentile(samples, 100*(1-alpha/2), axis=0)
    return lower, upper

def bootstrap_CI_at(b: Bootstrap, Ex: float | int, N: int, alpha=0.05,
                    space: VSpace = 'nu') -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the bootstrap confidence interval for a poisson sample.

    Parameters
    ----------
    b : Bootstrap
        The bootstrap distribution.
    N : int
        The number of bootstrap samples.
    alpha : float, optional
        The confidence level, by default 0.05.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The lower and upper bounds of the confidence interval.
    """
    i = b.base.raw.X_index.index_expression(Ex, strict=False)
    X = resolve_box_from_space(b, space)
    X = np.ascontiguousarray(X[:, i, :])
    j = last_nonzero(X)
    X = X[:, :j]
    @njit
    def fn(X):
        m = X.shape[0]
        M = N * m
        samples = np.empty((M, X.shape[1]))
        for i in range(N):
            for j in range(m):
                for k in range(X.shape[1]):
                    samples[i*m + j, k] = np.random.poisson(X[j, k])
        return samples
    samples = fn(X)
    lower = np.percentile(samples, 100*alpha/2, axis=0)
    upper = np.percentile(samples, 100*(1-alpha/2), axis=0)
    return lower, upper

def resolve_box_from_space(boot: Bootstrap, space: VSpace) -> np.ndarray:
    match space:
        case 'mu':
            return boot.ubox
        case 'eta':
            return boot.etabox
        case 'nu':
            return boot.nubox
        case _:
            raise ValueError(f'Unknown space: {space}')
