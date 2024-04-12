from __future__ import annotations

from ompy.array.ufunc import unpack_to_vectors
from .unfolder import Unfolder
from .result import Result, RESULT_CLASSES
from .result1d import UnfoldedResult1D, has_cost
from .result2d import UnfoldedResult2D
from tqdm.autonotebook import tqdm
import numpy as np
from scipy.stats import poisson
from .. import Matrix, Vector, ArrayList, H5PY_AVAILABLE, JAX_AVAILABLE
from ..version import FULLVERSION
from dataclasses import dataclass, field
from typing import Any, Literal, TypeAlias, overload, Self, TypeVar, Generic
from pathlib import Path
import warnings
import json
import matplotlib.pyplot as plt
from ..stubs import Axes, Lines, Plot1D, Plots1D, Unitlike
from ..array import AsymmetricVector
from ..numbalib import njit, prange
from abc import ABC, abstractmethod
from ..helpers import make_ax, maybe_set, readable_time, bytes_to_readable
from scipy.stats import norm
import logging
import time
import re

LOG = logging.getLogger(__name__)

if H5PY_AVAILABLE:
    import h5py

if JAX_AVAILABLE:
    import jax.numpy as jnp
    from jax.scipy.stats import norm as jax_norm
    from jax import device_put

"""
TODO
- [?] Measure bootstrap convergence
- [ ] Automatic coverage test
- [x] Vector bootstrap
- [ ] Covariance
- [ ] The bootstrap uses *a lot* of memory. Can we reduce it?
      Remove the _boxes and use custom methods to broadcast over the lists instead
      Sparse matrices?
- [ ] Use float16 or some other dtype Jax likes
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
                                  **kwargs)
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
    std = np.maximum(best/2, np.median(best)/2)
    mean = np.maximum(best, np.mean(best))
    initials = [best.clone(values=np.random.uniform(9, 5*mean)) for i in range(N)]
    #for n in range(N):
        #initial = best + np.random.normal(0, std)
        # Redistribute negative values
        #initial[initial < 0] = np.random.poisson(np.abs(initial[initial < 0]))
        #initials.append(initial)
    res_: UnfoldedResult2D = unfolder.unfold(A_boots, initial=initials, R=R, G=G,
                                             background=bg_boots, **kwargs)
    cost = None
    if has_cost(res_):
        cost = res_.cost
    unfolded_boots: list[Vector] = unpack_to_vectors(res_.best())
    bootstraped = BootstrapVector(base=res, bootstraps=A_boots, unfolded=unfolded_boots,  # type: ignore
                                  backgrounds=bgs, costs=cost, initials=initials,
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
    best = res.best().astype('float32')
    R = (res.meta.space, res.R.T.astype('float32'))
    G = res.G.T.astype('float32')
    G_ex = res.G_ex.astype('float32')
    kwargs = res.meta.kwargs | kwargs
    unfolder = Unfolder.resolve_method(res.meta.method)(R=res.R.T.astype('float32'), G=G.T.astype('float32'))
    A = res.raw.copy().astype('float32')
    mask = A.last_nonzeros()
    A.values = np.where(A <= 0, 3, A.values)
    A[~mask] = 0
    #TODO Add background
    A_boots: list[Matrix] = []
    unfolded_boot: list[Matrix] = []
    costs: list[np.ndarray] = []
    initials: list[Matrix] = []
    backgrounds: list[Matrix] = []
    best = best
    disable_tqdm = kwargs.pop('disable_tqdm', True)
    background = res.background
    if background is not None:
        background = background.astype('float32')
        background.values = np.where(background <= 0, 3, background.values)
        background[~mask] = 0

    for i in tqdm(range(N)):
        initial = best.clone(values=np.random.uniform(1, 3*best))
        A_boot = A.clone(values=np.random.poisson(A))
        if background is not None:
            bg = background.clone(values=np.random.poisson(background))
            backgrounds.append(bg)
        else:
            bg = None
        unf = unfolder.unfold(A_boot, initial=initial, R=R, G=G, G_ex=G_ex,
                              background=background,
                              disable_tqdm=disable_tqdm,
                              **kwargs)
        if False and unf.cost[-1] > unf.cost[0]:
            raise RuntimeError("Unfolding diverged")
        unfolded_boot.append(unf.best())
        A_boots.append(A_boot)

        # We do a rescaling to make the cost fit into fewer bytes to take up less space
        costs.append((unf.cost/unf.cost.max()).astype('float16'))
        initials.append(initial)
    bootstraped = BootstrapMatrix(base=res, bootstraps=A_boots, unfolded=unfolded_boot,
                                  kwargs=kwargs, costs=costs, initials=initials,
                                  backgrounds=backgrounds if background is not None else None)
    return bootstraped

T = TypeVar('T', bound=Matrix | Vector)
SaveFormat = Literal['hdf5', 'npz']

@dataclass(kw_only=True)
class Bootstrap(ABC, Generic[T]):
    base: Result[T]
    bootstraps: list[T]
    unfolded: list[T]
    initials: list[T]
    costs: np.ndarray | list[np.ndarray]
    backgrounds: list[T] | None = None
    kwargs: dict[str, Any] = field(default_factory=dict)
    ndim: int = field(init=False)
    _ubox: np.ndarray | None = None
    _etabox: np.ndarray | None = None
    _nubox: np.ndarray | None = None

    def save(self, path: str | Path, exist_ok: bool = False, format: SaveFormat = 'hdf5',
             **kwargs) -> None:
        format_ = format.lower()
        LOG.debug(f"Saving bootstrap to {path} in {format_} format(?)")
        start = time.time()
        if format_ == 'hdf5':
            self.save_hdf5(path, exist_ok=exist_ok, **kwargs)
        elif format_ == 'npz':
            self.save_npz(path, exist_ok=exist_ok)
        else:
            raise ValueError(f"Expected format {SaveFormat}, not {format}")
        LOG.debug(f"Saved bootstrap to {path} in {format_} format in {readable_time(time.time() - start)}")

    def save_hdf5(self, path: str | Path, exist_ok: bool = False, compression='gzip', **kwargs) -> None:
        path = Path(path)
        if not H5PY_AVAILABLE:
            LOG.error("h5py is not available. Install it or use `npz` format instead.")
            raise ImportError("h5py is not available. Install it or use `npz` format instead.")
        path.mkdir(parents=True, exist_ok=exist_ok)
        metadata = dict(version=FULLVERSION, base=self.base.__class__.__name__,
                        ndim=self.ndim)
        LOG.debug(f"Saving metadata to {path / 'metadata.json'}")
        with open(path / "metadata.json", "w") as f:
            json.dump(metadata, f)
        LOG.debug(f"Saving base to {path / 'base'}")
        self.base.save(path / "base", exist_ok=exist_ok)

        unfolded = ArrayList.from_list(self.unfolded)
        bootstraps = ArrayList.from_list(self.bootstraps)
        initials = ArrayList.from_list(self.initials)
        if self.backgrounds is not None and len(self.backgrounds) > 0:
            backgrounds = ArrayList.from_list(self.backgrounds)
        with h5py.File(path / "matrices.h5", "w") as f:

            LOG.debug(f"Saving `bootstraps` to {path / 'matrices.h5' / 'bootstraps'}"
                      f" with compression {compression}" + kwargs.get('compression_opts', ''))
            subg = f.create_group('bootstraps')
            bootstraps.insert_into_tree(f, 'bootstraps/', compression=compression, **kwargs)

            LOG.debug(f"Saving `unfolded` to {path / 'matrices.h5' / 'unfolded'}")
            f.create_group('unfolded')
            unfolded.insert_into_tree(f, "unfolded/", compression=compression, **kwargs)

            LOG.debug(f"Saving `initials` to {path / 'matrices.h5' / 'initials'}")
            f.create_group('initials')
            initials.insert_into_tree(f, "initials/", compression=compression, **kwargs)
            if self.backgrounds is not None:
                LOG.debug(f"Saving `backgrounds` to {path / 'matrices.h5' / 'backgrounds'}")
                f.create_group('backgrounds')
                backgrounds.insert_into_tree(f, "backgrounds/", compression=compression, **kwargs)  # type: ignore
            LOG.debug(f"Saving `costs` to {path / 'matrices.h5' / 'costs'}")
            f.create_dataset('costs', data=self.costs, compression=compression, **kwargs)
        LOG.warn("Saving `kwargs` is not implemented yet")


    def save_npz(self, path: str | Path, exist_ok: bool = False, disable_tqdm: bool = False) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=exist_ok)
        metadata = dict(version=FULLVERSION, base=self.base.__class__.__name__,
                        ndim=self.ndim)
        LOG.debug(f"Saving metadata to {path / 'metadata.json'}")
        with open(path / "metadata.json", "w") as f:
            json.dump(metadata, f)

        LOG.debug(f"Saving base to {path / 'base'}")
        self.base.save(path / "base", exist_ok=True)

        tqdm_ = tqdm if not disable_tqdm else lambda x: x
        LOG.debug(f"Saving {len(self.bootstraps)} matrices to {path}")
        for i in tqdm_(range(len(self.bootstraps))):
            self.bootstraps[i].save(path / f"boot_{i}.npz", exist_ok=True)
            self.unfolded[i].save(path / f"unfolded_{i}.npz", exist_ok=True)
            self.initials[i].save(path / f"initial_{i}.npz", exist_ok=True)
            if self.backgrounds is not None and len(self.backgrounds) > 0:
                self.backgrounds[i].save(path / f"background_{i}.npz", exist_ok=True)
            #np.save(path / f"cost_{i}.npy", self.costs[i])

        LOG.debug(f"Saving {len(self.costs)} `costs` to {path}")
        costs = {f"cost_{i}": self.costs[i] for i in range(len(self.costs))}
        np.savez(path / "costs.npz", **costs)

        LOG.warn("Not saving kwargs")

    @overload
    @classmethod
    def _load(cls, path: Path, arraytype: type[Matrix],
               basearray: type[BootstrapMatrix],
               read_only: int | None = None) -> BootstrapMatrix: ...

    @overload
    @classmethod
    def _load(cls, path: Path, arraytype: type[Vector],
               basearray: type[BootstrapVector],
               read_only: int | None = None) -> BootstrapVector: ...
    @classmethod
    def _load(cls, path: Path, arraytype: type[Matrix] | type[Vector],
               basearray: type[BootstrapMatrix] | type[BootstrapVector],
               read_only: int | None = None) -> BootstrapMatrix | BootstrapVector:
        if (path / "matrices.h5").exists():
            return cls._load_h5(path, arraytype, basearray, read_only)
        return cls._load_npz(path, arraytype, basearray, read_only)

    @classmethod
    def _load_npz(cls, path, arraytype, basearray, read_only):
        with open(path / "metadata.json", "r") as f:
            metadata = json.load(f)
        if metadata["version"] != FULLVERSION:
            warnings.warn(f"Version mismatch: {metadata['version']} != {FULLVERSION}")
        if metadata['ndim'] != basearray.ndim:
            raise ValueError(f"Wrong ndim: {metadata['ndim']} != {cls.ndim}")
        result_cls: type[Result] = RESULT_CLASSES[metadata["base"]]
        base = result_cls.from_path(path / "base")  # type: ignore
        unfolded = []
        bootstraps = []
        costs = []
        initials = []
        if (path / 'costs.npz').exists():
            costs = np.load(path / "costs.npz")
        backgrounds = []
        for i in range(len(list(path.glob("boot_*.npz")))):
            unfolded.append(arraytype.from_path(path / f"unfolded_{i}.npz"))
            bootstraps.append(arraytype.from_path(path / f"boot_{i}.npz"))
            if (path / 'background_i.npz').exists():
                backgrounds.append(arraytype.from_path(path / f"background_{i}.npz"))
            initials.append(arraytype.from_path(path / f"initial_{i}.npz"))
            if read_only is not None and i > read_only:
                break
        return basearray(base=base, bootstraps=bootstraps, unfolded=unfolded, costs=costs,
                         backgrounds=backgrounds if backgrounds else None, initials=initials)

    @classmethod
    def _load_h5(cls, path, arraytype, basearray, read_only):
        if not H5PY_AVAILABLE:
            raise ImportError("h5py is not available")
        with open(path / "metadata.json", "r") as f:
            metadata = json.load(f)
        if metadata["version"] != FULLVERSION:
            warnings.warn(f"Version mismatch: {metadata['version']} != {FULLVERSION}")
        if metadata['ndim'] != basearray.ndim:
            raise ValueError(f"Wrong ndim: {metadata['ndim']} != {cls.ndim}")
        result_cls: type[Result] = RESULT_CLASSES[metadata["base"]]
        base = result_cls.from_path(path / "base")  # type: ignore

        backgrounds = None
        with h5py.File(path / "matrices.h5", "r") as f:
            bootstraps = list(ArrayList.from_tree(f, "bootstraps/", read_only=read_only).to_arrays())
            unfolded = list(ArrayList.from_tree(f, "unfolded/", read_only=read_only).to_arrays())
            initials = list(ArrayList.from_tree(f, "initials/", read_only=read_only).to_arrays())
            costs = np.asarray(f['costs'])
            if "backgrounds" in f:
                backgrounds = list(ArrayList.from_tree(f, "backgrounds/", read_only=read_only).to_arrays())
        return basearray(base=base, bootstraps=bootstraps, unfolded=unfolded, costs=costs,
                         backgrounds=backgrounds, initials=initials)




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

    def memory_usage_rapport(self) -> None:
        g = bytes_to_readable
        memory_usage = {
            'bootstraps': sum([b.nbytes for b in self.bootstraps]),
            'unfolded': sum([b.nbytes for b in self.unfolded]),
            'costs': self.costs.nbytes if isinstance(self.costs, np.ndarray) else sum([c.nbytes for c in self.costs]),
            'backgrounds': sum([b.nbytes for b in self.backgrounds]) if self.backgrounds is not None else 0,
            #'base': self.base.nbytes,
            '_ubox': self._ubox.nbytes if self._ubox is not None else 0,
            '_etabox': self._etabox.nbytes if self._etabox is not None else 0,
            '_nubox': self._nubox.nbytes if self._nubox is not None else 0,
            'initial': sum([b.nbytes for b in self.initials])
        }
        rapport = "MEMORY USAGE RAPPORT\n"
        for attr, mem in memory_usage.items():
            rapport += f"{attr:<15} {g(mem)}\n"
        rapport += "=============================\n"
        rapport += f"{'Total':<15} {g(sum(memory_usage.values()))}"
        print(rapport)

@dataclass(kw_only=True)
class BootstrapVector(Bootstrap[Vector]):
    base: UnfoldedResult1D
    bootstraps: list[Vector]
    unfolded: list[Vector]
    costs: np.ndarray
    initials: list[Vector]
    kwargs: dict[str, Any] = field(default_factory=dict)
    ndim: Literal[1] = 1

    @classmethod
    def from_path(cls, path: str | Path, read_only: int | None = None) -> BootstrapVector:
        return Bootstrap._load(Path(path), Vector, BootstrapVector, read_only)  # type: ignore

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

    def plot_cost(self, ax: Axes | None = None, skip: float | int | None = None, **kwargs) -> Plots1D:
        ax = make_ax(ax)
        if len(self.costs) == 0:
            return ax, []
        cost = self.costs
        i = np.arange(len(cost))
        if skip is not None:
            if isinstance(skip, float):
                skip = int(skip * len(cost))
            cost = cost[skip:]
            i = i[skip:]
        l = ax.plot(i, cost, **kwargs)
        maybe_set(ax, xlabel='Iteration')
        maybe_set(ax, ylabel='Cost')
        return ax, l

    def plot_initials(self, ax: Axes | None = None, **kwargs) -> Plots1D:
        ax = make_ax(ax)
        lines = []
        kwargs = {'color': 'k', 'alpha': 1/10} | kwargs
        x = self.base.best().X
        dx = self.base.best().dX
        x = x + dx/2
        for initial in self.initials:
            l = ax.plot(initial.X, initial.values, '_', **kwargs)
            #l = initial.plot(ax=ax, **kwargs)
            #l = ax.plot(x, initial, '_', **kwargs)
            lines.extend(l)
        return ax, lines

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
    initials: list[Matrix]
    kwargs: dict[str, Any] = field(default_factory=dict)
    ndim: Literal[2] = 2

    @classmethod
    def from_path(cls, path: str | Path, read_only: int | None = None) -> BootstrapMatrix:
        return Bootstrap._load(Path(path), Matrix, BootstrapMatrix, read_only)

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

    def eta_ci(self, alpha=0.05, summary = np.median, as_matrix: bool = True,
               ) -> tuple[Matrix, Matrix] | tuple[np.ndarray, np.ndarray]:
        a_low = 100*alpha/2
        lower = np.percentile(self.etabox, a_low, axis=0)
        a_high = 100*(1-alpha/2)
        upper = np.percentile(self.etabox, a_high, axis=0)
        if as_matrix:
            lower = self.base.raw.clone(values=lower, name=f'Lower {100*(1-alpha):.0f}% PI')
            upper = self.base.raw.clone(values=upper, name=f'Upper {100*(1-alpha):.0f}% PI')
        return lower, upper

    def nu_mat(self, summary = np.median) -> Matrix:
        nu = summary(self.nubox, axis=0)
        nu = self.base.raw.clone(values=nu)
        return nu

    def get_eta(self, i: int) -> Matrix:
        return self.base.raw.clone(values=self.etabox[i, :, :],
                                   name=f'eta {i}')

    @property
    def ubox(self) -> np.ndarray:
        if self._ubox is None:
            self._ubox = np.stack(self.unfolded)  # type: ignore
        return self._ubox

    @property
    def etabox(self) -> np.ndarray:
        if self._etabox is None:
            self._etabox = gmul(self.ubox, self.G.values)
        return self._etabox

    @property
    def nubox(self) -> np.ndarray:
        if self._nubox is None:
            self._nubox = gmul(self.ubox, self.R.values)
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

def gmul(X, A):
    return np.einsum('ijk,kl->ijl', X, A)

#@njit
def mul(X, A):
    Y = np.zeros_like(X)
    for i in tqdm(range(X.shape[0])):
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


class Coverage:
    def __init__(self, path: Path):
        self.boots = self.load(path)

    @staticmethod
    def load(path: Path, pattern = 'boot*') -> list:
        path = Path(path)
        boots = []
        paths = list(path.glob(pattern))
        for boot_path in tqdm(paths):
            boot = BootstrapMatrix.from_path(boot_path)
            boots.append(boot)
        return boots

    def percentile(self, alpha: float = 0.05):
        # For each bootstrap, compute the CI
        pass

    def coverage(self, true: Matrix | np.ndarray, alpha: float = 0.05):
        # For all bootstrap CI, compute the coverage
        if isinstance(true, Matrix):
            true = true.values
        coverages = np.zeros((len(self.boots), *true.shape), dtype=bool)
        for i in tqdm(range(len(self.boots))):
            boot = self.boots[i]
            lower, upper = boot.eta_ci(as_matrix=False, alpha=alpha)
            coverages[i] = (lower <= true) & (true <= upper)
        return coverages



def bca_2(original_estimate: np.ndarray, bootstrap_samples: np.ndarray,  alpha=0.05):
    """
    Compute the Bias-Corrected and Accelerated (BCa) confidence intervals for each variable.

    :param bootstrap_samples: NxM numpy array of N bootstrap samples of M variables.
    :param original_estimate: M-dimensional vector of original estimates.
    :param alpha: Significance level for confidence intervals.
    :return: Mx2 numpy array of BCa confidence intervals for each variable.
    """
    N, M = bootstrap_samples.shape
    conf_intervals = np.zeros((M, 2))

    bias = np.zeros(M)
    bias_z0 = np.zeros(M)
    accelerations = np.zeros(M)

    for i in range(M):
        ci, p, z0, a = bca_var(original_estimate[i], bootstrap_samples[:, i], alpha=alpha)

        # BCa confidence intervals
        conf_intervals[i, 0] = ci[0]
        conf_intervals[i, 1] = ci[1]

        bias[i] = p
        bias_z0[i] = z0
        accelerations[i] = a

    return conf_intervals, bias, bias_z0, accelerations


def _bca_2(original_estimate: np.ndarray, bootstrap_samples: np.ndarray,  alpha=0.05):
    """
    Compute the Bias-Corrected and Accelerated (BCa) confidence intervals for each variable.

    :param original_estimate: M-dimensional vector of original estimates.
    :param bootstrap_samples: NxM numpy array of N bootstrap samples of M variables.
    :param alpha: Significance level for confidence intervals.
    :return: Mx2 numpy array of BCa confidence intervals for each variable.
    """
    theta_hat = original_estimate
    theta_star = bootstrap_samples
    N, M = bootstrap_samples.shape

    bias = np.mean(bootstrap_samples < original_estimate, axis=0)
    bias = np.clip(bias, 1e-5, 1-1e-5)
    z0 = norm.ppf(bias)

    # Acceleration by jackknife
    theta_total = np.sum(bootstrap_samples, axis=0)
    theta_jacks = (theta_total[np.newaxis, :] -  bootstrap_samples) / (N - 1)
    theta_jack_mean = np.mean(theta_jacks, axis=0)
    numerator = np.sum((theta_jack_mean - theta_jacks) ** 3, axis=0)
    denominator = 6 * np.sum((theta_jacks - theta_jack_mean) ** 2, axis=0) ** 1.5
    a = numerator / denominator

    # BCa confidence intervals
    z_alpha = norm.ppf(alpha / 2)
    z_1_alpha = norm.ppf(1 - alpha / 2)

    adjusted_lower = z0 + (z0 + z_alpha) / (1 - a * (z0 + z_alpha))
    adjusted_upper = z0 + (z0 + z_1_alpha) / (1 - a * (z0 + z_1_alpha))
    lower_percentile = 100 * norm.cdf(adjusted_lower)
    upper_percentile = 100 * norm.cdf(adjusted_upper)
    #lower_percentile = 100 * norm.cdf(2 * z0 + z_alpha)
    #upper_percentile = 100 * norm.cdf(2 * z0 + z_1_alpha)

    # BCa confidence intervals
    q = np.stack([lower_percentile, upper_percentile], axis=1)
    conf_intervals = np.zeros((M, 2))
    for m in range(M):
        conf_intervals[m] = np.percentile(theta_star[:, m], q[m])

    return conf_intervals, bias, z0, a


def bca(original_estimate: np.ndarray, bootstrap_samples: np.ndarray,  alpha=0.05, backend='numpy'):
    """
    Compute the Bias-Corrected and Accelerated (BCa) confidence intervals for each variable.

    :param original_estimate: M-dimensional vector of original estimates.
    :param bootstrap_samples: NxM numpy array of N bootstrap samples of M variables.
    :param alpha: Significance level for confidence intervals.
    :param backend: Backend to use for computation. Either 'numpy' or 'numba'.
    :return: Mx2 numpy array of BCa confidence intervals for each variable.
    """
    theta_hat = original_estimate
    theta_star = bootstrap_samples
    N, M = bootstrap_samples.shape

    #print(theta_star.shape)

    match backend:
        case 'numpy':
            bias = np.mean(bootstrap_samples < original_estimate, axis=0)
            bias = np.clip(bias, 1e-5, 1-1e-5)
            # Acceleration by jackknife
            theta_total = np.sum(bootstrap_samples, axis=0)
            theta_jacks = (theta_total[np.newaxis, :] -  bootstrap_samples) / (N - 1)
            theta_jack_mean = np.mean(theta_jacks, axis=0)
            numerator = np.sum((theta_jack_mean - theta_jacks) ** 3, axis=0)
            denominator = 6 * np.sum((theta_jacks - theta_jack_mean) ** 2, axis=0) ** 1.5
            a = numerator / denominator
        case 'numba':
            bias, a = _compute_bias_acceleration(theta_hat, theta_star)
        case 'jax':
            bias = jnp.mean(theta_star < theta_hat, axis=0)
            bias = jnp.clip(bias, 1e-5, 1-1e-5)
            
            theta_total = jnp.sum(theta_star, axis=0)
            theta_jacks = (theta_total - theta_star) / (N - 1)
            theta_jack_mean = jnp.mean(theta_jacks, axis=0)
            
            numerator = jnp.sum((theta_jack_mean - theta_jacks) ** 3, axis=0)
            denominator = 6 * jnp.sum((theta_jacks - theta_jack_mean) ** 2, axis=0) ** 1.5
            a = numerator / denominator
            
            z0 = jax_norm.ppf(bias)
            z_alpha = jax_norm.ppf(alpha / 2)
            z_1_alpha = jax_norm.ppf(1 - alpha / 2)

            adjusted_lower = z0 + (z0 + z_alpha) / (1 - a * (z0 + z_alpha))
            adjusted_upper = z0 + (z0 + z_1_alpha) / (1 - a * (z0 + z_1_alpha))
            
            lower_percentile = 100 * jax_norm.cdf(adjusted_lower)
            upper_percentile = 100 * jax_norm.cdf(adjusted_upper)
            
            q = jnp.stack([lower_percentile, upper_percentile], axis=1)
            conf_intervals = jnp.array([jnp.percentile(theta_star[:, m], q[m]) for m in range(M)])
            return conf_intervals, bias, z0, a
        case _:
            raise ValueError(f"Backend `{backend}` not supported.")

    # BCa confidence intervals
    z0 = norm.ppf(bias)
    z_alpha = norm.ppf(alpha / 2)
    z_1_alpha = norm.ppf(1 - alpha / 2)
    #a = 1e7

    adjusted_lower = z0 + (z0 + z_alpha) / (1 - a * (z0 + z_alpha))
    adjusted_upper = z0 + (z0 + z_1_alpha) / (1 - a * (z0 + z_1_alpha))
    lower_percentile = 100 * norm.cdf(adjusted_lower)
    upper_percentile = 100 * norm.cdf(adjusted_upper)

    # BCa confidence intervals
    q = np.stack([lower_percentile, upper_percentile], axis=1)

    match backend:
        case 'numpy':
            # this is >80% bottleneck
            conf_intervals = np.zeros((M, 2))
            for m in range(M):
                conf_intervals[m] = np.percentile(theta_star[:, m], q[m])
        case 'numba': # reduced to 50%, shared with bias and acceleration
            conf_intervals = _percentile_numba(theta_star, q)

    return conf_intervals, bias, z0, a

def _percentile_jax(X, q):
    pass

@njit(parallel=True)
def _percentile_numba(X, q):
    M = X.shape[1]
    conf = np.zeros((M, 2))
    for m in prange(M):
        conf[m] = np.percentile(X[:, m], q[m])
    return conf

@njit
def _compute_bias_acceleration(theta_hat, theta_star: np.ndarray):
    N = theta_star.shape[0]
    bias = numba_mean_axis_0((theta_star < theta_hat).astype(np.float64))
    bias = np.clip(bias, 1e-5, 1-1e-5)

    # Acceleration by jackknife
    theta_total = np.sum(theta_star, axis=0)
    theta_jacks = (theta_total[np.newaxis, :] -  theta_star) / (N - 1)
    theta_jack_mean = numba_mean_axis_0(theta_jacks)
    numerator = np.sum((theta_jack_mean - theta_jacks) ** 3, axis=0)
    denominator = 6 * np.sum((theta_jacks - theta_jack_mean) ** 2, axis=0) ** 1.5
    a = numerator / denominator
    return bias, a


@njit(parallel=True)
def numba_mean_axis_0(a):
    N = a.shape[1]
    res = np.zeros(N)
    for i in prange(N):
        res[i] = (a[:, i].mean())

    return res


def bca_var(theta_hat: float, theta_star: np.ndarray, alpha: float = 0.05) -> tuple[np.ndarray, float, float, float]:
    """ BCa for a single variable given bootstrap samples
    :param theta_hat: Original estimate for this variable
    :param theta_star: Bootstrap estimates for this variable
    :param alpha: Significance level for confidence intervals.
    :return: BCa confidence intervals for this variable, bias, bias_z0, acceleration
    """
    #theta_hat = np.mean(theta_star)

    # Bias correction z0
    p = np.mean(theta_star < theta_hat)
    p = np.clip(p, 1e-5, 1 - 1e-5)
    z0 = norm.ppf(p)

    # Acceleration by jackknife
    # assuming mean as the statistic
    # Can't allocate as the array is (N-1, N-1)

    theta_hat_jacks = np.zeros_like(theta_star)
    for i in range(len(theta_star)):
        theta_star_jack = np.delete(theta_star, i)
        theta_hat_jack = np.mean(theta_star_jack)
        theta_hat_jacks[i] = theta_hat_jack
    a = np.sum((np.mean(theta_hat_jacks) - theta_hat_jacks) ** 3) / (6 * np.sum((np.mean(theta_hat_jacks) - theta_hat_jacks) ** 2) ** 1.5)

    # Adjusted percentiles
    z_alpha = norm.ppf(alpha / 2)
    z_1_alpha = norm.ppf(1 - alpha / 2)
    adjusted_lower = z0 + (z0 + z_alpha) / (1 - a * (z0 + z_alpha))
    adjusted_upper = z0 + (z0 + z_1_alpha) / (1 - a * (z0 + z_1_alpha))
    lower_percentile = 100 * norm.cdf(adjusted_lower)
    upper_percentile = 100 * norm.cdf(adjusted_upper)
    #lower_percentile = 100 * norm.cdf(2 * z0 + z_alpha)
    #upper_percentile = 100 * norm.cdf(2 * z0 + z_1_alpha)

    # BCa confidence intervals
    conf_intervals = np.percentile(theta_star, [lower_percentile, upper_percentile])

    return conf_intervals, p, z0, a


def bca_var_2(theta_hat, theta_star, alpha: float = 0.05) -> tuple[np.ndarray, float, float, float]:
    """ BCa for a single variable given bootstrap samples
    :param theta_hat: Original estimate for this variable
    :param theta_star: Bootstrap estimates for this variable
    :param alpha: Significance level for confidence intervals.
    :return: BCa confidence intervals for this variable, bias, bias_z0, acceleration
    """
    #theta_hat = np.mean(theta_star)

    # Bias correction z0
    p = np.mean(theta_star < theta_hat)
    z0 = norm.ppf(p)

    # Acceleration by jackknife
    # assuming mean as the statistic
    # Can't allocate as the array is (N-1, N-1)
    # Acceleration by jackknife - optimized
    n = len(theta_star)
    theta_total_sum = np.sum(theta_star)
    theta_hat_jacks = (theta_total_sum - theta_star) / (n - 1)

    a = np.sum((np.mean(theta_hat_jacks) - theta_hat_jacks) ** 3) / (6 * np.sum((np.mean(theta_hat_jacks) - theta_hat_jacks) ** 2) ** 1.5)

    # Adjusted percentiles
    z_alpha = norm.ppf(alpha / 2)
    z_1_alpha = norm.ppf(1 - alpha / 2)
    lower_percentile = 100 * norm.cdf(2 * z0 + z_alpha)
    upper_percentile = 100 * norm.cdf(2 * z0 + z_1_alpha)

    # BCa confidence intervals
    conf_intervals = np.percentile(theta_star, [lower_percentile, upper_percentile])

    return conf_intervals, p, z0, a

