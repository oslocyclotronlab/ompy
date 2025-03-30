from __future__ import annotations

import json
import logging
import time
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Iterable,
    Literal,
    TypeAlias,
    overload,
    TypeVar,
    Generic,
    Callable,
)
import xarray as xr

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from scipy.stats import poisson
from tqdm.autonotebook import tqdm

from ompy.array.ufunc import unpack_to_vectors
from .result import Result, RESULT_CLASSES
from .result1d import UnfoldedResult1D, has_cost
from .result2d import UnfoldedResult2D
from .unfolder import Unfolder
from .. import Matrix, Vector, ArrayList, H5PY_AVAILABLE, JAX_AVAILABLE
from ..array import AsymmetricVector
from ..helpers import (
    make_ax,
    maybe_set,
    readable_time,
    bytes_to_readable,
    print_readable_time,
)
from ..numbalib import njit, prange
from ..stubs import Axes, Lines, Plots1D
from ..version import FULLVERSION
from cycler import cycler

LOG = logging.getLogger(__name__)

if H5PY_AVAILABLE:
    import h5py

if JAX_AVAILABLE:
    import jax.numpy as jnp
    from jax.scipy.stats import norm as jax_norm
    import jax
else:
    jnp = np

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
- [ ] Packed unfolding is not working. there is some scaling
      or cross row interaction that is not taken into account.
"""

VSpace: TypeAlias = Literal["mu", "eta", "nu"]
MV = TypeVar("MV", bound=Matrix | Vector)
CI_Method: TypeAlias = Literal["standard", "poisson", "bca", "supremum",
                            "studentized supremum", "bonferroni percentile",
                            "bonferroni bca", "hotelling T2"]


def bootstrap(
    res: Result, N: int, base: Literal["raw", "nu"] = "raw", **kwargs
) -> Bootstrap:
    match res:
        case UnfoldedResult1D():
            return bootstrap_vector(res, N, base=base, **kwargs)
        case UnfoldedResult2D():
            return bootstrap_matrix(res, N, base=base, **kwargs)
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
        res_ = unfolder.unfold(
            A_boot,
            initial=best,
            R=R,
            G=G,
            background=bg_boot,
            disable_tqdm=True,
            **kwargs,
        )
        if False and has_cost(res_):
            fig, ax = plt.subplots()
            ax.plot(res_.cost)
            plt.show()
            if res_.cost[-1] > res_.cost[0]:
                raise RuntimeError("Unfolding diverged")
            costs.append(res_.cost)
        unfolded_boot.append(res_.best())
        A_boots.append(A_boot)
    bootstraped = BootstrapVector(
        base=res,
        bootstraps=A_boots,
        unfolded=unfolded_boot,  # type: ignore
        backgrounds=bgs,
        costs=costs,
        **kwargs,
    )
    return bootstraped


def bootstrap_vector(
    res: UnfoldedResult1D,
    N: int,
    base: Literal["raw", "nu"] = "nu",
    bootstrap_background: bool = True,
    background_base: Literal["raw", "beta"] = "beta",
    **kwargs,
) -> BootstrapVector:
    A_boots: list[Vector] = []
    unfolded_boots: list[Vector] = []
    best = res.best()
    kwargs = res.meta.kwargs | kwargs
    unfolder = Unfolder.from_result_constructor(res)

    match base:
        case "raw":
            A = res.raw.copy()
        case "folded" | "nu":
            A = res.best_folded().copy()
        case _:
            raise ValueError(f"Unknown sample type {base}. Expected 'raw' or 'folded'")
    A_boots = A.sample(N)
    for i, boot in enumerate(A_boots):
        A_boots[i] = boot.astype("float32")

    bg: Vector | None = None
    if res.background is not None:
        bg = res.background.copy()
        assert bg is not None
        bg.values = np.where(bg <= 0, 0, bg.values)
    bgs: list[Vector] = []
    # To avoid Poisson(0)
    # A.values = np.where(A <= 0, 3, A.values)
    costs: list[np.ndarray] = []
    bg_boots = None
    if bg is not None:
        if bootstrap_background:
            match background_base:
                case "raw":
                    bg_boots = bg.sample(N)
                case "beta":
                    bg_boots = res.beta.sample(N)
        else:
            bg_boots = [bg] * N
    mean = np.maximum(best, np.mean(best))
    initials = [best.clone(values=np.random.uniform(0, 5 * mean)) for i in range(N)]
    mask_1d = res.meta.parameters.mask
    # Stack the mask to match the shape of the unfolded result
    # Stack the penalty mask to match the shape of the unfolded result

    mask = mask_1d
    start = time.time()
    costs: list[np.ndarray] = []
    auxs: list[dict[str, np.ndarray]] = []
    unfolded_betas: list[Vector] = []
    # TODO use unfolder.unfold_vectors()
    unf_res: list[UnfoldedResult1D] = unfolder.unfold_vectors(
        A_boots,
        initial=initials,
        background=bg_boots,
        mask=mask,
        **kwargs,
    )
    if False:
        for i in tqdm(range(N)):
            bg_boot = bg_boots[i] if bg_boots is not None else None
            res_ = unfolder.unfold(
                A_boots[i],
                initial=initials[i],
                background=bg_boot,
                mask=mask,
                penalty_mask=penalty_mask,
                **kwargs,
            )
            if has_cost(res_):
                costs.append(res_.cost)
            if hasattr(res_, "aux"):
                auxs.append(res_.aux)
            unfolded_boots.append(res_.best_mu())
            if hasattr(res_, "beta") and res_.beta is not None:
                unfolded_betas.append(res_.beta)
    elapsed = time.time() - start

    unfolded_boots = [res_.best_mu() for res_ in unf_res]
    costs = [res_.cost for res_ in unf_res]
    auxs = [res_.aux for res_ in unf_res]
    contaminants = [res_.xi for res_ in unf_res]

    bootstraped = BootstrapVector(
        base=res,
        bootstraps=A_boots,
        unfolded=unfolded_boots,  # type: ignore
        backgrounds=bgs,
        costs=costs,
        initials=initials,
        kwargs=kwargs,
        betas=unfolded_betas,
        elapsed_time=elapsed,
        aux=auxs,
        contaminants=contaminants,
    )
    return bootstraped  # , bg_boots


def bootstrap_matrix(
    res: UnfoldedResult2D, N: int, base: Literal["raw", "folded"] = "folded", **kwargs
) -> BootstrapMatrix:
    """Create Bootstrap ensemble of `A` using `res` method"""
    best = res.best().astype("float32")
    R = (res.meta.space, res.R.T.astype("float32"))
    G = res.G.T.astype("float32")
    G_ex = res.G_ex.astype("float32")
    kwargs = res.meta.kwargs | kwargs
    unfolder = Unfolder.resolve_method(res.meta.method)(
        R=res.R.T.astype("float32"), G=G.T.astype("float32")
    )

    match base:
        case "raw":
            A = res.raw.as_numpy().copy().astype("float32")
        case "folded":
            A = res.best_folded().as_numpy().copy().astype("float32")
        case _:
            raise ValueError(f"Unknown sample type {base}. Expected 'raw' or 'folded'")
    # TODO Add background
    A_boots: list[Matrix] = A.sample(N)
    unfolded_boot: list[Matrix] = []
    costs: list[np.ndarray] = []
    initials: list[Matrix] = []
    backgrounds: list[Matrix] = []
    best = best
    disable_tqdm = kwargs.pop("disable_tqdm", True)
    background = res.background
    if background is not None:
        raise NotImplementedError("Background not implemented yet")
        background = background.astype("float32")
        background.values = np.where(background <= 0, 3, background.values)
        background[~mask] = 0

    start = time.time()
    for i in tqdm(range(N)):
        initial = best.clone(values=np.random.uniform(1, 3 * best))
        if background is not None:
            bg = background.clone(values=np.random.poisson(background))
            backgrounds.append(bg)
        else:
            bg = None
        unf = unfolder.unfold(
            A_boots[i],
            initial=initial,
            R=R,
            G=G,
            G_ex=G_ex,
            background=background,
            disable_tqdm=disable_tqdm,
            **kwargs,
        )
        unf.to_device("cpu")
        unf.as_numpy()
        if False and unf.cost[-1] > unf.cost[0]:
            raise RuntimeError("Unfolding diverged")
        unfolded_boot.append(unf.best())

        # We do a rescaling to make the cost fit into fewer bytes to take up less space
        costs.append((unf.cost / unf.cost.max()).astype("float16"))
        initials.append(initial)
    elapsed = time.time() - start
    bootstraped = BootstrapMatrix(
        base=res,
        bootstraps=A_boots,
        unfolded=unfolded_boot,
        kwargs=kwargs,
        costs=costs,
        initials=initials,
        backgrounds=backgrounds if background is not None else None,
        elapsed_time=elapsed,
    )
    return bootstraped


T = TypeVar("T", bound=Matrix | Vector)
SaveFormat = Literal["hdf5", "npz"]


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
    _xi_eta_box: np.ndarray | None = None
    _xi_nu_box: np.ndarray | None = None
    elapsed_time: float | None = None
    aux: dict[str, Any] | None = None
    contaminants: list[list[T]] = field(default_factory=lambda: [[]])

    def save(
        self,
        path: str | Path,
        exist_ok: bool = False,
        format: SaveFormat = "hdf5",
        **kwargs,
    ) -> None:
        format_ = format.lower()
        LOG.debug(f"Saving bootstrap to {path} in {format_} format(?)")
        start = time.time()
        if format_ == "hdf5":
            self.save_hdf5(path, exist_ok=exist_ok, **kwargs)
        elif format_ == "npz":
            self.save_npz(path, exist_ok=exist_ok)
        else:
            raise ValueError(f"Expected format {SaveFormat}, not {format}")
        LOG.debug(
            f"Saved bootstrap to {path} in {format_} format in {readable_time(time.time() - start)}"
        )

    def save_hdf5(
        self, path: str | Path, exist_ok: bool = False, compression="gzip", **kwargs
    ) -> None:
        path = Path(path)
        if not H5PY_AVAILABLE:
            LOG.error("h5py is not available. Install it or use `npz` format instead.")
            raise ImportError(
                "h5py is not available. Install it or use `npz` format instead."
            )
        LOG.debug(f"Making directory {path}, exist_ok={exist_ok}")
        path.mkdir(parents=True, exist_ok=exist_ok)
        metadata = dict(
            version=FULLVERSION, base=self.base.__class__.__name__, ndim=self.ndim
        )
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

            LOG.debug(
                f"Saving `bootstraps` to {path / 'matrices.h5' / 'bootstraps'}"
                f" with compression {compression}" + kwargs.get("compression_opts", "")
            )
            subg = f.create_group("bootstraps")
            bootstraps.insert_into_tree(
                f, "bootstraps/", compression=compression, **kwargs
            )

            LOG.debug(f"Saving `unfolded` to {path / 'matrices.h5' / 'unfolded'}")
            f.create_group("unfolded")
            unfolded.insert_into_tree(f, "unfolded/", compression=compression, **kwargs)

            LOG.debug(f"Saving `initials` to {path / 'matrices.h5' / 'initials'}")
            f.create_group("initials")
            initials.insert_into_tree(f, "initials/", compression=compression, **kwargs)
            if self.backgrounds:
                LOG.debug(
                    f"Saving `backgrounds` to {path / 'matrices.h5' / 'backgrounds'}"
                )
                f.create_group("backgrounds")
                backgrounds.insert_into_tree(f, "backgrounds/", compression=compression, **kwargs)  # type: ignore
            LOG.debug(f"Saving `costs` to {path / 'matrices.h5' / 'costs'}")
            f.create_dataset(
                "costs", data=self.costs, compression=compression, **kwargs
            )
        LOG.warn("Saving `kwargs` is not implemented yet")

    def save_npz(
        self, path: str | Path, exist_ok: bool = False, disable_tqdm: bool = False
    ) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=exist_ok)
        metadata = dict(
            version=FULLVERSION, base=self.base.__class__.__name__, ndim=self.ndim
        )
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
                self.backgrounds[i].save(
                    path / f"background_{i}.npz", exist_ok=True
                )  # np.save(path / f"cost_{i}.npy", self.costs[i])

        LOG.debug(f"Saving {len(self.costs)} `costs` to {path}")
        costs = {f"cost_{i}": self.costs[i] for i in range(len(self.costs))}
        np.savez(path / "costs.npz", **costs)

        LOG.warn("Not saving kwargs")

    @overload
    @classmethod
    def _load(
        cls,
        path: Path,
        arraytype: type[Matrix],
        basearray: type[BootstrapMatrix],
        read_only: int | None = None,
    ) -> BootstrapMatrix: ...

    @overload
    @classmethod
    def _load(
        cls,
        path: Path,
        arraytype: type[Vector],
        basearray: type[BootstrapVector],
        read_only: int | None = None,
    ) -> BootstrapVector: ...

    @classmethod
    def _load(
        cls,
        path: Path,
        arraytype: type[Matrix] | type[Vector],
        basearray: type[BootstrapMatrix] | type[BootstrapVector],
        read_only: int | None = None,
    ) -> BootstrapMatrix | BootstrapVector:
        if (path / "matrices.h5").exists():
            return cls._load_h5(path, arraytype, basearray, read_only)
        return cls._load_npz(path, arraytype, basearray, read_only)

    @classmethod
    def _load_npz(cls, path, arraytype, basearray, read_only):
        with open(path / "metadata.json", "r") as f:
            metadata = json.load(f)
        if metadata["version"] != FULLVERSION:
            warnings.warn(f"Version mismatch: {metadata['version']} != {FULLVERSION}")
        if metadata["ndim"] != basearray.ndim:
            raise ValueError(f"Wrong ndim: {metadata['ndim']} != {cls.ndim}")
        result_cls: type[Result] = RESULT_CLASSES[metadata["base"]]
        base = result_cls.from_path(path / "base")  # type: ignore
        unfolded = []
        bootstraps = []
        costs = []
        initials = []
        if (path / "costs.npz").exists():
            costs = np.load(path / "costs.npz")
        backgrounds = []
        for i in range(len(list(path.glob("boot_*.npz")))):
            unfolded.append(arraytype.from_path(path / f"unfolded_{i}.npz"))
            bootstraps.append(arraytype.from_path(path / f"boot_{i}.npz"))
            if (path / "background_i.npz").exists():
                backgrounds.append(arraytype.from_path(path / f"background_{i}.npz"))
            initials.append(arraytype.from_path(path / f"initial_{i}.npz"))
            if read_only is not None and i > read_only:
                break
        return basearray(
            base=base,
            bootstraps=bootstraps,
            unfolded=unfolded,
            costs=costs,
            backgrounds=backgrounds if backgrounds else None,
            initials=initials,
        )

    @classmethod
    def _load_h5(cls, path, arraytype, basearray, read_only):
        if not H5PY_AVAILABLE:
            raise ImportError("h5py is not available")
        with open(path / "metadata.json", "r") as f:
            metadata = json.load(f)
        if metadata["version"] != FULLVERSION:
            warnings.warn(f"Version mismatch: {metadata['version']} != {FULLVERSION}")
        if metadata["ndim"] != basearray.ndim:
            raise ValueError(f"Wrong ndim: {metadata['ndim']} != {cls.ndim}")
        result_cls: type[Result] = RESULT_CLASSES[metadata["base"]]
        base = result_cls.from_path(path / "base")  # type: ignore

        backgrounds = None
        with h5py.File(path / "matrices.h5", "r") as f:
            bootstraps = list(
                ArrayList.from_tree(f, "bootstraps/", read_only=read_only).to_arrays()
            )
            unfolded = list(
                ArrayList.from_tree(f, "unfolded/", read_only=read_only).to_arrays()
            )
            initials = list(
                ArrayList.from_tree(f, "initials/", read_only=read_only).to_arrays()
            )
            costs = np.asarray(f["costs"])
            if "backgrounds" in f:
                backgrounds = list(
                    ArrayList.from_tree(
                        f, "backgrounds/", read_only=read_only
                    ).to_arrays()
                )
        return basearray(
            base=base,
            bootstraps=bootstraps,
            unfolded=unfolded,
            costs=costs,
            backgrounds=backgrounds,
            initials=initials,
        )

    @classmethod
    @abstractmethod
    def from_path(cls, path: Path | str, n: int | None = None) -> Bootstrap: ...

    @property
    def G_ex(self) -> Matrix:
        return self.base.G_ex

    def has_G_ex(self) -> bool:
        return hasattr(self.base, "G_ex") and self.G_ex is not None

    @property
    def G_eg(self) -> Matrix:
        return self.base.G_eg

    @property
    def GegD(self) -> Matrix:
        return self.base.GegD

    @property
    def D(self) -> Matrix:
        return self.base.D

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
            "bootstraps": sum([b.nbytes for b in self.bootstraps]),
            "unfolded": sum([b.nbytes for b in self.unfolded]),
            "costs": (
                self.costs.nbytes
                if isinstance(self.costs, np.ndarray)
                else sum([c.nbytes for c in self.costs])
            ),
            "backgrounds": (
                sum([b.nbytes for b in self.backgrounds])
                if self.backgrounds is not None
                else 0
            ),
            # 'base': self.base.nbytes,
            "_ubox": self._ubox.nbytes if self._ubox is not None else 0,
            "_etabox": self._etabox.nbytes if self._etabox is not None else 0,
            "_nubox": self._nubox.nbytes if self._nubox is not None else 0,
            "initial": sum([b.nbytes for b in self.initials]),
        }
        rapport = "MEMORY USAGE RAPPORT\n"
        for attr, mem in memory_usage.items():
            rapport += f"{attr:<15} {g(mem)}\n"
        rapport += "=============================\n"
        rapport += f"{'Total':<15} {g(sum(memory_usage.values()))}"
        print(rapport)

    def time(self) -> None:
        print_readable_time(self.elapsed_time)


@dataclass(kw_only=True)
class BootstrapVector(Bootstrap[Vector]):
    base: UnfoldedResult1D
    bootstraps: list[Vector]
    unfolded: list[Vector]
    costs: np.ndarray
    initials: list[Vector]
    betas: list[Vector] | None = None
    kwargs: dict[str, Any] = field(default_factory=dict)
    ndim: Literal[1] = 1
    contaminants: list[list[Vector]] = field(default_factory=lambda: [[]])

    @classmethod
    def from_path(
        cls, path: str | Path, read_only: int | None = None
    ) -> BootstrapVector:
        return Bootstrap._load(Path(path), Vector, BootstrapVector, read_only)  # type: ignore

    def plot_unfolded(self, ax: Axes | None = None, **kwargs) -> Plots1D:
        ax = make_ax(ax)
        mu: Vector = self.base.best()
        lines: list[Lines] = []
        _, l = mu.plot(ax=ax, label=r"$\hat{\mu}$")
        lines.append(l)
        for i in range(len(self)):
            u: Vector = self.unfolded[i]
            _, l = u.plot(ax=ax, color="k", alpha=1 / len(self))
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
        color = kwargs.pop("color", "k")
        alpha = kwargs.pop("alpha", 1 / len(self))
        for i in range(len(self)):
            _, l = self.backgrounds[i].plot(ax=ax, color=color, alpha=alpha, **kwargs)
            if i == 0:
                lines.append(l)
        return ax, (l, lines[-1])

    def plot_cost(
        self,
        ax: Axes | None = None,
        start: float | int = 0,
        auxiliary: bool | Iterable[str] = True,
        relative: bool = False,
        plot_each: bool = False,
        **kwargs,
    ) -> Plots1D:
        ax = make_ax(ax)
        if len(self.costs) == 0:
            return ax, []
        if isinstance(self.costs[0], Iterable):
            return self._plot_cost_list(
                ax, start, auxiliary, relative, plot_each, **kwargs
            )
        else:
            return self._plot_cost_single(ax, start, auxiliary, relative, **kwargs)

    def _plot_cost_single(
        self,
        ax: Axes | None = None,
        start: float | int = 0,
        auxiliary: bool | Iterable[str] = True,
        relative: bool = False,
        **kwargs,
    ) -> Plots1D:
        if isinstance(start, float):
            start = int(start * len(self.costs))
        cost = self.costs[start:]

        if isinstance(auxiliary, str):
            auxiliary = [auxiliary]
        aux = {}
        if len(self.aux) > 0:
            if isinstance(auxiliary, bool):
                keys = self.aux.keys() if auxiliary else []
            else:
                keys = auxiliary
            aux = {k: self.aux[k][start:] for k in keys}

        if relative:
            cost /= cost[0]
            for k in aux:
                aux[k] /= aux[k][0]

        i = np.arange(start, len(self.costs))
        lines: list[Lines] = []
        (l,) = ax.plot(i, cost, **kwargs)
        lines.append(l)
        for k, v in aux.items():
            (l,) = ax.plot(i, v, label=k)
            lines.append(l)
        maybe_set(ax, xlabel="Iteration")
        maybe_set(ax, ylabel="Cost")
        ax.legend()
        return ax, lines

    def _plot_cost_list(
        self,
        ax: Axes | None = None,
        start: float | int = 0,
        auxiliary: bool | Iterable[str] = True,
        relative: bool = False,
        plot_each: bool = False,
        **kwargs,
    ) -> Plots1D:
        if isinstance(start, float):
            start = int(start * len(self.costs[0]))

        # Process costs
        costs = np.array([cost[start:] for cost in self.costs])
        if relative:
            costs = costs / costs[:, 0:1]

        # Process auxiliary data
        if isinstance(auxiliary, str):
            auxiliary = [auxiliary]
        aux = {}
        if len(self.aux) > 0:
            if isinstance(auxiliary, bool):
                keys = self.aux.keys() if auxiliary else []
            else:
                keys = auxiliary
            aux = {k: np.array([a[start:] for a in self.aux[k]]) for k in keys}
            if relative:
                for k in aux:
                    aux[k] = aux[k] / aux[k][:, 0:1]

        i = np.arange(start, len(self.costs[0]))
        lines: list[Lines] = []

        default_cycler = cycler(color=["r", "g", "b", "y", "m", "c"])
        if plot_each:
            # Plot individual trajectories
            for j, cost in enumerate(costs):
                color = kwargs.get("color", None)
                (l,) = ax.plot(i, cost, color=color, alpha=0.2, **kwargs)
                if j == 0:  # Only add first line to legend
                    lines.append(l)

            # Plot auxiliary data
            for k, v in aux.items():
                color = next(default_cycler)["color"]
                for traj in v:
                    (l,) = ax.plot(
                        i,
                        traj,
                        color=color,
                        alpha=0.2,
                        label=k if len(lines) == 1 else "",
                    )
                    if len(lines) == 1:  # Only add first line of each type to legend
                        lines.append(l)
        else:
            # Plot mean and std band
            mean_cost = np.mean(costs, axis=0)
            std_cost = np.std(costs, axis=0)
            (l,) = ax.plot(i, mean_cost, **kwargs)
            lines.append(l)
            ax.fill_between(i, mean_cost - std_cost, mean_cost + std_cost, alpha=0.3)

            # Plot auxiliary data
            for k, v in aux.items():
                color = next(default_cycler)["color"]
                mean_aux = np.mean(v, axis=0)
                std_aux = np.std(v, axis=0)
                (l,) = ax.plot(i, mean_aux, color=color, label=k)
                lines.append(l)
                ax.fill_between(
                    i, mean_aux - std_aux, mean_aux + std_aux, color=color, alpha=0.3
                )

        maybe_set(ax, xlabel="Iteration")
        maybe_set(ax, ylabel="Cost")
        ax.legend()
        return ax, lines

    def plot_initials(self, ax: Axes | None = None, **kwargs) -> Plots1D:
        ax = make_ax(ax)
        lines = []
        kwargs = {"color": "k", "alpha": 1 / 10} | kwargs
        x = self.base.best().X
        dx = self.base.best().dX
        x = x + dx / 2
        for initial in self.initials:
            l = ax.plot(initial.X, initial.values, "_", **kwargs)
            # l = initial.plot(ax=ax, **kwargs)
            # l = ax.plot(x, initial, '_', **kwargs)
            lines.extend(l)
        return ax, lines

    def _make_ci(
        self,
        bootstraps: np.ndarray,
        original: np.ndarray,
        alpha=0.05,
        summary=np.median,
        method: CI_Method = "standard",
    ) -> AsymmetricVector:
        x = self.base.raw.clone(values=summary(bootstraps, axis=0))
        lower, upper = make_ci(
            bootstraps, original=original, alpha=alpha, method=method
        )
        return AsymmetricVector.from_CI(
            x, lower=lower, upper=upper, clip=True, order="K"
        )

    def mu(
        self, alpha=0.05, summary=np.median, method: CI_Method = "standard"
    ) -> AsymmetricVector:
        return self._make_ci(
            self.ubox,
            original=self.base.best().values,
            alpha=alpha,
            summary=summary,
            method=method,
        )

    def eta(
        self, alpha=0.05, summary=np.median, method: CI_Method = "standard"
    ) -> AsymmetricVector:
        return self._make_ci(
            self.etabox,
            original=self.base.best_eta().values,
            alpha=alpha,
            summary=summary,
            method=method,
        )

    def nu(
        self, alpha=0.05, summary=np.median, method: CI_Method = "standard"
    ) -> AsymmetricVector:
        return self._make_ci(
            self.nubox,
            original=self.base.best_folded().values,
            alpha=alpha,
            summary=summary,
            method=method,
        )

    def bg(
        self, alpha=0.05, summary=np.median, method: CI_Method = "standard"
    ) -> AsymmetricVector | None:
        if self.backgrounds is None:
            return None
        box = np.stack([b.values for b in self.backgrounds])
        return self._make_ci(
            box,
            self.base.background.values,
            alpha=alpha,
            summary=summary,
            method=method,
        )

    def beta(
        self, alpha=0.05, summary=np.median, method: CI_Method = "standard"
    ) -> AsymmetricVector | None:
        if self.betas is None:
            return None
        box = np.stack([beta.values for beta in self.betas])
        return self._make_ci(
            box,
            self.base.best_beta().values,
            alpha=alpha,
            summary=summary,
            method=method,
        )

    def xi_eta(
        self,
        i: int | None = None,
        alpha=0.05,
        summary=np.median,
        method: CI_Method = "standard",
    ) -> list[AsymmetricVector] | AsymmetricVector:
        xibox = self.xi_eta_box()
        if i is None:
            ci: list[AsymmetricVector] = []
            for i in range(xibox.shape[1]):
                ci.append(self._make_ci(
                xibox[:, i, :],
                self.base.best_xi_eta(i).values,
                alpha=alpha,
                summary=summary,
                    method=method,
                ))
            return ci
        else:
            return self._make_ci(
                xibox[:, i, :],
                self.base.best_xi_eta(i).values,
                alpha=alpha,
                summary=summary,
                method=method,
            )

    def xi_nu(
        self,
        i: int | None = None,
        alpha=0.05,
        summary=np.median,
        method: CI_Method = "standard",
    ) -> list[AsymmetricVector] | AsymmetricVector:
        xibox = self.xi_nu_box()
        if i is None:
            ci: list[AsymmetricVector] = []
            for i in range(xibox.shape[1]):
                ci.append(self._make_ci(
                    xibox[:, i, :],
                    self.base.best_xi_folded(i).values,
                    alpha=alpha,
                    summary=summary,
                    method=method,
                ))
            return ci
        else:
            return self._make_ci(
                xibox[:, i, :],
                self.base.best_xi_folded(i).values,
                alpha=alpha,
                summary=summary,
                method=method,
            )

    def total_nu(self,
                 alpha=0.05,
                 summary=np.median,
                 method: CI_Method = "standard",
                 ) -> AsymmetricVector:
        x = self.nubox + self.xi_nu_box().sum(axis=1)
        y = self.base.best_folded().values + np.sum([self.base.best_xi_folded(i) for i in range(self.xi_nu_box().shape[1])])
        return self._make_ci(
            x,
            y,
            alpha=alpha,
            summary=summary,
            method=method,
        )
                

    @property
    def ubox(self) -> np.ndarray:
        if self._ubox is None:
            self._ubox = np.stack(self.unfolded)  # type: ignore
        return self._ubox

    @property
    def etabox(self) -> np.ndarray:
        if self._etabox is None:
            self._etabox = jnp.einsum("ji,kj->ki", self.G_eg.values, self.ubox)
        return self._etabox

    @property
    def nubox(self) -> np.ndarray:
        if self._nubox is None:
            self._nubox = jnp.einsum("ji,kj->ki", self.GegD.values, self.ubox)
        return self._nubox

    def xi_eta_box(self) -> np.ndarray:
        if self._xi_eta_box is None:
            box = []
            for sample in self.contaminants:
                box.append(jnp.stack([c.values for c in sample]))
            box = jnp.stack(box)
            # Contaminants is a list of lists of vectors
            # We want to apply the same function to each vector in the list
            einsum_op = lambda x: jnp.einsum("ji,kj->ki", self.G_eg.values, x)
            self._xi_eta_box = jax.vmap(einsum_op)(box)
        return self._xi_eta_box

    def xi_nu_box(self) -> np.ndarray:
        if self._xi_nu_box is None:
            box = []
            for sample in self.contaminants:
                box.append(jnp.stack([c.values for c in sample]))
            box = jnp.stack(box)
            # Contaminants is a list of lists of vectors
            # We want to apply the same function to each vector in the list
            einsum_op = lambda x: jnp.einsum("ji,kj->ki", self.GegD.values, x)
            self._xi_nu_box = jax.vmap(einsum_op)(box)
        return self._xi_nu_box

@dataclass(kw_only=True)
class BootstrapMatrix(Bootstrap[Matrix]):
    base: UnfoldedResult2D
    bootstraps: list[Matrix]
    unfolded: list[Matrix]
    initials: list[Matrix]
    kwargs: dict[str, Any] = field(default_factory=dict)
    ndim: Literal[2] = 2

    @classmethod
    def from_path(
        cls, path: str | Path, read_only: int | None = None
    ) -> BootstrapMatrix:
        return Bootstrap._load(Path(path), Matrix, BootstrapMatrix, read_only)

    def plot_unfolded(self, Ex: float | int, ax: Axes | None = None) -> Plots1D:
        ax = make_ax(ax)
        mu: Vector = self.base.best().loc[Ex, :]
        j = mu.last_nonzero()
        mu = mu.iloc[:j]
        lines: list[Lines] = []
        _, l = mu.plot(ax=ax, label=r"$\hat{\mu}$")
        lines.append(l)
        for i in range(len(self)):
            u: Vector = self.unfolded[i].loc[Ex, :j]
            _, l = u.plot(ax=ax, color="k", alpha=0.01)
            if i == 0:
                lines.append(l)
        return ax, lines

    def plot_folded(self, Ex: float | int, ax: Axes | None = None) -> Plots1D:
        ax = make_ax(ax)
        nu: Vector = self.base.best_folded().loc[Ex, :]
        j = nu.last_nonzero()
        nu = nu.iloc[:j]
        lines: list[Lines] = []
        _, l = nu.plot(ax=ax, label=r"$\hat{\mu}$")
        lines.append(l)
        for i in range(len(self)):
            u: Vector = ((self.base.R @ (self.unfolded[i]).T).T).loc[Ex, :j]
            _, l = u.plot(ax=ax, color="k", alpha=0.01)
            if i == 0:
                lines.append(l)
        return ax, lines

    def plot_eta(self, Ex: float | int, ax: Axes | None = None, **kwargs) -> Plots1D:
        ax = make_ax(ax)
        eta: Vector = (self.base.best() @ self.G_eg).loc[Ex, :]
        j = eta.last_nonzero()
        eta = eta.iloc[:j]
        lines: list[Lines] = []
        N = len(self)
        alpha = kwargs.pop("alpha", 1 / (N * 0.5))
        for i in range(N):
            u: Vector = ((self.base.G @ (self.unfolded[i]).T).T).loc[Ex, :j]
            _, l = u.plot(
                ax=ax, color="k", alpha=alpha, label=r"$\hat{\eta}_\mathrm{boot}$"
            )
            if i == 0:
                lines.append(l)
        _, l = eta.plot(ax=ax, label=r"$\hat{\eta}$")
        lines.append(l)
        return ax, lines

    def eta_vec(
        self, Ex: float | int, alpha=0.05, summary=np.median
    ) -> AsymmetricVector:
        i = self.base.raw.X_index.index_expression(Ex, strict=False)
        j = last_nonzero(self.etabox[:, i, :])
        eta = summary(self.etabox[:, i, :j], axis=0)
        eta = self.base.raw.iloc[i, :j].clone(values=eta)
        lower = np.percentile(self.etabox[:, i, :j], 100 * alpha / 2, axis=0)
        upper = np.percentile(self.etabox[:, i, :j], 100 * (1 - alpha / 2), axis=0)
        eta = AsymmetricVector.from_CI(eta, lower=lower, upper=upper, clip=True)
        return eta

    def nu_vec(
        self, Ex: float | int, alpha=0.05, summary=np.median
    ) -> AsymmetricVector:
        i = self.base.raw.X_index.index_expression(Ex, strict=False)
        nubox = self.nubox
        j = last_nonzero(nubox[:, i, :])
        nu = summary(nubox[:, i, :j], axis=0)
        nu = self.base.raw.iloc[i, :j].clone(values=nu)
        lower = np.percentile(nubox[:, i, :j], 100 * alpha / 2, axis=0)
        upper = np.percentile(nubox[:, i, :j], 100 * (1 - alpha / 2), axis=0)
        nu = AsymmetricVector.from_CI(nu, lower=lower, upper=upper, clip=True)
        return nu

    def eta_mat(self, summary=np.median) -> Matrix:
        eta = summary(self.etabox, axis=0)
        eta = self.base.raw.clone(values=eta)
        return eta

    def eta_ci(
        self,
        alpha=0.05,
        summary=np.median,
        as_matrix: bool = True,
    ) -> tuple[Matrix, Matrix] | tuple[np.ndarray, np.ndarray]:
        a_low = 100 * alpha / 2
        lower = np.percentile(self.etabox, a_low, axis=0)
        a_high = 100 * (1 - alpha / 2)
        upper = np.percentile(self.etabox, a_high, axis=0)
        if as_matrix:
            lower = self.base.raw.clone(
                values=lower, name=f"Lower {100 * (1 - alpha):.0f}% PI"
            )
            upper = self.base.raw.clone(
                values=upper, name=f"Upper {100 * (1 - alpha):.0f}% PI"
            )
        return lower, upper

    def nu_mat(self, summary=np.median) -> Matrix:
        nu = summary(self.nubox, axis=0)
        nu = self.base.raw.clone(values=nu)
        return nu

    def mu_mat(self, summary=np.median) -> Matrix:
        mu = summary(self.ubox, axis=0)
        mu = self.base.raw.clone(values=mu)
        return mu

    def mu_vec(
        self, Ex: float | int, alpha=0.05, summary=np.median
    ) -> AsymmetricVector:
        i = self.base.raw.X_index.index_expression(Ex, strict=False)
        j = last_nonzero(self.ubox[:, i, :])
        mu = summary(self.ubox[:, i, :j], axis=0)
        mu = self.base.raw.iloc[i, :j].clone(values=mu)
        lower = np.percentile(self.ubox[:, i, :j], 100 * alpha / 2, axis=0)
        upper = np.percentile(self.ubox[:, i, :j], 100 * (1 - alpha / 2), axis=0)
        mu = AsymmetricVector.from_CI(mu, lower=lower, upper=upper, clip=True)
        return mu

    def get_eta(self, i: int) -> Matrix:
        return self.base.raw.clone(values=self.etabox[i, :, :], name=f"eta {i}")

    @property
    def ubox(self) -> np.ndarray:
        if self._ubox is None:
            self._ubox = np.stack(self.unfolded)  # type: ignore
        return self._ubox

    @property
    def etabox(self) -> np.ndarray:
        if self.base.meta.space != "RG":
            warnings.warn(
                f"eta is only properly defined for the RG space, not {self.base.meta.space}."
            )
        if self._etabox is None:
            self._etabox = gmul(self.ubox, self.G_eg.values, self.G_ex.values)
        return self._etabox

    @property
    def nubox(self) -> np.ndarray:
        if self._nubox is None:
            self._nubox = gmul(self.ubox, self.GegD.values.T, self.G_ex.values)
        return self._nubox


def resolve_ci_method(
    method: CI_Method,
) -> Callable[[np.ndarray, float], tuple[np.ndarray, np.ndarray]]:
    match method:
        case "standard" | "percentile":
            return standard_ci
        case "poisson":
            return poisson_ci
        case "bca":
            return bca
        case "supremum":
            return supremum_ci
        case "studentized supremum":
            return studentized_supremum_ci
        case "bonferroni percentile":
            return bonferroni_percentile_ci
        case "bonferroni bca":
            return bonferroni_bca_ci
        case "hotelling T2":
            return hotelling_T2_ci
        case _:
            raise ValueError(f"Unknown CI method: {method}. "
                            f"Must be one of: {', '.join(CI_Method.__args__)}")


def make_ci(
    data: np.ndarray,
    original: np.ndarray,
    alpha: float = 0.05,
    method: CI_Method = "standard",
    **kwargs,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate confidence intervals for bootstrap data.

    The CI are not necessarily probablistic, and *they do not describe the
    uncertainty about the central estimate*. They are simply the lower and
    upper bounds of the data.

    Args:
        data: Bootstrap samples array
        original: Original estimate of the statistic
        alpha: Significance level for confidence intervals (default: 0.05)
        method: Method to calculate confidence intervals (default: 'standard')
            One of: 'standard', 'poisson', 'bca'

    Returns:
        Tuple containing:
        - Lower confidence bound array
        - Upper confidence bound array
    """
    match method:
        case "bca":
            # q = bca_2(original, data, alpha, backend='jax')[0]
            q = bca_2(original, data, alpha, **kwargs)[0]
            print(q.shape)
            return q[:, 0], q[:, 1]
        case _:
            pass
    lower, upper = resolve_ci_method(method)(data, original, alpha, **kwargs)
    return lower, upper


def standard_ci(data: np.ndarray, orginial: np.ndarray = None, alpha=0.05) -> tuple[np.ndarray, np.ndarray]:
    lower = np.percentile(data, 100 * alpha / 2, axis=0)
    upper = np.percentile(data, 100 * (1 - alpha / 2), axis=0)
    return lower, upper


def poisson_ci(lambdas, original, alpha=0.05):
    lower_bounds = poisson.ppf(alpha / 2, lambdas)
    upper_bounds = poisson.ppf(1 - alpha / 2, lambdas)
    return lower_bounds, upper_bounds


def supremum_ci(data: np.ndarray, original: np.ndarray, alpha=0.05, clip: bool = True):
    f_hat = original
    delta_b = data - f_hat
    supremum = np.max(np.abs(delta_b), axis=1)
    c_alpha = np.quantile(supremum, 1 - alpha)
    bound = c_alpha

    lower = f_hat - bound if not clip else np.clip(f_hat - bound, 0, None)
    return lower, f_hat + bound

def studentized_supremum_ci(data: np.ndarray, original: np.ndarray, alpha=0.05):
    f_hat = original
    delta_b = data - f_hat
    std = np.std(data, axis=0)
    supremum = np.max(np.abs(delta_b / std), axis=0)
    c_alpha = np.quantile(supremum, 1 - alpha)
    bound = c_alpha*std
    return f_hat - bound, f_hat + bound

def bonferroni_percentile_ci(data: np.ndarray, original: np.ndarray, alpha=0.05):
    N = data.shape[1]
    return standard_ci(data, alpha=alpha/N)

def bonferroni_bca_ci(data: np.ndarray, original: np.ndarray, alpha=0.05):
    N = data.shape[1]
    q = bca_2(original, data, alpha/N)[0]
    return q[:, 0], q[:, 1]

def hotelling_T2_ci(data: np.ndarray, original: np.ndarray, alpha=0.05):
    # Suppose g_boot has shape (M, N), each row is a bootstrap replicate of g.
    M, N = data.shape

    # 1) Compute mean (the "center") of the bootstrap replicates
    mean = data.mean(axis=0)  # shape (N,)

    # 2) Compute the sample covariance
    # dev is shape (M, N), each row: g*_b - average
    dev = data - mean
    Sigma = (dev.T @ dev) / (M - 1)  # shape (N, N)

    # 3) Invert or pseudo-invert the covariance
    # If Sigma is ill-conditioned, np.linalg.inv might fail; so use np.linalg.pinv
    Sigma_inv = np.linalg.pinv(Sigma, rcond=1e-12)

    # 4) For each bootstrap replicate, compute T^2 distance from g_mean
    # T^2_b = dev[b,:]^T Sigma_inv dev[b,:]
    # We'll store them in T_sq array of length M
    T_sq = np.einsum('ij,jk,ik->i', dev, Sigma_inv, dev)  # shape (M,)

    # T_sq is the Mahalanobis distance from g_mean for each replicate
    # Now we can look at the empirical distribution of T_sq
    T_sq_sorted = np.sort(T_sq)

    cutoff_alpha = T_sq_sorted[int(np.floor((1 - alpha) * M))]

    T_sq_cut = np.quantile(T_sq, 1.0 - alpha)
    lower, upper = elliptical_subset_bounds(data, mean, Sigma_inv, T_sq_cut)
    return lower, upper

def elliptical_subset_bounds(boot, mean, sigma_inv, T_sq_cut):
    """
    Returns min/max of each dimension among all bootstrap points whose T^2 <= T_sq_cut.
    This yields a bounding box for the elliptical region, not necessarily a minimal bounding box.
    """
    dev = boot - mean
    # compute T^2 for each replicate
    T_sq = np.einsum('ij,jk,ik->i', dev, sigma_inv, dev)
    # keep those within the cutoff
    mask = (T_sq <= T_sq_cut)
    inside_points = boot[mask, :]  # shape (K, N), K ~ 0.95*M
    
    # bounding box for the ellipse
    lower_bounds = inside_points.min(axis=0)
    upper_bounds = inside_points.max(axis=0)
    
    return lower_bounds, upper_bounds


def coverage_rate_eta(
    boots: list[BootstrapVector],
    target: Vector,
    alpha: float = 0.05,
    abs_eps: float = 1,
    **kwargs,
) -> np.ndarray:
    coverage = np.zeros(len(target))
    for boot in tqdm(boots):
        eta = boot.eta(alpha=alpha, **kwargs)
        lower, upper = eta.lerr, eta.uerr
        # We accept an error of 1 count
        coverage += (target >= lower - abs_eps) & (target <= upper + abs_eps)
    return coverage / len(boots)


def cumulative_coverage_rate_eta(
    boots: list[BootstrapVector],
    target: Vector,
    alpha: float = 0.05,
    abs_eps: float = 1,
    normalize: bool = True,
    **kwargs,
) -> np.ndarray:
    coverage = np.zeros((len(boots), len(target)))
    for i, boot in enumerate(tqdm(boots)):
        eta = boot.eta(alpha=alpha, **kwargs)
        lower, upper = eta.lerr, eta.uerr
        # We accept an error of abs_eps count(s)
        coverage[i] = (target >= lower - abs_eps) & (target <= upper + abs_eps)
    coverage = np.cumsum(coverage, axis=0)
    if normalize:
        return coverage / np.arange(1, len(boots) + 1)[:, np.newaxis]
    else:
        return coverage


def rolling_coverage_rate_eta(
    boots: list[BootstrapVector],
    target: Vector,
    alpha: float = 0.05,
    abs_eps: float = 1,
    **kwargs,
) -> np.ndarray:
    coverage = np.zeros((len(boots), len(target)))
    for i, boot in enumerate(tqdm(boots)):
        eta = boot.eta(alpha=alpha, **kwargs)
        lower, upper = eta.lerr, eta.uerr
        # We accept an error of abs_eps count(s)
        coverage[i] = (target >= lower - abs_eps) & (target <= upper + abs_eps)
    return coverage


def total_coverage_rate_eta(
    boots: list[BootstrapVector],
    target: Vector,
    alphas: np.ndarray,
    abs_eps: float = 1,
    adjust_lower: bool = False,
    lower_eps: float = 0.1,
    upper_eps: float = 1e-3,
    **kwargs,
) -> xr.DataArray:

    coverage = np.zeros((len(target), len(alphas)))
    width = np.zeros((len(alphas)))
    for boot in tqdm(boots, desc="Bootstraps"):
        for j, alpha in enumerate(tqdm(alphas, desc="Alpha", leave=False)):
            eta = boot.eta(alpha=alpha, **kwargs)
            lower, upper = eta.lerr, eta.uerr
            # We accept an error of abs_eps count(s)
            if adjust_lower:
                lower = np.where(
                    ((lower <= lower_eps) & (upper - lower > lower_eps))
                    | (upper < upper_eps),
                    0,
                    lower,
                )

            lower_rate = target >= lower
            upper_rate = target <= upper
            coverage[:, j] += lower_rate & upper_rate
            width[j] += np.mean(upper - lower)
    coverage = coverage / len(boots)
    width = width / len(boots)
    return xr.DataArray(
        coverage,
        dims=["bin", "alpha"],
        coords={"bin": range(len(target)), "alpha": alphas},
    ), xr.DataArray(width, dims=["alpha"], coords={"alpha": alphas})


def MIS(vector: AsymmetricVector, target: np.ndarray, alpha: float) -> float:
    L, U = vector.lerr, vector.uerr
    return (
        (U - L)
        + np.where(target < L, 2 / alpha * (L - target), 0)
        + np.where(target > U, 2 / alpha * (target - U), 0)
    )


def mean_interval_score_eta(
    boots: list[BootstrapVector], target: Vector, alpha: float = 0.05, **kwargs
) -> xr.DataArray:
    scores = np.zeros(len(target))
    for boot in tqdm(boots, desc="Bootstraps"):
        eta = boot.eta(alpha=alpha, **kwargs)
        scores += MIS(eta, target.values, alpha)
    return xr.DataArray(
        scores / len(boots), dims=["bin"], coords={"bin": range(len(target))}
    )


def total_MIS_eta(
    boots: list[BootstrapVector],
    target: Vector,
    alphas: np.ndarray,
    abs_eps: float = 1,
    **kwargs,
) -> xr.DataArray:
    scores = np.zeros(((len(boots), len(alphas), len(target), 4)))
    for i, boot in enumerate(tqdm(boots)):
        for j, alpha in enumerate(tqdm(alphas, desc="Alpha", leave=False)):
            eta = boot.eta(alpha=alpha, **kwargs)
            lower, upper = eta.lerr, eta.uerr
            # We accept an error of abs_eps count(s)
            overshoot = np.where(
                target.values > upper, 2 / alpha * (target.values - upper), 0
            )
            undershoot = np.where(
                target.values < lower, 2 / alpha * (lower - target.values), 0
            )
            width = upper - lower
            scores[i, j, :, 0] += width
            scores[i, j, :, 1] += overshoot
            scores[i, j, :, 2] += undershoot
            scores[i, j, :, 3] += width + overshoot + undershoot
    scores = scores / len(boots)
    return xr.DataArray(
        scores,
        dims=["trial", "alpha", "bin", "score"],
        coords={
            "trial": range(len(boots)),
            "alpha": alphas,
            "bin": range(len(target)),
            "score": ["width", "overshoot", "undershoot", "total"],
        },
    )


@njit
def last_nonzero(box: np.ndarray) -> int:
    S = np.sum(box, axis=0)
    for i in range(len(S) - 1, -1, -1):
        if S[i] > 0:
            return i
    return 0


def gmul(X, A, B=None):
    if B is None:
        return np.einsum("ijk,kl->ijl", X, A)
    else:
        raise NotImplementedError("Numpy einsum just stalls. Use jax.")
        # For the daredevils, this is the einsum
        return np.einsum("ij,kjl,lm->kim", B, X, A)


if JAX_AVAILABLE:

    @jax.jit
    def _gmul(X, A, B=None):
        if B is None:
            return jnp.einsum("ijk,kl->ijl", X, A)
        else:
            return jnp.einsum("ij,kjl,lm->kim", B, X, A)

    def gmul(X, A, B=None):
        x = _gmul(X, A, B)
        return np.asarray(x)


# @njit
def mul(X, A):
    Y = np.zeros_like(X)
    for i in tqdm(range(X.shape[0])):
        Y[i, :, :] = (A @ (X[i, :, :].T)).T
    return np.ascontiguousarray(Y)


def bootstrap_CI(
    b: Bootstrap, N: int, alpha=0.05, space: VSpace = "nu"
) -> tuple[np.ndarray, np.ndarray]:
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
            samples[i * m + j, :, :] = np.random.poisson(X[j, :, :])
    lower = np.percentile(samples, 100 * alpha / 2, axis=0)
    upper = np.percentile(samples, 100 * (1 - alpha / 2), axis=0)
    return lower, upper


def bootstrap_CI_at(
    b: Bootstrap, Ex: float | int, N: int, alpha=0.05, space: VSpace = "nu"
) -> tuple[np.ndarray, np.ndarray]:
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
                    samples[i * m + j, k] = np.random.poisson(X[j, k])
        return samples

    samples = fn(X)
    lower = np.percentile(samples, 100 * alpha / 2, axis=0)
    upper = np.percentile(samples, 100 * (1 - alpha / 2), axis=0)
    return lower, upper


def resolve_box_from_space(boot: Bootstrap, space: VSpace) -> np.ndarray:
    match space:
        case "mu":
            return boot.ubox
        case "eta":
            return boot.etabox
        case "nu":
            return boot.nubox
        case _:
            raise ValueError(f"Unknown space: {space}")


class Coverage:
    def __init__(self, path: Path):
        self.boots = self.load(path)

    @staticmethod
    def load(path: Path, pattern="boot*") -> list:
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


def bca_2(original_estimate: np.ndarray, bootstrap_samples: np.ndarray, alpha=0.05):
    """
    Compute the Bias-Corrected and Accelerated (BCa) confidence intervals for each variable.

    :param bootstrap_samples: NxM numpy array of N bootstrap samples of M variables.
    :param original_estimate: M-dimensional vector of original estimates.
    :param alpha: Significance level for confidence intervals.
    :return: Mx2 numpy array of BCa confidence intervals for each variable.
    """
    N, M = bootstrap_samples.shape
    assert original_estimate.shape == (M,)
    conf_intervals = np.zeros((M, 2))

    bias = np.zeros(M)
    bias_z0 = np.zeros(M)
    accelerations = np.zeros(M)

    for i in range(M):
        try:
            ci, p, z0, a = bca_var(
                original_estimate[i], bootstrap_samples[:, i], alpha=alpha
            )
        except Exception as e:
            print(f"Index {i} failed")
            print(original_estimate[i])
            print(bootstrap_samples[:, i])
            print(original_estimate[i - 1])
            print(bootstrap_samples[:, i - 1])
            raise e

        # BCa confidence intervals
        conf_intervals[i, 0] = ci[0]
        conf_intervals[i, 1] = ci[1]

        bias[i] = p
        bias_z0[i] = z0
        accelerations[i] = a

    return conf_intervals, bias, bias_z0, accelerations


def _bca_2(original_estimate: np.ndarray, bootstrap_samples: np.ndarray, alpha=0.05):
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
    bias = np.clip(bias, 1e-5, 1 - 1e-5)
    z0 = norm.ppf(bias)

    # Acceleration by jackknife
    theta_total = np.sum(bootstrap_samples, axis=0)
    theta_jacks = (theta_total[np.newaxis, :] - bootstrap_samples) / (N - 1)
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
    # lower_percentile = 100 * norm.cdf(2 * z0 + z_alpha)
    # upper_percentile = 100 * norm.cdf(2 * z0 + z_1_alpha)

    # BCa confidence intervals
    q = np.stack([lower_percentile, upper_percentile], axis=1)
    conf_intervals = np.zeros((M, 2))
    for m in range(M):
        conf_intervals[m] = np.percentile(theta_star[:, m], q[m])

    return conf_intervals, bias, z0, a


def bca(
    original_estimate: np.ndarray,
    bootstrap_samples: np.ndarray,
    alpha=0.05,
    backend="numpy",
):
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

    # print(theta_star.shape)

    match backend:
        case "numpy":
            bias = np.mean(bootstrap_samples < original_estimate, axis=0)
            bias = np.clip(bias, 1e-5, 1 - 1e-5)
            # Acceleration by jackknife
            theta_total = np.sum(bootstrap_samples, axis=0)
            theta_jacks = (theta_total[np.newaxis, :] - bootstrap_samples) / (N - 1)
            theta_jack_mean = np.mean(theta_jacks, axis=0)
            numerator = np.sum((theta_jack_mean - theta_jacks) ** 3, axis=0)
            denominator = (
                6 * np.sum((theta_jacks - theta_jack_mean) ** 2, axis=0) ** 1.5
            )
            a = numerator / denominator
        case "numba":
            bias, a = _compute_bias_acceleration(theta_hat, theta_star)
        case "jax":
            bias = jnp.mean(theta_star < theta_hat, axis=0)
            bias = jnp.clip(bias, 1e-5, 1 - 1e-5)

            theta_total = jnp.sum(theta_star, axis=0)
            theta_jacks = (theta_total - theta_star) / (N - 1)
            theta_jack_mean = jnp.mean(theta_jacks, axis=0)

            numerator = jnp.sum((theta_jack_mean - theta_jacks) ** 3, axis=0)
            denominator = (
                6 * jnp.sum((theta_jacks - theta_jack_mean) ** 2, axis=0) ** 1.5
            )
            a = numerator / denominator

            z0 = jax_norm.ppf(bias)
            z_alpha = jax_norm.ppf(alpha / 2)
            z_1_alpha = jax_norm.ppf(1 - alpha / 2)

            adjusted_lower = z0 + (z0 + z_alpha) / (1 - a * (z0 + z_alpha))
            adjusted_upper = z0 + (z0 + z_1_alpha) / (1 - a * (z0 + z_1_alpha))

            lower_percentile = 100 * jax_norm.cdf(adjusted_lower)
            upper_percentile = 100 * jax_norm.cdf(adjusted_upper)

            q = jnp.stack([lower_percentile, upper_percentile], axis=1)
            conf_intervals = jnp.array(
                [jnp.percentile(theta_star[:, m], q[m]) for m in range(M)]
            )
            return conf_intervals, bias, z0, a
        case _:
            raise ValueError(f"Backend `{backend}` not supported.")

    # BCa confidence intervals
    z0 = norm.ppf(bias)
    z_alpha = norm.ppf(alpha / 2)
    z_1_alpha = norm.ppf(1 - alpha / 2)
    # a = 1e7

    adjusted_lower = z0 + (z0 + z_alpha) / (1 - a * (z0 + z_alpha))
    adjusted_upper = z0 + (z0 + z_1_alpha) / (1 - a * (z0 + z_1_alpha))
    lower_percentile = 100 * norm.cdf(adjusted_lower)
    upper_percentile = 100 * norm.cdf(adjusted_upper)

    # BCa confidence intervals
    q = np.stack([lower_percentile, upper_percentile], axis=1)

    match backend:
        case "numpy":
            # this is >80% bottleneck
            conf_intervals = np.zeros((M, 2))
            for m in range(M):
                conf_intervals[m] = np.percentile(theta_star[:, m], q[m])
        case "numba":  # reduced to 50%, shared with bias and acceleration
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
    bias = np.clip(bias, 1e-5, 1 - 1e-5)

    # Acceleration by jackknife
    theta_total = np.sum(theta_star, axis=0)
    theta_jacks = (theta_total[np.newaxis, :] - theta_star) / (N - 1)
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
        res[i] = a[:, i].mean()

    return res


def bca_var(
    theta_hat: float, theta_star: np.ndarray, alpha: float = 0.05
) -> tuple[np.ndarray, float, float, float]:
    """BCa for a single variable given bootstrap samples
    :param theta_hat: Original estimate for this variable
    :param theta_star: Bootstrap estimates for this variable
    :param alpha: Significance level for confidence intervals.
    :return: BCa confidence intervals for this variable, bias, bias_z0, acceleration
    """
    # theta_hat = np.mean(theta_star)

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
    a = np.sum((np.mean(theta_hat_jacks) - theta_hat_jacks) ** 3) / (
        6 * np.sum((np.mean(theta_hat_jacks) - theta_hat_jacks) ** 2) ** 1.5
    )

    # Adjusted percentiles
    z_alpha = norm.ppf(alpha / 2)
    z_1_alpha = norm.ppf(1 - alpha / 2)
    adjusted_lower = z0 + (z0 + z_alpha) / (1 - a * (z0 + z_alpha))
    adjusted_upper = z0 + (z0 + z_1_alpha) / (1 - a * (z0 + z_1_alpha))
    lower_percentile = 100 * norm.cdf(adjusted_lower)
    upper_percentile = 100 * norm.cdf(adjusted_upper)
    # lower_percentile = 100 * norm.cdf(2 * z0 + z_alpha)
    # upper_percentile = 100 * norm.cdf(2 * z0 + z_1_alpha)

    # BCa confidence intervals

    try:
        conf_intervals = np.percentile(theta_star, [lower_percentile, upper_percentile])
    except Exception as e:
        print(theta_hat)
        print(theta_star)
        print(z_alpha)
        print(adjusted_lower, adjusted_upper)
        print(lower_percentile, upper_percentile)
        raise e

    return conf_intervals, p, z0, a


def bca_var_2(
    theta_hat, theta_star, alpha: float = 0.05
) -> tuple[np.ndarray, float, float, float]:
    """BCa for a single variable given bootstrap samples
    :param theta_hat: Original estimate for this variable
    :param theta_star: Bootstrap estimates for this variable
    :param alpha: Significance level for confidence intervals.
    :return: BCa confidence intervals for this variable, bias, bias_z0, acceleration
    """
    # theta_hat = np.mean(theta_star)

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

    a = np.sum((np.mean(theta_hat_jacks) - theta_hat_jacks) ** 3) / (
        6 * np.sum((np.mean(theta_hat_jacks) - theta_hat_jacks) ** 2) ** 1.5
    )

    # Adjusted percentiles
    z_alpha = norm.ppf(alpha / 2)
    z_1_alpha = norm.ppf(1 - alpha / 2)
    lower_percentile = 100 * norm.cdf(2 * z0 + z_alpha)
    upper_percentile = 100 * norm.cdf(2 * z0 + z_1_alpha)

    # BCa confidence intervals
    conf_intervals = np.percentile(theta_star, [lower_percentile, upper_percentile])

    return conf_intervals, p, z0, a
