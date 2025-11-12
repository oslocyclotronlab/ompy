from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import (TYPE_CHECKING, Any, Generic, Never, Self, TypeAlias,
                    TypeVar, overload, Type)
from warnings import warn

import numpy as np

from ..accel import jax_working
from ..array import AbstractArray, Matrix, Vector, on_device
from ..helpers import print_readable_time
from ..pipeline.stage import Stage
from ..stubs import Axes, Plot1D, Plot2D
from ..version import FULLVERSION, warn_version
from .result_classes import RESULT_CLASSES
from .stubs import PlotSpace, Space

if TYPE_CHECKING:
    from .resampling.resampling import Resampling
    from .unfolder import Unfolder

if jax_working():
    import jax.numpy as jnp
    import jax
else:
    import numpy as jnp

from scipy.special import gammaln
import scipy.stats
from tqdm.autonotebook import tqdm
import html
import math
from matplotlib import pyplot as plt

T = TypeVar("T", bound=Matrix | Vector)
UnfolderMethod: TypeAlias = str | type["Unfolder"]


@dataclass(kw_only=True)
class ResultMeta(ABC, Generic[T]):
    time: float
    space: Space
    method: UnfolderMethod
    parameters: Parameters[T]
    stage: Stage = field(default_factory=lambda: Stage.UNFOLDED)

    @property
    def kwargs(self) -> dict[str, Any]:
        return self.parameters.kwargs

    def save(self, path: Path, exist_ok: bool = False) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=exist_ok)
        method = self.method if isinstance(self.method, str) else self.method.__name__
        meta = dict(
            version=FULLVERSION,
            name=self.__class__.__name__,
            time=self.time,
            space=self.space,
            method=method,
        )
        with open(path / "meta.json", "w") as f:
            json.dump(meta, f)
        self.parameters.save(path / "parameters", exist_ok=exist_ok)

    @classmethod
    def from_path(cls, path: Path) -> ResultMeta:
        path = Path(path)
        with open(path / "meta.json", "r") as f:
            meta = json.load(f)
        warn_version(meta["version"])
        if meta["name"] != cls.__name__:
            raise ValueError(f"Name mismatch: {meta['name']} != {cls.__name__}")
        paramfield = [f for f in fields(cls) if f.name == "parameters"][0]
        parameters = eval(paramfield.type + ".from_path(path / 'parameters')")
        return cls(
            time=meta["time"],
            space=meta["space"],
            method=meta["method"],
            parameters=parameters,
        )

    @classmethod
    def read_subclass(cls, path: Path) -> str:
        path = Path(path)
        with open(path / "meta.json", "r") as f:
            meta = json.load(f)
        warn_version(meta["version"])
        return meta["name"]


@dataclass(kw_only=True)
class ResultMeta1D(ResultMeta[Vector]):
    parameters: Parameters1D


@dataclass(kw_only=True)
class ResultMeta2D(ResultMeta[Matrix]):
    parameters: Parameters2D

def alias(*aliases: str):
    def wrapper(func):
        func._aliases = aliases
        return func
    return wrapper

def add_aliases(cls):
    d = cls.__dict__.copy().items()
    for name, method in d:
        if hasattr(method, '_aliases'):
            for alias in method._aliases:
                setattr(cls, alias, method)
    return cls

@add_aliases
@dataclass(kw_only=True)
class Result(ABC, Generic[T]):
    meta: ResultMeta[T]
    # Contaminant spectra
    contaminants: tuple[T, ...] = ()
    beta: T | None = None
    do_fold_beta: bool = field(default=False, repr=False)
    _ndim: int = 0

    @property
    def ndim(self) -> int:
        if self._ndim == 0:
            self._ndim = self.meta.parameters.raw.ndim
        return self._ndim

    @staticmethod
    @abstractmethod
    def _vec_or_mat() -> Type[Vector | Matrix]:
        pass


    @classmethod
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        RESULT_CLASSES[cls.__name__] = cls

        for name, method in cls.__dict__.items():
            if hasattr(method, "_aliases"):
                for alias in method._aliases:
                    setattr(cls, alias, method)

    def time(self) -> None:
        print_readable_time(self.meta.time)

    @abstractmethod
    def plot_comparison(
        self,
        ax: Axes | None = None,
        raw: bool = True,
        unfolded: bool = True,
        initial: bool = False,
        folded: bool = True,
        space: PlotSpace = "base",
        **kwargs,
    ) -> Plot1D: ...

    @abstractmethod
    def plot_comparison_to(
        self,
        other: Result | T,
        ax: Axes | None = None,
        space: PlotSpace = "eta",
        **kwargs,
    ) -> Plot1D | Plot2D: ...

    @abstractmethod
    def best(self) -> T: ...

    def best_folded(self, device="gpu?", correct_efficiency: bool = True) -> T:
        best = self.best()
        with on_device(device, self.GegD, best, endpoint="numpy"):
            nu = best @ self.GegD
        if correct_efficiency:
            return self.correct_efficiency(nu)
        else:
            return nu

    best_nu = best_folded
    nu = best_nu

    def best_eta(self, device="gpu?", correct_efficiency: bool = True) -> T:
        best = self.best()
        match self.meta.space:
            case "mu":
                with on_device(device, self.G_eg, best, endpoint="numpy"):
                    out = best @ self.G_eg
            case "eta":
                out = best
            case _:
                raise ValueError(f"Cannot map from {self.meta.space} to eta")

        if correct_efficiency:
            return self.correct_efficiency(out)
        else:
            return out
                
    eta = best_eta

    def best_contaminant_mu(self, i: int) -> T:
        return self.contaminants[i]

    def best_contaminant_eta(self, i: int, device="gpu?") -> T:
        if not self.contaminants:
            raise ValueError("No contaminants to unfold")
        if i >= len(self.contaminants):
            raise ValueError(f"Index {i} out of bounds for contaminants of length {len(self.contaminants)}")
        match self.meta.space:
            case "mu":
                with on_device(device, self.G_eg, self.contaminants[i], endpoint="numpy"):
                    return self.contaminants[i] @ self.G_eg
            case "eta":
                return self.contaminants[i]
            case _:
                raise ValueError(f"Cannot map from {self.meta.space} to eta")

    @alias("best_contaminant_nu")
    def best_contaminant_folded(self, i: int, device="gpu?") -> T:
        contaminant_mu = self.best_contaminant_mu(i)
        with on_device(device, self.GegD, contaminant_mu, endpoint="numpy"):
            return contaminant_mu @ self.GegD


    def best_mu(self) -> T:
        if not self.meta.space == "mu":
            raise ValueError(
                f"Cannot map to mu when unfolded with space {self.meta.space}"
            )
        return self.best()

    def folded_total(self, device="gpu?") -> T:
        # We can't do the efficiency correction here because
        # beta_folded might or might not get corrected
        nu = self.best_folded(device=device)
        for i in range(len(self.contaminants)):
            nu = nu + self.best_contaminant_folded(i, device=device)
        if self.beta is not None:
            nu = nu + self.beta_folded(device=device)
        nu.title = "Total (nu)"
        return nu

    def best_beta(self) -> T:
        return self.beta

    best_total = folded_total

    def resolve_spaces(self, target: PlotSpace) -> tuple[T, str]:
        label = "unfolded"
        match self.meta.space:
            case "mu":
                if target == "eta":
                    label = "G@" + label
                return self.best_eta(), label
            case "eta":
                if target in {"eta", "base"}:
                    return self.best_eta(), label
            case _:
                raise ValueError(f"Cannot map from {self.meta.space} to {target}")

    def residuals(self) -> T:
        return self.raw - self.best_folded()

    @property
    def D_eg(self) -> Matrix:
        return self.meta.parameters.D_eg

    @property
    def G_eg(self) -> Matrix:
        return self.meta.parameters.G_eg

    @property
    def G_ex(self) -> Matrix | None:
        return self.meta.parameters.G_ex

    @property
    def raw(self) -> T:
        return self.meta.parameters.raw

    @property
    def GegD(self) -> Matrix:
        return self.meta.parameters.GDeg

    @property
    def background(self) -> T | None:
        return self.meta.parameters.background

    @property
    def initial(self) -> T:
        return self.meta.parameters.initial

    def get_param(self, key: str) -> Any:
        return self.meta.parameters.kwargs[key]

    @classmethod
    def from_locals(cls, locals: dict[str, Any], **kwargs) -> Never:
        raise NotImplementedError()

        def resolve(key: str) -> Any:
            if key in kwargs:
                return kwargs.pop(key)
            elif key in locals:
                return locals.pop(key)
            else:
                raise ValueError(f"Missing parameter {key}")

    def save(self, path: Path, exist_ok: bool = False) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=exist_ok)
        # Meta
        # This base class saves the version and its own name, then
        # hands over the meta dict to its subclasses.
        # The subclasses must mutate the dict
        meta = dict(version=FULLVERSION, name=self.__class__.__name__)
        self._save(path, meta=meta, exist_ok=exist_ok)
        with open(path / "meta.json", "w") as f:
            json.dump(meta, f)
        if self.beta is not None:
            self.beta.save(path / "beta.npz", exist_ok=exist_ok)
        # Result Meta
        self.meta.save(path / "meta", exist_ok=exist_ok)

    @abstractmethod
    def _save(
        self, path: Path, meta: dict[str, Any], exist_ok: bool = False
    ) -> None: ...

    @classmethod
    def from_path(cls, path: Path) -> Result:
        path = Path(path)
        with open(path / "meta.json", "r") as f:
            meta = json.load(f)
        warn_version(meta["version"])
        if meta["name"] != cls.__name__:
            raise ValueError(f"Name mismatch: {meta['name']} != {cls.__name__}")
        if (path / "beta.npz").exists():
            beta = cls._vec_or_mat().from_path(path / "beta.npz")
        else:
            beta = None
        meta_cls: str = ResultMeta.read_subclass(path / "meta")
        meta_: ResultMeta = eval(meta_cls + ".from_path(path / 'meta')")
        other = cls._load(path, meta)
        return cls(meta=meta_, beta=beta, **other)

    @classmethod
    @abstractmethod
    def _load(cls, path: Path, meta: dict[str, Any]) -> dict[str, Any]: ...

    @overload
    def to_device(self, device, inplace: bool = True) -> None: ...
    @overload
    def to_device(self, device, inplace: bool = True) -> Self: ...

    def to_device(self, device, inplace: bool = True) -> None | Self:
        self.meta.parameters.to_device(device, inplace=inplace)
        if not inplace:
            return self

    @overload
    def as_numpy(self, inplace: bool = True) -> None: ...
    @overload
    def as_numpy(self, inplace: bool = True) -> Self: ...

    def as_numpy(self, inplace: bool = True) -> None | Self:
        self.meta.parameters.as_numpy(inplace=inplace)
        if self.background is not None:
            backgrounds = (bg.as_numpy(inplace=inplace) for bg in self.background)
        if not inplace:
            return self.__class__(
                raw=self.raw.as_numpy(inplace=inplace),
                background=backgrounds,
                initial=self.initial.as_numpy(inplace=inplace),
                D_eg=self.D_eg.as_numpy(inplace=inplace),
                G_eg=self.G_eg.as_numpy(inplace=inplace),
            )

    @property
    def efficiency(self) -> Vector | None:
        return self.meta.parameters.efficiency

    def correct_efficiency[T: Matrix | Vector](self, arr: T) -> T:
        if self.efficiency is None:
            return arr
        if arr.ndim == 1:
            return arr / self.efficiency
        else:
            return arr / self.efficiency[None, :]

    @abstractmethod
    def resample(self, N: int, **kwargs) -> Resampling: ...

    @property
    def mask(self) -> np.ndarray | None:
        return self.meta.parameters.mask
    
    @property
    def num_parameters(self) -> int:
        """Number of parameters in the model

        Note that this is not the same as effective degrees of freedom.
        
        """
        return np.sum(self.mask)

    def _apply_mask(self, arr: np.ndarray | T) -> np.ndarray:
        if isinstance(arr, AbstractArray):
            arr = arr.values
        return arr[~self.mask]

    def loglikelihood(self) -> T:
        ll = poisson_loglikelihood(self._apply_mask(self.raw), self._apply_mask(self.best_total()))
        x = np.zeros_like(self.raw)
        x[~self.mask] = ll
        return self.raw.clone(values=x, name='Loglikelihood')

    def deviance(self) -> T:
        dev = poisson_deviance(self._apply_mask(self.best_total()), self._apply_mask(self.raw))
        x = np.zeros_like(self.raw)
        x[~self.mask] = dev
        return self.raw.clone(values=x, name='Deviance')

    def pearson_chi2(self) -> T:
        x = np.zeros_like(self.raw)
        x[~self.mask] = pearson_chi2(self._apply_mask(self.best_total()), self._apply_mask(self.raw))
        return self.raw.clone(values=x, name='Pearson χ²')

    def rmse(self) -> float:
        return rmse(self._apply_mask(self.best_total()), self._apply_mask(self.raw))
    
    def mae(self) -> float:
        return mae(self._apply_mask(self.best_total()), self._apply_mask(self.raw))

    def mean_bias(self) -> float:
        return mean_bias(self._apply_mask(self.best_total()), self._apply_mask(self.raw))

    def randomized_pit(self, **kwargs) -> np.ndarray:
        return randomized_pit(self._apply_mask(self.best_total()), self._apply_mask(self.raw), **kwargs)

    def rootogram(self, **kwargs) -> Rootogram:
        return rootogram(self._apply_mask(self.best_total()), self._apply_mask(self.raw), **kwargs)

    def score(self) -> Score:
        return Score(
            name=self.__class__.__name__,
            n_obs=self.num_parameters,
            loglik=self.loglikelihood().sum(),
            deviance=self.deviance().sum(),
            pearson_chi2=self.pearson_chi2().sum(),
            rmse=self.rmse(),
            mae=self.mae(),
            mean_bias=self.mean_bias(),
            fit_time_s=self.meta.time,
        )

    def compare_with(self, other: Self, nested: bool | None = None) -> Comparison:
        """
        Compare two fitted results on the SAME data.
        - If nested is True, use Δdeviance ~ χ²_{Δdf} (df = Δ k_params).
        - If nested is False (or k unknown), report Vuong z (asymptotic).
        - Always report Δ loglik, Δ AIC/BIC when available.
        """
        if self.num_parameters != other.num_parameters:
            raise ValueError("Results must be scored on the same observations to compare.")
        delta_ll = self.loglikelihood().sum() - other.loglikelihood().sum()
        delta_dev = other.deviance().sum() - self.deviance().sum()  # positive favors self (lower deviance)

        # Decide nested if not given (only sensible if both k_params known and differ)
        if nested is None:
            nested = (self.num_parameters is not None and other.num_parameters is not None)

        p_nested = None
        vuong_z = None
        vuong_p = None
        df = None

        if nested and (self.num_parameters is not None and other.num_parameters is not None):
            # LRT via deviance difference: ΔD ~ χ²_{Δdf}; smaller deviance is better
            df = abs(self.num_parameters - other.num_parameters)
            if df == 0:
                p_nested = 1.0
            else:
                p_nested = float(1.0 - scipy.stats.chi2.cdf(abs(delta_dev), df))
        else:
            # Vuong test for non-nested models using pointwise log-likelihoods
            m = self.loglikelihood().values - other.loglikelihood().values
            m -= np.mean(m)  # center
            s = float(np.sqrt(np.mean(m**2)))
            if s > 0:
                vuong_z = float(np.sqrt(self.num_parameters) * (delta_ll / self.num_parameters) / s)
                vuong_p = float(2 * (1 - scipy.stats.norm.cdf(abs(vuong_z))))
            else:
                vuong_z, vuong_p = np.nan, np.nan

        return Comparison(
            model_a=self, model_b=other,
            delta_loglik=delta_ll,
            delta_deviance=delta_dev,
            nested=nested,
            lrt_df=df,
            lrt_p=p_nested,
            vuong_z=vuong_z,
            vuong_p=vuong_p,
        )

    def pressure(self) -> T:
        y = self.raw
        nu = self.best_total()
        ratio = y/(nu+1e-3)
        g = self.G_ex.T @ (1 - ratio) @ self.G_eg.T @ self.D_eg.T

        g[~self.mask] = 0

        g.name = 'Gradient pressure'
        g.ylabel = r'$E_\gamma$'
        g.xlabel = r'$E_{\mathrm{in}}$'
        return g

    def __unwrap__(self) -> T:
        """Unwrap protocol for pipeline compatibility.
        
        Returns the best estimate from the unfolding result.
        This allows Result objects to work with the pipeline lifting system.
        """
        return self.best()
    
    def __stage__(self) -> Stage:
        """Stage protocol for pipeline tracking.
        
        Returns the pipeline stage this result represents (typically UNFOLDED).
        """
        return self.meta.stage
    
    @property
    def stage(self) -> Stage:
        """The pipeline stage this result represents."""
        return self.__stage__()


@dataclass(kw_only=True)
class Parameters(ABC, Generic[T]):
    raw: T
    background: T | None = None
    initial: T
    D_eg: Matrix
    G_eg: Matrix
    G_ex: Matrix | None = None  # None should be interpreted as an identity matrix
    mask: np.ndarray | None = None
    _GDeg: Matrix | None = None  # Cache of G_eg @ D
    efficiency: Vector | None = None
    kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Can eat up GPU memory
        if False:
            self.to_device("cpu")

    @property
    def GDeg(self) -> Matrix:
        # Cache G_eg @ D because the product is so common
        if self._GDeg is None:
            self._GDeg = self.D_eg @ self.G_eg
        return self._GDeg

    @overload
    def to_device(self, device, inplace: bool = True) -> None: ...
    @overload
    def to_device(self, device, inplace: bool = True) -> Self: ...

    def to_device(self, device, inplace: bool = True) -> None | Self:
        raw = self.raw.to_device(device, inplace=inplace)
        if self.background is not None:
            backgrounds = (bg.to_device(device, inplace=inplace) for bg in self.background)
        initial = self.initial.to_device(device, inplace=inplace)
        D = self.D_eg.to_device(device, inplace=inplace)
        G_eg = self.G_eg.to_device(device, inplace=inplace)
        if self.G_ex is not None:
            G_ex = self.G_ex.to_device(device, inplace=inplace)
        if self._GDeg is not None:
            self._GDeg = self._GDeg.to_device(device, inplace=inplace)
        if not inplace:
            return self.__class__(
                raw=raw,
                background=backgrounds,
                initial=initial,
                D_eg=D,
                G_eg=G_eg,
                G_ex=G_ex,
                _GDeg=self._GDeg,
                mask=self.mask,
                **self.kwargs,
            )

    @overload
    def as_numpy(self, inplace: bool = True) -> None: ...
    @overload
    def as_numpy(self, inplace: bool = True) -> Self: ...

    def as_numpy(self, inplace: bool = True) -> None | Self:
        raw = self.raw.as_numpy(inplace=inplace)
        if self.background is not None:
            backgrounds = (bg.as_numpy(inplace=inplace) for bg in self.background)
        initial = self.initial.as_numpy(inplace=inplace)
        D = self.D_eg.as_numpy(inplace=inplace)
        G_eg = self.G_eg.as_numpy(inplace=inplace)
        if self.G_ex is not None:
            G_ex = self.G_ex.as_numpy(inplace=inplace)
        if self._GDeg is not None:
            self._GDeg = self._GDeg.as_numpy(inplace=inplace)
        if not inplace:
            return self.__class__(
                raw=raw,
                background=backgrounds,
                initial=initial,
                D_eg=D,
                G_eg=G_eg,
                G_ex=G_ex,
                _GDeg=self._GDeg,
                mask=self.mask,
                **self.kwargs,
            )

    def save(self, path: Path, exist_ok: bool = False) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=exist_ok)
        meta = dict(version=FULLVERSION, name=self.__class__.__name__)
        with open(path / "meta.json", "w") as f:
            json.dump(meta, f)

        self.raw.save(path / "raw.npz", exist_ok=exist_ok)
        if self.background is not None:
            self.background.save(path / "background.npz", exist_ok=exist_ok)
        self.initial.save(path / "initial.npz", exist_ok=exist_ok)
        self.D_eg.save(path / "D_eg.npz", exist_ok=exist_ok)
        self.G_eg.save(path / "G_eg.npz", exist_ok=exist_ok)
        if self.G_ex is not None:
            self.G_ex.save(path / "G_ex.npz", exist_ok=exist_ok)
        if self.mask is not None:
            np.savez(path / "mask.npz", mask=self.mask, version=FULLVERSION)
        if self.efficiency is not None:
            self.efficiency.save(path / "efficiency.npz", exist_ok=exist_ok)
        with open(path / "kwargs.json", "w") as f:
            kwargs = {}
            for k, v in self.kwargs.items():
                try:
                    json.dumps(v)
                except (TypeError, ValueError):
                    warn(f"Skipping saving of {k} because it is a {type(v)}")
                    continue
                kwargs[k] = v
            json.dump(kwargs, f)

    @classmethod
    def from_path(cls, path: Path) -> Parameters:
        path = Path(path)
        with open(path / "meta.json", "r") as f:
            meta = json.load(f)
        warn_version(meta["version"])
        if meta["name"] != cls.__name__:
            raise ValueError(f"Class mismatch: {meta['name']} != {cls.__name__}")

        D_eg = Matrix.from_path(path / "D_eg.npz")
        G_eg = Matrix.from_path(path / "G_eg.npz")
        G_ex = None
        if (path / "G_ex.npz").exists():
            G_ex = Matrix.from_path(path / "G_ex.npz")
        if (path / "mask.npz").exists():
            with np.load(path / "mask.npz", allow_pickle=False) as data:
                mask = data["mask"][()]
        if (path / "efficiency.npz").exists():
            efficiency = Vector.from_path(path / "efficiency.npz")
        with open(path / "kwargs.json", "r") as f:
            kwargs = json.load(f)

        raw, background, initial = cls._load(path)

        return cls(
            raw=raw,
            background=background,
            initial=initial,
            D_eg=D_eg,
            G_eg=G_eg,
            G_ex=G_ex,
            mask=mask,
            efficiency=efficiency,
            kwargs=kwargs,
        )

    @classmethod
    @abstractmethod
    def _load(cls, path: Path) -> tuple[T, T | None, T]: ...


@dataclass(kw_only=True)
class Parameters1D(Parameters[Vector]):
    raw: Vector
    background: Vector | None = None
    initial: Vector

    @classmethod
    def _load(cls, path: Path) -> tuple[Vector, Vector | None, Vector]:
        raw = Vector.from_path(path / "raw.npz")
        background = (
            Vector.from_path(path / "background.npz")
            if (path / "background.npz").exists()
            else None
        )
        initial = Vector.from_path(path / "initial.npz")
        return raw, background, initial


@dataclass(kw_only=True)
class Parameters2D(Parameters[Matrix]):
    raw: Matrix
    background: Matrix | None = None
    initial: Matrix

    @classmethod
    def _load(cls, path: Path) -> tuple[Matrix, Matrix | None, Matrix]:
        raw = Matrix.from_path(path / "raw.npz")
        background = (
            Matrix.from_path(path / "background.npz")
            if (path / "background.npz").exists()
            else None
        )
        initial = Matrix.from_path(path / "initial.npz")
        return raw, background, initial


def get_field(cls, name):
    matches = [f for f in fields(cls) if f.name == name]
    if not len(matches):
        raise ValueError(f"No field {name} in {cls}")
    return matches[0]



def poisson_loglikelihood(data, model):
    ll = data * np.log(model) - model
    # subtract log(n!) (constant w.r.t model for fixed n, but needed for Vuong)
    ll = ll - gammaln(data + 1.0)
    return ll

    
def poisson_deviance(nu_hat: np.ndarray, n: np.ndarray) -> np.ndarray:
    # D = 2 * sum( n*log(n/nu) - (n - nu) ), with 0*log(0) = 0
    with np.errstate(divide='ignore', invalid='ignore'):
        term = np.where(n > 0, n * np.log(n / nu_hat) - (n - nu_hat), - (n - nu_hat))
    return 2.0 * term


def pearson_chi2(nu_hat: np.ndarray, n: np.ndarray) -> np.ndarray:
    r = (n - nu_hat) / np.sqrt(nu_hat)
    return r ** 2


def rmse(nu_hat: np.ndarray, n: np.ndarray) -> float:
    return float(np.sqrt(np.mean((nu_hat - n) ** 2)))


def mae(nu_hat: np.ndarray, n: np.ndarray) -> float:
    return float(np.mean(np.abs(nu_hat - n)))


def mean_bias(nu_hat: np.ndarray, n: np.ndarray) -> float:
    return float(np.mean(nu_hat - n))

@dataclass(frozen=True, kw_only=True)
class RandomizedPIT:
    pit: np.ndarray

    def plot(self, ax: Axes | None = None) -> Axes:
        if ax is None:
            fig, ax = plt.subplots(nrows=3, ncols=1, constrained_layout=True, figsize=(6, 8))
        ax = ax.flatten()
        if len(ax) < 3:
            raise ValueError("Need at least 3 axes")
        self.plot_hist(ax=ax[0])
        self.plot_ecdf(ax=ax[1])
        self.plot_qq(ax=ax[2])
        return ax

    def plot_hist(self, bins: int = 100, ax: Axes | None = None) -> Axes:
        if ax is None:
            fig, ax = plt.subplots()

        ax.hist(self.pit, bins=bins, density=True, edgecolor='black', alpha=0.7)
        ax.axhline(1, color='red', linestyle='--', linewidth=1, label='Uniform density')  # uniform density line
        ax.set_title('Randomized PIT')
        ax.set_xlabel("PIT value")
        ax.set_ylabel("Density")
        return ax

    def plot_ecdf(self, ax: Axes | None = None) -> Axes:
        if ax is None:
            fig, ax = plt.subplots()
        pit = np.sort(self.pit)
        ecdf = np.arange(1, len(pit)+1) / len(pit)

        ax.plot(pit, ecdf, label="ECDF(PIT)")
        ax.plot([0,1], [0,1], linestyle='--', color='red', label="Uniform(0,1)")
        ax.set_title("PIT ECDF vs Uniform")
        ax.set_xlabel("PIT value")
        ax.set_ylabel("ECDF")
        ax.legend()
        return ax

    def plot_qq(self, ax: Axes | None = None) -> Axes:
        if ax is None:
            fig, ax = plt.subplots()
        pit = np.sort(self.pit)
        n = len(pit)
        uniform_q = (np.arange(1, n+1) - 0.5) / n

        ax.plot(uniform_q, pit, marker='o', linestyle='none', alpha=0.6, ms=1)
        ax.plot([0,1], [0,1], linestyle='--', color='red')
        ax.set_title("PIT QQ Plot")
        ax.set_xlabel("Theoretical Uniform Quantiles")
        ax.set_ylabel("Observed PIT Quantiles")
        
        return ax 


def randomized_pit(nu_hat: np.ndarray, n: np.ndarray, rng: np.random.Generator | None = None) -> RandomizedPIT:
    """
    Randomized PIT for discrete Poisson:
      U = F(n-1) + V * (F(n) - F(n-1)),  V ~ Uniform(0,1)
    Returns U in [0,1].
    """
    if rng is None:
        rng = np.random.default_rng()
    F_nm1 = scipy.stats.poisson.cdf(np.clip(n - 1, 0, None), mu=nu_hat)
    F_n = scipy.stats.poisson.cdf(n, mu=nu_hat)
    V = rng.random(size=n.shape)
    return RandomizedPIT(pit=F_nm1 + V * (F_n - F_nm1))


def rootogram_data(nu_hat: np.ndarray, n: np.ndarray, disable_tqdm: bool = False, leave_tqdm: bool = False) -> dict[str, Any]:
    """
    Returns binned observed vs expected counts suitable for a rootogram.
    """
    # Bin by observed n
    n_int = jnp.asarray(jnp.round(n), dtype=int)
    kmax = int(n_int.max())
    obs = jnp.bincount(n_int, minlength=kmax + 1).astype(float)
    # Expected under model: sum over i of P(N=k | nu_hat_i)
    ks = jnp.arange(kmax + 1)
    
    def body_fun(k):
        return jnp.sum(jax.scipy.stats.poisson.pmf(k, mu=nu_hat))
    
    exp_counts = jax.vmap(body_fun)(ks)
    #exp_counts = jax_tqdm(exp_counts, disable=disable_tqdm, leave=leave_tqdm)
    
    return {"k": ks, "observed": obs, "expected": exp_counts}

@dataclass(frozen=True, kw_only=True)
class Rootogram:
    k: np.ndarray
    observed: np.ndarray
    expected: np.ndarray

    def plot(self, ax: Axes | None = None) -> Axes:
        if ax is None:
            fig, ax = plt.subplots()
        obs = np.sqrt(self.observed) 
        exp = np.sqrt(self.expected)
        deviation = obs - exp
        ax.bar(self.k, deviation, width=0.9, align='center', alpha=0.7)
        ax.axhline(0, color='black', linewidth=0.5)
        ax.set_xlabel('Count value')
        ax.set_ylabel(r'Deviation $\sqrt{\mathrm{observed}} - \sqrt{\mathrm{expected}}$')
        ax.set_title('Hanging Rootogram')
        return ax

    def plot_mc(self, nu_hat: np.ndarray, B: int = 50, ax: Axes | None = None) -> Axes:
        dev_obs = np.sqrt(self.observed) - np.sqrt(self.expected)

        # 2) Bootstrap under the fitted model
        boot = np.empty((B, self.k.size))
        rng = np.random.default_rng()
        for b in tqdm(range(B)):
            n_sim = rng.poisson(nu_hat)           
            sim = rootogram_data(nu_hat, n_sim)   
            row = np.sqrt(sim["observed"]) - np.sqrt(sim["expected"])
            length = min(row.shape[0], dev_obs.shape[0])
            row[:length] = dev_obs[:length]
            boot[b] = row

        lo = np.percentile(boot, 2.5, axis=0)
        med = np.percentile(boot, 50, axis=0)
        hi = np.percentile(boot, 97.5, axis=0)

        # 3) Plot
        if ax is None:
            fig, ax = plt.subplots()

        ax.bar(self.k, dev_obs, width=0.9, align='center', alpha=0.7)
        ax.axhline(0, linewidth=1)

        # envelope as thin band and median line
        ax.fill_between(self.k, lo, hi, alpha=0.15, step='mid')
        ax.plot(self.k, med, linewidth=1)

        ax.set_xlabel("Count value $k$")
        ax.set_ylabel(r"$\sqrt{\mathrm{obs}_k}-\sqrt{\mathrm{exp}_k}$")
        ax.set_title('Rootogram with Monte Carlo Bootstrap')
        return ax

def rootogram(nu_hat: np.ndarray, n: np.ndarray) -> Rootogram:
    data = rootogram_data(nu_hat, n)
    return Rootogram(**data)

def _fmt(x, none="—", prec=3):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return none
    if isinstance(x, float):
        # compact but stable formatting
        return f"{x:.{prec}g}"
    return str(x)

def _favor_arrow(delta: float, higher_is_better: bool = True) -> str:
    """
    For loglik: higher is better (Δloglik = A-B).
    For deviance/AIC/BIC: lower is better but we report ΔX(B-A) so positive favors A.
    """
    if delta is None or (isinstance(delta, float) and math.isnan(delta)):
        return ""
    if higher_is_better:
        return "↑A" if delta > 0 else ("↑B" if delta < 0 else "—")
    else:
        # delta computed as (B - A); positive favors A (lower is better)
        return "↑A" if delta > 0 else ("↑B" if delta < 0 else "—")

def _kv_row(k: str, v: str) -> str:
    return f"<tr><th>{html.escape(k)}</th><td class='num'>{html.escape(v)}</td></tr>"

# ---------- Score dataclass ----------
@dataclass
class Score:
    name: str
    n_obs: int
    loglik: float                 # higher better
    deviance: float               # lower better
    pearson_chi2: float           # lower better
    rmse: float                   # lower better
    mae: float                    # lower better
    mean_bias: float              # ~0 better
    fit_time_s: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "n_obs": self.n_obs,
            "loglik": self.loglik,
            "deviance": self.deviance,
            "pearson_chi2": self.pearson_chi2,
            "rmse": self.rmse,
            "mae": self.mae,
            "mean_bias": self.mean_bias,
            "fit_time_s": self.fit_time_s,
        }

    # ---------- Terminal rendering ----------
    def summary_text(self, width: int = 80) -> str:
        bar = "─" * min(width, 80)
        lines = [
            f"Score: {self.name}",
            bar,
            f"loglik (↑ better):      {_fmt(self.loglik):>10}",
            f"deviance (↓ better):    {_fmt(self.deviance):>10}",
            f"Pearson χ² (↓ better):  {_fmt(self.pearson_chi2):>10}",
            f"RMSE (↓ better):        {_fmt(self.rmse):>10}",
            f"MAE (↓ better):         {_fmt(self.mae):>10}",
            f"Mean bias (~0):         {_fmt(self.mean_bias):>10}",
            bar,
            f"n_obs:                  {self.n_obs}",
            f"fit_time_s:             {_fmt(self.fit_time_s)}",
        ]
        return "\n".join(lines)

    # ---------- HTML / Notebook rendering ----------
    def _render_html_(self) -> str:
        style = """
<style>
.score-card { font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial;
              border:1px solid #e5e7eb; border-radius:10px; padding:12px; max-width:720px; }
.score-hdr { display:flex; justify-content:space-between; align-items:baseline; margin-bottom:6px; }
.score-hdr h3 { margin:0; font-size:1.05rem; }
.score-table { width:100%; border-collapse:collapse; margin-top:6px; }
.score-table th { text-align:left; font-weight:600; padding:6px 4px; border-bottom:1px solid #f3f4f6; }
.score-table td { text-align:right; padding:6px 4px; border-bottom:1px solid #f9fafb; }
.num { font-variant-numeric: tabular-nums; }
.dim { color:#6b7280; font-size:0.9rem; }
</style>
"""
        rows = []
        rows.append(_kv_row("loglik (higher better)", _fmt(self.loglik)))
        rows.append(_kv_row("deviance (lower better)", _fmt(self.deviance)))
        rows.append(_kv_row("Pearson χ² (lower better)", _fmt(self.pearson_chi2)))
        rows.append(_kv_row("RMSE (lower better)", _fmt(self.rmse)))
        rows.append(_kv_row("MAE (lower better)", _fmt(self.mae)))
        rows.append(_kv_row("Mean bias (~0)", _fmt(self.mean_bias)))
        rows.append(_kv_row("n_obs", str(self.n_obs)))
        rows.append(_kv_row("fit_time_s", _fmt(self.fit_time_s)))

        body = f"""
<div class="score-card">
  <div class="score-hdr">
    <h3>Model score</h3>
    <div class="dim">{html.escape(self.name)}</div>
  </div>
  <table class="score-table">
    <tbody>
      {''.join(rows)}
    </tbody>
  </table>
</div>
"""
        return style + body

    def _repr_html_(self) -> str:
        return self._render_html_()


@dataclass
class Comparison:
    model_a: Result
    model_b: Result
    delta_loglik: float           # A - B (higher better → positive favors A)
    delta_deviance: float         # B - A (lower better → positive favors A)
    nested: bool
    lrt_df: int | None
    lrt_p: float | None
    vuong_z: float | None
    vuong_p: float | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "A_name": self.model_a.__class__.__name__,
            "B_name": self.model_b.__class__.__name__,
            "Δloglik (A-B) [higher better]": self.delta_loglik,
            "Δdeviance (B-A) [lower better]": self.delta_deviance,
            "nested": self.nested,
            "LRT_df": self.lrt_df,
            "LRT_p": self.lrt_p,
            "Vuong_z": self.vuong_z,
            "Vuong_p": self.vuong_p,
        }

    # ---------- Terminal rendering ----------
    def summary_text(self, width: int = 80) -> str:
        if hasattr(self.model_a, 'name'):
            A = self.model_a.name
        else:
            A = self.model_a.__class__.__name__
        if hasattr(self.model_b, 'name'):
            B = self.model_b.name
        else:
            B = self.model_b.__class__.__name__
        lines = []
        bar = "─" * min(width, 80)
        lines.append(f"Comparison: {A}  vs  {B}")
        lines.append(bar)

        # Core metrics
        lines.append(
            f"Δloglik (A-B) [higher better]: {_fmt(self.delta_loglik):>8}  {_favor_arrow(self.delta_loglik, True)}"
        )
        lines.append(
            f"Δdeviance (B-A) [lower better]: {_fmt(self.delta_deviance):>8}  {_favor_arrow(self.delta_deviance, False)}"
        )

        # Tests
        lines.append(bar)
        if self.nested and self.lrt_p is not None:
            lines.append(f"LRT (nested): df={_fmt(self.lrt_df, prec=0)}, p={_fmt(self.lrt_p, prec=3)}")
            lines.append("  (Positive Δdeviance B−A suggests A fits significantly better if p is small.)")
        else:
            lines.append("LRT (nested): —")
        if self.vuong_z is not None and self.vuong_p is not None:
            lines.append(f"Vuong (non-nested): z={_fmt(self.vuong_z, prec=3)}, p={_fmt(self.vuong_p, prec=3)}")
            lines.append("  (Positive z favors A; negative favors B; small p indicates preference.)")
        else:
            lines.append("Vuong (non-nested): —")

        # Context
        lines.append(bar)
        if hasattr(self.model_a, 'loglikelihood'):
            lines.append(f"loglik_A={_fmt(self.model_a.loglikelihood().sum())}  loglik_B={_fmt(self.model_b.loglikelihood().sum())}")
        if hasattr(self.model_a, 'num_parameters'):
            lines.append(f"num_parameters={self.model_a.num_parameters}")
        else:
            lines.append("num_parameters=—")
        return "\n".join(lines)

    # ---------- HTML / Notebook rendering ----------
    def _render_html_(self) -> str:
        """
        Returns an HTML summary table. In Jupyter, this is also exposed via _repr_html_.
        """
        A = html.escape(self.model_a.__class__.__name__)
        B = html.escape(self.model_b.__class__.__name__)

        def cell(val, arrow=None, higher=True):
            txt = _fmt(val)
            arr = _favor_arrow(val, higher) if arrow else ""
            return f"<td class='num'>{html.escape(txt)} <span class='arrow'>{arr}</span></td>"

        # rows
        row_loglik = f"<tr><th>Δloglik (A−B) <small>higher better</small></th>{cell(self.delta_loglik, True, True)}</tr>"
        row_dev    = f"<tr><th>Δdeviance (B−A) <small>lower better</small></th>{cell(self.delta_deviance, True, False)}</tr>"

        # tests
        if self.nested and self.lrt_p is not None:
            lrt = f"df={_fmt(self.lrt_df, prec=0)}, p={_fmt(self.lrt_p)}"
        else:
            lrt = "—"
        if self.vuong_z is not None and self.vuong_p is not None:
            vuong = f"z={_fmt(self.vuong_z)}, p={_fmt(self.vuong_p)}"
        else:
            vuong = "—"

        if hasattr(self.model_a, 'loglikelihood'):
            llA, llB   = _fmt(self.model_a.loglikelihood().sum()), _fmt(self.model_b.loglikelihood().sum())
        else:
            llA, llB   = "—", "—"

        if hasattr(self.model_a, 'num_parameters'):
            num_parametersA = self.model_a.num_parameters
        else:
            num_parametersA = "—"
        if hasattr(self.model_b, 'num_parameters'):
            num_parametersB = self.model_b.num_parameters
        else:
            num_parametersB = "—"

        style = """
<style>
.cmp-card { font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial; 
            border: 1px solid #e5e7eb; border-radius: 10px; padding: 12px; max-width: 780px; }
.cmp-hdr { display:flex; justify-content:space-between; align-items:baseline; margin-bottom:6px;}
.cmp-hdr h3 { margin:0; font-size:1.05rem; }
.cmp-hdr small { color:#6b7280; }
.cmp-table { width:100%; border-collapse:collapse; margin-top:6px; }
.cmp-table th { text-align:left; font-weight:600; padding:6px 4px; border-bottom:1px solid #f3f4f6; }
.cmp-table td { text-align:right; padding:6px 4px; border-bottom:1px solid #f9fafb; }
.cmp-table .num { font-variant-numeric: tabular-nums; }
.cmp-foot { color:#6b7280; font-size:0.9rem; margin-top:8px; }
.badge { background:#f3f4f6; border:1px solid #e5e7eb; padding:2px 6px; border-radius:6px; }
.arrow { color:#6b7280; margin-left:6px; }
.kv { display:flex; gap:8px; flex-wrap:wrap; margin-top:6px;}
.kv span { background:#f9fafb; border:1px solid #eef2f7; padding:2px 6px; border-radius:6px; }
</style>
"""
        body = f"""
<div class="cmp-card">
  <div class="cmp-hdr">
    <h3>Model comparison</h3>
    <div class="badge">{A} vs {B}</div>
  </div>
  <table class="cmp-table">
    <tbody>
      {row_loglik}
      {row_dev}
      <tr><th>LRT (nested)</th><td class="num">{html.escape(lrt)}</td></tr>
      <tr><th>Vuong (non-nested)</th><td class="num">{html.escape(vuong)}</td></tr>
    </tbody>
  </table>
  <div class="kv">
    <span>loglik_A={html.escape(llA)}</span><span>loglik_B={html.escape(llB)}</span>
    <span>num_parameters={num_parametersA}</span><span>num_parameters={num_parametersB}</span>
  </div>
  <div class="cmp-foot">
    Conventions: Δloglik = A − B (higher is better). Δdeviance = B − A (lower is better → positive favors A).
  </div>
</div>
"""
        return style + body

    # Make it display automatically in notebooks
    def _repr_html_(self) -> str:
        return self._render_html_()

    @classmethod
    def from_scores(cls, score_a: Score, score_b: Score, nested: bool | None = None) -> Self:
        """
        Construct a Comparison directly from two Score objects.
        LRT used if k_params available; Vuong is not performed (no pointwise ll).
        """
        if score_a.n_obs != score_b.n_obs:
            raise ValueError("Scores must have same n_obs to compare.")

        # Determine nesting if not specified: only possible if both have AIC/BIC => k_params known
        lrt_df = None
        lrt_p = None
        delta_dev = score_b.deviance - score_a.deviance

        # Compute Δ loglik (A-B)
        delta_ll = score_a.loglik - score_b.loglik


        return cls(
            model_a=score_a,
            model_b=score_b,
            delta_loglik=delta_ll,
            delta_deviance=delta_dev,
            nested=nested,
            lrt_df=lrt_df,
            lrt_p=lrt_p,
            vuong_z=None,     # unknown without pointwise ll
            vuong_p=None,
        )

        
def _apply_mask_arrays(nu_hat: np.ndarray, n: np.ndarray, mask: np.ndarray | None):
    nu_hat = np.asarray(nu_hat, dtype=float)
    n = np.asarray(n, dtype=float)
    if mask is not None:
        mask = ~np.asarray(mask, dtype=bool)
        nu_hat = nu_hat[mask]
        n = n[mask]
    nu_hat = np.clip(nu_hat, 1e-12, None)  # avoid log(0)
    return nu_hat, n

def score(
    n: np.ndarray,
    nu: np.ndarray,
    mask: np.ndarray | None = None,
    name: str = "Model",
) -> Score:
    """
    Compute a Score from raw observed counts n and predicted means nu.
    mask optionally filters both.
    k_params and fit_time_s are optional metadata.
    """
    nu_hat, n = _apply_mask_arrays(nu, n, mask)

    # Compute metrics
    loglik = float(poisson_loglikelihood(n, nu_hat).sum())
    dev = float(poisson_deviance(nu_hat, n).sum())
    chi2 = float(pearson_chi2(nu_hat, n).sum())
    rmse_val = float(np.sqrt(np.mean((nu_hat - n) ** 2)))
    mae_val = float(np.mean(np.abs(nu_hat - n)))
    bias_val = float(np.mean(nu_hat - n))
    n_obs = int(n.size)


    return Score(
        name=name,
        n_obs=n_obs,
        loglik=loglik,
        deviance=dev,
        pearson_chi2=chi2,
        rmse=rmse_val,
        mae=mae_val,
        mean_bias=bias_val,
        fit_time_s=None,
    )