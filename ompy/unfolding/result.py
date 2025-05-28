from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import (TYPE_CHECKING, Any, Generic, Never, Self, TypeAlias,
                    TypeVar, overload)
from warnings import warn

import numpy as np

from .. import JAX_AVAILABLE
from ..array import AbstractArray, Matrix, Vector, on_device
from ..helpers import print_readable_time
from ..stubs import Axes, Plot1D, Plot2D
from ..version import FULLVERSION, warn_version
from .result_classes import RESULT_CLASSES
from .stubs import PlotSpace, Space

if TYPE_CHECKING:
    from .resampling.resampling import Resampling
    from .unfolder import Unfolder

if JAX_AVAILABLE:
    import jax.numpy as jnp
else:
    import numpy as jnp


T = TypeVar("T", bound=Matrix | Vector)
UnfolderMethod: TypeAlias = str | type["Unfolder"]


@dataclass(kw_only=True)
class ResultMeta(ABC, Generic[T]):
    time: float
    space: Space
    method: UnfolderMethod
    parameters: Parameters[T]

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
    xi: tuple[T, ...] = ()
    beta: T | None = None
    do_fold_beta: bool = False
    _ndim: int = 0

    @property
    def ndim(self) -> int:
        if self._ndim == 0:
            self._ndim = self.meta.parameters.raw.ndim
        return self._ndim

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

    def best_folded(self, device="gpu?") -> T:
        best = self.best()
        with on_device(device, self.GegD, best, endpoint="numpy"):
            nu = best @ self.GegD
        return nu

    best_nu = best_folded
    nu = best_nu

    def best_eta(self, device="gpu?") -> T:
        best = self.best()
        match self.meta.space:
            case "mu":
                with on_device(device, self.G_eg, best, endpoint="numpy"):
                    return best @ self.G_eg
            case "eta":
                return best
            case _:
                raise ValueError(f"Cannot map from {self.meta.space} to eta")
                
    eta = best_eta

    def best_xi_mu(self, i: int) -> T:
        return self.xi[i]

    def best_xi_eta(self, i: int, device="gpu?") -> T:
        if not self.xi:
            raise ValueError("No xi to unfold")
        if i >= len(self.xi):
            raise ValueError(f"Index {i} out of bounds for xi of length {len(self.xi)}")
        match self.meta.space:
            case "mu":
                with on_device(device, self.G_eg, self.xi[i], endpoint="numpy"):
                    return self.xi[i] @ self.G_eg
            case "eta":
                return self.xi[i]
            case _:
                raise ValueError(f"Cannot map from {self.meta.space} to eta")

    @alias("best_xi_nu")
    def best_xi_folded(self, i: int, device="gpu?") -> T:
        xi_mu = self.best_xi_mu(i)
        with on_device(device, self.GegD, xi_mu, endpoint="numpy"):
            return xi_mu @ self.GegD


    def best_mu(self) -> T:
        if not self.meta.space == "mu":
            raise ValueError(
                f"Cannot map to mu when unfolded with space {self.meta.space}"
            )
        return self.best()

    def folded_total(self, device="gpu?") -> T:
        nu = self.best_folded(device=device)
        for i in range(len(self.xi)):
            nu = nu + self.best_xi_folded(i, device=device)
        if self.beta is not None:
            nu = nu + self.beta_folded(device=device)
        nu.title = "Total (nu)"
        return nu

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
    def D(self) -> Matrix:
        return self.meta.parameters.D

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
        return self.meta.parameters.GegD

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
        meta_cls: str = ResultMeta.read_subclass(path / "meta")
        meta_: ResultMeta = eval(meta_cls + ".from_path(path / 'meta')")
        other = cls._load(path, meta)
        return cls(meta=meta_, **other)

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
        if not inplace:
            return self

    @abstractmethod
    def resample(self, N: int, **kwargs) -> Resampling: ...


@dataclass(kw_only=True)
class Parameters(ABC, Generic[T]):
    raw: T
    background: T | None = None
    initial: T
    D: Matrix
    G_eg: Matrix
    G_ex: Matrix | None = None  # None should be interpreted as an identity matrix
    mask: np.ndarray | None = None
    _GegD: Matrix | None = None  # Cache of G_eg @ D
    kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Can eat up GPU memory
        if False:
            self.to_device("cpu")

    @property
    def GegD(self) -> Matrix:
        # Cache G_eg @ D because the product is so common
        if self._GegD is None:
            self._GegD = self.D @ self.G_eg
        return self._GegD

    @overload
    def to_device(self, device, inplace: bool = True) -> None: ...
    @overload
    def to_device(self, device, inplace: bool = True) -> Self: ...

    def to_device(self, device, inplace: bool = True) -> None | Self:
        raw = self.raw.to_device(device, inplace=inplace)
        if self.background is not None:
            background = self.background.to_device(device, inplace=inplace)
        initial = self.initial.to_device(device, inplace=inplace)
        D = self.D.to_device(device, inplace=inplace)
        G_eg = self.G_eg.to_device(device, inplace=inplace)
        if self.G_ex is not None:
            G_ex = self.G_ex.to_device(device, inplace=inplace)
        if self._GegD is not None:
            self._GegD = self._GegD.to_device(device, inplace=inplace)
        if not inplace:
            return Parameters(
                raw=raw,
                background=background,
                initial=initial,
                D=D,
                G_eg=G_eg,
                G_ex=G_ex,
                _GegD=self._GegD,
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
            background = self.background.as_numpy(inplace=inplace)
        initial = self.initial.as_numpy(inplace=inplace)
        D = self.D.as_numpy(inplace=inplace)
        G_eg = self.G_eg.as_numpy(inplace=inplace)
        if self.G_ex is not None:
            G_ex = self.G_ex.as_numpy(inplace=inplace)
        if self._GegD is not None:
            self._GegD = self._GegD.as_numpy(inplace=inplace)
        if not inplace:
            return Parameters(
                raw=raw,
                background=background,
                initial=initial,
                D=D,
                G_eg=G_eg,
                G_ex=G_ex,
                _GegD=self._GegD,
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
        self.D.save(path / "D.npz", exist_ok=exist_ok)
        self.G_eg.save(path / "G_eg.npz", exist_ok=exist_ok)
        if self.G_ex is not None:
            self.G_ex.save(path / "G_ex.npz", exist_ok=exist_ok)
        with open(path / "kwargs.json", "w") as f:
            kwargs = {}
            for k, v in self.kwargs.items():
                if isinstance(v, (np.ndarray, Matrix, Vector, AbstractArray)):
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

        D = Matrix.from_path(path / "D.npz")
        G_eg = Matrix.from_path(path / "G_eg.npz")
        G_ex = None
        if (path / "G_ex.npz").exists():
            G_ex = Matrix.from_path(path / "G_ex.npz")
        with open(path / "kwargs.json", "r") as f:
            kwargs = json.load(f)

        raw, background, initial = cls._load(path)

        return cls(
            raw=raw,
            background=background,
            initial=initial,
            D=D,
            G_eg=G_eg,
            G_ex=G_ex,
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
