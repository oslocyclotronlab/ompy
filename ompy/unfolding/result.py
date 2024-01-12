from __future__ import annotations
from dataclasses import dataclass, field, fields
from ..array import Matrix, Vector, AbstractArray
from ..stubs import Axes
from typing import Any, TypeVar, Generic, Never, TypeAlias
from abc import ABC, abstractmethod
from .stubs import Space, PlotSpace
from ..stubs import Plot1D, Plot2D
from pathlib import Path
import numpy as np
import json
from ..version import FULLVERSION, warn_version
from warnings import warn
from .result_classes import RESULT_CLASSES
from typing import TYPE_CHECKING
from ..helpers import print_readable_time

if TYPE_CHECKING:
    from .unfolder import Unfolder

T = TypeVar('T', bound=Matrix | Vector)
UnfolderMethod: TypeAlias = str | type['Unfolder']


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
        meta = dict(version=FULLVERSION, name=self.__class__.__name__,
                    time=self.time, space=self.space, method=method)
        with open(path / 'meta.json', 'w') as f:
            json.dump(meta, f)
        self.parameters.save(path / 'parameters', exist_ok=exist_ok)

    @classmethod
    def from_path(cls, path: Path) -> ResultMeta:
        path = Path(path)
        with open(path / 'meta.json', 'r') as f:
            meta = json.load(f)
        warn_version(meta['version'])
        if meta['name'] != cls.__name__:
            raise ValueError(f"Name mismatch: {meta['name']} != {cls.__name__}")
        paramfield = [f for f in fields(cls) if f.name == 'parameters'][0]
        parameters = eval(paramfield.type + ".from_path(path / 'parameters')")
        return cls(time=meta['time'], space=meta['space'], method=meta['method'], parameters=parameters)

    @classmethod
    def read_subclass(cls, path: Path) -> str:
        path = Path(path)
        with open(path / 'meta.json', 'r') as f:
            meta = json.load(f)
        warn_version(meta['version'])
        return meta['name']


@dataclass(kw_only=True)
class ResultMeta1D(ResultMeta[Vector]):
    parameters: Parameters1D


@dataclass(kw_only=True)
class ResultMeta2D(ResultMeta[Matrix]):
    parameters: Parameters2D

@dataclass(kw_only=True)
class Result(ABC, Generic[T]):
    meta: ResultMeta[T]

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        RESULT_CLASSES[cls.__name__] = cls

    def time(self) -> None:
        print_readable_time(self.meta.time)

    @abstractmethod
    def plot_comparison(self, ax: Axes | None = None, raw: bool = True, unfolded: bool = True,
                        initial: bool = False, folded: bool = True,
                        space: PlotSpace = 'base',
                        **kwargs) -> Plot1D: ...

    @abstractmethod
    def plot_comparison_to(self,other: Result | T,  ax: Axes | None = None, 
                           space: PlotSpace = 'eta', **kwargs) -> Plot1D | Plot2D: ...

    @abstractmethod
    def best(self) -> T: ...

    def best_folded(self) -> T:
        return self.R@self.best()

    def best_eta(self) -> T:
        if self.meta.space in {'GR', 'RG'}:
            return self.G@self.best()
        elif self.meta.space == 'R':
            return self.best()
        else:
            raise ValueError(f"Cannot map from {self.meta.space} to eta")

    def resolve_spaces(self, target: PlotSpace) -> tuple[T, str]:
        label = 'unfolded'
        if self.meta.space in {'GR', 'RG'}:
            if target == 'eta':
                label = 'G@' + label
                return self.best_eta(), label
            elif target in {'base', 'mu'}:
                return self.best(), label
        elif self.meta.space == 'R':
            if target in {'eta', 'base'}:
                return self.best_eta(), label
        raise ValueError(f"Cannot map from {self.meta.space} to {target}")

    def residuals(self) -> T:
        return self.raw - self.best_folded()

    @property
    def R(self) -> Matrix:
        return self.meta.parameters.R

    @property
    def G(self) -> Matrix:
        return self.meta.parameters.G

    @property
    def G_ex(self) -> Matrix | None:
        return self.meta.parameters.G_ex

    @property
    def raw(self) -> T:
        return self.meta.parameters.raw

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
        meta = dict(version=FULLVERSION, name=self.__class__.__name__)
        with open(path / 'meta.json', 'w') as f:
            json.dump(meta, f)
        # Result Meta
        self.meta.save(path / 'meta', exist_ok=exist_ok)
        self._save(path, exist_ok=exist_ok)

    @abstractmethod
    def _save(self, path: Path, exist_ok: bool = False) -> None: ...

    @classmethod
    def from_path(cls, path: Path) -> Result:
        path = Path(path)
        with open(path / 'meta.json', 'r') as f:
            meta = json.load(f)
        warn_version(meta['version'])
        if meta['name'] != cls.__name__:
            raise ValueError(f"Name mismatch: {meta['name']} != {cls.__name__}")
        meta_cls: str = ResultMeta.read_subclass(path / 'meta')
        meta_: ResultMeta = eval(meta_cls + ".from_path(path / 'meta')")
        other = cls._load(path)
        return cls(meta=meta_, **other)

    @classmethod
    @abstractmethod
    def _load(cls, path: Path) -> dict[str, Any]: ...



@dataclass(kw_only=True)
class Parameters(ABC, Generic[T]):
    raw: T
    background: T | None = None
    initial: T
    R: Matrix
    G: Matrix
    G_ex: Matrix | None = None
    kwargs: dict[str, Any] = field(default_factory=dict)

    def save(self, path: Path, exist_ok: bool = False) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=exist_ok)
        meta = dict(version=FULLVERSION, name=self.__class__.__name__)
        with open(path / 'meta.json', 'w') as f:
            json.dump(meta, f)

        self.raw.save(path / 'raw.npz', exist_ok=exist_ok)
        if self.background is not None:
            self.background.save(path / 'background.npz', exist_ok=exist_ok)
        self.initial.save(path / 'initial.npz', exist_ok=exist_ok)
        self.R.save(path / 'R.npz', exist_ok=exist_ok)
        self.G.save(path / 'G.npz', exist_ok=exist_ok)
        if self.G_ex is not None:
            self.G_ex.save(path / 'G_ex.npz', exist_ok=exist_ok)
        with open(path / 'kwargs.json', 'w') as f:
            kwargs = {}
            for k, v in self.kwargs.items():
                if isinstance(v, (np.ndarray, Matrix, Vector, AbstractArray)):
                    continue
                kwargs[k] = v
            json.dump(kwargs, f)

    @classmethod
    def from_path(cls, path: Path) -> Parameters:
        path = Path(path)
        with open(path / 'meta.json', 'r') as f:
            meta = json.load(f)
        warn_version(meta['version'])
        if meta['name'] != cls.__name__:
            raise ValueError(f"Class mismatch: {meta['name']} != {cls.__name__}")

        R = Matrix.from_path(path / 'R.npz')
        G = Matrix.from_path(path / 'G.npz')
        G_ex = None
        if (path / 'G_ex.npz').exists():
            G_ex = Matrix.from_path(path / 'G_ex.npz')
        with open(path / 'kwargs.json', 'r') as f:
            kwargs = json.load(f)

        raw, background, initial = cls._load(path)

        return cls(raw=raw, background=background, initial=initial,
                   R=R, G=G, G_ex=G_ex, kwargs=kwargs)

    @classmethod
    @abstractmethod
    def _load(cls, path: Path) -> tuple[T, T| None, T]: ...


@dataclass(kw_only=True)
class Parameters1D(Parameters[Vector]):
    raw: Vector
    background: Vector | None = None
    initial: Vector

    @classmethod
    def _load(cls, path: Path) -> tuple[Vector, Vector | None, Vector]:
        raw = Vector.from_path(path / 'raw.npz')
        background = Vector.from_path(path / 'background.npz') if (path / 'background.npz').exists() else None
        initial = Vector.from_path(path / 'initial.npz')
        return raw, background, initial


@dataclass(kw_only=True)
class Parameters2D(Parameters[Matrix]):
    raw: Matrix
    background: Matrix | None = None
    initial: Matrix

    @classmethod
    def _load(cls, path: Path) -> tuple[Matrix, Matrix | None, Matrix]:
        raw = Matrix.from_path(path / 'raw.npz')
        background = Matrix.from_path(path / 'background.npz') if (path / 'background.npz').exists() else None
        initial = Matrix.from_path(path / 'initial.npz')
        return raw, background, initial




def get_field(cls, name):
    matches = [f for f in fields(cls) if f.name == name]
    if not len(matches):
        raise ValueError(f"No field {name} in {cls}")
    return matches[0]
