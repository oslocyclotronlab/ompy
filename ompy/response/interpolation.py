from __future__ import annotations
from .numbalib import index, jit, prange
from ..stubs import Axes, Pathlike, LineKwargs, ErrorBarKwargs, Unitlike
from .. import Vector, __full_version__
from dataclasses import dataclass
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import numpy as np
import warnings
from collections import OrderedDict
from typing import Iterable, Any, Self, overload, Literal
from abc import ABC, abstractmethod
from pathlib import Path
import json
from scipy.interpolate import UnivariateSpline, InterpolatedUnivariateSpline


@dataclass
class Interpolation(ABC):
    __subclasses = {}
    def __init__(self, points: Vector, copy: bool = False):
        self.points: Vector = points if not copy else points.copy()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.__subclasses[cls.__name__] = cls

    def __call__(self, points: float | Vector | np.ndarray) -> float | Vector | np.ndarray:
        match points:
            case Vector():
                y = self.eval(points.X)
                return points.clone(values=y)
            case _:
                return self.eval(np.atleast_1d(points))

    @abstractmethod
    def eval(self, points: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def _metadata(self) -> dict[str, Any]: ...

    def save(self, path: Pathlike, exist_ok: bool = True) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=exist_ok)
        # Save attributes to json
        meta = self._metadata()
        if 'class' in meta or 'datapath' in meta or 'version' in meta:
            raise AssertionError(f"save() and _metadata() implemented incorrectly in {self.__class__.__name__}")
        meta['class'] = self.__class__.__name__
        meta['datapath'] = 'data.npy'
        meta['version'] = __full_version__
        with (path / 'meta.json').open('w') as f:
            json.dump(meta, f)
        self.points.save(path / meta['datapath'])

    @classmethod
    def from_path(cls, path: Pathlike) -> Self:
        path = Path(path)
        with (path / 'meta.json').open() as f:
            meta = json.load(f)
        expected_class = meta['class']
        cls_ = cls.__subclasses.get(expected_class)
        if cls.from_path == cls_.from_path:
            raise NotImplementedError(f"from_path() not implemented in {cls_.__name__}")
        return cls_.from_path(path)

    @staticmethod
    def _load(path: Pathlike) -> tuple[Vector, dict[str, any]]:
        path = Path(path)
        with (path / 'meta.json').open() as f:
            meta = json.load(f)
        version = meta['version']
        if version != __full_version__:
            warnings.warn(f"Loading data from version {version} into version {__full_version__}")
        data = Vector.from_path(path / meta['datapath'])
        return data, meta

    def plot(self, ax: Axes = None, skw: LineKwargs | None = None,
             lkw: LineKwargs | None = None, data: bool = True, intp: bool = True, **kwargs) -> Axes:
        if ax is None:
            fig, ax = plt.subplots()
        if data:
            skw = kwargs | (skw or {})
            self.plot_data(ax, **skw)
        if intp:
            lkw = kwargs | (lkw or {})
            self.plot_intp(ax, **lkw)
        return ax

    def plot_intp(self, ax: Axes = None, emax: None | float = None, **kwargs) -> Axes:
        if emax is None:
            emax = self.points.E.max()
        assert emax is not None
        x = np.linspace(min(1e-3, self.points.E.min()), emax, 4000)
        y = self(x)
        ax.plot(x, y, **kwargs)
        return ax

    def plot_data(self, ax: Axes = None, **kwargs) -> Axes:
        X, Y = self.points.unpack()
        skw = {}
        skw.setdefault('marker', 'x')
        skw.setdefault('mew', 0.5)
        skw.setdefault('linestyle', '')
        skw = kwargs | skw
        ax.plot(X, Y, **skw)
        return ax


    def plot_residuals(self, ax: Axes = None, **kwargs) -> Axes:
        if ax is None:
            fig, ax = plt.subplots()
        X, Y = self.points.unpack()
        y = self(X)
        m = Y > 0
        r = (Y - y)[m] / Y[m]
        ax.axhline(0, color='k', linestyle='--')
        ax.scatter(X[m], r, **kwargs)
        ax.set_xlabel("E [keV]")
        ax.set_ylabel(r"$\frac{y - \hat{y}}{y}$")
        return ax

    @property
    def x(self) -> np.ndarray:
        return self.points.X

    @property
    def y(self) -> np.ndarray:
        return self.points.values

    def to_same_unit(self, unit: Unitlike) -> float:
        return self.points.X_index.to_same_unit(unit)

    def clone(self, points: Vector | None = None, copy: bool = False) -> Self:
        return self.__class__(points or self.points, copy=copy)

    def copy(self, **kwargs) -> Self:
        return self.clone(copy=True, **kwargs)


class Scalable(Interpolation):
    def __init__(self, intp: Interpolation, C: float = 1.0, copy: bool = False):
        super().__init__(intp.points, copy=copy)
        self.intp = intp
        self.C = C

    def eval(self, points: np.ndarray) -> np.ndarray:
        return self.C * self.intp(points)

    @overload
    def scale(self, C: float, inplace: Literal[False]) -> Self: ...

    @overload
    def scale(self, C: float, inplace: Literal[True]) -> None: ...

    def scale(self, C: float, inplace=False) -> Self | None:
        factor = self.C * C
        if inplace:
            self.C = factor
            return self
        return self.clone(C=factor)

    def _metadata(self) -> dict[str, any]:
        meta = {"C": self.C, 'intp_path': 'scaled_intp.npy',
                'intp_class': self.intp.__class__.__name__}
        return meta

    @classmethod
    def from_path(cls, path: Pathlike) -> Self:
        path = Path(path)
        points, meta = Interpolation._load(path)
        C = meta["C"]
        intp = Interpolation.from_path(path / meta['intp_path'])
        return cls(intp, C=C)

    def save(self, path: Pathlike, exist_ok: bool = True) -> None:
        path = Path(path)
        super().save(path, exist_ok)
        meta = self._metadata()
        self.intp.save(path / meta['intp_path'])

    def clone(self, points: Vector | None = None, intp: Interpolation | None = None,
                C: float | None = None, copy: bool = False) -> Self:
            return type(self)(intp if intp is not None else self.intp,
                            C if C is not None else self.C,
                            copy=copy)

class PoissonInterpolation(Interpolation):
    def plot(self, ax: Axes = None, ebkw: ErrorBarKwargs | None = None,
             lkw: LineKwargs | None = None, **kwargs) -> Axes:
        if ax is None:
            fig, ax = plt.subplots()
        x = np.linspace(min(1e-3, self.points.E.min()), max(self.points.E.max(), 3e4), 2000)
        y = self(x)
        X, Y = self.points.unpack()
        ebkw = ebkw or {}
        ebkw.setdefault('ms', 2)
        ebkw.setdefault('ls', '')
        ebkw.setdefault('marker', 'o')
        #ebkw.setdefault('fmt', '')
        #ebkw.setdefault('capsize', 2)
        #ebkw.setdefault('capthick', 0.5)
        #ebkw.setdefault('elinewidth', 0.5)
        #ebkw.setdefault('linestyle', '')
        ebkw = kwargs | ebkw
        #ax.errorbar(X, Y, yerr=np.sqrt(Y), **ebkw)
        ax.plot(X, Y, **ebkw)
        lkw = lkw or {}
        lkw = kwargs | lkw
        ax.plot(x, y, **lkw)
        ax.set_xlabel("E [keV]")
        ax.set_ylabel("Counts")
        return ax


class LinearInterpolation(Interpolation):
    def __init__(self, data: Vector, intp: interp1d, copy: bool = False):
        super().__init__(data, copy=copy)
        self.intp = intp

    def eval(self, points: np.ndarray) -> np.ndarray:
        return self.intp(points)

    def _metadata(self) -> dict[str, any]:
        return {}

    @classmethod
    def from_path(cls, path: Pathlike) -> Self:
        data, meta = Interpolation._load(path)
        intp = interp1d(data.X, data.values, bounds_error=False, fill_value='extrapolate')
        return cls(data, intp)

    def clone(self, points: Vector | None = None, intp: interp1d | None = None,
              copy: bool = False) -> Self:
        return LinearInterpolation(points if points is not None else self.points,
                                   intp if intp is not None else self.intp,
                                   copy=copy)


class SplineInterpolation(Interpolation):
    def __init__(self, data: Vector, intp: UnivariateSpline, copy: bool = False,
                 kwargs = None):
        super().__init__(data, copy=copy)
        self.intp = intp
        self.kwargs = kwargs

    def eval(self, points: np.ndarray) -> np.ndarray:
        return self.intp(points)

    def _metadata(self) -> dict[str, any]:
        return self.kwargs

    @classmethod
    def from_path(cls, path: Pathlike) -> Self:
        data, meta = Interpolation._load(path)
        intp = UnivariateSpline(data.X, data.values)
        return cls(data, intp)

    def clone(self, points: Vector | None = None, intp: UnivariateSpline | None = None,
              copy: bool = False) -> Self:
        return SplineInterpolation(points if points is not None else self.points,
                                   intp if intp is not None else self.intp,
                                   copy=copy)


class ISplineInterpolation(Interpolation):
    def __init__(self, data: Vector, intp: InterpolatedUnivariateSpline, copy: bool = False,
                 kwargs = None):
        super().__init__(data, copy=copy)
        self.intp = intp
        self.kwargs = kwargs

    def eval(self, points: np.ndarray) -> np.ndarray:
        return self.intp(points)

    def _metadata(self) -> dict[str, any]:
        return self.kwargs

    @classmethod
    def from_path(cls, path: Pathlike) -> Self:
        data, meta = Interpolation._load(path)
        intp = InterpolatedUnivariateSpline(data.X, data.values)
        return cls(data, intp)

    def clone(self, points: Vector | None = None, intp: InterpolatedUnivariateSpline | None = None,
              copy: bool = False) -> Self:
        return type(self)(points if points is not None else self.points,
                          intp if intp is not None else self.intp,
                          copy=copy)


class CompoundInterpolation(Interpolation):
    def __init__(self, points: Vector, interpolations: OrderedDict[float, Interpolation]):
        self.points= points
        self.interpolations = interpolations

    def eval(self, points: np.ndarray) -> np.ndarray:
        boundaries = np.array(list(self.interpolations.keys()))
        intp = self.interpolations.values()
        return case_eval(points, boundaries, intp)


@jit(parallal=True)
def case_eval(X: np.ndarray, cases: np.ndarray, interpolations: Iterable[Interpolation]) -> np.ndarray:
    Y = np.empty_like(X)
    for i in prange(len(X)):
        x = X[i]
        if x < cases[0]:
            Y[i] = interpolations[0](x)
            continue
        if cases[-1] < x:
            Y[i] = interpolations[-1](x)
            continue
        for j in range(1, len(cases)):
            if cases[j-1] < x < cases[j]:
                Y[i] = interpolations[j](x)
                break
    return Y


class Interpolator(ABC):
    def __init__(self, points: Vector):
        self.points = points
        self.cov: None | np.ndarray = None

    def __call__(self) -> Interpolation:
        return self.interpolate()

    @abstractmethod
    def interpolate(self) -> Interpolation: ...

    @property
    def x(self) -> np.ndarray:
        return self.points.X

    @property
    def y(self) -> np.ndarray:
        return self.points.values

    def plot_cor(self, ax: Axes = None, **kwargs) -> Axes:
        if ax is None:
            fig, ax = plt.subplots()
        cor = self.cov / np.sqrt(np.outer(self.cov.diagonal(), self.cov.diagonal()))
        kwargs.setdefault('cmap', 'RdBu_r')
        kwargs.setdefault('vmin', -1)
        kwargs.setdefault('vmax', 1)
        m = ax.matshow(cor, **kwargs)
        ax.figure.colorbar(m, ax=ax)
        return ax


class LinearInterpolator(Interpolator):
    def interpolate(self) -> LinearInterpolation:
        intp = interp1d(self.x, self.y, kind="linear", fill_value="extrapolate")
        return LinearInterpolation(self.points, intp)


class SplineInterpolator(Interpolator):
    def interpolate(self, **kwargs) -> SplineInterpolation:
        intp = UnivariateSpline(self.x, self.y, **kwargs)
        return SplineInterpolation(self.points, intp, kwargs=kwargs)


class ISplineInterpolator(Interpolator):
    def interpolate(self, **kwargs) -> ISplineInterpolation:
        intp = InterpolatedUnivariateSpline(self.x, self.y, **kwargs)
        return ISplineInterpolation(self.points, intp, kwargs=kwargs)


class Lerp:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __call__(self, x):
        return self.call(x)

    def call(self, x):
        i = index(self.x, x)
        if x < self.x[i]:
            return 0.0
        if x > self.x[-1]:
            return 0.0

        x0 = self.x[i]
        x1 = self.x[i + 1]
        y0 = self.y[i]
        y1 = self.y[i + 1]
        t = (x - x0) / (x1 - x0)
        return (1 - t) * y0 + t * y1
