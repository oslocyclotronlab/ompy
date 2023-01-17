from __future__ import annotations
from . import ResponseData
from .numbalib import index, jit, prange
from ..stubs import Axes, Pathlike
from .. import Vector
from dataclasses import dataclass
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import numpy as np
import warnings
from collections import OrderedDict
from typing import TypeAlias, Iterable
from abc import ABC, abstractmethod
from pathlib import Path


try:
    from numba import njit, int32, float32, float64
    from numba.experimental import jitclass
except ImportError:
    warnings.warn("Numba could not be imported. Falling back to non-jiting which will be much slower")
    int32 = np.int32
    float32 = np.float32
    float64 = np.float64

    def nop_decorator(func, *aargs, **kkwargs):
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    njit = nop_decorator
    jitclass = nop_decorator


@dataclass
class Interpolation(ABC):
    def __init__(self, points: Vector):
        self.points: Vector = points

    def __call__(self, points: float | Vector | np.ndarray) -> float | Vector | np.ndarray:
        match points:
            case Vector():
                y = self.eval(points.E)
                return points.clone(values=y)
            case _:
                return self.eval(points)

    @abstractmethod
    def eval(self, points: np.ndarray) -> np.ndarray: ...

    def save(self, path: Pathlike, exist_ok: bool = True) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=exist_ok)
        raise NotImplementedError()

    @staticmethod
    def from_file(path: Pathlike) -> Interpolation: ...

    def plot(self, ax: Axes = None, **kwargs) -> Axes:
        if ax is None:
            fig, ax = plt.subplots()
        x = np.linspace(min(1e-3, self.points.E.min()), max(self.points.E.max(), 3e4), 2000)
        y = self(x)
        X, Y = self.points.unpack()
        kwargs.setdefault('marker', 'x')
        #self.points.plot(ax=ax, kind='scatter')
        ax.plot(x, y)
        ax.scatter(X, Y, **kwargs)
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
        return self.points.E

    @property
    def y(self) -> np.ndarray:
        return self.points.values


class PoissonInterpolation(Interpolation):
    def plot(self, ax: Axes = None, **kwargs) -> Axes:
        if ax is None:
            fig, ax = plt.subplots()
        x = np.linspace(min(1e-3, self.points.E.min()), max(self.points.E.max(), 3e4), 2000)
        y = self(x)
        X, Y = self.points.unpack()
        ax.errorbar(X, Y, yerr=np.sqrt(Y), fmt='', capsize=5, linestyle='')
        ax.plot(x, y, **kwargs)
        ax.set_xlabel("E [keV]")
        ax.set_ylabel("Counts")
        return ax

class LinearInterpolation(Interpolation):
    def __init__(self, data: Vector, intp: interp1d):
        super().__init__(data)
        self.intp = intp

    def eval(self, points: np.ndarray) -> np.ndarray:
        return self.intp(points)

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
        return self.points.E

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


spec2 = OrderedDict()
spec2['x'] = float64[::1]
spec2['y'] = float64[::1]


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


# class MultiLerp:
#     """
#     Interpolate N points between X and Y parameterized by x and y
#
#     """
#     def __init__(self, x: float, X: np.ndarray,
#                  y: float, Y: np.ndarray):
#         self.x = x
#         self.y = y
#         self.X = X
#         self.Y = Y
#
#     def __call__(self, z: float) -> np.ndarray:
#         return self.call(z)
#
#     def call(self, z: float) -> np.ndarray:
#         if z < self.x or z > self.y:
#             raise ValueError("z is outside of interpolation range.")
#         Z = np.empty_like(self.x)
#         for i in range(len(Z)):


