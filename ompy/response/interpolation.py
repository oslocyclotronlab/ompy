from __future__ import annotations
from . import ResponseData
from .numbalib import index
from ..stubs import Axes
from dataclasses import dataclass
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import numpy as np
import warnings
from collections import OrderedDict

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
class ResponseInterpolation:
    data: ResponseData
    FE: interp1d
    SE: interp1d
    DE: interp1d
    AP: interp1d
    Eff: interp1d
    FWHM: interp1d

    @staticmethod
    def from_data(data: ResponseData) -> ResponseInterpolation:
        # TODO Fix interpolation
        return ResponseInterpolation(
            data=data,
            FE=interp1d(data.E, data.FE),
            SE=interp1d(data.E, data.SE),
            DE=interp1d(data.E, data.DE),
            AP=interp1d(data.E, data.AP),
            Eff=interp1d(data.E, data.Eff),
            FWHM=interp1d(data.E, data.FWHM, fill_value="extrapolate"),
        )

    def plot(self, ax: Axes | None = None, **kwargs):
        if ax is None:
            _, ax = plt.subplots(3, 2)
        ax = ax.flatten()
        if len(ax) < 5:
            raise ValueError("Need at least 5 axes")
        E = self.E
        ax[0].plot(E, self.FE(E), **kwargs)
        ax[1].plot(E, self.SE(E), **kwargs)
        ax[2].plot(E, self.DE(E), **kwargs)
        ax[3].plot(E, self.AP(E), **kwargs)
        ax[4].plot(E, self.Eff(E), **kwargs)
        ax[5].plot(E, self.FWHM(E), **kwargs)
        return ax

    @property
    def E(self) -> np.ndarray:
        return self.data.E

    def sigma(self, E: np.ndarray) -> np.ndarray:
        return E*self.FWHM(E) / 2.355


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


