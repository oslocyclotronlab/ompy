from __future__ import annotations
import numpy as np
from ..numbalib import njit, prange, NumpyArray
from .. import Vector, Matrix, Response, zeros_like
from ..stubs import Axes
from abc import ABC, abstractmethod
from dataclasses import dataclass
import matplotlib.pyplot as plt
from matplotlib import cm


class Unfolder(ABC):
    """ Abstract base class for unfolding algorithms

    Parameters
    ----------
    R: Matrix
        The unsmoothed response matrix
    G: Matrix
        The gaussian smoothing matrix
    """

    def __init__(self, R: Matrix, G: Matrix):
        if R.shape != G.shape:
            raise ValueError(f"R and G must have the same shape, got {R.shape} and {G.shape}")
        if R.X_index != R.Y_index:
            raise ValueError("R must be square")
        if R.X_index != G.X_index:
            raise ValueError("R and G must have the same axes")
        self.R: Matrix = R
        self.G: Matrix = G

    @classmethod
    def from_response(cls, response: Response, data: Matrix | Vector, **kwargs) -> Unfolder:
        R = response.specialize_like(data)
        G = response.gaussian_like(data)
        return cls(R, G, **kwargs)

    def unfold(self, data: Matrix | Vector, **kwargs) -> Matrix | Vector:
        match data:
            case Matrix():
                return self.unfold_matrix(data, **kwargs)
            case Vector():
                return self.unfold_vector(data, **kwargs)
            case _:
                raise ValueError(f"Expected Matrix or Vector, got {type(data)}")

    def unfold_vector(self, data: Vector, initial, R: str | Matrix = 'R', **kwargs) -> UnfoldedResult1D:
        match R:
            case Matrix():
                pass
            case 'R':
                R = self.R
            case 'G':
                R = self.G
            case 'GR':
                R = self.R@self.G
            case 'RG':
                R = self.G@self.R
            case _:
                raise ValueError(f"Invalid R {R}")
        if R.X_index != data.X_index:
            raise ValueError("R and data must have the same axes")
        R = R.T

        initial = initial_vector(data, initial)
        return self._unfold_vector(R, data, initial, **kwargs)

    @abstractmethod
    def _unfold_vector(self, R: Matrix, data: Vector, initial: np.ndarray, **kwargs) -> UnfoldedResult1D: ...

    def unfold_matrix(self, data: Matrix, **kwargs) -> Matrix:
        if self.R.X_index != data.X_index or self.R.Y_index != data.Y_index:
            raise ValueError("R and data must have the same axes")
        return self._unfold_matrix(data)

    @abstractmethod
    def _unfold_matrix(self, data: Matrix, **kwargs) -> Matrix: ...


def initial_vector(data: Vector, initial: str | float | np.ndarray | Vector) -> Vector:
    match initial:
        case 'raw':
            return data.copy()
        case float():
            return float(initial) + zeros_like(data)
        case np.ndarray():
            return initial.copy()
        case Vector():
            return initial.values.copy()
        case 'random':
            return np.random.poisson(np.median(data), len(data))
        case _:
            raise ValueError(f"Invalid initial value {initial}")

@dataclass(frozen=True, slots=True)
class UnfoldedResult1D(ABC):
    R: Matrix
    raw: Vector
    initial: Vector
    uall: np.ndarray
    cost: np.ndarray

    @abstractmethod
    def best(self) -> Vector: ...

    def plot_cost(self, ax: Axes | None = None, **kwargs) -> Axes:
        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(self.cost, **kwargs)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Cost")
        return ax

    def unfolded(self, index: int) -> Vector:
        return Vector(X=self.raw.X_index, values=self.uall[index])

    def plot_uall(self, ax: Axes | None = None, N: int | None = None, **kwargs) -> Axes:
        if ax is None:
            fig, ax = plt.subplots()
        M = N
        N = len(self)
        step = 1 if M is None else N // M
        M = N if M is None else M

        cmap = cm.get_cmap('Blues')
        vmin = 0.1
        vmax = 0.9
        cmap = cmap(np.linspace(vmin, vmax, M))  # Extract the colormap values within the range
        cmap = cm.colors.ListedColormap(cmap)
        for i in range(0, N, step):
            j = i // step
            self.unfolded(i).plot(ax=ax, color=cmap(j), **kwargs)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm._A = []  # Create empty array to avoid warning
        plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
        cax = plt.axes([0.85, 0.1, 0.075, 0.8])
        cbar = ax.figure.colorbar(sm, cax=cax)
        cbar.set_label('Iteration')
        #cbar.set_tick_labels(np.linspace(0, N, 10))
        ax.figure.suptitle('Unfolded')
        return ax

    def plot_fall(self, ax: Axes | None = None, N: int | None = None, **kwargs) -> Axes:
        if ax is None:
            fig, ax = plt.subplots()
        M = N
        N = len(self)
        step = 1 if M is None else N // M
        M = N if M is None else M

        cmap = cm.get_cmap('Blues')
        vmin = 0.1
        vmax = 0.9
        cmap = cmap(np.linspace(vmin, vmax, M))  # Extract the colormap values within the range
        cmap = cm.colors.ListedColormap(cmap)
        for i in range(0, N, step):
            (self.R@self.unfolded(i)).plot(ax=ax, color=cmap(i//step), **kwargs)
        self.raw.plot(ax=ax, ls='--', label='Raw', color='r')
        ax.figure.suptitle('Refolded')
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm._A = []  # Create empty array to avoid warning
        plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
        cax = plt.axes([0.85, 0.1, 0.075, 0.8])
        cbar = ax.figure.colorbar(sm, cax=cax)
        cbar.set_label('Iteration')
        ax.set_title('')
        #cbar.set_tick_labels(np.linspace(0, N, 10))
        return ax

    def __len__(self) -> int:
        return len(self.uall)
