from __future__ import annotations
import numpy as np
from ..numbalib import njit, prange, NumpyArray
from .. import Vector, Matrix, Response, zeros_like
from ..stubs import Axes, Unitlike
from abc import ABC, abstractmethod
from dataclasses import dataclass
import matplotlib.pyplot as plt
from matplotlib import cm
from typing import Literal, TypeAlias
from tqdm.autonotebook import tqdm


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
        if not R.X_index.is_compatible_with(R.Y_index):
            raise ValueError("R must be square")
        if not R.X_index.is_compatible_with(G.X_index):
            raise ValueError(f"R and G must have the same axes.\n{R.X_index.summary()}\n{G.X_index.summary()}")
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

    def unfold_vector(self, data: Vector, initial: InitialVector = 'raw', R: str | Matrix = 'R', **kwargs) -> UnfoldedResult1D:
        R = self._resolve_response(R)
        if not R.is_compatible_with(data.X_index):
            raise ValueError("R and data must have the same axes")
        R = R.T

        initial = initial_vector(data, initial)
        return self._unfold_vector(R, data, initial, **kwargs)

    @abstractmethod
    def _unfold_vector(self, R: Matrix, data: Vector, initial: np.ndarray, **kwargs) -> UnfoldedResult1D: ...

    def unfold_matrix(self, data: Matrix, initial: InitialMatrix = 'raw', R: str | Matrix = 'R', **kwargs) -> Matrix:
        R = self._resolve_response(R)
        if self.R.X_index != data.Y_index:
            raise ValueError("R and data must have the same axes")
        R = R.T
        use_previous, initial = initial_matrix(data, initial)
        return self._unfold_matrix(R, data, initial, use_previous, **kwargs)

    def _unfold_matrix(self, R: Matrix, data: Matrix, initial: Matrix,
                       use_previous: bool, **kwargs) -> UnfoldedResult2D:
        """ A default, simple implementation of unfolding a matrix

         """
        best = np.zeros((data.shape[0], R.shape[1]))
        for i in tqdm(range(data.shape[0])):
            vec = data.iloc[i, :]
            # We only want to unfold up to the diagonal + resolution
            j = vec.last_nonzero()
            vec = vec.iloc[:j]
            if use_previous and i > 0:
                init = best[i-1, :j]
            else:
                init = initial.iloc[i, :j]
            R_ = R.iloc[:j, :j]
            res = self._unfold_vector(R_, vec, init, **kwargs)
            best[i, :j] = res.best()
        return UnfoldedResult2DSimple(R=R, raw=data, initial=initial, uall=best)

    def _resolve_response(self, R: str | Matrix) -> Matrix:
        match R:
            case Matrix():
                return R
            case 'R':
                return self.R
            case 'G':
                return self.G
            case 'GR':
                return self.G@self.R
            case _:
                raise ValueError(f"Invalid R {R}")


InitialVector: TypeAlias = Literal['raw', 'random'] | float | np.ndarray | Vector
InitialMatrix: TypeAlias = Literal['raw', 'random'] | float | np.ndarray | Matrix


def initial_vector(data: Vector, initial: InitialVector) -> Vector:
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


def initial_matrix(data: Matrix, initial: InitialMatrix) -> tuple[bool, Matrix]:
    match initial:
        case 'raw':
            return False, data.copy()
        case float():
            return False, float(initial) + zeros_like(data)
        case np.ndarray():
            return False, initial.copy()
        case Matrix():
            return False, initial.values.copy()
        case 'random':
            return False, np.random.poisson(np.median(data), data.shape)
        case 'previous':
            return True, data.copy()
        case _:
            raise ValueError(f"Invalid initial value {initial}")


@dataclass#(frozen=True, slots=True)
class UnfoldedResult1D(ABC):
    R: Matrix
    raw: Vector
    initial: Vector
    uall: np.ndarray
    cost: np.ndarray

    @abstractmethod
    def best(self) -> Vector: ...

    def best_fold(self) -> Vector:
        return self.R@self.best()

    def plot_cost(self, ax: Axes | None = None, **kwargs) -> Axes:
        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(self.cost[1:], **kwargs)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Cost")
        return ax

    def unfolded(self, index: int) -> Vector:
        return Vector(X=self.raw.X_index, values=self.uall[index])

    def folded(self, index: int) -> Vector:
        return self.R @ self.unfolded(index)

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
        self.initial.plot(ax=ax, color='k', label='Initial', **kwargs)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm._A = []  # Create empty array to avoid warning
        plt.subplots_adjust(bottom=0.1, right=0.85, top=0.9)
        cax = plt.axes([0.85, 0.1, 0.02, 0.8])
        cbar = ax.figure.colorbar(sm, cax=cax)
        cbar.set_label('Iteration')
        ticks = np.linspace(0.1, 0.9, 10)
        tick_labels = np.linspace(0, N, 10)
        cbar.set_ticks(ticks)
        cbar.set_ticklabels([f'{int(t)}' for t in tick_labels])
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
        (self.R@self.initial).plot(ax=ax, ls='-.', label='Initial', color='g')
        self.raw.plot(ax=ax, ls='--', label='Raw', color='r')
        ax.figure.suptitle('Refolded')
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm._A = []  # Create empty array to avoid warning

        plt.subplots_adjust(bottom=0.1, right=0.85, top=0.9)
        cax = plt.axes([0.85, 0.1, 0.02, 0.8])
        cbar = ax.figure.colorbar(sm, cax=cax)
        cbar.set_label('Iteration')
        ticks = np.linspace(0.1, 0.9, 10)
        tick_labels = np.linspace(0, N, 10)
        cbar.set_ticks(ticks)
        cbar.set_ticklabels([f'{int(t)}' for t in tick_labels])
        ax.set_title('')
        #cbar.set_tick_labels(np.linspace(0, N, 10))
        return ax

    def __len__(self) -> int:
        return len(self.uall)


@dataclass#(frozen=True, slots=True)
class UnfoldedResult2D(ABC):
    R: Matrix
    raw: Matrix
    initial: Matrix
    uall: np.ndarray

    @abstractmethod
    def best(self) -> Matrix: ...

    def best_fold(self) -> Matrix:
        return self.R@self.best()

    @abstractmethod
    def unfolded(self, Ex: Unitlike | slice | None = None) -> Matrix | Vector: ...

    @abstractmethod
    def folded(self, Ex: Unitlike | slice | None = None) -> Vector: ...

    def subset(self, mat: np.ndarray, Ex: Unitlike | slice | None = None) -> Matrix | Vector:
        if Ex is not None:
            i = self.raw.Ex_index.index_expression(Ex)
            mat = mat[i, :]
            if mat.ndim == 1:
                return Vector(X=self.raw.Y_index, values=mat)
            X = self.raw.Ex_index[Ex]
            return Matrix(X=X, Y=self.raw.Y_index, values=mat)
        return Matrix(X=self.raw.X_index, Y=self.raw.Y_index, values=mat)

    def __len__(self) -> int:
        return self.uall.shape[0]


@dataclass#(frozen=True, slots=True)
class UnfoldedResult2DCost(UnfoldedResult2D):
    cost: np.ndarray

    def best(self) -> Matrix:
        return self.unfolded(np.argmin(self.cost))

    def unfolded(self, index: int, Ex: Unitlike | slice | None = None) -> Matrix | Vector:
        mat = self.uall[index]
        return self.subset(mat, Ex)

    def folded(self, index: int, Ex: Unitlike | slice | None = None) -> Vector:
        return self.unfolded(index, Ex) @ self.R.T

    def plot_cost(self, ax: Axes | None = None, **kwargs) -> Axes:
        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(self.cost, **kwargs)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Cost")
        return ax


@dataclass#(frozen=True, slots=True)
class UnfoldedResult2DSimple(UnfoldedResult2D):
    def best(self) -> Matrix:
        return self.unfolded()

    def unfolded(self, Ex: Unitlike | slice | None = None) -> Matrix | Vector:
        mat = self.uall
        return self.subset(mat, Ex)

    def folded(self, Ex: Unitlike | slice | None = None) -> Vector:
        return self.unfolded(Ex) @ self.R.T
