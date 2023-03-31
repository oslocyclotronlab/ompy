from __future__ import annotations
import numpy as np
from ..numbalib import njit, prange
from .. import Vector, Matrix, Response, zeros_like, make_axes
from ..stubs import Axes, Unitlike
from abc import ABC, abstractmethod
from dataclasses import dataclass
import matplotlib.pyplot as plt
from matplotlib import cm
from typing import Literal, TypeAlias, overload
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

    @abstractmethod
    def supports_background() -> bool: ...

    @classmethod
    def from_response(cls, response: Response, data: Matrix | Vector, **kwargs) -> Unfolder:
        R = response.specialize_like(data)
        G = response.gaussian_like(data)
        return cls(R, G, **kwargs)

    @overload
    def unfold(self, data: Matrix, background: Matrix | None = None, **kwargs) -> Matrix:
        ...

    @overload
    def unfold(self, data: Vector, background: Vector | None = None, **kwargs) -> Vector:
        ...

    def unfold(self, data: Matrix | Vector, background: Matrix | Vector | None = None, **kwargs) -> Matrix | Vector:
        match data:
            case Matrix():
                return self.unfold_matrix(data, background, **kwargs)
            case Vector():
                return self.unfold_vector(data, background, **kwargs)
            case _:
                raise ValueError(f"Expected Matrix or Vector, got {type(data)}")

    def unfold_vector(self, data: Vector, background: Vector | None = None, initial: InitialVector = 'raw', R: str | Matrix = 'R', ignore_511: bool = False, **kwargs) -> UnfoldedResult1D:
        R = self._resolve_response(R)
        if not R.is_compatible_with(data.X_index):
            raise ValueError("R and data must have the same axes")
        if background is not None and not R.is_compatible_with(background.X_index):
            raise ValueError("The background has different index from the data.")
        R = R.T

        initial: Vector = initial_vector(data, initial)
        mask: Vector = make_mask(data, kwargs.pop('mask', None))
        if ignore_511:
            mask &= mask_511(data)
        return self._unfold_vector(R, data, background, initial, mask, **kwargs)

    @abstractmethod
    def _unfold_vector(self, R: Matrix, data: Vector, background: Vector, initial: np.ndarray, mask: Vector, **kwargs) -> UnfoldedResult1D: ...

    def unfold_matrix(self, data: Matrix, background: Matrix | None = None, initial: InitialMatrix = 'raw', R: str | Matrix = 'R', **kwargs) -> UnfoldedResult2D:
        R = self._resolve_response(R)
        if not self.R.X_index.is_compatible_with(data.Y_index):
            raise ValueError("R and data must have the same axes."
                             f"\n\nThe index of R:\n{R.X_index.summary()}"
                             f"\n\nThe index of data:\n{data.Y_index.summary()}")
        if background is not None and not background.is_compatible_with(data):
            raise ValueError("The background has different indices from the data.")
        R = R.T
        use_previous, initial = initial_matrix(data, initial)
        return self._unfold_matrix(R, data, background, initial, use_previous, **kwargs)

    def _unfold_matrix(self, R: Matrix, data: Matrix, background: Matrix | None, initial: Matrix,
                       use_previous: bool, **kwargs) -> UnfoldedResult2D:
        """ A default, simple implementation of unfolding a matrix

         """
        best = np.zeros((data.shape[0], R.shape[1]))
        mask_ = kwargs.pop('mask', None)
        do_ignore_511 = kwargs.pop('ignore_511', False)
        N = data.shape[0]
        time = np.zeros(N)
        bins = np.zeros(N)
        masks = np.zeros_like(data)
        for i in tqdm(range(N)):
            vec: Vector = data.iloc[i, :]
            # We only want to unfold up to the diagonal + resolution
            j = vec.last_nonzero()
            vec: Vector = vec.iloc[:j]
            if background is not None:
                bvec: Vector | None = background.iloc[i, :j]
            else:
                bvec = None
            if use_previous and i > 0:
                init = best[i-1, :j]
            else:
                init = initial.iloc[i, :j]
            R_: Matrix = R.iloc[:j, :j]
            mask = make_mask(vec, mask_)
            if do_ignore_511:
                mask &= mask_511(vec)
            res = self._unfold_vector(R_, vec, bvec, init, mask, **kwargs)
            best[i, :j] = res.best()
            masks[i, :j] = mask
            time[i] = res.meta.time
            bins[i] = j
        return UnfoldedResult2DSimple(ResultMeta2D(time=time, bins=bins), R=R, raw=data, background=background, initial=initial, uall=best, mask=mask)

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
            case 'RG':
                return self.R@self.G
            case _:
                raise ValueError(f"Invalid R {R}")


InitialVector: TypeAlias = Literal['raw', 'random'] | float | np.ndarray | Vector
InitialMatrix: TypeAlias = Literal['raw', 'random'] | float | np.ndarray | Matrix


def initial_vector(data: Vector, initial: InitialVector) -> Vector:
    match initial:
        case 'raw':
            return data.copy()
        case float():
            return data.clone(values=float(initial) + zeros_like(data))
        case np.ndarray():
            return data.clone(values=initial.copy())
        case Vector():
            return initial.copy()
        case 'random':
            return data.copy(values=np.random.poisson(np.median(data.values), len(data)))
        case _:
            raise ValueError(f"Invalid initial value {initial}")


def initial_matrix(data: Matrix, initial: InitialMatrix) -> tuple[bool, Matrix]:
    match initial:
        case 'raw':
            return False, data.copy()
        case float():
            return False, zeros_like(data) + initial
        case np.ndarray():
            return False, data.copy(values=initial)
        case Matrix():
            return False, initial.copy()
        case 'random':
            return False, data.copy(values=np.random.poisson(np.median(data), data.shape))
        case 'previous':
            return True, data.copy()
        case _:
            raise ValueError(f"Invalid initial value {initial}")


def mask_511(data: Vector) -> np.ndarray:
    mask = np.ones_like(data.values, dtype=bool)
    eps = 50
    start = 510-eps
    stop = 510+eps
    if stop < data.X_index.leftmost:
        return mask
    if start > data.X_index.rightmost:
        return mask
    stop = min(stop, data.X_index[-1])
    start = max(start, data.X_index[0])
    start = data.X_index.index(start)
    stop = data.X_index.index(stop)
    mask[start:stop] = False
    return mask

def make_mask(data: Vector, mask) -> Vector:
    match mask:
        case np.ndarray():
            return data.clone(values=mask, dtype=bool)
        case Vector():
            return mask
        case None:
            return data.clone(values=np.ones_like(data.values, dtype=bool),
                              dtype=bool)
        case _:
            return data.clone(values=mask(data), dtype=bool)


@dataclass(kw_only=True)
class ResultMeta:
    time: float
    ignore_511: bool = False


@dataclass(kw_only=True)
class UnfoldedResult1D(ABC):
    meta: ResultMeta
    R: Matrix
    raw: Vector
    background: Vector | None
    mask: Vector
    initial: Vector

    @abstractmethod
    def unfolded(self, *args, **kwargs) -> Vector: ...

    def folded(self, *args, **kwargs) -> Vector:
        return self.R @ self.unfolded(*args, **kwargs)

    @abstractmethod
    def best(self) -> Vector: ...

    def best_fold(self) -> Vector:
        return self.R@self.best()

    #@make_axes
    def plot_comparison(self, ax: Axes, mask: bool = True,
                        unfolded: bool = True, folded: bool = True,
                        raw: bool = True, error: bool = False,
                        **kwargs) -> Axes:
        if self.background is not None:
            self.background.plot(ax=ax, label='background')
            (self.raw - self.background).plot(ax=ax, label='raw - background')
        if raw:
            self.raw.plot(ax=ax, label="raw")
        if unfolded:
            self.best().plot(ax=ax, label="unfolded")
        if folded:
            self.best_fold().plot(ax=ax, label="refold")
        if mask and np.any(~self.mask.values):
            kwargs = {'color': 'gray', 'alpha': 0.1, 'edgecolor': None}
            ylim = ax.get_ylim()
            ax.fill_between(self.raw.X[~self.mask.values], *ylim, **kwargs)

        return ax

@dataclass(kw_only=True)
class UnfoldedResult1DSimple(UnfoldedResult1D):
    u: Vector

    def unfolded(self) -> Vector:
        return self.raw.clone(values=self.u)

    def best(self) -> Vector:
        return self.unfolded()


@dataclass(kw_only=True)
class Errors1DABC(UnfoldedResult1D):
    @abstractmethod
    def error(self) -> np.ndarray | None: ...

    def folded_error(self) -> np.ndarray | None:
        return None

    #@make_axes
    def plot_comparison(self, ax: Axes, fill: bool = False,
                        raw: bool = True, background: bool = True,
                        unfolded: bool = True, folded: bool = True,
                        mask: bool = True,
                        error: bool = True,
                        **kwargs) -> Axes:
        if self.error() is None or not error:
            return super().plot_comparison(ax=ax, raw=raw, **kwargs)
        if self.background is not None and background:
            self.background.plot(ax=ax, label='background')
            (self.raw - self.background).plot(ax=ax, label='raw - background')
        if raw:
            self.raw.plot(ax=ax, label="raw")
        x = self.raw.X + self.raw.dX/2
        y = self.best().values
        yerr = self.error()
        if unfolded:
            self.best().plot(ax=ax, label='unfolded')
            color = ax.lines[-1].get_color()
            if fill:
                ax.fill_between(x, y-yerr, y+yerr, alpha=0.2, color=color,
                                step='mid', edgecolor=None)
            else:
                kwargs = {'capsize': 2, 'fmt': 'none', 'capthick': 0.5, 'ms': 1} | kwargs
                ax.errorbar(x, self.best(), yerr=self.error(),
                            color=color, **kwargs)
        if folded:
            folded_error = self.folded_error()
            y = self.best_fold()
            y.plot(ax=ax, label='refolded')
            color = ax.lines[-1].get_color()
            y = y.values
            if folded_error is not None:
                if fill:
                    ax.fill_between(x, y - folded_error, y + folded_error,
                                    alpha=0.2, color=color, step='mid', edgecolor=None)
                else:
                    kwargs['marker'] = '^'
                    ax.errorbar(x, y, yerr=folded_error,
                                color=color,
                                **kwargs)

        if mask and np.any(~self.mask.values):
            maskkw = {'color': 'gray', 'alpha': 0.2, 'edgecolor': None}
            ylim = ax.get_ylim()
            ax.fill_between(self.raw.X[~self.mask.values], *ylim, **maskkw)
        return ax

@dataclass(kw_only=True)
class Errors1DSimple(Errors1DABC):
    err: np.ndarray
    def error(self) -> np.ndarray:
        return self.err

@dataclass(kw_only=True)
class Errors1DCovariance(Errors1DABC):
    cov: np.ndarray
    def error(self) -> np.ndarray | None:
        if self.cov is None:
            return None
        return np.sqrt(np.diag(self.cov))

    def folded_error(self) -> np.ndarray | None:
        if self.cov is None:
            return None
        V = self.R @ self.cov @ self.R.T
        return np.sqrt(np.diag(V))

    def cov_mat(self) -> Matrix:
        X = self.raw.X_index
        mat = Matrix(X=X, Y=X, values=self.cov)
        return mat

    def cor(self) -> np.ndarray:
        diag = self.cov.diagonal()
        return self.cov / np.sqrt(diag*diag.reshape(-1, 1))

    def cor_mat(self) -> Matrix:
        X = self.raw.X_index
        mat = Matrix(X=X, Y=X, values=self.cor())
        return mat



@dataclass#(frozen=True, slots=True)
class UnfoldedResult1DAll(UnfoldedResult1D):
    uall: np.ndarray
    cost: np.ndarray

    def unfolded(self, index: int) -> Vector:
        return Vector(X=self.raw.X_index, values=self.uall[index])


    @make_axes
    def plot_uall(self, ax: Axes, N: int | None = None, **kwargs) -> Axes:
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

    @make_axes
    def plot_fall(self, ax: Axes, N: int | None = None, **kwargs) -> Axes:
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


    @make_axes
    def plot_cost(self, ax: Axes, **kwargs) -> Axes:
        ax.plot(self.cost[1:], **kwargs)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Cost")
        return ax

    def __len__(self) -> int:
        return len(self.uall)


@dataclass(kw_only=True)
class ResultMeta2D:
    time: np.ndarray
    bins: np.ndarray
    ignore_511: bool = False


@dataclass(kw_only=True)
class UnfoldedResult2D(ABC):
    meta: ResultMeta2D
    R: Matrix
    raw: Matrix
    initial: Matrix
    background: Matrix
    mask: Matrix
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

    @make_axes
    def plot_time(self, ax: Axes, **kwargs) -> Axes:
        ax.plot(self.meta.bins, self.meta.time, **kwargs)
        ax.set_xlabel('Bins')
        ax.set_ylabel('Time (s)')
        return ax

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

    @make_axes
    def plot_cost(self, ax: Axes, **kwargs) -> Axes:
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
