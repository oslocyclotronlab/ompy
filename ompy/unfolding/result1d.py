from __future__ import annotations
from .result import Result, PlotSpace, ResultMeta1D, Parameters1D, Result
from .. import Matrix, Vector
from ..helpers import make_ax, maybe_set
from ..stubs import Lines, Plot1D, Plots1D, Axes
from ..array import ErrorVector, SymmetricVector, ErrorPlotKind, CorrelationMatrix
from ..response import Components
from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import overload, TypeGuard, TypeVar, Sequence, TypeAlias




@dataclass(kw_only=True)
class UnfoldedResult1D(Result[Vector]):
    meta: ResultMeta1D

    #@overload
    #def background(self) -> Vector | None: ...

    @abstractmethod
    def unfolded(self, *args, **kwargs) -> Vector: ...

    def folded(self, *args, **kwargs) -> Vector:
        return self.R @ self.unfolded(*args, **kwargs)

    # @make_axes
    def plot_comparison(self, ax: Axes | None = None,
                        unfolded: bool = True, folded: bool = True,
                        raw: bool = True,
                        initial: bool = False,
                        space: PlotSpace = 'base',
                        **kwargs) -> Plots1D:
        ax = make_ax(ax)
        lines: list[Lines] = []
        if self.background is not None:
            _,  line = self.background.plot(ax=ax, label='background')
            lines.append(line)
            _, line = (self.raw - self.background).plot(ax=ax, label='raw - background')
            lines.append(line)
        if raw:
            _, line = self.raw.plot(ax=ax, label="raw")
            lines.append(line)
        if initial:
            _, line = self.initial.plot(ax=ax, label="initial")
            lines.append(line)
        if unfolded:
            vec, label = self.resolve_spaces(space)
            _, line = vec.plot(ax=ax, label=label)
            lines.append(line)
        if folded:
            _, line = self.best_folded().plot(ax=ax, label="refold")
            lines.append(line)

        return ax, lines

    def plot_comparison_to(self, other: UnfoldedResult1D | Vector, ax: Axes | None = None,
                           space: PlotSpace = 'eta',
                           raw: bool = True,
                           unfolded: bool = True,
                           **kwargs) -> Plots1D:
        ax = make_ax(ax)
        if isinstance(other, UnfoldedResult1D):
            other_vec: Vector = other.resolve_spaces(space)[0]
        else:
            other_vec = other

        lines: list[Lines] = []
        self_vec, label = self.resolve_spaces(space)
        _, line = self_vec.plot(ax=ax, label=label)
        return ax, lines

    def plot_residuals(self, ax: Sequence[Axes] | None = None,
                      absolute: bool = True, relative: bool = True,
                    space: PlotSpace = 'base',
                    **kwargs) -> Plots1D:
        if ax is None:
            fig, ax = plt.subplots(nrows=absolute + relative, sharex=True,
                                   constrained_layout = True)  # type: ignore
        assert ax is not None
        ax = ax.flatten()  # type: ignore
        lines: list[Lines] = []
        raw = self.raw
        nu = self.best_folded()
        if self.background is not None:
            raw = raw - self.background
        if absolute:
            maybe_set(ax[0], ylabel=r"$\mathrm{raw}- \hat{\nu}$")
            _, line = (raw - nu).plot(ax=ax[0], **kwargs)
            lines.append(line)
        if relative:
            maybe_set(ax[1], ylabel=r"$\left|\mathrm{raw} - \hat{\nu}\right|/\mathrm{raw}$")
            _, line = (abs(raw - nu)/raw).plot(ax=ax[1], **kwargs)
            lines.append(line)

        return ax, lines

@dataclass(kw_only=True)
class UnfoldedResult1DSimple(UnfoldedResult1D):
    u: Vector

    def unfolded(self, *args, **kwargs) -> Vector:
        return self.u

    def best(self) -> Vector:
        return self.u.copy()

    def _save(self, path: Path, exist_ok: bool = False):
        self.u.save(path / 'u.npz', exist_ok=exist_ok)

    @classmethod
    def _load(cls, path: Path) -> dict[str, Vector]:
        u = Vector.from_path(path / 'u.npz')
        return {'u': u}

@dataclass(kw_only=True)
class UnfoldedResult1DMultiple(UnfoldedResult1D):
    u: np.ndarray

    def unfolded(self, i: int) -> Vector:
        return self.raw.clone(values=self.u[i, :])

    def _save(self, path: Path, exist_ok: bool = False):
        np.save(path / 'u.npy', self.u)

    @classmethod
    def _load(cls, path: Path) -> dict[str, np.ndarray]:
        u = np.load(path / 'u.npy')
        return {'u': u}


@dataclass(kw_only=True)
class Errors1DABC(UnfoldedResult1D):
    def good_cov(self) -> bool:
        return hasattr(self, 'cov') and self.cov is not None

    @abstractmethod
    def error(self) -> np.ndarray | None: ...

    def best_eta(self) -> Vector:
        eta = super().best_eta()
        if self.good_cov():
            cov = self.cov_eta()
            transformed = SymmetricVector.add_error(eta, cov)
            return transformed
        return eta

    def cov_eta(self) -> Matrix:
        if not self.good_cov():
            raise RuntimeError("Can not transform without covariance matrix")
        if self.meta.space in {'GR', 'RG'}:
            cov = self.G@self.cov@self.G.T
            return cov
        return self.cov

    def folded_error(self) -> np.ndarray | None:
        return None

    def best_error(self) -> ErrorVector:
        error = self.error()
        if error is None:
            raise RuntimeError("No error stored")
        return SymmetricVector.add_error(self.best(), error)

    def best_folded_error(self) -> ErrorVector:
        error = self.error()
        if error is None:
            raise RuntimeError("No error stored")
        return SymmetricVector.add_error(self.best_folded(), error)

    # @make_axes
    def plot_comparison(self, ax: Axes, fill: bool = False,
                        raw: bool = True, background: bool = True,
                        unfolded: bool = True, folded: bool = True,
                        error: bool = True,
                        initial: bool = False,
                        ekind: ErrorPlotKind = 'line',
                        space: PlotSpace = 'base',
                        **kwargs) -> Plot1D:
        if self.error() is None or not error:
            return super().plot_comparison(ax=ax, raw=raw, **kwargs)

        lines: list[Lines] = []
        if self.background is not None and background:
            _, line = self.background.plot(ax=ax, label='background')
            lines.append(line)
            _, line = (self.raw - self.background).plot(ax=ax, label='raw - background')
            lines.append(line)
        if raw:
            _, line = self.raw.plot(ax=ax, label="raw")
            lines.append(line)
        if initial:
            _, line = self.initial.plot(ax=ax, label="initial")
            lines.append(line)
        if unfolded:
            vec, label = self.resolve_spaces(space)
            _, line = vec.plot(ax=ax, label=label, ekind=ekind)
            lines.append(line)
        if folded:
            if self.folded_error() is None:
                _, line = self.best_folded().plot(ax=ax, label='refolded')
                lines.append(line)
            else:
                _, line = self.best_folded_error().plot(ax=ax, label='refolded', ekind=ekind)
                lines.append(line)
        return ax, lines


@dataclass(kw_only=True)
class Errors1DSimple(Errors1DABC):
    err: np.ndarray

    def error(self) -> np.ndarray:
        return self.err

    def _save(self, path: Path, exist_ok: bool = False):
        np.save(path / 'err.npy', self.err)

    @classmethod
    def _load(cls, path: Path) -> dict[str, np.ndarray]:
        err = np.load(path / 'err.npy')
        return {'err': err}

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
        mat = CorrelationMatrix(X=X, Y=X, values=self.cor())
        return mat

    def _save(self, path: Path, exist_ok: bool = False):
        np.save(path / 'cov.npy', self.cov)

    @classmethod
    def _load(cls, path: Path) -> dict[str, np.ndarray]:
        cov = np.load(path / 'cov.npy')
        return {'cov': cov}

def has_cost(res: Result) -> TypeGuard[Cost1D]:
    if hasattr(res, 'cost'):
        return True
    return False

T = TypeVar('T', bound=Matrix | Vector)

@dataclass(kw_only=True)
class Cost1D(Result[T]):
    cost: np.ndarray

    def plot_cost(self, ax: Axes | None = None, start: int | float = 0, relative: bool = False, **kwargs) -> Plot1D:
        ax = make_ax(ax)
        if isinstance(start, float):
            start = int(start*len(self.cost))
        cost = self.cost[start:]
        if relative:
            cost /= cost[0]
            
        x = np.arange(start, len(self.cost))
        line, = ax.plot(x, cost, **kwargs)
        maybe_set(ax, xlabel='iteration', ylabel='cost')
        return ax, line

    def _save(self, path: Path, exist_ok: bool = False):
        np.save(path / 'cost.npy', self.cost)

    @classmethod
    def _load(cls, path: Path) -> dict[str, np.ndarray]:
        cov = np.load(path / 'cost.npy')
        return {'cost': cov}


@dataclass(kw_only=True)
class ComponentsRes(Result[T]):
    components: Components

    def _save(self, path: Path, exist_ok: bool = False) -> None:
        self.components.save(path / 'components', exist_ok=exist_ok)

    @classmethod
    def _load(cls, path: Path) -> dict[str, Components]:
        components = Components.from_path(path / 'components')
        return {'components': components}

