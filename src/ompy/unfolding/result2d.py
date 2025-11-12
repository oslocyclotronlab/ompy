from __future__ import annotations
from typing import TYPE_CHECKING, Any, Iterable, Type
from .result import Result, PlotSpace, ResultMeta2D
from ..array import Matrix, on_device
from ..stubs import Lines, Plots2D, Plot1D, array2D, Axes
from dataclasses import dataclass, field
from abc import abstractmethod
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

if TYPE_CHECKING:
    from .resampling.resampling import Resampling2D


def cache(func):
    """Decorator to cache the result of a function call.
    Like functools, but with a simpler implementation.
    """
    value = None
    is_set = False
    def wrapper(*args, **kwargs):
        nonlocal value, is_set
        if not is_set:
            value = func(*args, **kwargs)
            is_set = True
        return value
    return wrapper


@dataclass(kw_only=True)
class UnfoldedResult2D(Result):
    meta: ResultMeta2D

    @staticmethod
    def _vec_or_mat() -> Type[Matrix]:
        return Matrix

    @abstractmethod
    def best(self) -> Matrix: ...

    #@cache
    def best_folded(self, device='gpu?') -> Matrix:
        best = self.best()
        if self.G_ex is None:
            with on_device(device, best, self.GegD, endpoint='numpy'):
                m = best@self.GegD
        else:
            with on_device(device, best, self.GegD, self.G_ex, endpoint='numpy'):
                m = self.G_ex@best@self.GegD
        return self.raw.clone(values=m, name='nu')  # Fix labels
    best_nu = best_folded

    #@cache
    def best_eta(self, device='gpu?') -> Matrix:
        match self.meta.space:
            case 'mu':
                best = self.best()
                if self.G_ex is None:
                    with on_device(device, best, self.G_eg, endpoint='numpy'):
                        m = best@self.G_eg
                else:
                    with on_device(device, best, self.G_eg, self.G_ex, endpoint='numpy'):
                        m = self.G_ex@best@self.G_eg
            case 'eta':
                m = self.best()
            case _:
                raise ValueError(f"Cannot map from {self.meta.space} to eta")
        return self.raw.clone(values=m, name='eta')  # Fix labels

    def beta_eta(self, device='gpu?') -> Matrix:
        if self.beta is None:
            raise ValueError("No beta to fold")
        if self.do_fold_beta:
            m = self.G_ex@self.beta@self.G_eg
        else:
            return self.beta
        return self.raw.clone(values=m, name='beta (eta)')  # Fix labels

    def beta_nu(self, device='gpu?') -> Matrix:
        if self.beta is None:
            raise ValueError("No beta to fold")
        if self.do_fold_beta:
            m = self.G_ex@self.beta@self.GegD
        else:
            return self.beta
        return self.raw.clone(values=m, name='beta (nu)')  # Fix labels

    beta_folded = beta_nu

    def best_contaminant_eta(self, i: int, device='gpu?') -> Matrix:
        match self.meta.space:
            case 'mu':
                best = self.best_contaminant_mu(i)
                if self.G_ex is None:
                    with on_device(device, best, self.G_eg, endpoint='numpy'):
                        m = best@self.G_eg
                else:
                    with on_device(device, best, self.G_eg, self.G_ex, endpoint='numpy'):
                        m = self.G_ex@best@self.G_eg
            case 'eta':
                m = self.best_contaminant_mu(i)
            case _:
                raise ValueError(f"Cannot map from {self.meta.space} to eta")
        return self.raw.clone(values=m, name='eta')  # Fix labels

    def best_contaminant_nu(self, i: int, device='gpu?') -> Matrix:
        best = self.best_contaminant_mu(i)
        if self.G_ex is None:
            with on_device(device, best, self.GegD, endpoint='numpy'):
                m = best@self.GegD
        else:
            
            with on_device(device, best, self.GegD, self.G_ex, endpoint='numpy'):
                m = self.G_ex@best@self.GegD
        return self.raw.clone(values=m, name='nu')  # Fix labels


    def plot_comparison(self, ax: Axes | None = None, raw: bool = True, unfolded: bool = True,
                        initial: bool = False, folded: bool = True,
                        space: PlotSpace = 'base',
                        **kwargs) -> Plots2D:
        vmin = np.inf
        vmax = -np.inf
        if raw:
            vmin = min(vmin, self.raw.values.min())
            vmax = max(vmax, self.raw.values.max())
        if unfolded:
            vmin = min(vmin, self.best().values.min())
            vmax = max(vmax, self.best().values.max())
        if initial:
            vmin = min(vmin, self.initial.values.min())
            vmax = max(vmax, self.initial.values.max())
        if folded:
            vmin = min(vmin, self.best_folded().values.min())
            vmax = max(vmax, self.best_folded().values.max())
        if vmin <= 0 or vmax <= 0:
            kwargs.setdefault('scale', 'symlog')
        kwargs.setdefault('vmin', vmin)
        kwargs.setdefault('vmax', vmax)
        N = raw + unfolded + initial + folded
        i = 0
        if ax is None:
            fig, ax = plt.subplots(sharex=True, sharey=True, ncols=N, layout='constrained', figsize=(10, 5))
        assert ax is not None
        ax: list[Axes] = ax.flatten()
        res = []
        add_cbar_ = kwargs.pop('add_cbar', True)
        if raw:
            add_cbar = i == N-1 and add_cbar_
            _, r = self.raw.plot(ax=ax[i], add_cbar=add_cbar, **kwargs)
            res.append(r)
            ax[i].set_title('Raw')
            i += 1
        if folded:
            add_cbar = i == N-1 and add_cbar_
            _, r = self.best_folded().plot(ax=ax[i], add_cbar=add_cbar, **kwargs)
            res.append(r)
            ax[i].set_title('Folded')
            i += 1
        if unfolded:
            add_cbar = i == N-1 and add_cbar_
            mat, title = self.resolve_spaces(space)
            _, r = mat.plot(ax=ax[i], add_cbar=add_cbar, **kwargs)
            res.append(r)
            ax[i].set_title(title)
            i += 1
        if initial:
            add_cbar = i == N-1 and add_cbar_
            mat, label = self.resolve_spaces(space)
            _, r = self.initial.plot(ax=ax[i], add_cbar=add_cbar, **kwargs)
            res.append(r)
            ax[i].set_title('Initial')
            i += 1
        xlabel = ax[0].get_xlabel()
        ylabel = ax[0].get_ylabel()
        for a in ax:
            a.set_xlabel('')
            a.set_ylabel('')
        ax[0].figure.supxlabel(xlabel)
        ax[0].figure.supylabel(ylabel)
        return ax, res


    def plot_comparison_at(self, at, ax: Axes | None = None,
                           raw: bool = True, unfolded: bool = True,
                           initial: bool = False, folded: bool = True,
                           space: PlotSpace = 'base',
                           **kwargs) -> Plot1D:

        if ax is None:
            fig, ax = plt.subplots()
        assert ax is not None
        U = self.best().loc[at, :]
        F = self.best_folded().loc[at, :]
        R = self.raw.loc[at, :]
        I = self.initial.loc[at, :]
        lines: list[Lines] = []
        if raw:
            _, line = R.plot(ax=ax, label="raw")
            lines.append(line)
        if initial:
            _, line = I.plot(ax=ax, label="initial")
            lines.append(line)
        if unfolded:
            label = 'unfolded'
            if space == 'eta':
                if self.meta.space in {'GR', 'RG'}:
                    U = self.G@U
                    label = 'G@unfolded'
                _, line = U.plot(ax=ax, label=label)
            else:
                _, line = U.plot(ax=ax, label=label)
            lines.append(line)
        if folded:
            _, line = F.plot(ax=ax, label="refold")
            lines.append(line)

        return ax, lines


    def plot_comparison_to(self, other: Result, ax: Axes | None = None, space: PlotSpace = 'eta', **kwargs) -> Plot1D | Plots2D:
        raise NotImplementedError()

    def resample(self, N: int, **kwargs) -> Resampling2D:
        from .resampling.resample2d import resample_matrix 
        return resample_matrix(self, N, **kwargs)


@dataclass(kw_only=True)
class UnfoldedResult2DSimple(UnfoldedResult2D):
    u: Matrix
    def best(self) -> Matrix:
        return self.u.copy()

    def _save(self, path: Path, meta: dict[str, Any], exist_ok: bool = False):
        self.u.save(path / 'u.npz', exist_ok=exist_ok)

    @classmethod
    def _load(cls, path: Path, meta: dict[str, Any]) -> dict[str, Matrix]:
        u = Matrix.from_path(path / 'u.npz')
        return {'u': u}


@dataclass(kw_only=True)
class Cost2D(Result):
    cost: array2D
    aux: dict[str, np.ndarray] = field(default_factory=dict)

    def plot_cost(self, ax: Axes | None = None, auxiliary: bool | Iterable[str] = True, **kwargs) -> Plot1D:
        if ax is None:
            fig, ax = plt.subplots()
        assert ax is not None

        if isinstance(start, float):
            start = int(start*len(self.cost))
        cost = self.cost[start:]

        lines = []
        cmap = kwargs.pop('cmap', 'turbo')
        colormap = plt.get_cmap(cmap)
        N = self.cost.shape[1]
        colors = [colormap(i) for i in np.linspace(0, 1, N)]
        for i, c in enumerate(self.cost.T):
            ax.plot(c, color=colors[i],  **kwargs)
        # Create a "fake" mappable for the colorbar
        index = self.raw.Y
        norm = plt.Normalize(index.min(), index.max())
        sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
        sm.set_array([])

        # Add the colorbar
        cbar = ax.figure.colorbar(sm, ax=ax, orientation='vertical')
        cbar.set_label(self.raw.get_xlabel())
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Cost")
        return ax, lines

    def _save(self, path: Path, meta: dict[str, Any], exist_ok: bool = False):
        np.save(path / 'cost.npy', self.cost)
        for k, v in self.aux.items():
            np.save(path / f'{k}.npy', v)
        meta['aux'] = self.aux.keys()

    @classmethod
    def _load(cls, path: Path, meta: dict[str, Any]) -> dict[str, np.ndarray]:
        cost = np.load(path / 'cost.npy')
        aux = {}
        for k in meta['aux']:
            aux[k] = np.load(path / f'{k}.npy')
        return {'cost': cost, 'aux': aux}
