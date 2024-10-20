from __future__ import annotations
from .index import Index
from .vector import Vector, VectorMetadata
from ..stubs import Axes, Plot1D, ErrorPlotKind
from typing import Iterable, Literal, Self
import numpy as np
from ..helpers import maybe_set
import matplotlib.pyplot as plt
from abc import abstractmethod
from scipy.stats import poisson

"""

TODO Clean up plotting code
TODO Add a PoissonVector to plot Poisson error bars and CI.
"""


class ErrorVector(Vector):
    @classmethod
    @abstractmethod
    def add_error(cls, other: Vector, err: np.ndarray) -> Self: ...

    @classmethod
    def from_vector(cls, other: Vector, *args, **kwargs) -> Self:
        return cls.add_error(other, *args, **kwargs)

    def save(self, *args, **kwargs) -> None:
        raise NotImplementedError()

    @classmethod
    def from_path(cls, *args, **kwargs) -> Self:
        raise NotImplementedError()



class AsymmetricVector(ErrorVector):
    def __init__(self, *, lerr: Iterable[float],
                 uerr: Iterable[float],
                 copy: bool = False,
                 dtype: type = float,
                 order: Literal['C', 'F'] = 'C',
                 **kwargs):
        kwargs['copy'] = copy
        kwargs['dtype'] = dtype
        kwargs['order'] = order
        super().__init__(**kwargs)
        if copy:
            def fetch(x):
                return np.asarray(x, dtype=dtype, order=order).copy()
        else:
            def fetch(x):
                return np.asarray(x, dtype=dtype, order=order)
        self.lerr: np.ndarray = fetch(lerr)
        self.uerr: np.ndarray = fetch(uerr)
        if self.lerr.shape != self.values.shape:
            raise ValueError("lerr must have the same shape as values. Got"
                             f" {self.lerr.shape} and {self.values.shape}")
        if self.uerr.shape != self.values.shape:
            raise ValueError("uerr must have the same shape as values. Got"
                             f" {self.uerr.shape} and {self.values.shape}")

    @classmethod
    def add_error(cls, other: Vector, lerr: np.ndarray, uerr: np.ndarray,
                  **kwargs) -> AsymmetricVector:
        return cls(X=other.X_index, values=other.values, lerr=lerr, uerr=uerr, **kwargs)

    @classmethod
    def from_CI(cls, other: Vector, lower: np.ndarray,
                upper: np.ndarray, clip: bool = False, **kwargs) -> AsymmetricVector:
        lerr = other.values - lower
        uerr = upper - other.values
        if np.any(lerr < 0) or np.any(uerr < 0):
            if clip:
                lerr = np.maximum(lerr, 0)
                uerr = np.maximum(uerr, 0)
            else:
                raise ValueError(("CI must be greater than or equal to the "
                                  "values. Might be due to numerical precision. "
                                  "Consider setting `clip=True`"))
        return cls(X=other.X_index, values=other.values, lerr=lerr, uerr=uerr, **kwargs)

    @classmethod
    def from_list(cls, vectors: list[Vector], summary=np.median, alpha: float = 0.68, **kwargs) -> AsymmetricVector:
        box = np.array([v.values for v in vectors])
        summaried = summary(box, axis=0)
        # create a alpha% percentile interval:
        lerr = np.percentile(box, (1-alpha)/2*100, axis=0)
        uerr = np.percentile(box, (1+alpha)/2*100, axis=0)
        vec = vectors[0].clone(values=summaried)
        return cls.from_CI(vec, lerr, uerr, **kwargs)


    def clone_from_slice(self, slice_: slice) -> Self:
        """ Returns a new vector with the given slice applied.
        """
        index: Index = self._index.__getitem__(slice_)
        values = self.values.__getitem__(slice_)
        lerr = self.lerr.__getitem__(slice_)
        uerr = self.uerr.__getitem__(slice_)
        return self.clone(X=index, values=values,
                              lerr=lerr, uerr=uerr)
    def get_CI(self):
        return self.values - self.lerr, self.values + self.uerr

    def plot(self, ax: Axes | None = None,
             kind: str = 'step',
             ekind: ErrorPlotKind = 'line',
             ekwargs: dict | None = None,
             e_offset: float | np.ndarray = 0,
             **kwargs) -> Plot1D:
        """ Plots the vector

        Args:
            ax (matplotlib axis, optional): The axis to plot onto. If not
                provided, a new figure is created
            kind (str, optional):
                - 'line' : line plot (default) evokes `ax.plot`
                - 'plot' : same as 'line'
                - 'step' : step plot
                - 'bar' : vertical bar plot
            ekwargs (dict, optional): Keyword arguments for the error bars.
            e_offset (float or array-like, optional): Offset for the error bars.
                Useful for plotting several sets of error bars.
            kwargs (optional): Additional kwargs to plot command.

        Returns:
            The figure and axis used.
        """
        if ax is None:
            ax: Axes = plt.subplots()[1]
        #ax, lines = super().plot(ax=ax, kind=kind, **kwargs)
        ekwargs = {} if ekwargs is None else ekwargs

        match kind:
            case "plot" | "line":
                raise NotImplementedError("Not implemented yet")
                if self._index.is_left():
                    bins = self.X + self.dX/2
                else:
                    bins = self.X
                kwargs.setdefault("markersize", 3)
                kwargs.setdefault("marker", ".")
                kwargs.setdefault("linestyle", "-")
                if self.std is not None:
                    ax.errorbar(bins, self.values, yerr=self.std,
                                **kwargs)
                else:
                    ax.plot(bins, self.values, **kwargs)
            case "step":
                step = 'post' if self._index.is_left() else 'mid'
                bins = self._index.ticks()
                if self._index.is_left():
                    values = np.append(self.values, self.values[-1])
                else:
                    values = np.append(np.append(self.values[0], self.values), self.values[-1])
                kw = dict(where=step) | kwargs
                line, = ax.step(bins, values, **kw)

                # Error
                color = line.get_color()
                if ekind == 'line':
                    if self._index.is_left():
                        bins = self.X + self.dX/2
                    else:
                        bins = self.X
                    kw = {'capsize': 1, 'ls': 'none', 'color': color,
                          'capthick': 0.5, 'ms': 1} | ekwargs
                    err = np.array([self.lerr, self.uerr])
                    #err = np.append(err, err[:, -1:], axis=1)
                    eline = ax.errorbar(bins + e_offset, self.values, yerr=err, **kw)
                    lines = (line, eline)
                elif ekind == 'fill':
                    kw = {'step': step, 'edgecolor': None, 'alpha': 0.2, 'color': color} | ekwargs
                    lerr = np.append(self.lerr, self.lerr[-1])
                    uerr = np.append(self.uerr, self.uerr[-1])
                    eline = ax.fill_between(bins, values - lerr, values + uerr,
                                           **kw)
                    lines = (line, eline)
                else:
                    raise ValueError(f"Invalid ekind: {ekind}")
            case "bar":
                raise NotImplementedError("Not implemented yet")
                align = 'center' if self._index.is_mid() else 'edge'
                kwargs.setdefault("align", align)
                kwargs.setdefault('width', self.dX)
                kwargs.setdefault('yerr', self.std)
                ax.bar(self.X, self.values, **kwargs)
            case "dot" | "scatter":
                raise NotImplementedError("Not implemented yet")
                if self._index.is_left():
                    bins = self.X + self.dX/2
                else:
                    bins = self.X
                kwargs.setdefault("marker", ".")
                ax.scatter(bins, self.values, **kwargs)
            case _:
                raise ValueError(f"Invalid kind: {kind}")
        maybe_set(ax, xlabel=self.xlabel + f" [${self.unit:~L}$]",
                  ylabel=self.ylabel, title=self.name)
        return ax, lines

    def clone(self, X=None, values=None, lerr=None, uerr=None, order: Literal['C', 'F'] ='C',
              metadata=None, copy=False, **kwargs) -> AsymmetricVector:
        """ Copies the object.

        Any keyword argument will override the equivalent
        attribute in the copy. For example, vector.clone(E=[1,2,3])
        tries to set the energy to [1,2,3].

        kwargs: Any keyword argument is overwritten
            in the copy.
        Returns:
            The copy
        """
        X = X if X is not None else self._index
        values = values if values is not None else self.values
        metadata = metadata if metadata is not None else self.metadata
        lerr = lerr if lerr is not None else self.lerr
        uerr = uerr if uerr is not None else self.uerr
        metakwargs = VectorMetadata.__slots__
        # Extract all keyword argumetns that are in metakwargs from kwargs
        for key in metakwargs:
            if key in kwargs:
                metadata = metadata.update(key=kwargs.pop(key))
        return AsymmetricVector(X=X, values=values, lerr=lerr, uerr=uerr, order=order,
                      metadata=metadata, copy=copy, **kwargs)


class SymmetricVector(AsymmetricVector):
    def __init__(self, *, err: Iterable[float] | None = None,
                 **kwargs):
        super().__init__(lerr=err, uerr=err, **kwargs)

    @classmethod
    def add_error(cls, other: Vector, err: np.ndarray,
                  **kwargs) -> SymmetricVector:
        # Covariance matrix
        if err.ndim == 2:
            err = np.sqrt(np.diag(err))
        return cls(X=other.X_index, values=other.values, err=err, **kwargs)

    def clone(self, X=None, values=None, err=None, order: Literal['C', 'F'] ='C',
              metadata=None, copy=False, **kwargs) -> SymmetricVector:
        """ Copies the object.

        Any keyword argument will override the equivalent
        attribute in the copy. For example, vector.clone(E=[1,2,3])
        tries to set the energy to [1,2,3].

        kwargs: Any keyword argument is overwritten
            in the copy.
        Returns:
            The copy
        """
        X = X if X is not None else self._index
        values = values if values is not None else self.values
        metadata = metadata if metadata is not None else self.metadata
        err = err if err is not None else self.lerr
        metakwargs = VectorMetadata.__slots__
        # Extract all keyword argumetns that are in metakwargs from kwargs
        for key in metakwargs:
            if key in kwargs:
                metadata = metadata.update(key=kwargs.pop(key))
        return SymmetricVector(X=X, values=values, err=err, order=order,
                      metadata=metadata, copy=copy, **kwargs)


class PoissonVector(ErrorVector):
    @classmethod
    def add_error(cls, other: Vector, **kwargs) -> PoissonVector:
        return cls(X=other.X_index, values=other.values, **kwargs)

    @classmethod
    def from_vector(cls, other: Vector, clip: bool = False, **kwargs) -> PoissonVector:
        if np.any(other.values < 0):
            if clip:
                values = np.clip(other.values, 0, None)
            else:
                raise ValueError("Cannot create PoissonVector from negative values. "
                                 "Consider using `clip=True`")

        return cls(X=other.X_index, values=other.values, **kwargs)


    def plot(self, ax: Axes | None = None,
             kind: str = 'step',
             ekind: ErrorPlotKind = 'line',
             confidence: float = 0.05,
             **kwargs) -> Plot1D:
        """ Plots the vector

        Args:
            ax (matplotlib axis, optional): The axis to plot onto. If not
                provided, a new figure is created
            kind (str, optional):
                - 'line' : line plot (default) evokes `ax.plot`
                - 'plot' : same as 'line'
                - 'step' : step plot
                - 'bar' : vertical bar plot
            kwargs (optional): Additional kwargs to plot command.

        Returns:
            The figure and axis used.
        """
        if ax is None:
            ax: Axes = plt.subplots()[1]
        #ax, lines = super().plot(ax=ax, kind=kind, **kwargs)

        lerr, uerr = poisson.interval(1-confidence, self.values)
        lerr = self.values - lerr
        uerr = uerr - self.values
        # Numerical errors
        lerr = np.clip(lerr, 0, None)
        uerr = np.clip(uerr, 0, None)
        match kind:
            case "plot" | "line":
                raise NotImplementedError("Not implemented yet")
                if self._index.is_left():
                    bins = self.X + self.dX/2
                else:
                    bins = self.X
                kwargs.setdefault("markersize", 3)
                kwargs.setdefault("marker", ".")
                kwargs.setdefault("linestyle", "-")
                if self.std is not None:
                    ax.errorbar(bins, self.values, yerr=self.std,
                                **kwargs)
                else:
                    ax.plot(bins, self.values, **kwargs)
            case "step":
                step = 'post' if self._index.is_left() else 'mid'
                bins = self._index.ticks()
                if self._index.is_left():
                    values = np.append(self.values, self.values[-1])
                else:
                    values = np.append(np.append(self.values[0], self.values), self.values[-1])
                kw = dict(where=step) | kwargs
                line, = ax.step(bins, values, **kw)

                # Error
                color = line.get_color()
                if ekind == 'line':
                    if self._index.is_left():
                        bins = self.X + self.dX/2
                    else:
                        bins = self.X
                    kw = {'capsize': 1, 'ls': 'none', 'color': color,
                          'capthick': 0.5, 'ms': 1} | kwargs
                    err = np.array([lerr, uerr])
                    #err = np.append(err, err[:, -1:], axis=1)
                    eline = ax.errorbar(bins, self.values, yerr=err, **kw)
                    lines = (line, eline)
                elif ekind == 'fill':
                    kw = {'step': step, 'edgecolor': None, 'alpha': 0.2, 'color': color} | kwargs
                    lerr = np.append(lerr, lerr[-1])
                    uerr = np.append(uerr, uerr[-1])
                    eline = ax.fill_between(bins, values - lerr, values + uerr,
                                           **kw)
                    lines = (line, eline)
                else:
                    raise ValueError(f"Invalid ekind: {ekind}")
            case "bar":
                raise NotImplementedError("Not implemented yet")
            case "dot" | "scatter":
                raise NotImplementedError("Not implemented yet")
            case _:
                raise ValueError(f"Invalid kind: {kind}")
        maybe_set(ax, xlabel=self.xlabel + f" [${self.unit:~L}$]",
                  ylabel=self.ylabel, title=self.name)
        return ax, lines
