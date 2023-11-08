from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any, Iterable, Literal, overload, TypeAlias, Self, TypeVar, Generic, Never
from typing import TYPE_CHECKING

import numpy as np
from numpy import ndarray
from .. import XARRAY_AVAILABLE
from .abstractarray import AbstractArray
from .filehandling import (load_csv_1D, load_numpy_1D,
                           load_tar, load_txt_1D, mama_read, mama_write,
                           save_csv_1D, save_numpy_1D, save_root_1D, save_tar, save_txt_1D,
                           save_npz_1D, load_npz_1D, resolve_filetype, load_root_1D)
from .index import Index, make_or_update_index, Edges, Index, is_uniform
from ..library import div0
from ..helpers import maybe_set, ensure_path
from ..stubs import Unitlike, arraylike, Axes, Pathlike, Plot1D, QuantityLike, array1D, VectorPlot, Line2D, is_lines
from ..stubs import Plot1D, PlotError1D, PlotScatter1D, PlotBar1D
from typing import TypeVar
from .vectormetadata import VectorMetadata
from .rebin import Preserve
from .vectorprotocol import VectorProtocol
import matplotlib.pyplot as plt
from matplotlib.container import ErrorbarContainer, BarContainer
from matplotlib.collections import PathCollection

if TYPE_CHECKING:
    from .matrix import Matrix

LOG = logging.getLogger(__name__)
logging.captureWarnings(True)

"""
-[x] Constructor
-[x] __getitem__
-[x] __setitem__
-[x] index
-[x] vector index. Need to fix index' index first.
-[x] rebin
-[-] plot
-[x] save
-[x] load
-[ ] ROOT load/save
-[ ] Batch rebinning
"""

VectorPlotKind: TypeAlias = Literal['step', 'plot', 'line', 'bar', 'dot', 'scatter', 'poisson']

KwargsDict: TypeAlias = dict[str, Any]
T = TypeVar('T')

@overload
def maybe_pop_from_kwargs(kwargs: KwargsDict, item: T, name: str, alias: str) -> tuple[KwargsDict, T, None]: ...

@overload
def maybe_pop_from_kwargs(kwargs: KwargsDict, item: None, name: str, alias: str) -> tuple[KwargsDict, Any, str]: ...

def maybe_pop_from_kwargs(kwargs: KwargsDict, item: T | None, name: str, alias: str) -> tuple[KwargsDict, T, str | None]:
    alias_value: None | str = None
    iter = kwargs.items().__iter__()
    if item is None:
        try:
            alias_value, item = next(iter)
        except StopIteration:
            raise ValueError(f"Missing argument {name}")
        if alias in kwargs:
            raise ValueError(f"Duplicate argument {alias} and {kwargs[alias]}")
        if item is None:
            raise ValueError(f"Missing argument {name}")
    kwargs = dict(iter)
    return kwargs, item, alias_value

class Vector(AbstractArray, VectorProtocol):
    """ Stores 1d array with energy axes (a vector)

    Attributes:
        values (np.ndarray): The values at each bin.
    """
    _ndim = 1

    # HACK: Descriptors really don't work well with %autoreload.
    # comment / uncomment this to silence the errors when developing
    # __slots__ = ('_X', 'values', 'std', 'loc', 'iloc', 'metadata')

    def __init__(self, *, X: arraylike | Index | None = None,
                 values: arraylike | None = None,
                 copy: bool = False,
                 unit: Unitlike | None = None,
                 order: np._OrderKACF = 'C',
                 edge: Edges = 'left',
                 boundary: bool = False,
                 metadata: VectorMetadata = VectorMetadata(),
                 indexkwargs: dict[str, Any] | None = None,
                 dtype: np.dtype | str = np.dtype('float32'),
                 **kwargs):
        """
        If no `std` is given, it will default to None

        Args:
            values: see above
            E: see above
            std: see above
            copy: Whether to copy `values` and `E` or by reference.
                Defaults to True.

        Raises:
           ValueError if the runtime lengths of the arrays are different.
           ValueError if incompatible arguments are provided.

        """
        # Resolve aliasing
        # First keyword argument is the alias
        #kwiter = kwargs.items().__iter__()
        kwargs, X, xalias = maybe_pop_from_kwargs(kwargs, X, 'X', 'xalias')
        kwargs, values, valias = maybe_pop_from_kwargs(kwargs, values, 'values', 'valias')

        xalias = xalias or kwargs.pop('xalias', '')
        # Put back on kwargs for metadata to handle
        if valias is not None:
            kwargs['valias'] = valias

        if copy:
            def fetch(x):
                return np.asarray(x, dtype=dtype, order=order).copy()
        else:
            def fetch(x):
                return np.asarray(x, dtype=dtype, order=order)

        super().__init__(fetch(values))

        # Create an index from array or update existing index
        default_label = 'xlabel' not in kwargs
        xlabel = kwargs.pop('xlabel', 'Energy')
        # Pop a set of keys from kwargs if kwargs has these keys
        indexkwargs = indexkwargs or {}
        default_unit = False if unit is not None else True
        unit = 'keV' if default_unit else unit  # Not elegant. Index will overwrite anyway.
        assert X is not None
        self._index = make_or_update_index(X, unit=unit, alias=xalias, label=xlabel,
                                           default_label=default_label,
                                           default_unit=default_unit,
                                           edge=edge, boundary=boundary,
                                           **indexkwargs)
        _xalias = '' if not xalias else f' (`{xalias}`)'
        _valias = '' if not valias else f' (`{valias}`)'
        if len(self._index) != len(self.values):
            raise ValueError(
                f"Length of index{_xalias} and values{_valias} must be the same. Got {len(self._index)} and {len(self.values)}")
        wrong_kw = set(kwargs) - set(VectorMetadata.__slots__)
        if wrong_kw:
            raise ValueError(f"Invalid keyword arguments: {', '.join(wrong_kw)}")
        self.metadata = metadata.update(**kwargs)

        self.loc: ValueLocator = ValueLocator(self, strict=False)
        self.vloc: ValueLocator = ValueLocator(self, strict=True)
        self.iloc: IndexLocator = IndexLocator(self)

    def __getattr__(self, item) -> Any:
        meta: VectorMetadata = self.__dict__['metadata']
        alias: str = self.__dict__['_index'].alias
        if item == alias:
            x = self.X
        elif item == meta.valias:
            x = self.__dict__['values']
        elif item == 'd' + alias:
            x = self.dX
        else:
            x = super().__getattr__(item)
        return x


    @ensure_path
    def save(self, path: Path,
             filetype: str | None = None,
             exist_ok: bool = True, **kwargs) -> None:
        """Save to a file of specified format

        Args:
            path (str or Path): Path to save
            filetype (str, optional): Filetype. Default uses
                auto-recognition from suffix.
                Options: ["numpy", "txt", "tar", "mama", "csv"]
            **kwargs: additional keyword arguments

        Raises:
            ValueError: Filetype is not supported
        """
        path = Path(path)
        path, filetype = resolve_filetype(path, filetype)

        E = self._index.to_unit('keV').bins
        match filetype:
            case "npy":
                warnings.warn("Saving as .npy is deprecated. Use .npz instead.")
                save_numpy_1D(self.values, E, path)
            case 'npz':
                save_npz_1D(path, self, exist_ok=exist_ok)
            case "txt":
                warnings.warn("Saving to .txt does not preserve metadata. Use .npz instead.")
                save_txt_1D(self.values, E, path, **kwargs)
            case 'tar':
                warnings.warn("Saving to .tar does not preserve metadata. Use .npz instead.")
                save_tar([self.values, E], path)
            case 'mama':
                warnings.warn("MAMA format does not preserve metadata.")
                mama_write(self, path, **kwargs)
            case 'csv':
                warnings.warn("CSV format does not preserve metadata.")
                save_csv_1D(self.values, E, path)
            case 'root':
                save_root_1D(self, path, exist_ok=exist_ok)
            case _:
                raise ValueError(f"Unknown filetype {filetype}")

    @classmethod
    def from_path(cls, path: Pathlike, filetype: str | None = None) -> Self:
        """Load to a file of specified format

        Units assumed to be keV.

        Args:
            path (str or Path): Path to Load
            filetype (str, optional): Filetype. Default uses
                auto-recognition from suffix.

        Raises:
            ValueError: Filetype is not supported
        """
        path = Path(path)
        path, filetype = resolve_filetype(path, filetype)
        LOG.debug(f"Loading {path} as {filetype}")

        match filetype:
            case 'npy':
                values, E = load_numpy_1D(path)
            case 'npz':
                return load_npz_1D(path, Vector)
            case 'txt':
                values, E = load_txt_1D(path)
            case 'tar':
                from_file = load_tar(path)
                if len(from_file) == 3:
                    values, E = from_file
                elif len(from_file) == 2:
                    values, E = from_file
                else:
                    raise ValueError(f"Expected two or three columns\
                     in file '{path}', got {len(from_file)}")
            case 'mama':
                ret = mama_read(str(path))
                if len(ret) == 2:
                    values, E = ret
                else:
                    raise ValueError(f"Expected two columns in mama, got {len(ret)}")
            case 'csv':
                values, E = load_csv_1D(path)
            case 'root':
                return load_root_1D(path, Vector)
            case _:
                try:
                    ret = mama_read(str(path))
                    if len(ret) == 2:
                        values, E = ret
                    else:
                        raise ValueError(f"Expected two columns in mama, got {len(ret)}")
                    return Vector(E=E, values=values, edge='mid')
                except ValueError:  # from within ValueError
                    raise ValueError(f"Unknown filetype {filetype}")
        return Vector(E=E, values=values)

    @overload
    def drop_nan(self, inplace: Literal[False] = ...) -> Self: ...

    @overload
    def drop_nan(self, inplace: Literal[True] = ...) -> None: ...

    def drop_nan(self, inplace: bool = False) -> Self | None:
        """ Drop the elements that are `np.nan`

        Args:
            inplace (bool, optional): If `True` perform the cut on this vector
                or if `False` returns a copy. Defaults to True
        Returns:
            The cut vector if `inplace` is True.
        """
        return self.from_mask(~np.isnan(self.values))

    @overload
    def rebin(self, bins: arraylike | Index | None = None,
              factor: float | None = None,
              binwidth: QuantityLike | None = None,
              numbins: int | None = None,
              preserve: Preserve = 'counts',
              inplace: Literal[False] = ...) -> Self:
        ...

    @overload
    def rebin(self, bins: arraylike | Index | None = None,
              factor: float | None = None,
              binwidth: QuantityLike | None = None,
              numbins: int | None = None,
              preserve: Preserve = 'counts',
              inplace: Literal[True] = ...) -> None:
        ...

    def rebin(self, bins: arraylike | Index | None = None,
              factor: float | None = None,
              binwidth: QuantityLike | None = None,
              numbins: int | None = None,
              preserve: Preserve = 'counts',
              inplace: bool = False) -> Self | None:
        """ Rebins vector, assuming equidistant binning

        Args:
            bins: The new energy bins. Can not be
                given alongside 'factor' or `binwidth`.
            factor: The factor by which the step size shall be
                changed. E.g `factor=2.0` yields twice as large
                bins. Can not be given alongside 'bins' or `binwidth`.
            binwidth: The new bin width. Can not be given
                alongside `factor` or `bins`.
            numbins: The new number of bins. Must be fewer than before.
            inplace: Whether to change E and values
                inplace or return the rebinned vector.
                Defaults to `false`.
        Returns:
            The rebinned vector if inplace is 'False'.
        """
        bins_: Index = self._index.handle_rebin_arguments(bins=bins, factor=factor, binwidth=binwidth, numbins=numbins)
        _, rebinned = self._index.rebin(bins_, self.values, preserve=preserve)

        if inplace:
            self.values = rebinned
            self._index = bins_
        else:
            return self.clone(X=bins_, values=rebinned)

    @overload
    def rebin_like(self, other: Vector, inplace: Literal[False] = ...) -> Self:
        ...

    @overload
    def rebin_like(self, other: Vector, inplace: Literal[True] = ...) -> None:
        ...

    def rebin_like(self, other: Vector | Index, inplace: bool = False, preserve: Preserve = 'counts') -> Self | None:
        """ Rebin to match the binning of `other`.

        Args:
            other: Rebin to the bin width of the provided vector.
            inplace: Whether to rebin inplace or return a copy.
                Defaults to `False`.
        """
        match other:
            case Vector():
                index = other._index
            case Index():
                index = other
            case _:
                raise TypeError(f"Can not rebin like {type(other)}")
        index = index.to_unit(self.unit)
        _, rebinned = self._index.rebin(index, self.values, preserve=preserve)
        index = index.copy(meta=self._index.meta)
        if inplace:
            self.values = rebinned
            self._index = index
        else:
            return self.clone(X=index, values=rebinned)

    def closest(self, E: ndarray, side: np._SortSide = 'right',
                inplace=False) -> Self | None:
        """ Re-bin the vector without merging bins.

            The resulting vector will have E as the x-axis while
            the jth y-value will be given by the ith value of the original
            y-values where E[i] < E_new[j] <= E[i+1] or
            E[i] <= E_new[j] < E[i+1].

            If E is dimensionless, it is assumed to be in the same unit
            as `Vector.E`

            Args:
                E: Bin value to find. Value or array.
                side: 'left': E[i] < E[j] <= E[i+1],
                      'right': E[i] <= E[j] <= E[i+1]
                inplace: Whether to make the change inplace or not.
            Returns:
                Vector with the new E axis and the bin content of the bins
                that contains E.
            Raises:
                RuntimeError if the x-axis of the original vector is
                not sorted.
        """

        if not np.all(self.X[:-1] <= self.X[1:]):
            raise RuntimeError("x-axis not sorted.")

        # Convert to same units at strip
        E_old = self.X
        E = self.to_same(E)
        indices = np.searchsorted(E_old, E, side=side)

        # Ensure that any element outside the range of E will get index
        # -1.
        indices[indices >= len(self.X)] = 0
        indices -= 1

        # We need to append 0 to the end to ensure that we fill 0 if any
        # element E_new is outside of the bounds of self.E
        values = np.append(self.values, [0])
        values = values[indices]

        std = None
        if self.std is not None:
            std = np.append(self.std, [0])
            std = std[indices]

        E *= self.unit
        if inplace:
            self._X = E
            self.values = values
            self.std = std
        else:
            return self.clone(values=values, X=E, std=std)

    def cumulative(self, factor: float | Literal['de'] = 1.0,
                   inplace: bool = False) -> Self | None:
        """ Cumulative sum of the vector.

            Args:
                factor: A factor to multiply to the resulting vector. Possible
                values are a float or string 'de'. If 'de' the
                factor will be calculated by E[1] - E[0]. The default is 1.0.
                inplace: Whether to make the change inplace or not.
            Returns:
                The cumulative sum vector if inplace is 'False'
            Raises:
                RuntimeError if elements in self.E are not equidistant
                and factor='de'.
                ValueError if factor is a string other than 'de'.
        """
        if isinstance(factor, str):
            if factor.lower() != 'de':
                raise ValueError(f"Unkown option for factor {factor}")
            factor = self.de

        cumsum = factor * self.values.cumsum()
        assert isinstance(cumsum, np.ndarray)
        cumerr = None
        if self.std is not None:
            cumerr = np.sqrt(np.cumsum(self.std ** 2)) * factor

        if inplace:
            self.values = cumsum
            self.std = cumerr
        else:
            return self.clone(values=cumsum, std=cumerr)

    def set_order(self, order: np._OrderKACF) -> None:
        """ Wrapper around numpy to set the alignment """
        self.values = self.values.copy(order=order)
        self._index = self._index.copy(order=order)

    @property
    def dX(self) -> float | np.ndarray:
        if is_uniform(self._index):
            return self._index.dX
        return self._index.steps()

    def last_nonzero(self) -> int:
        """ Returns the index of the last nonzero value """
        j = len(self)
        while (j := j - 1) >= 0:
            if self[j] != 0:
                break
        return j

    def cut_at_last_nonzero(self) -> Self:
        return self.iloc[:self.last_nonzero() + 1]

    def update(self, xlabel: str | None = None, vlabel: str | None = None,
               name: str | None = None, misc: dict[str, Any] | None = None,
               inplace: bool = False, title: str | None = None) -> None | Self:
        index = self._index.update(label=xlabel)
        if title is not None:
            if name is not None:
                if name != title:
                    raise ValueError("`name` and `title` alias each other. Only provide one")
            name = title
        meta = self.metadata.update(vlabel=vlabel, name=name, misc=misc)
        if inplace:
            self._index = index
            self.metadata = meta
        else:
            return self.clone(X=index, metadata=meta)

    def add_comment(self, key: str, comment: Any, inplace: bool = False) -> None | Self:
        meta = self.metadata.add_comment(key, comment)
        if inplace:
            self.metadata = meta
        else:
            return self.clone(metadata=meta)

    @property
    def _summary(self) -> str:
        s = self._index.summary()
        s += f'\nValue alias: {self.metadata.valias}\n'
        s += f'ylabel: {self.metadata.vlabel}\n'
        if len(self.metadata.misc) > 0:
            s += "Metadata:\n"
            for key, val in self.metadata.misc.items():
                s += f'\t{key}: {val}\n'
        s += f"Total counts: {self.sum():.3g}\n"
        return s

    def summary(self) -> None:
        print(self._summary)

    def __str__(self) -> str:
        summary = self._summary
        summary += "\nValues:\n"
        return summary + str(self.values)

    def clone(self, X=None, values=None, order: Literal['C', 'F'] ='C',
              metadata=None, copy=False, dtype: np.dtype | None = None,
              **kwargs) -> Self:
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
        metakwargs = VectorMetadata.__slots__
        # Extract all keyword argumetns that are in metakwargs from kwargs
        for key in metakwargs:
            if key in kwargs:
                metadata = metadata.update(**{key: kwargs.pop(key)})
        return Vector(X=X, values=values, order=order,
                      metadata=metadata, copy=copy, dtype=dtype, **kwargs)

    def copy(self, **kwargs) -> Self:
        return self.clone(copy=True, **kwargs)

    @property
    def unit(self) -> Any:
        return self._index.unit

    @property
    def xlabel(self) -> str:
        return self._index.label

    @xlabel.setter
    def xlabel(self, value: str) -> None:
        self.update(xlabel=value, inplace=True)

    def get_xlabel(self) -> str:
        return self.xlabel + f" [${self.unit:~L}$]"

    @property
    def ylabel(self) -> str:
        return self.vlabel

    @ylabel.setter
    def ylabel(self, value: str) -> None:
        self.update(vlabel=value, inplace=True)

    def get_ylabel(self) -> str:
        return self.ylabel

    @property
    def alias(self) -> str:
        return self._index.alias

    @property
    def X(self) -> np.ndarray:
        return np.array(self._index.bins, dtype=self.dtype)

    @property
    def X_index(self) -> Index:
        return self._index

    def enumerate(self) -> Iterable[tuple[int, float, float]]:
        """ Returns an iterator over the indices and values """
        for i, x in enumerate(self.X):
            yield i, x, self.values[i]

    def unpack(self) -> tuple[np.ndarray, np.ndarray]:
        """ Returns the energy and values as separate arrays """
        return self.X, self.values


    def index(self, x: float) -> int:
        """ Returns the index of the bin containing x """
        return self._index.index(x)

    def is_compatible_with(self, other: AbstractArray | Index) -> bool:
        match other:
            case Vector():
                return self._index.is_compatible_with(other._index)
            case Index():
                return self._index.is_compatible_with(other)
            case _:
                return False

    @overload
    def to_unit(self, unit: Unitlike, inplace: Literal[False] = ...) -> Self: ...

    @overload
    def to_unit(self, unit: Unitlike, inplace: Literal[True] = ...) -> None: ...

    def to_unit(self, unit: Unitlike, inplace: bool = False) -> None | Self:
        """ Converts the index to the given unit """
        index = self._index.to_unit(unit)
        if inplace:
            self._index = index
        else:
            return self.clone(X=index)

    def to_edge(self, edge: Edges, inplace: bool = False) -> None | Self:
        """ Converts the index to the given edge """
        index = self._index.to_edge(edge)
        if inplace:
            self._index = index
        else:
            return self.clone(X=index)

    def to_left(self, inplace: bool = False) -> None | Self:
        """ Converts the index to the left edge """
        return self.to_edge('left', inplace=inplace)

    @overload
    def to_same_edge(self, other: Vector, inplace: Literal[False] = ...) -> Self: ...

    @overload
    def to_same_edge(self, other: Vector, inplace: Literal[True] = ...) -> None: ...

    def to_same_edge(self, other: Vector, inplace: bool = False) -> None | Self:
        """ Converts the index to the same edge as other """
        index = self._index.to_same_edge(other._index)
        if inplace:
            self._index = index
        else:
            return self.clone(X=index)

    def to_same(self, other: Vector) -> Self:
        return self.to_same_edge(other).to_unit(other.unit)

    @overload
    def to_mid(self, inplace: Literal[False] = ...) -> Self: ...

    @overload
    def to_mid(self, inplace: Literal[True] = ...) -> Self: ...

    def to_mid(self, inplace: bool = False) -> None | Self:
        """ Converts the index to the middle """
        return self.to_edge('mid', inplace=inplace)

    @overload
    def plot(self, ax: Axes | None = None,
             kind: Literal['step', 'plot', 'line'] = ...,
             **kwargs) -> Plot1D : ...

    @overload
    def plot(self, ax: Axes | None = None,
             kind: Literal['dot', 'scatter'] = ...,
             **kwargs) -> PlotScatter1D : ...

    @overload
    def plot(self, ax: Axes | None = None,
             kind: Literal['bar'] = ...,
             **kwargs) -> PlotBar1D: ...

    @overload
    def plot(self, ax: Axes | None = None,
             kind: Literal['poisson'] = ...,
             **kwargs) -> PlotError1D: ...

    def plot(self, ax: Axes | None = None,
             kind: VectorPlotKind = 'step',
             **kwargs) -> VectorPlot:
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
            _, _ax = plt.subplots()
            assert isinstance(_ax, Axes)
            ax = _ax

        match kind:
            case "plot" | "line":
                if self._index.is_left():
                    bins = self.X + self.dX/2
                else:
                    bins = self.X
                kwargs.setdefault("markersize", 3)
                kwargs.setdefault("marker", ".")
                kwargs.setdefault("linestyle", "-")
                line = ax.plot(bins, self.values, **kwargs)
                assert isinstance(line, Line2D)
                maybe_set(ax, xlabel=self.get_xlabel(),
                        ylabel=self.get_ylabel(), title=self.name)
                return ax, line
            case "step":
                step = 'post' if self._index.is_left() else 'mid'
                bins = self._index.ticks()
                if self._index.is_left():
                    values = np.append(self.values, self.values[-1])
                else:
                    values = np.append(np.append(self.values[0], self.values), self.values[-1])
                kwargs.setdefault("where", step)
                line = ax.step(bins, values, **kwargs)
                assert is_lines(line)
                maybe_set(ax, xlabel=self.get_xlabel(),
                        ylabel=self.get_ylabel(), title=self.name)
                return ax, line
            case "bar":
                align = 'center' if self._index.is_mid() else 'edge'
                kwargs.setdefault("align", align)
                kwargs.setdefault('width', self.dX)
                line = ax.bar(self.X, self.values, **kwargs)
                assert isinstance(line, BarContainer)
                maybe_set(ax, xlabel=self.get_xlabel(),
                        ylabel=self.get_ylabel(), title=self.name)
                return ax, line
            case "dot" | "scatter":
                if self._index.is_left():
                    bins = self.X + self.dX/2
                else:
                    bins = self.X
                kwargs.setdefault("marker", ".")
                line = ax.scatter(bins, self.values, **kwargs)
                assert isinstance(line, PathCollection)
                maybe_set(ax, xlabel=self.get_xlabel(),
                        ylabel=self.get_ylabel(), title=self.name)
                return ax, line
            case "poisson":
                if self._index.is_left():
                    bins = self.X + self.dX/2
                else:
                    bins = self.X
                kw = dict(marker = 'o', ls='none', capsize=2, capthick=0.5, ms=3, lw=1)
                kw |= kwargs
                line = ax.errorbar(bins, self.values, yerr=np.sqrt(self.values), **kw)  # type: ignore
                assert isinstance(line, ErrorbarContainer)
                maybe_set(ax, xlabel=self.get_xlabel(),
                        ylabel=self.get_ylabel(), title=self.name)
                return ax, line
            case _:
                raise ValueError(f"Invalid kind: {kind}")

    def integrate(self) -> np.float_:
        """ Returns the integral of the vector """
        x: array1D = self.values
        y: array1D | float = self.dX
        return np.sum(x*y)

    @overload
    def __matmul__(self, other: Matrix) -> Self: ...
    @overload
    def __matmul__(self, other: Vector) -> float: ...
    @overload
    def __matmul__(self, other: np.ndarray) -> np.ndarray | float: ...

    def __matmul__(self, other: Matrix | Vector | np.ndarray) -> Self | float | np.ndarray:
        match other:
            case Vector():
                self.check_or_assert(other)
                return self.values @ other.values
            case AbstractArray():
                if self.shape[0] != other.shape[0]:
                    raise ValueError(f"Shape mismatch {self.shape} @ {other.shape}")
                if not self.is_compatible_with(other.X_index):
                    raise ValueError(f"Index mismatch {self._index} @ {other.X_index}")
                return Vector(X=other.Y_index, values=self.values @ other.values)
            case _:
                return self.values @ other
    @overload
    def from_mask(self, mask: np.ndarray, inplace: Literal[False] = ...) -> Self: ...

    @overload
    def from_mask(self, mask: np.ndarray, inplace: Literal[True] = ...) -> None: ...

    def from_mask(self, mask: np.ndarray, inplace: bool = False) -> None | Self:
        """ Returns a new vector with the given mask applied """
        # Check that the True are contiguous
        if not check_contiguous(mask):
            raise ValueError("Mask must be contiguous")
        indices = np.argwhere(mask).ravel()
        start = indices[0]
        stop = indices[-1]+1
        vec = self.iloc[start:stop]
        if inplace:
            self.values = vec.values
            self._index = vec._index
        else:
            return vec

    def clone_from_slice(self, slice_: slice) -> Self:
        """ Returns a new vector with the given slice applied.

        Mainly used for the Locators to handle the creation of
        new vectors, particularly subclasses.
        """
        index: Index = self._index.__getitem__(slice_)
        values = self.values.__getitem__(slice_)
        return self.clone(X=index, values=values)


    def to_xarray(self):
        return to_xarray_vector(self)


if XARRAY_AVAILABLE:
    import xarray as xr
    def to_xarray_vector(vec) -> xr.DataArray:
        """ Convert to xarray DataArray """
        return xr.DataArray(vec.values, coords=[vec.X], dims=[vec.alias])
else:
    def to_xarray_vector(vec) -> Never:
        raise NotImplementedError("xarray is not installed")

VT = TypeVar('VT', bound=Vector)
class ValueLocator(Generic[VT]):
    def __init__(self, vector: VT, strict: bool = True):
        self.vec: VT = vector
        self.strict: bool = strict

    @overload
    def __getitem__(self, key: int) -> float: ...

    @overload
    def __getitem__(self, key: slice) -> VT: ...

    @overload
    def __getitem__(self, key: np.ndarray) -> np.ndarray: ...

    def __getitem__(self, key: int | slice | np.ndarray) -> VT | float | np.ndarray:
        match key:
            case slice():
                s: slice = self.vec._index.index_slice(key, strict=self.strict)
                return self.vec.clone_from_slice(s)
            case Index():
                start = self.vec._index.index(key[0])
                stop = self.vec._index.index(key[-1]) + 1
                return self.vec.clone_from_slice(slice(start, stop))
            case _:
                # TODO What happens with key: np.ndarray?
                i: int = self.vec._index.index_expression(key, strict=self.strict)
                return self.vec.values.__getitem__((i,))

    def __setitem__(self, key, val) -> None:
        match key:
            case slice():
                s: slice = self.vec._index.index_slice(key, strict=self.strict)
                self.vec.values.__setitem__((s,), val)
            case Index():
                start = self.vec._index.index(key[0])
                stop = self.vec._index.index(key[-1]) + 1
                self.vec.values.__setitem__((slice(start, stop),), val)
            case _:
                i: int = self.vec._index.index_expression(key, strict=self.strict)
                self.vec.values.__setitem__((i,), val)


class IndexLocator(Generic[VT]):
    def __init__(self, vector: VT):
        self.vector: VT = vector

    @overload
    def __getitem__(self, key: int) -> float: ...

    @overload
    def __getitem__(self, key: slice) -> VT: ...

    @overload
    def __getitem__(self, key: np.ndarray) -> np.ndarray: ...

    def __getitem__(self, key: int | slice | np.ndarray) -> VT | float | np.ndarray:
        match key:
            case slice():
                return self.vector.clone_from_slice(key)
            case _:
                return self.vector.values.__getitem__(key)

    def __setitem__(self, key: int | slice, val) -> None:
        self.vector.values.__setitem__(key, val)



def check_contiguous(arr: array1D) -> bool:
    # Find indices of all True values
    true_indices = np.where(arr)[0]

    # If there are no True values or only one True value at the edges, it's valid
    if true_indices.size == 0 or (true_indices.size == 1 and (true_indices[0] == 0 or true_indices[0] == len(arr) - 1)):
        return True

    # Check if all True values are contiguous
    if true_indices[-1] - true_indices[0] + 1 != true_indices.size:
        return False

    # Check if the contiguous run of True values starts or ends at an edge
    #if true_indices[0] == 0 or true_indices[-1] == len(arr) - 1:
    #   return True

    return True
