from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generic,
    Iterable,
    Literal,
    Never,
    Self,
    TypeAlias,
    TypeVar,
    overload,
)

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PathCollection
from matplotlib.container import BarContainer, ErrorbarContainer
from numpy import ndarray
from numpy.typing import DTypeLike, NDArray

from ..accel import numba_available, xarray_available, jax_available
from ..helpers import ensure_path, maybe_set
from ..rendering.html import collapse, collapsible, table
from ..stubs import (
    Axes,
    Line2D,
    Pathlike,
    Plot1D,
    PlotBar1D,
    PlotError1D,
    PlotScatter1D,
    QuantityLike,
    Unitlike,
    VectorPlot,
    array1D,
    arraylike,
    is_lines,
)
from .abstractarray import AbstractArray
from .abstractarray import fetch as _fetch
from .filehandling import (
    load_csv_1D,
    load_npz_1D,
    load_numpy_1D,
    load_root_1D,
    load_tar,
    load_txt_1D,
    mama_read,
    mama_write,
    resolve_filetype,
    save_csv_1D,
    save_npz_1D,
    save_numpy_1D,
    save_root_1D,
    save_tar,
    save_txt_1D,
)
from .index import Edges, Index, is_uniform, make_or_update_index
from .plotsettings import PlotSettings
from .rebin import Preserve
from .vectormetadata import VectorMetadata
from .vectorpreset import VectorPreset, get_preset
from .vectorprotocol import VectorProtocol

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

VectorPlotKind: TypeAlias = Literal[
    "step", "plot", "line", "bar", "dot", "scatter", "poisson"
]

KwargsDict: TypeAlias = dict[str, Any]
T = TypeVar("T")


@overload
def maybe_pop_from_kwargs(
    kwargs: KwargsDict, item: T, name: str, alias: str
) -> tuple[KwargsDict, T, None]: ...


@overload
def maybe_pop_from_kwargs(
    kwargs: KwargsDict, item: None, name: str, alias: str
) -> tuple[KwargsDict, Any, str]: ...


def maybe_pop_from_kwargs(
    kwargs: KwargsDict, item: T | None, name: str, alias: str
) -> tuple[KwargsDict, T, str | None]:
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


KwargsDict: TypeAlias = dict[str, Any]
T = TypeVar("T")
NPOrder: TypeAlias = Literal["K", "A", "C", "F"]
Array1D: TypeAlias = NDArray[Any]


class Vector(AbstractArray, VectorProtocol):
    """Stores 1d array with energy axes (a vector)

    Attributes:
        values (np.ndarray): The values at each bin.
    """

    _ndim = 1

    # HACK: Descriptors really don't work well with %autoreload.
    # comment / uncomment this to silence the errors when developing
    # __slots__ = ('_X', 'values', 'std', 'loc', 'iloc', 'metadata')

    def __init__(
        self,
        *,
        X: arraylike | Index | None = None,
        values: arraylike | None = None,
        copy: bool = False,
        unit: Unitlike | None = None,
        vunit: Unitlike | None = None,
        order: NPOrder | None = None,
        edge: Edges = "left",
        boundary: bool = False,
        metadata: VectorMetadata | dict[str, Any] = VectorMetadata(),
        indexkwargs: dict[str, Any] | None = None,
        dtype: DTypeLike | str = np.dtype("float32"),
        plot_settings: PlotSettings | dict[str, Any] | None = None,
        **kwargs: Any,
    ):
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
        # kwiter = kwargs.items().__iter__()
        kwargs, X, xalias = maybe_pop_from_kwargs(kwargs, X, "X", "xalias")
        kwargs, values, valias = maybe_pop_from_kwargs(
            kwargs, values, "values", "valias"
        )

        xalias = xalias or kwargs.pop("xalias", "")
        # Put back on kwargs for metadata to handle
        if valias is not None:
            kwargs["valias"] = valias

        if copy:

            def fetch(x: Array1D) -> Array1D:
                return _fetch(x, dtype=dtype, order=order).copy()

        else:

            def fetch(x: Array1D) -> Array1D:
                return _fetch(x, dtype=dtype, order=order)

        super().__init__(fetch(values))

        # Create an index from array or update existing index
        default_label = "xlabel" not in kwargs
        xlabel = kwargs.pop("xlabel", "Energy")
        # Pop a set of keys from kwargs if kwargs has these keys
        indexkwargs = indexkwargs or {}
        default_unit = False if unit is not None else True
        unit = (
            "keV" if default_unit else unit
        )  # Not elegant. Index will overwrite anyway.
        assert X is not None
        self._index = make_or_update_index(
            X,
            unit=unit,
            alias=xalias,
            label=xlabel,
            default_label=default_label,
            default_unit=default_unit,
            edge=edge,
            boundary=boundary,
            **indexkwargs,
        )
        _xalias = "" if not xalias else f" (`{xalias}`)"
        _valias = "" if not valias else f" (`{valias}`)"
        if np.ndim(self._index) != 1:
            raise ValueError(f"Index must be 1D, got {np.ndim(self._index)}")
        if np.ndim(self.values) != 1:
            raise ValueError(f"Values must be 1D, got {np.ndim(self.values)}")
        if np.size(self._index) != np.size(self.values):
            raise ValueError(
                f"Length of index{_xalias} and values{_valias} must be the same. Got {len(self._index)} and {len(self.values)}"
            )
        if "ylabel" in kwargs:
            ylabel = kwargs.pop("ylabel")
            if "vlabel" in kwargs:
                raise ValueError("Can not specify both `ylabel` and `vlabel`")
            kwargs["vlabel"] = ylabel

        # Handle vunit parameter
        if vunit is not None:
            kwargs['vunit'] = vunit

        # Name and title are aliases
        if 'name' in kwargs and 'title' in kwargs:
            raise ValueError("Can not specify both `name` and `title`")
        if 'title' in kwargs:
            kwargs["name"] = kwargs.pop("title")

        # Add plot_settings to kwargs if provided
        if plot_settings is not None:
            kwargs['plot_settings'] = plot_settings

        wrong_kw = set(kwargs) - set(VectorMetadata.__slots__)
        if wrong_kw:
            raise ValueError(f"Invalid keyword arguments: {', '.join(wrong_kw)}")

        if not isinstance(metadata, VectorMetadata):
            metadata = VectorMetadata(**metadata)
        self.metadata: VectorMetadata = metadata.update(**kwargs)

        self.loc: ValueLocator = ValueLocator(self, strict=False)
        self.vloc: ValueLocator = ValueLocator(self, strict=True)
        self.iloc: IndexLocator = IndexLocator(self)

    def __getattr__(self, item) -> Any:
        meta: VectorMetadata = self.__dict__["metadata"]
        alias: str = self.__dict__["_index"].alias
        if item == alias:
            x = self.X
        elif item == meta.valias:
            x = self.__dict__["values"]
        elif item == "d" + alias:
            x = self.dX
        else:
            x = super().__getattr__(item)
        return x

    @ensure_path
    def save(
        self, path: Path, filetype: str | None = None, exist_ok: bool = True, **kwargs
    ) -> None:
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

        E = self._index.to_unit("keV").bins
        match filetype:
            case "npy":
                warnings.warn("Saving as .npy is deprecated. Use .npz instead.")
                save_numpy_1D(self.values, E, path)
            case "npz":
                save_npz_1D(path, self, exist_ok=exist_ok)
            case "txt":
                warnings.warn(
                    "Saving to .txt does not preserve metadata. Use .npz instead."
                )
                save_txt_1D(self.values, E, path, **kwargs)
            case "tar":
                warnings.warn(
                    "Saving to .tar does not preserve metadata. Use .npz instead."
                )
                save_tar([self.values, E], path)
            case "mama":
                warnings.warn("MAMA format does not preserve metadata.")
                mama_write(self, path, **kwargs)
            case "csv":
                warnings.warn("CSV format does not preserve metadata.")
                save_csv_1D(self.values, E, path)
            case "root":
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
            case "npy":
                values, E = load_numpy_1D(path)
            case "npz":
                return load_npz_1D(path, Vector)
            case "txt":
                values, E = load_txt_1D(path)
            case "tar":
                from_file = load_tar(path)
                if len(from_file) == 3:
                    values, E = from_file
                elif len(from_file) == 2:
                    values, E = from_file
                else:
                    raise ValueError(
                        f"Expected two or three columns\
                     in file '{path}', got {len(from_file)}"
                    )
            case "mama":
                ret = mama_read(str(path))
                if len(ret) == 2:
                    values, E = ret
                else:
                    raise ValueError(f"Expected two columns in mama, got {len(ret)}")
            case "csv":
                values, E = load_csv_1D(path)
            case "root":
                return load_root_1D(path, Vector)
            case _:
                try:
                    ret = mama_read(str(path))
                    if len(ret) == 2:
                        values, E = ret
                    else:
                        raise ValueError(
                            f"Expected two columns in mama, got {len(ret)}"
                        )
                    return Vector(E=E, values=values, edge="mid")
                except ValueError:  # from within ValueError
                    raise ValueError(f"Unknown filetype {filetype}")
        return Vector(E=E, values=values)

    @overload
    def drop_nan(self, inplace: Literal[False] = ...) -> Self: ...

    @overload
    def drop_nan(self, inplace: Literal[True] = ...) -> None: ...

    def drop_nan(self, inplace: bool = False) -> Self | None:
        """Drop the elements that are `np.nan`

        Args:
            inplace (bool, optional): If `True` perform the cut on this vector
                or if `False` returns a copy. Defaults to True
        Returns:
            The cut vector if `inplace` is True.
        """
        return self.from_mask(~np.isnan(self.values), inplace=inplace)

    @overload
    def rebin(
        self,
        bins: arraylike | Index | None = None,
        factor: float | None = None,
        binwidth: QuantityLike | None = None,
        numbins: int | None = None,
        preserve: Preserve = "counts",
        inplace: Literal[False] = ...,
    ) -> Self: ...

    @overload
    def rebin(
        self,
        bins: arraylike | Index | None = None,
        factor: float | None = None,
        binwidth: QuantityLike | None = None,
        numbins: int | None = None,
        preserve: Preserve = "counts",
        inplace: Literal[True] = ...,
    ) -> None: ...

    def rebin(
        self,
        bins: arraylike | Index | None = None,
        factor: float | None = None,
        binwidth: QuantityLike | None = None,
        numbins: int | None = None,
        preserve: Preserve = "counts",
        inplace: bool = False,
    ) -> Self | None:
        """Rebins vector, assuming equidistant binning

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
        bins_: Index = self._index.handle_rebin_arguments(
            bins=bins, factor=factor, binwidth=binwidth, numbins=numbins
        )
        _, rebinned = self._index.rebin(bins_, self.values, preserve=preserve)

        if inplace:
            self.values = rebinned
            self._index = bins_
        else:
            return self.clone(X=bins_, values=rebinned)

    @overload
    def rebin_like(self, other: Vector, inplace: Literal[False] = ...) -> Self: ...

    @overload
    def rebin_like(self, other: Vector, inplace: Literal[True] = ...) -> None: ...

    def rebin_like(
        self,
        other: Vector | Index,
        inplace: bool = False,
        preserve: Preserve = "counts",
    ) -> Self | None:
        """Rebin to match the binning of `other`.

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

    @overload
    def reshape_like(self, other: Vector, inplace: Literal[False] = ...) -> Self: ...

    @overload
    def reshape_like(self, other: Vector, inplace: Literal[True] = ...) -> None: ...

    def reshape_like(self, other: Vector | Index, inplace: bool = False) -> Self | None:
        """Reshape the vector so its index becomes congruent with ``other``.

        This method cuts the vector to match another vector's binning structure.
        Both vectors must have the same bin widths and share a congruent lattice.

        Args:
            other: Vector (or Index) providing the target binning.
            inplace: Update this vector in-place when True. Returns a new vector otherwise.

        Raises:
            ValueError: If the indices cannot be made congruent through cutting.
        """

        def _slice_to_match(source: Index, target: Index) -> tuple[slice, Index]:
            aligned_target = target.to_unit(source).to_same_edge(source)
            if len(aligned_target) == 0:
                raise ValueError("Cannot align to an empty index.")
            if len(aligned_target) > len(source):
                raise ValueError("Other index has more bins than self.")
            if not (source.is_uniform() and aligned_target.is_uniform()):
                raise ValueError("Cannot cut non-uniform indices to be congruent.")
            dx_source = float(source.dX)
            dx_target = float(aligned_target.dX)
            if not np.isclose(dx_source, dx_target):
                raise ValueError(
                    f"Incompatible bin widths: {dx_source} vs {dx_target}."
                )
            start_value = float(aligned_target[0])
            origin = float(source[0])
            offset = (start_value - origin) / dx_source
            offset_rounded = round(offset)
            if not np.isclose(offset, offset_rounded):
                raise ValueError("Indices do not share a congruent lattice.")
            start = int(offset_rounded)
            stop = start + len(aligned_target)
            if start < 0 or stop > len(source):
                raise ValueError("Requested cut falls outside bounds.")
            return slice(start, stop)

        match other:
            case Vector():
                target_index = other._index
            case Index():
                target_index = other
            case _:
                raise TypeError(f"Cannot cut like {type(other)}")

        s = _slice_to_match(self._index, target_index)
        values = self.values[s]
        if inplace:
            self.values = values
            self._index = target_index
            return None
        return self.clone(X=target_index, values=values)

    @overload
    def cut_like(self, other: Vector, inplace: Literal[False] = ..., fill: int | None = None) -> Self: ...

    @overload
    def cut_like(self, other: Vector, inplace: Literal[True] = ..., fill: int | None = None) -> None: ...

    def cut_like(self, other: Vector | Index, inplace: bool = False, fill: int | None = None) -> Self | None:
        """Cut the vector to span the same range as ``other``.

        Unlike reshape_like, this method does not require matching bin widths.
        It only ensures that the vector spans the same range as the other vector/index.

        Args:
            other: Vector (or Index) providing the target range.
            inplace: Update this vector in-place when True. Returns a new vector otherwise.
            fill: Value to use when other has a larger range. If None and other is larger,
                  raises ValueError. If provided, extends the vector with this fill value.

        Raises:
            ValueError: If other has a larger range and fill is None.
        """
        match other:
            case Vector():
                target_index = other._index
            case Index():
                target_index = other
            case _:
                raise TypeError(f"Cannot cut like {type(other)}")

        # Convert to same units and edge type
        aligned_target = target_index.to_unit(self._index.unit).to_same_edge(self._index)
        
        if len(aligned_target) == 0:
            raise ValueError("Cannot align to an empty index.")
        
        # Get the range boundaries
        source_left = float(self._index.leftmost)
        source_right = float(self._index.rightmost)
        target_left = float(aligned_target.leftmost)
        target_right = float(aligned_target.rightmost)
        
        # Check if target extends beyond source
        extends_left = target_left < source_left
        extends_right = target_right > source_right
        
        if (extends_left or extends_right) and fill is None:
            raise ValueError(
                f"Target has larger range [{target_left}, {target_right}] than source "
                f"[{source_left}, {source_right}] and no fill value provided."
            )
        
        # Find overlapping range
        overlap_left = max(source_left, target_left)
        overlap_right = min(source_right, target_right)
        
        if overlap_left >= overlap_right:
            raise ValueError("No overlap between source and target")
        
        # Find indices in source that fall within target range
        source_bins = self._index.bins
        mask = (source_bins >= target_left) & (source_bins <= target_right)
        
        if not np.any(mask):
            # No bins in range, try a more lenient check
            mask = (source_bins >= overlap_left) & (source_bins <= overlap_right)
        
        if not np.any(mask):
            raise ValueError("No bins from source fall within target range")
        
        # Get the slice of values
        indices = np.where(mask)[0]
        start_idx = indices[0]
        stop_idx = indices[-1] + 1
        
        cut_values = self.values[start_idx:stop_idx]
        cut_index = self._index[start_idx:stop_idx]
        
        # If target is larger and fill is provided, we need to extend
        if fill is not None and (extends_left or extends_right):
            # Extend source's binning to cover target's range, keeping source's bin width
            # Get source's bin width
            source_dX = self._index.steps()
            if isinstance(source_dX, np.ndarray):
                # Non-uniform binning - use the most common bin width
                bin_width = float(np.median(source_dX))
            else:
                bin_width = float(source_dX)
            
            # Determine how many bins to add on each side
            n_left = 0
            n_right = 0
            
            if extends_left:
                # Calculate bins needed to extend left
                gap_left = source_left - target_left
                n_left = int(np.ceil(gap_left / bin_width))
            
            if extends_right:
                # Calculate bins needed to extend right
                gap_right = target_right - source_right
                n_right = int(np.ceil(gap_right / bin_width))
            
            # Create extended bins array
            total_bins = len(cut_index) + n_left + n_right
            
            # Build the extended index using source's binning class
            new_leftmost = cut_index.leftmost - n_left * bin_width
            extended_bins = new_leftmost + np.arange(total_bins) * bin_width
            # Use cut_index's class to preserve the exact Index type (Left/Mid, Uniform/NonUniform)
            new_index = cut_index.__class__.from_array(
                extended_bins,
                extrapolate_boundary=True,
                unit=cut_index.unit,
                label=cut_index.label,
                alias=cut_index.alias
            )
            
            # Create array with fill values
            new_values = np.full(len(new_index), fill, dtype=self.values.dtype)
            
            # Place cut_values in the correct position
            # cut_values starts at n_left
            new_values[n_left:n_left + len(cut_index)] = cut_values
            
            if inplace:
                self.values = new_values
                self._index = new_index
                return None
            return self.clone(X=new_index, values=new_values)
        else:
            if inplace:
                self.values = cut_values
                self._index = cut_index
                return None
            return self.clone(X=cut_index, values=cut_values)

    @overload
    def align_with(self, other: Vector, fill: None = None) -> tuple[Self, Self]: ...

    @overload
    def align_with(self, other: Vector, fill: int) -> tuple[Self, Self]: ...

    def align_with(self, other: Vector | Index, fill: int | None = None) -> tuple[Self, Self]:
        """Align both vectors to a common congruent binning structure.
        
        Both vectors are rebinned to the coarser (larger) bin width and cut/extended
        to a common range, ensuring they become congruent (identical binning structure).
        
        Args:
            other: The vector or index to align with
            fill: If None, uses intersection of ranges only.
                  If provided, extends to union of ranges with this fill value.
        
        Returns:
            Tuple of (self_aligned, other_aligned) with congruent binning
            
        Raises:
            ValueError: If there's no overlapping range between vectors
            
        Example:
            >>> v1 = Vector(X=np.arange(0, 100, 2), ...)  # bin width 2
            >>> v2 = Vector(X=np.arange(20, 80, 5), ...)  # bin width 5
            >>> v1_aligned, v2_aligned = v1.align_with(v2)
            >>> # Both now have bin width 5, range [20, 80]
            >>> v1_aligned.X_index == v2_aligned.X_index  # True
        """
        # Get the other as a Vector
        match other:
            case Vector():
                other_vec = other
            case Index():
                # Create a dummy vector from index
                other_vec = Vector(X=other, values=np.zeros(len(other)))
            case _:
                raise TypeError(f"Cannot align with {type(other)}")
        
        # Translate fill parameter: fill=None → 'intersection', fill=int → 'union'
        mode = 'intersection' if fill is None else 'union'
        
        # Use Index.align_with() to get common index (handles coarser bin width and lattice alignment)
        common_index = self._index.align_with(other_vec._index, fill=mode)
        
        # Rebin both vectors directly to the common index
        # This automatically handles the lattice alignment
        v1 = self.rebin(bins=common_index)
        v2 = other_vec.rebin(bins=common_index)
        
        # For union mode with non-zero fill, rebin fills with 0 by default
        # We need to replace the 0s in extended regions with the fill value
        if fill is not None and fill != 0:
            # Identify regions that were extended (outside original range)
            mask_v1 = (common_index.bins < self._index.leftmost) | (common_index.bins > self._index.rightmost)
            mask_v2 = (common_index.bins < other_vec._index.leftmost) | (common_index.bins > other_vec._index.rightmost)
            
            # Fill extended regions
            if np.any(mask_v1):
                v1.values[mask_v1] = fill
            if np.any(mask_v2):
                v2.values[mask_v2] = fill
        
        return v1, v2

    def set_order(self, order: np._OrderKACF) -> None:
        """Wrapper around numpy to set the alignment"""
        self.values = self.values.copy(order=order)
        self._index = self._index.copy(order=order)

    @property
    def dX(self) -> float | np.ndarray:
        if is_uniform(self._index):
            return self._index.dX
        return self._index.steps()

    def last_nonzero(self, eps: float = 0) -> int:
        """Returns the index of the last nonzero value"""
        j = len(self)
        while (j := j - 1) >= 0:
            if self[j] > eps:
                break
        return j

    def cut_at_last_nonzero(self, **kwargs) -> Self:
        return self.iloc[: self.last_nonzero(**kwargs) + 1]

    def update(
        self,
        xlabel: str | None = None,
        vlabel: str | None = None,
        vunit: Unitlike | None = None,
        name: str | None = None,
        misc: dict[str, Any] | None = None,
        inplace: bool = False,
        title: str | None = None,
    ) -> None | Self:
        index = self._index.update(label=xlabel)
        if title is not None:
            if name is not None:
                if name != title:
                    raise ValueError(
                        "`name` and `title` alias each other. Only provide one"
                    )
            name = title
        meta = self.metadata.update(vlabel=vlabel, vunit=vunit, name=name, misc=misc)
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
        s += f"\nValue alias: {self.metadata.valias}\n"
        s += f"ylabel: {self.metadata.vlabel}\n"
        if len(self.metadata.misc) > 0:
            s += "Metadata:\n"
            for key, val in self.metadata.misc.items():
                s += f"\t{key}: {val}\n"
        s += f"Total counts: {self.sum():.3g}\n"
        s += f"NaN counts: {np.isnan(self.values).sum()}\n"
        return s

    def summary(self) -> None:
        print(self._summary)

    def _repr_html_(self) -> str:
        """
        HTML representation for notebook display.
        Mirrors Matrix._repr_html_ with vector-specific details.
        """
        array_info = [
            ("Array type", self.values.__class__.__name__),
            ("Total counts", f"{self.sum():.3g}"),
            ("NaN counts", f"{np.isnan(self.values).sum()}"),
        ]

        metadata_html = ""
        if (
            hasattr(self, "metadata")
            and hasattr(self.metadata, "misc")
            and len(self.metadata.misc) > 0
        ):
            metadata_items = [(key, str(val)) for key, val in self.metadata.misc.items()]
            metadata_html = f"""
            <div class="metadata-section">
                <h4>Metadata:</h4>
                {table(metadata_items, color="#f0f0f0")}
            </div>
            """

        html = f"""
        <div class="array-container" style="margin: 10px 0;">
            <div class="array-info">
                {table(array_info, color="#e6f7ff")}
            </div>
            <div class="indices-section" style="margin-top: 10px;">
                {collapse(self._index._repr_html_(), "Index")}
            </div>

            {metadata_html}

            <div class="values-section" style="margin-top: 10px;">
                {collapsible(self.values, "Array Values")}
            </div>
        </div>
        """

        return html

    def __str__(self) -> str:
        summary = self._summary
        summary += "\nValues:\n"
        return summary + str(self.values)

    def clone(
        self,
        X=None,
        values=None,
        order: np._OrderKACF | None = None,
        metadata=None,
        copy=False,
        dtype: np.dtype | None = None,
        **kwargs,
    ) -> Self:
        """Copies the object.

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
        return Vector(
            X=X,
            values=values,
            order=order,
            metadata=metadata,
            copy=copy,
            dtype=dtype,
            **kwargs,
        )

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
        unit = f"{self.unit:~L}"
        unit = f" [${unit}$]" if unit else ""
        return self.xlabel + unit

    @property
    def ylabel(self) -> str:
        return self.vlabel

    @ylabel.setter
    def ylabel(self, value: str) -> None:
        self.update(vlabel=value, inplace=True)

    @property
    def vlabel(self) -> str:
        return self.metadata.vlabel

    @vlabel.setter
    def vlabel(self, value: str) -> None:
        self.update(vlabel=value, inplace=True)

    def get_ylabel(self) -> str:
        """Get formatted y-axis label with unit"""
        unit = f"{self.vunit:~L}"
        unit = f" [${unit}$]" if unit else ""
        return self.ylabel + unit

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
        """Returns an iterator over the indices and values"""
        for i, x in enumerate(self.X):
            yield i, x, self.values[i]

    def unpack(self) -> tuple[np.ndarray, np.ndarray]:
        """Returns the energy and values as separate arrays"""
        return self.X, self.values

    def index(self, x: float) -> int:
        """Returns the index of the bin containing x"""
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
        """Converts the index to the given unit"""
        index = self._index.to_unit(unit)
        if inplace:
            self._index = index
        else:
            return self.clone(X=index)

    def to_edge(self, edge: Edges, inplace: bool = False) -> None | Self:
        """Converts the index to the given edge"""
        index = self._index.to_edge(edge)
        if inplace:
            self._index = index
        else:
            return self.clone(X=index)

    def to_left(self, inplace: bool = False) -> None | Self:
        """Converts the index to the left edge"""
        return self.to_edge("left", inplace=inplace)

    @overload
    def to_same_edge(self, other: Vector, inplace: Literal[False] = ...) -> Self: ...

    @overload
    def to_same_edge(self, other: Vector, inplace: Literal[True] = ...) -> None: ...

    def to_same_edge(self, other: Vector, inplace: bool = False) -> None | Self:
        """Converts the index to the same edge as other"""
        index = self._index.to_same_edge(other._index)
        if inplace:
            self._index = index
        else:
            return self.clone(X=index)

    def to_same(self, other: Vector) -> Self:
        return self.to_same_edge(other).to_unit(other.unit)

    @overload
    def shift_index(self, offset: QuantityLike, inplace: Literal[False] = ...) -> Self: ...

    @overload
    def shift_index(self, offset: QuantityLike, inplace: Literal[True] = ...) -> None: ...

    def shift_index(self, offset: QuantityLike, inplace: bool = False) -> Self | None:
        """Shift the index by a constant offset.

        Args:
            offset: The offset to shift the index by. Will be converted to the vector's unit.
            inplace: If True, modify this vector in place. If False, return a new vector.

        Returns:
            A new Vector with shifted index if inplace=False, None otherwise.

        Example:
            >>> vec = Vector(X=[0, 1, 2, 3], values=[10, 20, 30, 40], unit="keV")
            >>> shifted = vec.shift_index(100)  # Shift index by 100 keV
            >>> shifted.X[0]
            100.0
        """
        shifted_index = self._index.shift(offset)
        if inplace:
            self._index = shifted_index
            return None
        else:
            return self.clone(X=shifted_index)

    @overload
    def to_mid(self, inplace: Literal[False] = ...) -> Self: ...

    @overload
    def to_mid(self, inplace: Literal[True] = ...) -> Self: ...

    def to_mid(self, inplace: bool = False) -> None | Self:
        """Converts the index to the middle"""
        return self.to_edge("mid", inplace=inplace)

    @overload
    def plot(
        self,
        ax: Axes | None = None,
        kind: Literal["step", "plot", "line"] = ...,
        **kwargs,
    ) -> Plot1D: ...

    @overload
    def plot(
        self, ax: Axes | None = None, kind: Literal["dot", "scatter"] = ..., **kwargs
    ) -> PlotScatter1D: ...

    @overload
    def plot(
        self, ax: Axes | None = None, kind: Literal["bar"] = ..., **kwargs
    ) -> PlotBar1D: ...

    @overload
    def plot(
        self, ax: Axes | None = None, kind: Literal["poisson"] = ..., **kwargs
    ) -> PlotError1D: ...

    def plot(
        self, ax: Axes | None = None, kind: VectorPlotKind | None = None, 
        scale: str | None = None, **kwargs
    ) -> VectorPlot:
        """Plots the vector

        Args:
            ax (matplotlib axis, optional): The axis to plot onto. If not
                provided, a new figure is created
            kind (str, optional):
                - 'line' : line plot (default) evokes `ax.plot`
                - 'plot' : same as 'line'
                - 'step' : step plot
                - 'bar' : vertical bar plot
                If None, uses plot_settings.kind or defaults to 'step'
            scale (str, optional): Y-axis scale ('linear', 'log', 'symlog').
                If None, uses plot_settings.yscale
            kwargs (optional): Additional kwargs to plot command.

        Returns:
            The figure and axis used.
        """
        if ax is None:
            _, _ax = plt.subplots()
            assert isinstance(_ax, Axes)
            ax = _ax

        # Get plot settings
        settings = self.metadata.plot_settings
        
        # Priority: explicit args > plot_settings > defaults
        effective_scale = scale or (settings.yscale if settings else None)
        effective_kind = kind or (settings.kind if settings else None) or "step"
        
        # Merge plot_kwargs from settings with user kwargs
        if settings and settings.plot_kwargs:
            merged_kwargs = settings.plot_kwargs | kwargs
        else:
            merged_kwargs = kwargs

        # in case `values` is on the gpu
        values = np.asarray(self.values)

        match effective_kind:
            case "plot" | "line":
                if self._index.is_left():
                    bins = self.X + self.dX / 2
                else:
                    bins = self.X
                merged_kwargs.setdefault("markersize", 3)
                merged_kwargs.setdefault("marker", ".")
                merged_kwargs.setdefault("linestyle", "-")
                line = ax.plot(bins, values, **merged_kwargs)[0]
                assert isinstance(line, Line2D)
                maybe_set(
                    ax,
                    xlabel=self.get_xlabel(),
                    ylabel=self.get_ylabel(),
                    title=self.name,
                )
                if effective_scale:
                    ax.set_yscale(effective_scale)
                return ax, line
            case "step":
                step = "post" if self._index.is_left() else "mid"
                bins = self._index.ticks()
                if self._index.is_left():
                    values = np.append(values, values[-1])
                else:
                    values = np.append(np.append(values[0], values), values[-1])
                merged_kwargs.setdefault("where", step)
                line = ax.step(bins, values, **merged_kwargs)
                assert is_lines(line)
                maybe_set(
                    ax,
                    xlabel=self.get_xlabel(),
                    ylabel=self.get_ylabel(),
                    title=self.name,
                )
                if effective_scale:
                    ax.set_yscale(effective_scale)
                return ax, line
            case "bar":
                align = "center" if self._index.is_mid() else "edge"
                merged_kwargs.setdefault("align", align)
                merged_kwargs.setdefault("width", self.dX)
                line = ax.bar(self.X, values, **merged_kwargs)
                assert isinstance(line, BarContainer)
                maybe_set(
                    ax,
                    xlabel=self.get_xlabel(),
                    ylabel=self.get_ylabel(),
                    title=self.name,
                )
                if effective_scale:
                    ax.set_yscale(effective_scale)
                return ax, line
            case "dot" | "scatter":
                if self._index.is_left():
                    bins = self.X + self.dX / 2
                else:
                    bins = self.X
                merged_kwargs.setdefault("marker", ".")
                line = ax.scatter(bins, values, **merged_kwargs)
                assert isinstance(line, PathCollection)
                maybe_set(
                    ax,
                    xlabel=self.get_xlabel(),
                    ylabel=self.get_ylabel(),
                    title=self.name,
                )
                if effective_scale:
                    ax.set_yscale(effective_scale)
                return ax, line
            case "poisson":
                if self._index.is_left():
                    bins = self.X + self.dX / 2
                else:
                    bins = self.X
                kw = dict(marker="o", ls="none", capsize=2, capthick=0.5, ms=3, lw=1)
                kw |= merged_kwargs
                line = ax.errorbar(bins, values, yerr=np.sqrt(values), **kw)  # type: ignore
                assert isinstance(line, ErrorbarContainer)
                maybe_set(
                    ax,
                    xlabel=self.get_xlabel(),
                    ylabel=self.get_ylabel(),
                    title=self.name,
                )
                if effective_scale:
                    ax.set_yscale(effective_scale)
                return ax, line
            case _:
                raise ValueError(f"Invalid kind: {kind}")

    @overload
    def __matmul__(self, other: Matrix) -> Self: ...
    @overload
    def __matmul__(self, other: Vector) -> float: ...
    @overload
    def __matmul__(self, other: Array1D) -> Array1D | float: ...

    def __matmul__(self, other: Matrix | Vector | Array1D) -> Self | float | Array1D:
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
        """Returns a new vector with the given mask applied"""
        # Check that the True are contiguous
        if not check_contiguous(mask):
            raise ValueError("Mask must be contiguous")
        indices = np.argwhere(mask).ravel()
        start = indices[0]
        stop = indices[-1] + 1
        vec = self.iloc[start:stop]
        if inplace:
            self.values = vec.values
            self._index = vec._index
        else:
            return vec

    def clone_from_slice(self, slice_: slice) -> Self:
        """Returns a new vector with the given slice applied.

        Mainly used for the Locators to handle the creation of
        new vectors, particularly subclasses.
        """
        index: Index = self._index.__getitem__(slice_)
        values = self.values.__getitem__(slice_)
        return self.clone(X=index, values=values)

    def to_xarray(self):
        return to_xarray_vector(self)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}({len(self)})[{self.device}]{self.name} at {hex(id(self))}>"

    def integrate(
        self, method: Callable[[np.ndarray, np.ndarray], np.float_] = np.trapz
    ) -> np.float_:
        """Returns the integral of the vector

        Args:
            method: The integration method to use. Default is `np.trapz`
        """
        return method(self.values, self.X)

    # ============ CONVENIENCE CONSTRUCTORS WITH PRESETS ============

    @classmethod
    def from_preset(
        cls,
        preset: str | VectorPreset,
        X: arraylike,
        values: arraylike,
        *,
        ylabel: str | None = None,
        plot_settings: PlotSettings | dict[str, Any] | None = None,
        **kwargs: Any
    ) -> Self:
        """Create vector from a preset.
        
        Presets combine semantic info (xalias, labels) with visual defaults.
        
        Args:
            preset: Preset name (str) or VectorPreset object
            X: X-axis values
            values: Y-axis values
            ylabel: Override preset's ylabel
            plot_settings: Override preset's plot_settings (PlotSettings object or dict)
            **kwargs: Additional Vector constructor kwargs (unit, name, etc.)
        
        Examples:
            >>> # Use preset as-is
            >>> loss = Vector.from_preset('iterations', range(100), losses)
            >>> loss.plot()  # Automatically log scale, "Iterations" label
            
            >>> # Override ylabel
            >>> chi2 = Vector.from_preset('iterations', range(100), chi2_vals,
            ...                           ylabel=r'$\\chi^2$')
            
            >>> # Override plot settings (PlotSettings object)
            >>> loss = Vector.from_preset(
            ...     'iterations', range(100), losses,
            ...     plot_settings=PlotSettings.log_line().with_kwargs(color='red')
            ... )
            
            >>> # Override plot settings (dict - convenient!)
            >>> loss = Vector.from_preset(
            ...     'iterations', range(100), losses,
            ...     plot_settings={'yscale': 'linear', 'kind': 'scatter'}
            ... )
            
            >>> # Custom preset
            >>> my_preset = VectorPreset(
            ...     xalias='t', xlabel='Time [s]', ylabel='Voltage',
            ...     plot_settings=PlotSettings.linear_line()
            ... )
            >>> vec = Vector.from_preset(my_preset, times, voltages)
        """
        # Get preset
        if isinstance(preset, str):
            preset_obj = get_preset(preset)
        else:
            preset_obj = preset
        
        # Apply preset to kwargs
        preset_kwargs = preset_obj.apply_to_vector_kwargs()
        
        # Override ylabel if provided
        if ylabel is not None:
            preset_kwargs['vlabel'] = ylabel
        
        # Override plot_settings if provided
        if plot_settings is not None:
            preset_kwargs['plot_settings'] = plot_settings
        
        # Merge with user kwargs (user kwargs take precedence)
        preset_kwargs.update(kwargs)
        
        return cls(X=X, values=values, **preset_kwargs)
    
    @classmethod
    def for_iterations(
        cls,
        values: arraylike,
        *,
        iterations: arraylike | None = None,
        ylabel: str = "Loss",
        name: str = "",
        plot_settings: PlotSettings | dict[str, Any] | None = None,
        **kwargs: Any
    ) -> Self:
        """Create a vector for iteration/training curves.
        
        Convenience constructor that sets:
        - xalias='iteration', xlabel='Iterations'
        - ylabel (customizable, default='Loss')
        - Default plot: log scale, line plot
        
        Args:
            iterations: Iteration numbers (e.g., range(100))
            values: Values at each iteration
            ylabel: Y-axis label (default: "Loss")
            name: Vector name/title
            plot_settings: Override default plot settings (log line)
            **kwargs: Additional Vector constructor kwargs
        
        Examples:
            >>> # Simple usage
            >>> loss = Vector.for_iterations(range(100), losses)
            >>> loss.plot()  # Automatic log scale, proper labels
            
            >>> # Custom ylabel
            >>> chi2 = Vector.for_iterations(range(100), chi2_vals, 
            ...                              ylabel=r'$\\chi^2$')
            
            >>> # Linear scale instead
            >>> acc = Vector.for_iterations(range(100), accuracy,
            ...                             ylabel='Accuracy',
            ...                             plot_settings=PlotSettings.linear_line())
            
            >>> # With styling
            >>> loss = Vector.for_iterations(
            ...     range(100), losses,
            ...     plot_settings=PlotSettings.log_line().with_kwargs(
            ...         color='red', linewidth=2
            ...     )
            ... )
        """
        if plot_settings is None:
            plot_settings = PlotSettings()

        if iterations is None:
            iterations = np.arange(len(values))
        
        return cls(
            X=iterations,
            values=values,
            xalias='iteration',
            xlabel='Iterations',
            vlabel=ylabel,
            name=name,
            plot_settings=plot_settings,
            unit='',  # Iterations are dimensionless
            vunit='',  # Loss/metrics are dimensionless
            **kwargs
        )
    
    @classmethod
    def log_scale(
        cls,
        X: arraylike,
        values: arraylike,
        *,
        xlabel: str = "Energy",
        ylabel: str = "Counts",
        plot_settings: PlotSettings | dict[str, Any] | None = None,
        **kwargs: Any
    ) -> Self:
        """Create a vector with log scale by default.
        
        Args:
            X: X-axis values
            values: Y-axis values
            xlabel: X-axis label (default: "Energy")
            ylabel: Y-axis label (default: "Counts")
            plot_settings: Override default plot settings (log step)
            **kwargs: Additional Vector constructor kwargs
        
        Examples:
            >>> spectrum = Vector.log_scale(energies, counts)
            >>> spectrum.plot()  # Log scale, step plot
            
            >>> # Custom labels
            >>> nld = Vector.log_scale(E, rho,
            ...                        xlabel='Excitation Energy',
            ...                        ylabel=r'$\\rho$ [MeV$^{-1}$]')
        """
        if plot_settings is None:
            plot_settings = PlotSettings.log_step()
        
        return cls(
            X=X,
            values=values,
            xlabel=xlabel,
            vlabel=ylabel,
            plot_settings=plot_settings,
            **kwargs
        )

    def to_numba(self):
        return to_numba(self)

    def _coerce_other(self, other: AbstractArray | NDArray[Any] | float) -> NDArray[Any] | float:
        if isinstance(other, AbstractArray):
            # Only allow elementwise with another Vector
            if isinstance(other, Vector):
                if len(self) != len(other):
                    raise ValueError(f"Incompatible lengths: {len(self)} != {len(other)}")
                if not self.is_compatible_with(other._index):
                    raise ValueError("Incompatible vector indices.")
                return other.values  # 1D -> standard NumPy elementwise
            # Anything else falls back to ndarray behavior (e.g., Matrix ops will handle themselves)
            return other.values
        # ndarray/float: let NumPy broadcast
        return other


def to_numba(vec: Vector) -> None:
    raise NotImplementedError("Numba is not available")


if numba_available():
    from .numba_array import Vector as NumbaVector

    def to_numba(vec: Vector) -> NumbaVector:
        return NumbaVector(vec.X, vec.values)


if xarray_available():
    import xarray as xr

    def to_xarray_vector(vec) -> xr.DataArray:
        """Convert to xarray DataArray"""
        return xr.DataArray(vec.values, coords=[vec.X], dims=[vec.alias])

else:

    def to_xarray_vector(vec) -> Never:
        raise NotImplementedError("xarray is not installed")


if jax_available():
    import jax

    # Make it compatible as a pytree
    def flatten(obj) -> tuple[tuple[ndarray], dict[str, Any]]:
        aux = {
            "index": obj._index,
            "metadata": obj.metadata,
            "class": obj.__class__,
        }
        return (obj.values,), aux

    def unflatten(aux_data: dict[str, Any], children: tuple[ndarray]) -> Vector:
        return aux_data["class"](
            X=aux_data["index"],
            values=children[0],
            metadata=aux_data["metadata"],
        )

    jax.tree_util.register_pytree_node(Vector, flatten, unflatten)

VT = TypeVar("VT", bound=Vector)


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
    if true_indices.size == 0 or (
        true_indices.size == 1
        and (true_indices[0] == 0 or true_indices[0] == len(arr) - 1)
    ):
        return True

    # Check if all True values are contiguous
    if true_indices[-1] - true_indices[0] + 1 != true_indices.size:
        return False

    # Check if the contiguous run of True values starts or ends at an edge
    # if true_indices[0] == 0 or true_indices[-1] == len(arr) - 1:
    #   return True

    return True
