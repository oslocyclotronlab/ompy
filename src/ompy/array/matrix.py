from __future__ import annotations

import copy
import logging
import warnings
from pathlib import Path
from typing import Any, Callable, Literal, Never, Self, TypeAlias, overload

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps as cmaps
from matplotlib import ticker
from matplotlib.colors import Colormap, LogNorm, Normalize, SymLogNorm, LinearSegmentedColormap
from numpy.typing import DTypeLike

from ..accel import ROOT_imported, xarray_available, jax_available, jax_working
from ..units import Unit

from ..helpers import (
    AnnotatedColorbar,
IQR_range,
ensure_path,
make_ax,
maybe_set,
robust_z_score,
robust_z_score_i,
)
from ..accel import njit
from ..rendering.html import collapse, collapsible, table
from ..stubs import (
    ArrayBool,
    Axes,
    Colorbar,
    Figure,
    Pathlike,
    QuadMesh,
    QuantityLike,
    Unitlike,
    arraylike,
)
from .abstractarray import AbstractArray
from .abstractarray import fetch as _fetch
from .filehandling import (
    Filetype,
    load_hdf5_2D,
    load_npz_2D,
    load_numpy_2D,
    load_tar,
    load_txt_2D,
    mama_read,
    mama_write,
    resolve_filetype,
    save_hdf5_2D,
    save_npz_2D,
    save_numpy_2D,
    save_tar,
    save_txt_2D,
    load_root_2D,
)
from .index import Edges, Index, make_or_update_index
from .matrixmetadata import MatrixMetadata
from .matrixprotocol import MatrixProtocol
from .rebin import Preserve, rebin_2D
from .vector import Vector, maybe_pop_from_kwargs

LOG = logging.getLogger(__name__)
logging.captureWarnings(True)

# TODO mat*vec[:, None[ doesn't work
AxisEither: TypeAlias = Literal[0, 1]
AxisBoth: TypeAlias = Literal[0, 1, 2]
Axis: TypeAlias = AxisEither | AxisBoth
ColorByArg: TypeAlias = Literal["values", "z-score", "IQR", "IQR2"]
ColorBy: TypeAlias = ColorByArg | tuple[ColorByArg, ...]


class _BiLogNorm(Normalize):
    def __init__(
        self,
        neg_min: float,
        neg_max: float,
        pos_min: float,
        pos_max: float,
        clip: bool = False,
    ):
        super().__init__(vmin=-neg_max, vmax=pos_max, clip=clip)
        self.neg_min = float(neg_min)
        self.neg_max = float(neg_max)
        self.pos_min = float(pos_min)
        self.pos_max = float(pos_max)
        self._neg_log_min = np.log10(self.neg_min)
        self._neg_log_max = np.log10(self.neg_max)
        self._pos_log_min = np.log10(self.pos_min)
        self._pos_log_max = np.log10(self.pos_max)
        self._neg_span = self._neg_log_max - self._neg_log_min
        self._pos_span = self._pos_log_max - self._pos_log_min

    def __call__(self, value, clip=None):
        arr = np.ma.array(value, copy=False)
        result = np.zeros(arr.shape, dtype=float)
        mask = np.ma.getmask(arr)

        pos_mask = np.asarray(arr > 0)
        neg_mask = np.asarray(arr < 0)

        if pos_mask.any():
            log_vals = np.log10(arr[pos_mask])
            if np.isclose(self._pos_span, 0):
                result[pos_mask] = 1.0
            else:
                result[pos_mask] = 0.5 + 0.5 * (log_vals - self._pos_log_min) / self._pos_span

        if neg_mask.any():
            log_vals = np.log10(-arr[neg_mask])
            if np.isclose(self._neg_span, 0):
                result[neg_mask] = 0.0
            else:
                result[neg_mask] = 0.5 - 0.5 * (log_vals - self._neg_log_min) / self._neg_span

        zero_mask = np.asarray(arr == 0)
        result[zero_mask] = 0.5
        return np.ma.array(result, mask=mask)

    def inverse(self, value):
        val = np.asarray(value)
        out = np.zeros_like(val, dtype=float)

        pos_mask = val >= 0.5
        neg_mask = val < 0.5

        if pos_mask.any():
            if np.isclose(self._pos_span, 0):
                out[pos_mask] = 10 ** self._pos_log_min
            else:
                scaled = (val[pos_mask] - 0.5) / 0.5
                out[pos_mask] = 10 ** (scaled * self._pos_span + self._pos_log_min)

        if neg_mask.any():
            if np.isclose(self._neg_span, 0):
                out[neg_mask] = -(10 ** self._neg_log_min)
            else:
                scaled = (0.5 - val[neg_mask]) / 0.5
                out[neg_mask] = -(10 ** (scaled * self._neg_span + self._neg_log_min))

        zero_mask = np.isclose(val, 0.5)
        out[zero_mask] = 0.0
        return out


def _make_bilog_cmap(
    neg_cmap: Colormap, pos_cmap: Colormap, *, steps: int = 256
) -> Colormap:
    steps = max(steps, 2)
    half = steps // 2
    neg_samples = np.linspace(0, 1, half, endpoint=False)
    pos_samples = np.linspace(0, 1, steps - half)
    neg_colors = neg_cmap(neg_samples)[::-1]
    pos_colors = pos_cmap(pos_samples)
    colors_combined = np.vstack([neg_colors, pos_colors])
    cmap = LinearSegmentedColormap.from_list("bilog", colors_combined)
    return cmap


class Matrix(AbstractArray, MatrixProtocol):
    """Stores 2d array with energy axes (a matrix).

    Stores matrices along with calibration and energy axis arrays. Performs
    several integrity checks to verify that the arrays makes sense in relation
    to each other.


    Note that since a matrix is numbered NxM where N is rows going in the
    y-direction and M is columns going in the x-direction, the "x-dimension"
    of the matrix has the same shape as the Ex array (Excitation axis)

    Note:
        Many functions will implicitly assume linear binning.

    .. parsed-literal::
                   Diagonal Ex=Eγ
                                  v
         a y E │██████▓▓██████▓▓▓█░   ░
         x   x │██ █████▓████████░   ░░
         i a   │█████████████▓▓░░░░░
         s x i │███▓████▓████░░░░░ ░░░░
           i n │███████████░░░░   ░░░░░
         1 s d │███▓█████░░   ░░░░ ░░░░ <-- Counts
             e │███████▓░░░░░░░░░░░░░░░
             x │█████░     ░░░░ ░░  ░░
               │███░░░░░░░░ ░░░░░░  ░░░
             N │█▓░░  ░░░  ░░░░░  ░░░░░
               └───────────────────────
                     Eγ, index M
                     x axis
                     axis 0 of plot
                     axis 1 of matrix

    Attributes:
        values: 2D matrix storing the counting data
        Eg: The gamma energy along the x-axis (mid-bin calibration)
        Ex: The excitation energy along the y-axis (mid-bin calibration)
        path: Load a Matrix from a given path
        state: An enum to keep track of what has been done to the matrix
        shape: Tuple (len(Ex), len(Eg)), the shape of `values`


    TODO:
        - Synchronize cuts. When a cut is made along one axis,
          such as values[min:max, :] = 0, make cuts to the
          other relevant variables
        - Make values, Ex and Eg to properties so that
          the integrity of the matrix can be ensured.
    """

    _ndim = 2

    def __init__(
        self,
        *,
        X: arraylike | Index | None = None,
        Y: arraylike | Index | None = None,
        values: np.ndarray | None = None,
        X_unit: Unitlike | None = None,
        Y_unit: Unitlike | None = None,
        vunit: Unitlike | None = None,
        edge: Edges = "left",
        boundary: bool = False,
        metadata: MatrixMetadata = MatrixMetadata(),
        order: np._OrderKACF | None = None,
        copy: bool = False,
        indexkwargs: dict[str, Any] | None = None,
        dtype: DTypeLike = np.float32,
        **kwargs,
    ):
        # Resolve aliasing
        kwargs, X, xalias = maybe_pop_from_kwargs(kwargs, X, "X", "xalias")
        kwargs, Y, yalias = maybe_pop_from_kwargs(kwargs, Y, "Y", "yalias")
        kwargs, values, valias = maybe_pop_from_kwargs(
            kwargs, values, "values", "valias"
        )
        xalias = xalias or kwargs.pop("xalias", None)
        yalias = yalias or kwargs.pop("yalias", None)
        if valias is not None:
            kwargs["valias"] = valias

        if copy:

            def fetch(x):
                return _fetch(x, dtype, order).copy()

        else:

            def fetch(x):
                return _fetch(x, dtype, order)

        super().__init__(fetch(values))
        # self.values = fetch(values)
        if self.values.ndim != 2:
            raise ValueError(f"values must be 2D, not {self.values.ndim}")
        indexkwargs = indexkwargs or {}
        # If no `label` is given, set it to the default
        # but let Index.label override of given an Index
        default_xlabel = "xlabel" not in kwargs
        xlabel = kwargs.pop("xlabel", r"Excitation energy")
        default_ylabel = "ylabel" not in kwargs
        ylabel = kwargs.pop("ylabel", r"$\gamma$-energy")

        # If no `unit` is given, set it to the default
        # but let Index.unit override of given an Index
        default_X_unit = X_unit is None
        default_Y_unit = Y_unit is None
        X_unit = "keV" if default_X_unit else X_unit
        Y_unit = "keV" if default_Y_unit else Y_unit
        assert X_unit is not None
        assert Y_unit is not None
        # BUG: The dtype of the values should not be the same as the dtype of the index.
        dtype_index = None
        self.X_index: Index = make_or_update_index(
            X,
            unit=Unit(X_unit),
            alias=xalias,
            label=xlabel,
            default_label=default_xlabel,
            default_unit=default_X_unit,
            edge=edge,
            boundary=boundary,
            dtype=dtype_index,
            **indexkwargs,
        )
        self.Y_index: Index = make_or_update_index(
            Y,
            unit=Unit(Y_unit),
            alias=yalias,
            label=ylabel,
            default_label=default_ylabel,
            default_unit=default_Y_unit,
            edge=edge,
            boundary=boundary,
            dtype=dtype_index,
            **indexkwargs,
        )
        if len(self.X_index) != self.values.shape[0]:
            _alias = f" ({xalias})" if xalias else ""
            _valias = f" ({valias})" if valias else ""
            raise ValueError(
                f"Length of X_index{_alias} must match first dimension of values{_valias}, expected {self.values.shape[0]}, got {len(self.X_index)}"
            )
        if len(self.Y_index) != self.values.shape[1]:
            _alias = f" ({yalias})" if yalias else ""
            _valias = f" ({valias})" if valias else ""
            raise ValueError(
                f"Length of Y_index{_alias} must match second dimension of values{_valias}, expected {self.values.shape[1]}, got {len(self.Y_index)}"
            )
        # Handle vunit parameter
        if vunit is not None:
            kwargs['vunit'] = vunit
        wrong_kw = set(kwargs) - set(MatrixMetadata.__slots__)
        if wrong_kw:
            raise ValueError(f"Invalid keyword arguments: {', '.join(wrong_kw)}")
        if isinstance(metadata, dict):
            metadata = MatrixMetadata(**metadata)
        if not isinstance(metadata, MatrixMetadata):
            raise TypeError(f"metadata must be a MatrixMetadata, not {type(metadata)}")
        self.metadata = metadata.update(**kwargs)
        if self.xalias == self.yalias and xalias:
            raise ValueError(
                f"Aliases must be unique. Got {self.xalias} == {self.yalias}"
            )
        self.iloc = IndexLocator(self)
        self.vloc = ValueLocator(self)
        self.loc = ValueLocator(self, strict=False)

    def __getattr__(self, item) -> Any:
        meta: MatrixMetadata = self.__dict__["metadata"]
        xalias: str = self.__dict__["X_index"].alias
        yalias: str = self.__dict__["Y_index"].alias
        if item == xalias:
            x = self.X
        elif item == yalias:
            x = self.Y
        elif item == meta.valias:
            x = self.__dict__["values"]
        elif item == "d" + xalias:
            x = self.dX
        elif item == "d" + yalias:
            x = self.dY
        elif item == "index_" + xalias:
            return self.index_X
        elif item == "index_" + yalias:
            return self.index_Y
        elif item == xalias + "_index":
            return self.X_index
        elif item == yalias + "_index":
            return self.Y_index
        else:
            x = super().__getattr__(item)
        return x

    def __dir__(self) -> list[str]:
        base = super().__dir__()
        meta = self.__dict__.get("metadata")
        xi = self.__dict__.get("X_index")
        yi = self.__dict__.get("Y_index")
        extras = []
        if xi:
            extras += [xi.alias, "d"+xi.alias, xi.alias+"_index", "index_"+xi.alias]
        if yi:
            extras += [yi.alias, "d"+yi.alias, yi.alias+"_index", "index_"+yi.alias]
        if meta:
            extras += [meta.valias]
        return sorted(set(base) | set(extras))


    @classmethod
    def from_path(
        cls, path: Pathlike, filetype: Filetype | None = None, **kwargs
    ) -> Self:
        """Load matrix from specified filetype

        Args:
            path (str or Path): path to file to load
            filetype (str, optional): Filetype to load. Has an
                auto-recognition.

        Raises:
            ValueError: If filetype is unknown
        """
        path = Path(path)
        path, filetype = resolve_filetype(path, filetype)

        match filetype:
            case "npz":
                return cls.from_npz(path, **kwargs)
            case "npy":
                return cls.from_npy(path)
            case "txt":
                return cls.from_txt(path)
            case "tar":
                return cls.from_tar(path)
            case "mama":
                return cls.from_mama(path)
            case "hdf5":
                return cls.from_hdf5(path, **kwargs)
            case "root":
                return cls.from_root(path, **kwargs)
            case _:
                raise ValueError(f"Unknown filetype: {filetype}")

    @classmethod
    @ensure_path
    def from_npz(cls, path: Path, **kwargs) -> Self:
        return load_npz_2D(path, cls, **kwargs)

    @classmethod
    @ensure_path
    def from_npy(cls, path: Path) -> Self:
        values, Y, X = load_numpy_2D(path)
        return cls(Ex=X, Eg=Y, values=values)

    @classmethod
    @ensure_path
    def from_txt(cls, path: Path) -> Self:
        values, Y, X = load_txt_2D(path)
        return cls(Ex=X, Eg=Y, values=values)

    @classmethod
    @ensure_path
    def from_tar(cls, path: Path) -> Self:
        values, Y, X = load_tar(path)
        return cls(Ex=X, Eg=Y, values=values)

    @classmethod
    @ensure_path
    def from_mama(cls, path: Path) -> Self:
        ret = mama_read(path)
        if len(ret) == 3:
            values, Y, X = ret
            return cls(Ex=X, Eg=Y, values=values)
        else:
            raise RuntimeError("Could not interpret mama file")

    @classmethod
    @ensure_path
    def from_hdf5(cls, path: Path, **kwargs) -> Self:
        return load_hdf5_2D(path, cls, **kwargs)

    @classmethod
    @ensure_path
    def from_root(cls, path: Path, what: str, **kwargs) -> Self:
        return load_root_2D(path, what, cls, **kwargs)

    @classmethod
    def from_hist(cls, **kwargs) -> Self:
        # Two first kwargs are X and Y
        kw = iter(kwargs)
        Xalias = next(kw)
        Yalias = next(kw)
        xval = kwargs.pop(Xalias)
        yval = kwargs.pop(Yalias)
        values, xedges, yedges = jnp.histogram2d(xval, yval, **kwargs)
        cls_kwargs = {Xalias: np.asarray(xedges), Yalias: np.asarray(yedges), 'boundary': True}
        return cls(values=values.T, **cls_kwargs)

    @ensure_path
    def save(self, path: Path, filetype: Filetype | None = None, **kwargs) -> None:
        """Save matrix to file

        Legacy method. Prefer `to_<format>(path)` methods instead.

        Args:
            path (str or Path): path to file to save
            filetype (str, optional): Filetype to save. Has an
                auto-recognition. Options: ["numpy", "npz", "tar", "mama", "txt", "hdf5"]
            **kwargs: additional keyword arguments
        Raises:
            ValueError: If filetype is unknown
        """
        path, filetype = resolve_filetype(path, filetype)

        match filetype:
            case "npz":
                self.to_npz(path, **kwargs)
            case "npy":
                warnings.warn("Saving as numpy is deprecated, use npz instead")
                self.to_numpy(path, **kwargs)
            case "txt":
                self.to_txt(path, **kwargs)
            case "tar":
                warnings.warn(
                    "Saving to .tar does not preserve metadata. Use .npz instead."
                )
                save_tar([self.values, self.Y_index.bins, self.X_index.bins], path)
            case "mama":
                self.to_mama(path, **kwargs)
            case "hdf5":
                self.to_hdf5(path, **kwargs)
            case _:
                raise ValueError(f"Unknown filetype: {filetype}")

    @ensure_path
    def to_npz(self, path: Path, **kwargs) -> None:
        """Save matrix to NPZ file format.

        Args:
            path (Pathlike): Path to save the file.
            **kwargs: Additional keyword arguments to pass to save_npz_2D.
        """
        save_npz_2D(path, self, **kwargs)

    @ensure_path
    def to_hdf5(self, path: Path, **kwargs) -> None:
        """Save matrix to HDF5 file format.

        Args:
            path (Pathlike): Path to save the file.
            **kwargs: Additional keyword arguments to pass to save_hdf5_2D.
        """
        save_hdf5_2D(self, path, **kwargs)

    @ensure_path
    def to_txt(self, path: Path, **kwargs) -> None:
        """Save matrix to TXT file format.

        Args:
            path (Pathlike): Path to save the file.
            **kwargs: Additional keyword arguments to pass to save_txt_2D.
        """
        X = self.X_index.to_unit("keV").bins
        Y = self.Y_index.to_unit("keV").bins
        save_txt_2D(self.values, Y, X, path, **kwargs)

    @ensure_path
    def to_npy(self, path: Path) -> None:
        """Save matrix to NumPy binary file format.

        Args:
            path (Pathlike): Path to save the file.
            **kwargs: Additional keyword arguments to pass to save_numpy_2D.
        """
        X = self.X_index.to_unit("keV").bins
        Y = self.Y_index.to_unit("keV").bins
        save_numpy_2D(self.values, Y, X, path)

    @ensure_path
    def to_mama(self, path: Path, **kwargs) -> None:
        """Save matrix to MAMA file format.

        Args:
            path (Pathlike): Path to save the file.
            **kwargs: Additional keyword arguments to pass to mama_write.
        """
        mama_write(self, path, comment="Made by OMpy", **kwargs)

    @ensure_path
    def to_tar(self, path: Path) -> None:
        """Save matrix as tarball

        Args:
             path (Pathlike): Path to save the file
        """
        X = self.X_index.to_unit("keV").bins
        Y = self.Y_index.to_unit("keV").bins
        save_tar([self.values, Y, X])

    @overload
    def reshape_like(self, other: Matrix, inplace: Literal[False] = ...) -> Matrix: ...

    @overload
    def reshape_like(self, other: Matrix, inplace: Literal[True] = ...) -> None: ...

    def reshape_like(self, other: Matrix, inplace: bool = False) -> Matrix | None:
        """Reshape the matrix so its axes become congruent with ``other``.

        This method cuts the matrix to match another matrix's binning structure.
        Both matrices must have the same bin widths and share a congruent lattice.

        Args:
            other: Matrix providing the target binning.
            inplace: Update this matrix in-place when True. Returns a new matrix otherwise.

        Raises:
            ValueError: If the axes cannot be made congruent through cutting.
        """

        def _axis_slice(
            source: Index, target: Index, axis_label: str
        ) -> tuple[slice, Index]:
            aligned_target = target.to_unit(source).to_same_edge(source)
            if len(aligned_target) == 0:
                raise ValueError(f"Cannot align empty axis {axis_label!r}.")
            if len(aligned_target) > len(source):
                raise ValueError(
                    f"Other matrix has more bins than self along axis {axis_label!r}."
                )
            if not (source.is_uniform() and aligned_target.is_uniform()):
                raise ValueError(
                    f"Cannot cut non-uniform axis {axis_label!r} to be congruent."
                )
            dx_source = float(source.dX)
            dx_target = float(aligned_target.dX)
            if not np.isclose(dx_source, dx_target):
                raise ValueError(
                    f"Incompatible bin width along axis {axis_label!r}: "
                    f"{dx_source} vs {dx_target}."
                )
            start_value = float(aligned_target[0])
            origin = float(source[0])
            offset = (start_value - origin) / dx_source
            offset_rounded = round(offset)
            if not np.isclose(offset, offset_rounded):
                raise ValueError(
                    f"Axis {axis_label!r} does not share a congruent lattice "
                    "with the target."
                )
            start = int(offset_rounded)
            stop = start + len(aligned_target)
            if start < 0 or stop > len(source):
                raise ValueError(
                    f"Requested cut for axis {axis_label!r} falls outside bounds."
                )
            sliced = source[start:stop]
            comparison = sliced.to_unit(target).to_same_edge(target)
            if comparison != target:
                raise ValueError(
                    f"Unable to reproduce target axis {axis_label!r} via cutting."
                )
            return slice(start, stop), sliced

        sx, x_index = _axis_slice(self.X_index, other.X_index, "X")
        sy, y_index = _axis_slice(self.Y_index, other.Y_index, "Y")
        values = self.values[sx, sy]
        if inplace:
            self.values = values
            self.X_index = x_index
            self.Y_index = y_index
            return None
        return self.clone(X=x_index, Y=y_index, values=values)

    @overload
    def cut_like(self, other: Matrix, inplace: Literal[False] = ..., fill: int | None = None) -> Matrix: ...

    @overload
    def cut_like(self, other: Matrix, inplace: Literal[True] = ..., fill: int | None = None) -> None: ...

    def cut_like(self, other: Matrix, inplace: bool = False, fill: int | None = None) -> Matrix | None:
        """Cut the matrix to span the same range as ``other``.

        Unlike reshape_like, this method does not require matching bin widths.
        It only ensures that X and Y span the same range as the other matrix.

        Args:
            other: Matrix providing the target range.
            inplace: Update this matrix in-place when True. Returns a new matrix otherwise.
            fill: Value to use when other has a larger range. If None and other is larger,
                  raises ValueError. If provided, extends the matrix with this fill value.

        Raises:
            ValueError: If other has a larger range and fill is None.
        """

        def _cut_axis(
            source: Index, target: Index, source_values: np.ndarray, axis: int, axis_label: str
        ) -> tuple[Index, np.ndarray]:
            # Convert to same units and edge type
            aligned_target = target.to_unit(source.unit).to_same_edge(source)
            
            if len(aligned_target) == 0:
                raise ValueError(f"Cannot align to empty axis {axis_label!r}.")
            
            # Get the range boundaries
            source_left = float(source.leftmost)
            source_right = float(source.rightmost)
            target_left = float(aligned_target.leftmost)
            target_right = float(aligned_target.rightmost)
            
            # Check if target extends beyond source
            extends_left = target_left < source_left
            extends_right = target_right > source_right
            
            if (extends_left or extends_right) and fill is None:
                raise ValueError(
                    f"Target axis {axis_label!r} has larger range "
                    f"[{target_left}, {target_right}] than source "
                    f"[{source_left}, {source_right}] and no fill value provided."
                )
            
            # Find overlapping range
            overlap_left = max(source_left, target_left)
            overlap_right = min(source_right, target_right)
            
            if overlap_left >= overlap_right:
                raise ValueError(
                    f"No overlap between source and target on axis {axis_label!r}"
                )
            
            # Find indices in source that fall within target range
            # We want bins whose centers or edges fall within [target_left, target_right]
            source_bins = source.bins
            mask = (source_bins >= target_left) & (source_bins <= target_right)
            
            if not np.any(mask):
                # No bins in range, try a more lenient check
                mask = (source_bins >= overlap_left) & (source_bins <= overlap_right)
            
            if not np.any(mask):
                raise ValueError(
                    f"No bins from source fall within target range on axis {axis_label!r}"
                )
            
            # Get the slice of values
            indices = np.where(mask)[0]
            start_idx = indices[0]
            stop_idx = indices[-1] + 1
            
            if axis == 0:
                cut_values = source_values[start_idx:stop_idx, :]
            else:
                cut_values = source_values[:, start_idx:stop_idx]
            
            cut_index = source[start_idx:stop_idx]
            
            # If target is larger and fill is provided, we need to extend
            if fill is not None and (extends_left or extends_right):
                # Extend source's binning to cover target's range, keeping source's bin width
                # Get source's bin width
                source_dX = source.steps()
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
                if axis == 0:
                    new_shape = (len(new_index), source_values.shape[1])
                else:
                    new_shape = (source_values.shape[0], len(new_index))
                new_values = np.full(new_shape, fill, dtype=source_values.dtype)
                
                # Place cut_values in the correct position
                # cut_values starts at n_left
                if axis == 0:
                    new_values[n_left:n_left + len(cut_index), :] = cut_values
                else:
                    new_values[:, n_left:n_left + len(cut_index)] = cut_values
                
                return new_index, new_values
            else:
                return cut_index, cut_values
        
        # Process both axes
        x_index, values_x_cut = _cut_axis(self.X_index, other.X_index, self.values, 0, "X")
        y_index, values_final = _cut_axis(self.Y_index, other.Y_index, values_x_cut, 1, "Y")
        
        if inplace:
            self.values = values_final
            self.X_index = x_index
            self.Y_index = y_index
            return None
        return self.clone(X=x_index, Y=y_index, values=values_final)

    @overload
    def align_with(self, other: Matrix, fill: None = None) -> tuple[Matrix, Matrix]: ...

    @overload  
    def align_with(self, other: Matrix, fill: int) -> tuple[Matrix, Matrix]: ...

    def align_with(self, other: Matrix, fill: int | None = None) -> tuple[Matrix, Matrix]:
        """Align both matrices to a common congruent binning structure.
        
        Both matrices are rebinned to the coarser (larger) bin width and cut/extended
        to a common range, ensuring they become congruent (identical binning structure).
        
        Args:
            other: The matrix to align with
            fill: If None, uses intersection of ranges only (no extension).
                  If provided, extends to union of ranges, filling with this value.
        
        Returns:
            Tuple of (self_aligned, other_aligned) with congruent binning
            
        Raises:
            ValueError: If there's no overlapping range between matrices
            
        Example:
            >>> mat1 = Matrix(X=np.arange(0, 100, 2), ...)  # bin width 2, range [0,100]
            >>> mat2 = Matrix(X=np.arange(20, 80, 5), ...)  # bin width 5, range [20,80]
            >>> m1, m2 = mat1.align_with(mat2)  # Intersection
            >>> # Both now have bin width 5 (coarser), range [20,80] (overlap)
            >>> m1.X_index == m2.X_index  # True
            >>> m1.Y_index == m2.Y_index  # True
            >>> 
            >>> m1, m2 = mat1.align_with(mat2, fill=0)  # Union with fill
            >>> # Both now have bin width 5, range [0,100] (full coverage)
        """
        # Translate fill parameter: fill=None → 'intersection', fill=int → 'union'
        mode = 'intersection' if fill is None else 'union'
        
        # Use Index.align_with() to get common indices (handles coarser bin width and lattice alignment)
        common_X = self.X_index.align_with(other.X_index, fill=mode)
        common_Y = self.Y_index.align_with(other.Y_index, fill=mode)
        
        # Rebin both matrices directly to the common indices
        # This automatically handles the lattice alignment
        m1 = self.rebin(axis=0, bins=common_X).rebin(axis=1, bins=common_Y)
        m2 = other.rebin(axis=0, bins=common_X).rebin(axis=1, bins=common_Y)
        
        # For union mode with non-zero fill, rebin fills with 0 by default
        # We need to replace the 0s in extended regions with the fill value
        if fill is not None and fill != 0:
            # Identify regions that were extended (outside original range)
            X_mask_m1 = (common_X.bins < self.X_index.leftmost) | (common_X.bins > self.X_index.rightmost)
            Y_mask_m1 = (common_Y.bins < self.Y_index.leftmost) | (common_Y.bins > self.Y_index.rightmost)
            X_mask_m2 = (common_X.bins < other.X_index.leftmost) | (common_X.bins > other.X_index.rightmost)
            Y_mask_m2 = (common_Y.bins < other.Y_index.leftmost) | (common_Y.bins > other.Y_index.rightmost)
            
            # Fill extended regions for m1
            if np.any(X_mask_m1):
                m1.values[X_mask_m1, :] = fill
            if np.any(Y_mask_m1):
                m1.values[:, Y_mask_m1] = fill
                
            # Fill extended regions for m2
            if np.any(X_mask_m2):
                m2.values[X_mask_m2, :] = fill
            if np.any(Y_mask_m2):
                m2.values[:, Y_mask_m2] = fill
        
        return m1, m2

    @overload
    def rebin(
        self,
        axis: int | str,
        *,
        bins: arraylike | Index | None = None,
        factor: float | None = None,
        binwidth: QuantityLike | None = None,
        numbins: int | None = None,
        preserve: Preserve = "counts",
        inplace: Literal[False] = ...,
    ) -> Matrix: ...

    @overload
    def rebin(
        self,
        axis: int | str,
        *,
        bins: arraylike | Index | None = None,
        factor: float | None = None,
        binwidth: QuantityLike | None = None,
        numbins: int | None = None,
        preserve: Preserve = "counts",
        inplace: Literal[True] = ...,
    ) -> None: ...

    def rebin(
        self,
        axis: int | str,
        *,
        bins: arraylike | Index | None = None,
        factor: float | None = None,
        binwidth: QuantityLike | None = None,
        numbins: int | None = None,
        preserve: Preserve = "counts",
        inplace: bool = False,
    ) -> Self | None:
        """Rebins one axis of the matrix

        Args:
            axis: the axis to rebin.
            bins: The new mids along the axis. Can not be
                given alongside 'factor' or 'binwidth'.
                Note that the edges represented by the bins are assumed
                to be in the same edge type (left or mid) like the original index.
            factor: The factor by which the step size shall be
                changed. Can not be given alongside 'mids'
                or 'binwidth'.
            binwidth: The new bin width. Can not be given
                alongside `factor` or `mids`.
            inplace: Whether to change the axis and values
                inplace or return the rebinned matrix.
                Defaults to `False`.
        Returns:
            The rebinned Matrix if inplace is 'False'.
        Raises:
            ValueError if the axis is not a valid axis.
        """

        axis_: AxisBoth = self.axis_to_int(axis, allow_both=True)
        if axis_ == 2:
            if inplace:
                self.rebin(
                    axis=0,
                    bins=bins,
                    factor=factor,
                    binwidth=binwidth,
                    inplace=True,
                    numbins=numbins,
                )
                self.rebin(
                    axis=1,
                    bins=bins,
                    factor=factor,
                    binwidth=binwidth,
                    inplace=True,
                    numbins=numbins,
                )
                return
            else:
                new = self.rebin(
                    axis=0,
                    bins=bins,
                    factor=factor,
                    binwidth=binwidth,
                    inplace=False,
                    numbins=numbins,
                )
                new.rebin(
                    axis=1,
                    bins=bins,
                    factor=factor,
                    binwidth=binwidth,
                    inplace=True,
                    numbins=numbins,
                )
                return new
        elif axis_ == 0:
            index: Index = self.X_index.handle_rebin_arguments(
                bins=bins, factor=factor, binwidth=binwidth, numbins=numbins
            )
            rebinned = rebin_2D(
                self.X_index, index.bins, self.values, axis=0, preserve=preserve
            )
            if inplace:
                self.values = rebinned
                self.X_index = index
            else:
                return self.clone(X=index, values=rebinned)
        else:
            index: Index = self.Y_index.handle_rebin_arguments(
                bins=bins, factor=factor, binwidth=binwidth, numbins=numbins
            )
            rebinned = rebin_2D(
                self.Y_index, index.bins, self.values, axis=1, preserve=preserve
            )
            if inplace:
                self.values = rebinned
                self.Y_index = index
            else:
                return self.clone(Y=index, values=rebinned)

    @overload
    def rebin_coarsest(self, inplace: Literal[False] = ..., preserve: Preserve = "counts") -> Matrix: ...

    @overload
    def rebin_coarsest(self, inplace: Literal[True] = ..., preserve: Preserve = "counts") -> None: ...

    def rebin_coarsest(
        self, 
        inplace: bool = False, 
        preserve: Preserve = "counts"
    ) -> Self | None:
        """Rebin one axis to match the coarsest (largest) binwidth of the two axes.
        
        This method compares the binwidths of both axes and rebins the finer-resolution
        axis to match the coarser one. If both axes already have the same binwidth,
        no operation is performed.
        
        For uniform binning, the scalar binwidth is compared directly. For non-uniform
        binning, the maximum binwidth is used as representative.
        
        Args:
            inplace: Whether to change the matrix in-place or return a new one.
                Defaults to False.
            preserve: What to preserve during rebinning ('counts' or 'density').
                Defaults to 'counts'.
                
        Returns:
            The rebinned Matrix if inplace is False, None otherwise.
            
        Examples:
            >>> # If X has binwidth 10 keV and Y has binwidth 20 keV,
            >>> # X will be rebinned to 20 keV
            >>> mat_coarse = mat.rebin_coarsest()
        """
        # Get representative binwidths for each axis
        # dX and dY can be scalars (for uniform) or arrays (for non-uniform or from steps())
        dx = self.dX
        dy = self.dY
        
        # Convert to scalar representative values
        # For arrays, use the maximum (coarsest) binwidth as representative
        if isinstance(dx, np.ndarray):
            if len(dx) == 0:
                raise ValueError("Empty X axis")
            dx_repr = float(np.max(dx))
        else:
            dx_repr = float(dx)
            
        if isinstance(dy, np.ndarray):
            if len(dy) == 0:
                raise ValueError("Empty Y axis")
            dy_repr = float(np.max(dy))
        else:
            dy_repr = float(dy)
        
        # If they're already essentially equal, do nothing
        if np.isclose(dx_repr, dy_repr):
            if inplace:
                return None
            else:
                return self.clone()
        
        # Rebin the finer axis to match the coarser one
        if dx_repr < dy_repr:
            # X is finer, rebin it to match Y's binwidth
            return self.rebin(axis=0, binwidth=dy_repr, preserve=preserve, inplace=inplace)
        else:
            # Y is finer, rebin it to match X's binwidth
            return self.rebin(axis=1, binwidth=dx_repr, preserve=preserve, inplace=inplace)

    @overload
    def rebin_identical(self, inplace: Literal[False] = ..., preserve: Preserve = "counts") -> Matrix: ...

    @overload
    def rebin_identical(self, inplace: Literal[True] = ..., preserve: Preserve = "counts") -> None: ...

    def rebin_identical(
        self, 
        inplace: bool = False, 
        preserve: Preserve = "counts"
    ) -> Self | None:
        """Rebin and cut both axes to make them identical.
        
        This method performs two operations:
        1. Rebins to equalize binwidths using the coarsest (largest) binwidth
        2. Cuts to the smallest common range between the two axes
        
        The result is a matrix where both axes have identical bins, making it suitable
        for operations that require symmetric matrices (e.g., diagonal operations).
        
        Since rebinning can only reduce resolution (not increase it), we always
        rebin down to the coarsest binwidth and cut to the smallest range.
        
        Args:
            inplace: Whether to change the matrix in-place or return a new one.
                Defaults to False.
            preserve: What to preserve during rebinning ('counts' or 'density').
                Defaults to 'counts'.
                
        Returns:
            The rebinned and cut Matrix if inplace is False, None otherwise.
            
        Raises:
            ValueError: If the axes cannot be made identical (e.g., incompatible units).
            
        Examples:
            >>> # Create a symmetric matrix suitable for diagonal analysis
            >>> symmetric_mat = mat.rebin_identical()
        """
        # First, equalize the binwidths
        if inplace:
            self.rebin_coarsest(inplace=True, preserve=preserve)
            result = self
        else:
            result = self.rebin_coarsest(inplace=False, preserve=preserve)
        
        # Now both axes have the same binwidth, but may have different ranges
        # Find the common range (intersection)
        x_left = result.X_index.leftmost
        x_right = result.X_index.rightmost
        y_left = result.Y_index.leftmost
        y_right = result.Y_index.rightmost
        
        # Common range is from the rightmost left edge to the leftmost right edge
        common_left = max(x_left, y_left)
        common_right = min(x_right, y_right)
        
        if common_left >= common_right:
            raise ValueError(
                f"No overlap between axes: X=[{x_left}, {x_right}], "
                f"Y=[{y_left}, {y_right}]"
            )
        
        # Get the binwidth (should be the same for both now)
        dx = result.dX
        if isinstance(dx, np.ndarray):
            binwidth = float(np.max(dx))
        else:
            binwidth = float(dx)
        
        # Create new bins for the common range
        # Make sure we start on a bin edge compatible with both axes
        n_bins = int(np.floor((common_right - common_left) / binwidth))
        if n_bins < 1:
            raise ValueError(
                f"Common range [{common_left}, {common_right}] is too small "
                f"for binwidth {binwidth}"
            )
        
        # Use the coarser left edge as starting point and create uniform bins
        new_bins = common_left + np.arange(n_bins + 1) * binwidth
        
        # Rebin both axes to the new common bins
        if inplace:
            result.rebin(axis=0, bins=new_bins[:-1], preserve=preserve, inplace=True)
            result.rebin(axis=1, bins=new_bins[:-1], preserve=preserve, inplace=True)
            return None
        else:
            result = result.rebin(axis=0, bins=new_bins[:-1], preserve=preserve, inplace=False)
            result.rebin(axis=1, bins=new_bins[:-1], preserve=preserve, inplace=True)
            return result

    def index_X(self, x: float) -> int:
        return self.X_index.index_expression(x)

    def index_Y(self, x: float) -> int:
        return self.Y_index.index_expression(x)

    def to_unit(
        self, unit: Unitlike, axis: str | int = "both", inplace: bool = False
    ) -> None | Matrix:
        """Returns a copy with units set to `unit`.

        Args:
            unit: The unit to transform to.
        Returns:
            A copy of the matrix with the unit of `Ex` and
            `Eg` set to `unit`.
        """
        axis: AxisBoth = self.axis_to_int(axis, allow_both=True)
        xindex = self.X_index
        yindex = self.Y_index
        match axis:
            case 0:
                xindex = xindex.to_unit(unit)
            case 1:
                yindex = yindex.to_unit(unit)
            case 2:
                xindex = xindex.to_unit(unit)
                yindex = yindex.to_unit(unit)
        if inplace:
            self.X_index = xindex
            self.Y_index = yindex
        else:
            return self.clone(X=xindex, Y=yindex)

    def to_mid(self, axis: int | str = "both", inplace: bool = False) -> None | Matrix:
        """Returns a copy with the bins set to the midpoints of the bins.

        Args:
            axis: The axis to transform. Defaults to both.
            inplace: Change the matrix inplace or return a copy.
        Returns:
            A copy of the matrix with the bins set to the midpoints of the bins.
        """
        return self.to_edge("mid", axis=axis, inplace=inplace)

    @overload
    def to_left(
        self, axis: int | str = ..., inplace: Literal[False] = ...
    ) -> Matrix: ...

    @overload
    def to_left(self, axis: int | str = ..., inplace: Literal[True] = ...) -> None: ...

    def to_left(self, axis: int | str = "both", inplace: bool = False) -> None | Matrix:
        """Returns a copy with the bins set to the left edges of the bins.

        Args:
            axis: The axis to transform. Defaults to both.
            inplace: Change the matrix inplace or return a copy.
        Returns:
            A copy of the matrix with the bins set to the left edges of the bins.
        """
        return self.to_edge("left", axis=axis, inplace=inplace)

    def to_edge(
        self, edge: Edges, axis: int | str = "both", inplace: bool = False
    ) -> None | Matrix:
        """Returns a copy with the bins set to the left or mid edges of the bins.

        Args:
            edge: The edge to transform to. Either 'left' or 'mid'.
            axis: The axis to transform. Defaults to both.
            inplace: Change the matrix inplace or return a copy.
        Returns:
            A copy of the matrix with the bins set to the left or mid edges of the bins.
        """
        axis_: AxisBoth = self.axis_to_int(axis, allow_both=True)
        xindex = self.X_index
        yindex = self.Y_index
        match axis_:
            case 0:
                xindex = xindex.to_edge(edge)
            case 1:
                yindex = yindex.to_edge(edge)
            case 2:
                xindex = xindex.to_edge(edge)
                yindex = yindex.to_edge(edge)
        if inplace:
            self.X_index = xindex
            self.Y_index = yindex
        else:
            return self.clone(X=xindex, Y=yindex)

    @overload
    def shift_index(
        self, which: Literal[0, 1, "X", "Y"], offset: QuantityLike, inplace: Literal[False] = ...
    ) -> Matrix: ...

    @overload
    def shift_index(
        self, which: Literal[0, 1, "X", "Y"], offset: QuantityLike, inplace: Literal[True] = ...
    ) -> None: ...

    def shift_index(
        self, which: int | str, offset: QuantityLike, inplace: bool = False
    ) -> None | Matrix:
        """Shift the specified index by a constant offset.

        Args:
            which: Which axis to shift. Can be 0/"X" for X-axis or 1/"Y" for Y-axis.
            offset: The offset to shift by. Will be converted to the axis's unit.
            inplace: If True, modify this matrix in place. If False, return a new matrix.

        Returns:
            A new Matrix with shifted index if inplace=False, None otherwise.

        Example:
            >>> mat = Matrix(X=[0, 1, 2], Y=[0, 10, 20], values=np.random.rand(3, 3))
            >>> shifted = mat.shift_index("X", 100)  # Shift X-axis by 100 keV
            >>> shifted.X[0]
            100.0
        """
        axis = self.axis_to_int(which, allow_both=False)
        xindex = self.X_index
        yindex = self.Y_index
        match axis:
            case 0:
                xindex = xindex.shift(offset)
            case 1:
                yindex = yindex.shift(offset)
        if inplace:
            self.X_index = xindex
            self.Y_index = yindex
            return None
        else:
            return self.clone(X=xindex, Y=yindex)

    def set_order(self, order: np._OrderKACF) -> None:
        self.values = self.values.copy(order=order)
        self.X_index = self.X_index.copy(order=order)
        self.Y_index = self.Y_index.copy(order=order)

    @property
    def dX(self) -> float | np.ndarray:
        return self.X_index.steps()

    @property
    def dY(self) -> float | np.ndarray:
        return self.Y_index.steps()

    def from_mask(self, mask: ArrayBool) -> Matrix | Vector:
        """Returns a copy of the matrix with only the rows and columns  where `mask` is True.

        A Vector is returned if the matrix is 1D.
        Can only return a Matrix if the mask selects a continuous 2D region.
        """
        warnings.warn("`from_mask` Not tested!")

        if not np.all(np.diff(np.nonzero(mask)[0]) == 1):
            raise ValueError("Mask must be contiguous")
        values = self.values[mask]

        if values.ndim == 1:
            # Project into vector
            raise NotImplementedError()
        elif values.ndim == 2:
            return self.clone(X=self.X_index[mask], Y=self.Y_index[mask], values=values)
        else:
            raise ValueError("Only supports 1D or 2D arrays")

    @property
    def T(self) -> Self:
        values = self.values.T
        return self.clone(values=values, X=self.Y_index, Y=self.X_index)

    @property
    def _summary(self) -> str:
        s = f"Array type: {self.values.__class__.__name__}\n"
        s += f"X index:\n{self.X_index.summary()}\n"
        s += f"Y index:\n{self.Y_index.summary()}\n"
        if len(self.metadata.misc) > 0:
            s += "Metadata:\n"
            for key, val in self.metadata.misc.items():
                s += f"\t{key}: {val}\n"
        s += f"Total counts: {self.sum():.3g}"
        return s

    def summary(self):
        print(self._summary)

    def _repr_html_(self) -> str:
        """
        Generate HTML representation for Jupyter notebook display.
        Uses table() for metadata display and collapsible() for array values.
        Composes X_index and Y_index _repr_html_ output.
        """
        # Get array type and total counts
        array_info = [
            ("Array type", self.values.__class__.__name__),
            ("Total counts", f"{self.sum():.3g}"),
        ]

        # Create metadata table if available
        metadata_html = ""
        if (
            hasattr(self, "metadata")
            and hasattr(self.metadata, "misc")
            and len(self.metadata.misc) > 0
        ):
            metadata_items = [
                (key, str(val)) for key, val in self.metadata.misc.items()
            ]
            metadata_html = f"""
            <div class="metadata-section">
                <h4>Metadata:</h4>
                {table(metadata_items, color="#f0f0f0")}
            </div>
            """

        # Main HTML structure
        html = f"""
        <div class="array-container" style="margin: 10px 0;">
            <div class="array-info">
                {table(array_info, color="#e6f7ff")}
            </div>
            <div class="indices-section" style="margin-top: 10px;">
                {collapse(self.X_index._repr_html_(), "X index")}
                <div style="margin-top: 10px;">
                    {collapse(self.Y_index._repr_html_(), "Y index")}
                </div>
            </div>
            
            {metadata_html}
            
            <div class="values-section" style="margin-top: 10px;">
                {collapsible(self.values, "Array Values")}
            </div>
        </div>
        """

        return html

    @overload
    def sum(
        self, axis: Literal["both"] = ..., out: np.ndarray | None = ...
    ) -> float: ...

    @overload
    def sum(self, axis: Literal[2] = ..., out: np.ndarray | None = ...) -> float: ...

    @overload
    def sum(
        self, axis: Literal[0, 1] | str = ..., out: np.ndarray | None = ...
    ) -> Vector: ...

    def sum(
        self, axis: int | str = "both", out: np.ndarray | None = None
    ) -> Vector | float:
        if out is not None:
            raise NotImplementedError("ops")
        axis_: AxisBoth = self.axis_to_int(axis, allow_both=True)
        if axis_ == 2:
            return self.values.sum()
        values = self.values.sum(axis=axis_)
        index = self.X_index if axis_ else self.Y_index
        return self.meta_into_vector(index=index, values=values)

    def __str__(self) -> str:
        summary = self._summary
        summary += "\nValues:\n"
        return summary + str(self.values)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.shape})[{self.device}]{self.title} at {hex(id(self))}"

    @property
    def X(self) -> np.ndarray:
        return self.X_index.bins

    @property
    def Y(self) -> np.ndarray:
        return self.Y_index.bins

    def clone(
        self,
        X: Index | None = None,
        Y: Index | None = None,
        values: np.ndarray | None = None,
        metadata: MatrixMetadata | None = None,
        copy: bool = False,
        dtype: DTypeLike | None = None,
        **kwargs,
    ) -> Self:
        """Copies the object.

        Any keyword argument will override the equivalent
        attribute in the copy. For example, matrix.clone(Eg=[1,2,3])
        tries to set the gamma energy to [1,2,3].

        kwargs: Any keyword argument is overwritten
            in the copy.
        Returns:
            The copy
        """
        X = X if X is not None else self.X_index
        Y = Y if Y is not None else self.Y_index
        values = values if values is not None else self.values
        metadata = metadata if metadata is not None else self.metadata
        metadata = metadata.update(**kwargs)
        if dtype is None:
            dtype = values.dtype
        return type(self)(
            X=X, Y=Y, values=values, metadata=metadata, copy=copy, dtype=dtype
        )

    def _wrap_locator_result(
        self,
        *,
        axis: int | None,
        key: tuple[slice | int | None, slice | int | None],
        values: np.ndarray | float,
        result: Matrix | Vector | float,
    ) -> Matrix | Vector | float:
        """Hook for subclasses to post-process locator results."""
        return result

    def _plot_bilog(
        self,
        *,
        ax: Axes,
        X: np.ndarray,
        Y: np.ndarray,
        values: np.ndarray,
        mask: np.ndarray,
        pos_vals: np.ndarray,
        neg_vals: np.ndarray,
        add_cbar: bool,
        cbarkwargs: dict[str, Any] | None,
        bilog_kwargs: dict[str, Any],
        all_bad: bool,
    ) -> tuple[Axes, tuple[tuple[QuadMesh, QuadMesh], tuple[Colorbar | None, Colorbar | None]]]:
        pos_min = bilog_kwargs.get("pos_min", pos_vals.min())
        pos_max = bilog_kwargs.get("pos_max", pos_vals.max())
        neg_min = bilog_kwargs.get("neg_min", neg_vals.min())
        neg_max = bilog_kwargs.get("neg_max", neg_vals.max())

        pos_norm = LogNorm(vmin=pos_min, vmax=pos_max)
        neg_norm = LogNorm(vmin=neg_min, vmax=neg_max)

        pos_cmap_name = bilog_kwargs.get("cmap_pos") or bilog_kwargs.get("cmap_positive")
        neg_cmap_name = bilog_kwargs.get("cmap_neg") or bilog_kwargs.get("cmap_negative")
        pos_cmap = cm.get_cmap(pos_cmap_name) if pos_cmap_name else cm.get_cmap("viridis")
        neg_cmap = cm.get_cmap(neg_cmap_name) if neg_cmap_name else cm.get_cmap("magma")

        neg_data = np.ma.array(-values, mask=(values >= 0) | mask)
        pos_data = np.ma.array(values, mask=(values <= 0) | mask)

        neg_mesh = ax.pcolormesh(
            Y,
            X,
            neg_data,
            cmap=neg_cmap,
            norm=neg_norm,
        )
        pos_mesh = ax.pcolormesh(
            Y,
            X,
            pos_data,
            cmap=pos_cmap,
            norm=pos_norm,
        )

        cbar_pair: tuple[Colorbar | None, Colorbar | None]
        if add_cbar and not all_bad:
            cbarkwargs = {} if cbarkwargs is None else dict(cbarkwargs)
            fig = ax.figure

            neg_cbar_kwargs = dict(bilog_kwargs.get("colorbar_neg", {}))
            pos_cbar_kwargs = dict(bilog_kwargs.get("colorbar_pos", {}))

            location_neg = neg_cbar_kwargs.pop("location", "left")
            location_pos = pos_cbar_kwargs.pop("location", "right")
            pad_neg = neg_cbar_kwargs.pop("pad", 0.08)
            pad_pos = pos_cbar_kwargs.pop("pad", 0.08)

            cb_neg = fig.colorbar(
                neg_mesh,
                ax=ax,
                location=location_neg,
                pad=pad_neg,
                **neg_cbar_kwargs,
            )
            cb_pos = fig.colorbar(
                pos_mesh,
                ax=ax,
                location=location_pos,
                pad=pad_pos,
                **pos_cbar_kwargs,
            )

            base_neg = bilog_kwargs.get("base_neg", bilog_kwargs.get("base", 10))
            num_neg = bilog_kwargs.get("numticks_neg", bilog_kwargs.get("numticks", None))
            neg_locator = ticker.LogLocator(base=base_neg, numticks=num_neg)
            cb_neg.ax.yaxis.set_major_locator(neg_locator)
            neg_formatter_raw = ticker.LogFormatter(base=base_neg)
            cb_neg.ax.yaxis.set_major_formatter(
                ticker.FuncFormatter(lambda v, _: f"-{neg_formatter_raw(v):s}")
            )

            base_pos = bilog_kwargs.get("base_pos", bilog_kwargs.get("base", 10))
            num_pos = bilog_kwargs.get("numticks_pos", bilog_kwargs.get("numticks", None))
            pos_locator = ticker.LogLocator(base=base_pos, numticks=num_pos)
            cb_pos.ax.yaxis.set_major_locator(pos_locator)
            cb_pos.ax.yaxis.set_major_formatter(ticker.LogFormatter(base=base_pos))

            cbar_pair = (cb_neg, cb_pos)
        else:
            cbar_pair = (None, None)

        return ax, ((neg_mesh, pos_mesh), cbar_pair)

    def is_compatible_with(self, other: AbstractArray | Index) -> bool:
        return self.is_compatible_with_X(other) or self.is_compatible_with_Y(other)

    def is_compatible_with_X(self, other: AbstractArray | Index) -> bool:
        match other:
            case Index():
                return self.X_index.is_compatible_with(other)
            case Matrix():
                return self.X_index.is_compatible_with(other.X_index)
            case Vector():
                return self.X_index.is_compatible_with(other._index)
            case _:
                return False

    def is_compatible_with_Y(self, other: AbstractArray | Index) -> bool:
        match other:
            case Index():
                return self.Y_index.is_compatible_with(other)
            case Matrix():
                return self.Y_index.is_compatible_with(other.Y_index)
            case Vector():
                return self.Y_index.is_compatible_with(other._index)
            case _:
                return False

    def normalize(self, axis: str | Literal[0, 1, 2], inplace=False) -> Self | None:
        axis_: AxisBoth = self.axis_to_int(axis, allow_both=True)
        if not inplace:
            match axis_:
                case 0:
                    s = self.values.sum(axis=0)
                    s[s == 0] = 1
                    values = self.values / s[np.newaxis, :]
                case 1:
                    s = self.values.sum(axis=1)
                    s[s == 0] = 1
                    values = self.values / s[:, np.newaxis]
                case 2:
                    s = sum(self.values)
                    values = self.values / s
            return self.clone(values=values)
        else:
            match axis_:
                case 0:
                    s = self.values.sum(axis=0)
                    s[s == 0] = 1
                    self.values /= s[np.newaxis, :]
                case 1:
                    s = self.values.sum(axis=1)
                    s[s == 0] = 1
                    self.values /= s[:, np.newaxis]
                case 2:
                    s = sum(self.values)
                    self.values /= s

    @overload
    def plot(
        self,
        ax: Axes,
        *,
        scale: str | None = None,
        vmin: float | None = None,
        vmax: float | None = None,
        add_cbar: Literal[True] = ...,
        cbarkwargs: dict[str, Any] | None = None,
        bad_map: Callable[[Matrix], ArrayBool | bool] = lambda x: False,
        color_by: ColorBy = ...,
        **kwargs,
    ) -> tuple[Axes, tuple[QuadMesh, Colorbar]]: ...

    @overload
    def plot(
        self,
        ax: Axes,
        *,
        scale: str | None = None,
        vmin: float | None = None,
        vmax: float | None = None,
        add_cbar: Literal[False] = ...,
        cbarkwargs: dict[str, Any] | None = None,
        bad_map: Callable[[Matrix], ArrayBool | bool] = lambda x: False,
        color_by: ColorBy = ...,
        **kwargs,
    ) -> tuple[Axes, tuple[QuadMesh, None]]: ...

    def plot(
        self,
        ax: Axes | None = None,
        *,
        scale: str | None = None,
        vmin: float | None = None,
        vmax: float | None = None,
        add_cbar: bool = True,
        cbarkwargs: dict[str, Any] | None = None,
        bad_map: Callable[[Matrix], ArrayBool | bool] = lambda x: False,
        color_by: ColorBy = "IQR",
        **kwargs,
    ) -> tuple[Axes, tuple[QuadMesh, Colorbar | None]]:
        """Plots the matrix with the energy along the axis

        Args:
            ax: A matplotlib axis to plot onto
            title: Defaults to the current matrix state
            scale: Scale along the z-axis. Can be either "log"
                or "linear". Defaults to logarithmic
                if number of counts > 1000
            vmin: Minimum value for coloring in scaling
            vmax Maximum value for coloring in scaling
            add_cbar: Whether to add a colorbar. Defaults to True.
            **kwargs: Additional kwargs to plot command.

        Returns:
            The ax used for plotting

        Raises:
            ValueError: If scale is unsupported
        """
        ax = make_ax(ax)

        # In case `values` is on the gpu
        values = np.asarray(self.values)
        if np.all(~np.isfinite(values)):
            raise ValueError("Matrix contains only NaN or infinite values")

        # Simple heuristic to determine scale
        if scale is None:
            if np.any(values < 0):
                if np.sum(abs(values)) > 1e3:
                    scale = "symlog"
                else:
                    scale = "linear"
            else:
                if np.sum(values) > 1e3:
                    scale = "log"
                else:
                    scale = "linear"

        mask = np.isnan(values) | (values == 0) | bad_map(self)
        masked = np.ma.array(values, mask=mask)
        X_mesh, Y_mesh = self._plot_mesh()

        # Try methods to determine colorscale to prevent
        # outliers from skewing the data
        if isinstance(color_by, str):
            color_args = []
        else:
            color_args = color_by[1:] if len(color_by) > 1 else []
            color_by = color_by[0]

        # If there are no good values, dont try to set vmin and vmax
        all_bad = np.all(mask)
        if all_bad:
            color_by = "values"

        if color_by == "IQR":
            factor = color_args[0] if color_args else 1.5
            vmin_IQR, vmax_IQR = IQR_range(values[~mask].ravel(), factor)
            if scale == "log":
                vmin_IQR = max(vmin_IQR, 1e-1)
            vmin = vmin if vmin is not None else vmin_IQR
            vmax = vmax if vmax is not None else vmax_IQR
        elif color_by == "z-score":
            z = np.zeros_like(values)
            x = values[~mask]
            z[~mask] = robust_z_score(x)
            zmin = color_args[0] if color_args else -2
            zmax = color_args[1] if len(color_args) > 1 else 2
            vmin_z = max(zmin, z.min())
            vmax_z = min(zmax, z.max())

            vmin_z = robust_z_score_i(vmin_z, x)
            vmax_z = robust_z_score_i(vmax_z, x)
            if scale == "log":
                vmin_z = max(vmin_z, 1e-1)

            vmin = vmin if vmin is not None else vmin_z
            vmax = vmax if vmax is not None else vmax_z
        elif color_by == "percentile":
            lower: float = color_args[0] if color_args else 0.5
            upper: float = color_args[1] if len(color_args) > 1 else 99.5
            vmin_p = np.percentile(values[~mask], lower)
            vmax_p = np.percentile(values[~mask], upper)

            vmin = vmin if vmin is not None else vmin_p
            vmax = vmax if vmax is not None else vmax_p
        elif color_by == "values":
            pass
        else:
            raise ValueError(
                f"Unknown color_by: {color_by}. Supported are"
                " 'IQR', 'z-score', 'percentile' and 'values'"
            )

        bilog_ticks: list[float] | None = None

        if scale == "log":
            if vmin is not None and vmin <= 0:
                raise ValueError("`vmin` must be positive for log-scale")
            if vmin is None:
                _max = np.log10(self.max())
                _min = np.log10(values[values > 0].min())
                if _max - _min > 10:
                    vmin = 10 ** (int(_max - 6))
            norm = LogNorm(vmin=vmin, vmax=vmax)
        elif scale == "symlog":
            lintresh = kwargs.pop("lintresh", 1e-1)
            linscale = kwargs.pop("linscale", 1)
            norm = SymLogNorm(lintresh, linscale, vmin, vmax)
        elif scale == "bilog":
            pos_vals = values[values > 0]
            neg_vals = -values[values < 0]
            if pos_vals.size == 0 or neg_vals.size == 0:
                warnings.warn(
                    "Bilog scale requires both positive and negative values; "
                    "falling back to symlog.",
                    RuntimeWarning,
                )
                lintresh = kwargs.pop("lintresh", 1e-1)
                linscale = kwargs.pop("linscale", 1)
                norm = SymLogNorm(lintresh, linscale, vmin, vmax)
            else:
                bilog_kwargs = kwargs.pop("bilog", {})
                return self._plot_bilog(
                    ax=ax,
                    X=X_mesh,
                    Y=Y_mesh,
                    values=values,
                    mask=mask,
                    pos_vals=pos_vals,
                    neg_vals=neg_vals,
                    add_cbar=add_cbar,
                    cbarkwargs=cbarkwargs,
                    bilog_kwargs=bilog_kwargs,
                    all_bad=all_bad,
                )
        elif scale == "linear":
            norm = Normalize(vmin=vmin, vmax=vmax)
        else:
            norm = scale(vmin=vmin, vmax=vmax)
        norm = kwargs.pop("norm", norm)
        # Set entries of 0 to white
        current_cmap = copy.copy(cm.get_cmap())
        current_cmap.set_bad(color="white")
        cmap = plt.get_cmap(kwargs.pop("cmap", current_cmap))
        mesh = ax.pcolormesh(Y_mesh, X_mesh, masked, cmap=cmap, norm=norm, **kwargs)

        # TODO: Let the index handle the ticks?
        if self.Y_index.is_mid():
            ax.xaxis.set_major_locator(MeshLocator(self.Y))
            ax.tick_params(axis="x")
        if self.X_index.is_mid():
            ax.yaxis.set_major_locator(MeshLocator(self.X))
        if hasattr(self.X_index, "scale"):
            if self.X_index.scale is not None:
                ax.set_yscale(self.X_index.scale)
        if hasattr(self.Y_index, "scale"):
            if self.Y_index.scale is not None:
                ax.set_xscale(self.Y_index.scale)

        maybe_set(ax, title=self.name)
        maybe_set(ax, ylabel=self.get_xlabel())
        maybe_set(ax, xlabel=self.get_ylabel())

        # show z-value in status bar
        # https://stackoverflow.com/questions/42577204/show-z-value-at-mouse-pointer-position-in-status-line-with-matplotlibs-pcolorme
        def format_coord(x, y):
            xarr = Y
            yarr = X
            if (x > xarr[0]) & (x <= xarr[-1]) & (y > yarr[0]) & (y <= yarr[-1]):
                col = np.searchsorted(xarr, x) - 1
                row = np.searchsorted(yarr, y) - 1
                z = masked[row, col]
                return f"{self.yalias}={x:1.0f}{self.X_index.unit:~}, {self.xalias}={y:1.0f}{self.Y_index.unit:~}, z={z:1.2E}"
                # return f'x={x:1.0f}, y={y:1.0f}, z={z:1.3f}   [{row},{col}]'
            else:
                return f"x={x:1.0f}, y={y:1.0f}"

        # TODO: Takes waaaay to much CPU
        #ax.format_coord = format_coord

        cbar: Colorbar | tuple[cm.ScalarMappable, Normalize] = None
        if add_cbar and not all_bad:
            if cbarkwargs is None:
                cbarkwargs = {}
            if bilog_ticks is not None:
                cbarkwargs.setdefault("ticks", bilog_ticks)
            kwargs = dict(ax=ax) | cbarkwargs
            cbar = AnnotatedColorbar(mesh, **kwargs)
        else:
            cbar = (cmap, norm)

        return ax, (mesh, cbar)

    def plot_3d(
        self,
        ax: Axes | None = None,
        vmin: float | None = None,
        vmax: float | None = None,
        scale="linear",
        add_cbar: bool = True,
        cbarkwargs: dict | None = None,
        **kwargs,
    ):
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
        fig: Figure = ax.figure  # type: ignore

        if scale == "log":
            if vmin is not None and vmin <= 0:
                raise ValueError("`vmin` must be positive for log-scale")
            if vmin is None:
                _max = np.log10(self.max())
                _min = np.log10(self.values[self.values > 0].min())
                if _max - _min > 10:
                    vmin = 10 ** (int(_max - 6))
            norm = LogNorm(vmin=vmin, vmax=vmax)
        elif scale == "symlog":
            lintresh = kwargs.pop("lintresh", 1e-1)
            linscale = kwargs.pop("linscale", 1)
            norm = SymLogNorm(lintresh, linscale, vmin, vmax)
        elif scale == "linear":
            norm = Normalize(vmin=vmin, vmax=vmax)
        else:
            raise ValueError("Unsupported zscale ", scale)
        norm = kwargs.pop("norm", norm)
        cmap = cmaps.get_cmap(kwargs.pop("cmap", cm.get_cmap()))
        X, Y = self.X, self.Y  # self._plot_mesh()
        dx = X[1] - X[0]
        dy = Y[1] - Y[0]
        Y, X = np.meshgrid(Y, X)
        Z = self.values
        colors = cmap(norm(Z.flatten()))

        # Plotting a 3D histogram with color based on height (log normalized)
        mesh = ax.bar3d(
            Y.flatten(),
            X.flatten(),
            np.zeros_like(Z).flatten(),
            dy,
            dx,
            Z.flatten(),
            shade=True,
            color=colors,
        )

        if False:
            if self.Y_index.is_mid():
                ax.xaxis.set_major_locator(MeshLocator(self.Y))
                ax.tick_params(axis="x", rotation=40)
            if self.X_index.is_mid():
                ax.yaxis.set_major_locator(MeshLocator(self.X))

        cbar: Colorbar | None = None
        if add_cbar:
            if cbarkwargs is None:
                cbarkwargs = {}
            cbarkwargs.setdefault("fraction", 0.03)
            cbarkwargs.setdefault("pad", 0.04)

            mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
            mappable.set_array(Z)
            if vmin is not None and vmax is not None:
                cbar = fig.colorbar(mappable, ax=ax, extend="both", **cbarkwargs)
            elif vmin is not None:
                cbar = fig.colorbar(mappable, ax=ax, extend="min", **cbarkwargs)
            elif vmax is not None:
                cbar = fig.colorbar(mappable, ax=ax, extend="max", **cbarkwargs)
            else:
                cbar = fig.colorbar(mappable, ax=ax, **cbarkwargs)

            maybe_set(cbar.ax, ylabel=self.get_vlabel())

        maybe_set(ax, title=self.name)
        maybe_set(ax, ylabel=self.get_xlabel())
        maybe_set(ax, xlabel=self.get_ylabel())

        return ax, (mesh, cbar)

    def get_ylabel(self) -> str:
        unit = f"{self.Y_index.unit:~L}"
        if unit:
            return self.ylabel + f" [${unit}$]"
        return self.ylabel

    def get_xlabel(self) -> str:
        unit = f"{self.X_index.unit:~L}"
        if unit:
            return self.xlabel + f" [${unit}$]"
        return self.xlabel

    def get_vlabel(self) -> str:
        """Get formatted colorbar label with unit"""
        unit = f"{self.vunit:~L}"
        if unit:
            return self.vlabel + f" [${unit}$]"
        return self.vlabel

    def set_xlabel(self, label: str) -> Self:
        self.X_index = self.X_index.update_metadata(label=label)
        return self

    def set_ylabel(self, label: str) -> Self:
        self.Y_index = self.Y_index.update_metadata(label=label)
        return self

    def _plot_mesh(self) -> tuple[np.ndarray, np.ndarray]:
        if self.X_index.is_mid():
            X = self.X_index.to_left().ticks()
        else:
            X = self.X_index.ticks()

        if self.Y_index.is_mid():
            Y = self.Y_index.to_left().ticks()
        else:
            Y = self.Y_index.ticks()
        return X, Y

    def meta_into_vector(self, index: np.ndarray | Index, values: np.ndarray) -> Vector:
        # TODO Unclear if name should be transferred
        return Vector(X=index, values=values, vlabel=self.vlabel, valias=self.valias)
        # name=self.name)

    @overload
    def axis_to_int(
        self, axis: int | str, allow_both: Literal[False] = ...
    ) -> AxisEither: ...

    @overload
    def axis_to_int(
        self, axis: int | str, allow_both: Literal[True] = ...
    ) -> AxisBoth: ...

    def axis_to_int(self, axis: int | str, allow_both: bool = False) -> Axis:
        match axis:
            case 0 | 1:
                return axis
            case 2:
                if allow_both:
                    return axis
                raise ValueError("Cannot use axis 2 for matrix")
            case "x" | "X" | self.xalias:
                return 0
            case "y" | "Y" | self.yalias:
                return 1
            case "both":
                if allow_both:
                    return 2
                raise ValueError("Cannot use axis 2 for matrix")
            case _:
                raise ValueError(f"Unknown axis {axis}")

    @property
    def xalias(self) -> str:
        return self.X_index.alias

    @property
    def yalias(self) -> str:
        return self.Y_index.alias

    @property
    def xlabel(self) -> str:
        return self.X_index.label

    @xlabel.setter
    def xlabel(self, value: str) -> None:
        self.X_index = self.X_index.update_metadata(label=value)

    @property
    def ylabel(self) -> str:
        return self.Y_index.label

    @ylabel.setter
    def ylabel(self, value: str) -> None:
        self.Y_index = self.Y_index.update_metadata(label=value)

    @overload
    def __matmul__(self, other: Matrix) -> Self: ...

    @overload
    def __matmul__(self, other: Vector) -> Vector: ...

    @overload
    def __matmul__(self, other: np.ndarray) -> np.ndarray: ...

    def __matmul__(
        self, other: Matrix | Vector | np.ndarray
    ) -> Self | Vector | np.ndarray:
        match other:
            case Matrix():
                if self.shape[1] != other.shape[0]:
                    raise ValueError(f"Shape mismatch {self.shape} @ {other.shape}")
                if not self.is_compatible_with_Y(other.X_index):
                    raise ValueError(
                        f"Y index mismatch\n({self.Y_index})\n @\n({other.X_index})"
                    )
                arr = self.values @ other.values
                return type(self)(
                    X=self.X_index, Y=other.Y_index, values=arr, dtype=arr.dtype
                )
            case Vector():
                if self.shape[1] != other.shape[0]:
                    raise ValueError(f"Shape mismatch {self.shape} @ {other.shape}")
                if not self.is_compatible_with_Y(other._index):
                    raise ValueError(
                        f"Y index mismatch between:\n{self.Y_index.summary()}\n\nand:\n{other._index.summary()}"
                    )
                return self.meta_into_vector(self.X_index, self.values @ other.values)
            case np.ndarray():
                return self.values @ other
            case _:
                if not hasattr(other, "__rmatmul__"):
                    raise TypeError(
                        f"Cannot multiply {self.__class__.__name__} with {other.__class__.__name__}"
                    )
                return other.__rmatmul__(self)

    def last_nonzero(self, i: int, eps: float = 0) -> int:
        """Returns the index of the last non-zero element"""
        j = self.shape[1]
        while (j := j - 1) >= 0:
            if abs(self[i, j]) > eps:
                break
        return j

    def last_nonzeros(self, eps: float = 0.0) -> np.ndarray:
        return last_nonzeros(self.values, eps=eps)

    def to_xarray(self) -> "xr.DataArray":
        return to_xarray_matrix(self)

    def to_root(self, identifier: str | None = None):
        return to_root_matrix(self, identifier)

    def set_xalias(self, alias: str, label: str | None = None) -> Self:
        if label is None:
            label = self.xlabel
        index = self.X_index.update(alias=alias, label=label)
        return self.clone(X=index)

    def set_yalias(self, alias: str, label: str | None = None) -> Self:
        if label is None:
            label = self.ylabel
        index = self.Y_index.update(alias=alias, label=label)
        return self.clone(Y=index)

    @classmethod
    def from_matrix(
        cls,
        other: Matrix,
        X: Index | None = None,
        Y: Index | None = None,
        values: np.ndarray | None = None,
        **kwargs,
    ) -> Self:
        if X is None:
            X = other.X_index
        if Y is None:
            Y = other.Y_index
        if values is None:
            values = other.values
        return cls(X=X, Y=Y, values=values, **kwargs)

    @classmethod
    def from_vector(
        cls,
        other: Vector,
        values: np.ndarray,
        X: Index | None = None,
        **kwargs,
    ) -> Self:
        """ Square matrix with indices like vector """
        if X is None:
            X = other.X_index
        return cls(X=X, Y=X, values=values, **kwargs)

    def _coerce_other(self, other: AbstractArray | np.ndarray | float) -> np.ndarray | float:
        if isinstance(other, AbstractArray):
            if isinstance(other, Vector):
                v = np.asarray(other.values)
                # Column-wise (broadcast across rows) if vector matches Y
                if self.is_compatible_with_Y(other._index):
                    if self.shape[1] != v.shape[0]:
                        raise ValueError(f"Length mismatch: {self.shape[1]} vs {v.shape[0]}")
                    return v[np.newaxis, :]  # (1, n)
                # Row-wise (broadcast across columns) if vector matches X
                if self.is_compatible_with_X(other._index):
                    if self.shape[0] != v.shape[0]:
                        raise ValueError(f"Length mismatch: {self.shape[0]} vs {v.shape[0]}")
                    return v[:, np.newaxis]  # (m, 1)
                raise ValueError("Vector index not compatible with Matrix X or Y.")

            if isinstance(other, Matrix):
                A = self.shape
                B = other.shape
                # NumPy-style broadcastability
                m_ok = (A[0] == B[0]) or (A[0] == 1) or (B[0] == 1)
                n_ok = (A[1] == B[1]) or (A[1] == 1) or (B[1] == 1)
                if not (m_ok and n_ok):
                    raise ValueError(f"Shapes not broadcastable: {A} and {B}")

                # Index compat for non-broadcast axes only
                if A[0] != 1 and B[0] != 1:
                    if not self.is_compatible_with_X(other.X_index):
                        raise ValueError("Incompatible X indices for Matrix op.")
                if A[1] != 1 and B[1] != 1:
                    if not self.is_compatible_with_Y(other.Y_index):
                        raise ValueError("Incompatible Y indices for Matrix op.")
                return other.values

            # Any other AbstractArray: fall through to its values and let NumPy decide
            return other.values

        # ndarray/float: allow normal NumPy broadcasting
        return other


if xarray_available():
    import xarray as xr

    def to_xarray_matrix(mat) -> xr.DataArray:  # type: ignore
        xalias = mat.xalias if mat.xalias else 'y'
        yalias = mat.yalias if mat.yalias else 'x'
        return xr.DataArray(
            mat.values, coords=[mat.X, mat.Y], dims=[xalias, yalias]
        )

else:

    def to_xarray_matrix(mat) -> Never:
        raise NotImplementedError("xarray not available")


if ROOT_imported():
    from ROOT import TH2D  # type: ignore

    def to_root_matrix(mat, identifier: str | None = None) -> TH2D:  # type: ignore
        mat = mat.to_left()
        if identifier is None:
            identifier = mat.name
        hist = TH2D(
            identifier, identifier, len(mat.Y) - 1, mat.Y, len(mat.X) - 1, mat.X
        )
        for i in range(len(mat.Y)):
            for j in range(len(mat.X)):
                hist.SetBinContent(i + 1, j + 1, mat[j, i])
        hist.GetXaxis().SetTitle(mat.ylabel)
        hist.GetYaxis().SetTitle(mat.xlabel)
        hist.SetTitle(mat.name)
        return hist

else:

    def to_root_matrix(mat, *args, **kwargs) -> Never:
        raise NotImplementedError("ROOT not imported")

if jax_available():
    import jax

    # Make it compatible as a pytree
    def flatten(obj) -> tuple[tuple[np.ndarray], dict[str, Any]]:
        aux = {
            "X_index": obj.X_index,
            "Y_index": obj.Y_index,
            "metadata": obj.metadata,
            "class": obj.__class__,
        }
        return (obj.values,), aux

    def unflatten(aux_data: dict[str, Any], children: tuple[np.ndarray]) -> Matrix:
        return aux_data["class"](
            X=aux_data["X_index"],
            Y=aux_data["Y_index"],
            values=children[0],
            metadata=aux_data["metadata"],
        )

    jax.tree_util.register_pytree_node(Matrix, flatten, unflatten)

class IndexLocator:
    def __init__(self, matrix: Matrix):
        self.mat = matrix

    @overload
    def __getitem__(self, key: tuple[slice, slice]) -> Matrix: ...

    @overload
    def __getitem__(self, key: tuple[slice, int | slice]) -> Vector: ...

    @overload
    def __getitem__(self, key: tuple[int | slice, slice]) -> Vector: ...

    @overload
    def __getitem__(self, key: tuple[int, int]) -> float: ...

    def __getitem__(self, key):
        if not isinstance(key, np.ndarray):
            values = self.mat.values.__getitem__(key)
        match key:
            case np.ndarray():
                return self.linear_index(key)
            case slice() as x, slice() as y:
                X = self.mat.X_index[x]
                Y = self.mat.Y_index[y]
                return self.mat.clone(values=values, X=X, Y=Y)
            case slice() as x, y:
                X = self.mat.X_index[x]
                return self.mat.meta_into_vector(X, values)
            case x, slice() as y:
                Y = self.mat.Y_index[y]
                return self.mat.meta_into_vector(Y, values)
            case x, y:
                return values

    def linear_index(self, indices) -> Matrix:
        values = np.where(indices, self.mat.values, 0)
        return self.mat.clone(values=values)


class ValueLocator:
    def __init__(self, matrix: Matrix, strict: bool = True):
        self.mat = matrix
        self.strict = strict

    @overload
    def __getitem__(self, key: tuple[slice, slice]) -> Matrix: ...

    @overload
    def __getitem__(self, key: tuple[slice, int | float | slice]) -> Vector: ...

    @overload
    def __getitem__(self, key: tuple[int | float | slice, slice]) -> Vector: ...

    @overload
    def __getitem__(self, key: tuple[int | float, int | float]) -> float: ...

    def __getitem__(self, key):
        match key:
            case slice() as x, slice() as y:
                sx: slice = self.mat.X_index.index_slice(x, strict=self.strict)
                sy: slice = self.mat.Y_index.index_slice(y, strict=self.strict)
                xindex = self.mat.X_index[sx]
                yindex = self.mat.Y_index[sy]
                values = self.mat.values.__getitem__((sx, sy))
                result = self.mat.clone(values=values, X=xindex, Y=yindex)
                return self.mat._wrap_locator_result(
                    axis=None,
                    key=(sx, sy),
                    values=values,
                    result=result,
                )
            case slice() as x, y:
                sx: slice = self.mat.X_index.index_slice(x, strict=self.strict)
                xindex = self.mat.X_index[sx]
                j: int = self.mat.Y_index.index_expression(y, strict=self.strict)
                values = self.mat.values.__getitem__((sx, j))
                result = self.mat.meta_into_vector(xindex, values)
                return self.mat._wrap_locator_result(
                    axis=0,
                    key=(sx, j),
                    values=values,
                    result=result,
                )
            case x, slice() as y:
                sy: slice = self.mat.Y_index.index_slice(y, strict=self.strict)
                yindex = self.mat.Y_index[sy]
                i: int = self.mat.X_index.index_expression(x, strict=self.strict)
                values = self.mat.values.__getitem__((i, sy))
                result = self.mat.meta_into_vector(yindex, values)
                return self.mat._wrap_locator_result(
                    axis=1,
                    key=(i, sy),
                    values=values,
                    result=result,
                )
            case x, y:
                i: int = self.mat.X_index.index_expression(x, strict=self.strict)
                j: int = self.mat.Y_index.index_expression(y, strict=self.strict)
                values = self.mat.values.__getitem__((i, j))
                return self.mat._wrap_locator_result(
                    axis=None,
                    key=(i, j),
                    values=values,
                    result=values,
                )

    def __setitem__(self, key, val):
        match key:
            case slice() as x, slice() as y:
                sx: slice = self.mat.X_index.index_slice(x, strict=self.strict)
                sy: slice = self.mat.Y_index.index_slice(y, strict=self.strict)
                self.mat.values.__setitem__((sx, sy), val)
            case slice() as x, y:
                sx: slice = self.mat.X_index.index_slice(x, strict=self.strict)
                j: int = self.mat.Y_index.index_expression(y, strict=self.strict)
                self.mat.values.__setitem__((sx, j), val)
            case x, slice() as y:
                sy: slice = self.mat.Y_index.index_slice(y, strict=self.strict)
                i: int = self.mat.X_index.index_expression(x, strict=self.strict)
                self.mat.values.__setitem__((i, sy), val)
            case x, y:
                i: int = self.mat.X_index.index_expression(x, strict=self.strict)
                j: int = self.mat.Y_index.index_expression(y, strict=self.strict)
                self.mat.values.__setitem__((i, j), val)

    def __call__(
        self, **mappings: dict[str, slice | int | float | str]
    ) -> Matrix | Vector | None:
        """
        Return a slice of the array using its aliases.

        Args:
            mappings (dict): Map from alias to slice, such as `(Ex=slice(0, 10), Eg='5MeV')`.

        Returns:
            Matrix | Vector | None: Returns a Matrix or Vector object based on the provided mappings.

        Raises:
            ValueError: If no mappings are provided, more than two mappings are provided, or if the provided indices are invalid.
        """
        if len(mappings) == 0:
            raise ValueError("No mappings provided")
        if len(mappings) > 2:
            raise ValueError("Only two mappings allowed.")
        if len(mappings) == 1:
            index, value = mappings.popitem()
            if self.is_x_index(index):
                return self.__getitem__((value, slice(None)))
            elif self.is_y_index(index):
                return self.__getitem__((slice(None), value))
            else:
                raise ValueError(f"Invalid index: {index}")
        else:
            index0, value0 = mappings.popitem()
            index1, value1 = mappings.popitem()
            fn = (self.is_x_index, self.is_y_index)
            match [f(index0) for f in fn], [f(index1) for f in fn]:
                case [True, False], [False, True]:
                    return self.__getitem__((value0, value1))
                case [False, True], [True, False]:
                    return self.__getitem__((value1, value0))
                case [False, False], [False, False]:
                    raise ValueError(f"Invalid indices {index0}, {index1}")
                case [True, False], [True, False]:
                    raise ValueError(
                        f"Indices must be different: {index0} maps to {index1}"
                    )
                case [False, True], [False, True]:
                    raise ValueError(
                        f"Indices must be different: {index0} maps to {index1}"
                    )
                case [False, False], _:
                    raise ValueError(f"Invalid index: {index0}")
                case _, [False, False]:
                    raise ValueError(f"Invalid index: {index1}")

    def is_x_index(self, x) -> bool:
        return x in {"x", "X", self.mat.xalias}

    def is_y_index(self, x) -> bool:
        return x in {"y", "y", self.mat.yalias}


class MeshLocator(ticker.Locator):
    # Unrelated to the other locators. Named from matplotlib.ticker.MeshLocator
    def __init__(self, locs, nbins=10):
        "place ticks on the i-th data points where (i-offset)%base==0"
        self.locs = locs
        self.nbins = nbins

    def __call__(self):
        """Return the locations of the ticks"""
        vmin, vmax = self.axis.get_view_interval()
        return self.tick_values(vmin, vmax)

    def tick_values(self, vmin, vmax):

        if vmax < vmin:
            vmin, vmax = vmax, vmin

        if vmin == vmax:
            vmin -= 1
            vmax += 1

        dmin, dmax = self.axis.get_data_interval()

        imin = np.abs(self.locs - vmin).argmin()
        imax = np.abs(self.locs - vmax).argmin()
        step = max(int(np.ceil((imax - imin) / self.nbins)), 1)
        ticks = self.locs[imin : imax + 1 : step]
        if vmax - vmin > 0.8 * (dmax - dmin) and imax - imin > 20:
            # Round to the nearest "nicest" number
            # TODO Could be improved by taking vmin into account
            i = min(int(np.log10(abs(self.locs[imax]))), 2)
            i = max(i, 1)
            ticks = np.unique(np.around(ticks, -i))
        return self.raise_if_exceeds(ticks)


@njit
def last_nonzeros(x: np.ndarray, eps: float = 0.0) -> np.ndarray:
    """Returns a mask with 1 up to the last nonzero value in each row"""
    mask = np.zeros_like(x, dtype=np.bool_)
    for i in range(x.shape[0]):
        for j in range(x.shape[1] - 1, -1, -1):
            if abs(x[i, j]) > eps:
                mask[i, :j] = True
                break
    return mask


if jax_working():
    import jax
    import jax.numpy as jnp
    @jax.jit
    def last_nonzeros(x: jnp.ndarray, eps: float = 0.0) -> jnp.ndarray:
        """
        Returns a boolean mask of the same shape as `x`, where for each row
        all columns before the last element whose absolute value exceeds `eps`
        are True, and the rest are False.
        """
        n_cols = x.shape[1]
        # For entries with |x|>eps use their column index, else -1
        idx = jnp.where(jnp.abs(x) > eps, jnp.arange(n_cols), -1)
        # Find the last index > eps in each row (or -1 if none)
        last = jnp.max(idx, axis=1)        # shape (n_rows,)
        # Column indices 0,1,...,n_cols-1
        cols = jnp.arange(n_cols)          # shape (n_cols,)
        # For each row i, cols < last[i] gives True up to (but excluding) the last non-zero
        mask = cols < last[:, None]        # shape (n_rows, n_cols), dtype=bool
        return mask
