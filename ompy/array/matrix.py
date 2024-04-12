from __future__ import annotations

import copy
import logging
import warnings
from pathlib import Path
from typing import (overload, Literal, TypeAlias, Callable, Any, Never, Self)

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps as cmaps
from matplotlib import ticker
from matplotlib.colors import LogNorm, Normalize, SymLogNorm

from .abstractarray import AbstractArray
from .abstractarray import fetch as _fetch
from .filehandling import (load_numpy_2D, load_tar,
                           load_txt_2D, mama_read, mama_write, save_numpy_2D,
                           save_tar, save_txt_2D, Filetype, resolve_filetype,
                           load_npz_2D, save_npz_2D, save_hdf5_2D, load_hdf5_2D)
from .index import make_or_update_index, Edges, Index
from .matrixmetadata import MatrixMetadata
from .matrixprotocol import MatrixProtocol
from .rebin import rebin_2D, Preserve
from .vector import Vector, maybe_pop_from_kwargs
from .. import XARRAY_AVAILABLE, Unit, ROOT_IMPORTED
from ..helpers import maybe_set, make_ax, IQR_range
from ..helpers import robust_z_score, robust_z_score_i, AnnotatedColorbar
from ..numbalib import njit
from ..stubs import (Unitlike, Pathlike, Axes, Figure,
                     Colorbar, QuadMesh, arraylike, ArrayBool, QuantityLike)

LOG = logging.getLogger(__name__)
logging.captureWarnings(True)

# TODO mat*vec[:, None[ doesn't work
AxisEither: TypeAlias = Literal[0, 1]
AxisBoth: TypeAlias = Literal[0, 1, 2]
Axis: TypeAlias = AxisEither | AxisBoth
ColorByArg: TypeAlias = Literal['values', 'z-score', 'IQR', 'IQR2']
ColorBy: TypeAlias = ColorByArg | tuple[ColorByArg, ...]


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

    def __init__(self, *,
                 X: arraylike | Index | None = None,
                 Y: arraylike | Index | None = None,
                 values: np.ndarray | None = None,
                 X_unit: Unitlike | None = None,
                 Y_unit: Unitlike | None = None,
                 edge: Edges = 'left',
                 boundary: bool = False,
                 metadata: MatrixMetadata = MatrixMetadata(),
                 order: np._OrderKACF | None = None,
                 copy: bool = False,
                 indexkwargs: dict[str, Any] | None = None,
                 dtype: np.dtype | str = 'float32',
                 **kwargs):
        # Resolve aliasing
        kwargs, X, xalias = maybe_pop_from_kwargs(kwargs, X, 'X', 'xalias')
        kwargs, Y, yalias = maybe_pop_from_kwargs(kwargs, Y, 'Y', 'yalias')
        kwargs, values, valias = maybe_pop_from_kwargs(
            kwargs, values, 'values', 'valias')
        xalias = xalias or kwargs.pop('xalias', None)
        yalias = yalias or kwargs.pop('yalias', None)
        if valias is not None:
            kwargs['valias'] = valias

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
        default_xlabel = 'xlabel' not in kwargs
        xlabel = kwargs.pop('xlabel', r"Excitation energy")
        default_ylabel = 'ylabel' not in kwargs
        ylabel = kwargs.pop('ylabel', r"$\gamma$-energy")

        # If no `unit` is given, set it to the default
        # but let Index.unit override of given an Index
        default_X_unit = X_unit is None
        default_Y_unit = Y_unit is None
        X_unit = 'keV' if default_X_unit else X_unit
        Y_unit = 'keV' if default_Y_unit else Y_unit
        assert X_unit is not None
        assert Y_unit is not None
        self.X_index: Index = make_or_update_index(X, unit=Unit(X_unit), alias=xalias, label=xlabel,
                                                   default_label=default_xlabel,
                                                   default_unit=default_X_unit,
                                                   edge=edge, boundary=boundary,
                                                   **indexkwargs)
        self.Y_index: Index = make_or_update_index(Y, unit=Unit(Y_unit), alias=yalias, label=ylabel,
                                                   default_label=default_ylabel,
                                                   default_unit=default_Y_unit,
                                                   edge=edge, boundary=boundary,
                                                   **indexkwargs)
        if len(self.X_index) != self.values.shape[0]:
            _alias = f' ({xalias})' if xalias else ''
            _valias = f' ({valias})' if valias else ''
            raise ValueError(
                f"Length of X_index{_alias} must match first dimension of values{_valias}, expected {self.values.shape[0]}, got {len(self.X_index)}")
        if len(self.Y_index) != self.values.shape[1]:
            _alias = f' ({yalias})' if yalias else ''
            _valias = f' ({valias})' if valias else ''
            raise ValueError(
                f"Length of Y_index{_alias} must match second dimension of values{_valias}, expected {self.values.shape[1]}, got {len(self.Y_index)}")
        wrong_kw = set(kwargs) - set(MatrixMetadata.__slots__)
        if wrong_kw:
            raise ValueError(
                f"Invalid keyword arguments: {', '.join(wrong_kw)}")
        if isinstance(metadata, dict):
            metadata = MatrixMetadata(**metadata)
        if not isinstance(metadata, MatrixMetadata):
            raise TypeError(f"metadata must be a MatrixMetadata, not {type(metadata)}")
        self.metadata = metadata.update(**kwargs)
        if self.xalias == self.yalias and xalias:
            raise ValueError(f"Aliases must be unique. Got {self.xalias} == {self.yalias}")
        self.iloc = IndexLocator(self)
        self.vloc = ValueLocator(self)
        self.loc = ValueLocator(self, strict=False)

    def __getattr__(self, item) -> Any:
        meta: MatrixMetadata = self.__dict__['metadata']
        xalias: str = self.__dict__['X_index'].alias
        yalias: str = self.__dict__['Y_index'].alias
        if item == xalias:
            x = self.X
        elif item == yalias:
            x = self.Y
        elif item == meta.valias:
            x = self.__dict__['values']
        elif item == 'd' + xalias:
            x = self.dX
        elif item == 'd' + yalias:
            x = self.dY
        elif item == 'index_' + xalias:
            return self.index_X
        elif item == 'index_' + yalias:
            return self.index_Y
        elif item == xalias + '_index':
            return self.X_index
        elif item == yalias + '_index':
            return self.Y_index
        else:
            x = super().__getattr__(item)
        return x

    @classmethod
    def from_path(cls, path: Pathlike, filetype: Filetype | None = None, **kwargs) -> Self:
        """ Load matrix from specified filetype

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
            case 'npz':
                return cls.from_npz(path, **kwargs)
            case 'npy':
                return cls.from_npy(path)
            case 'txt':
                return cls.from_txt(path)
            case 'tar':
                return cls.from_tar(path)
            case 'mama':
                return cls.from_mama(path)
            case 'hdf5':
                return cls.from_hdf5(path, **kwargs)
            case _:
                raise ValueError(f"Unknown filetype: {filetype}")

    @classmethod
    def from_npz(cls, path: Pathlike, **kwargs) -> Self:
        return load_npz_2D(path, cls, **kwargs)

    @classmethod
    def from_npy(cls, path: Pathlike) -> Self:
        values, Y, X = load_numpy_2D(path)
        return cls(Ex=X, Eg=Y, values=values)

    @classmethod
    def from_txt(cls, path: Pathlike) -> Self:
        values, Y, X = load_txt_2D(path)
        return cls(Ex=X, Eg=Y, values=values)

    @classmethod
    def from_tar(cls, path: Pathlike) -> Self:
        values, Y, X = load_tar(path)
        return cls(Ex=X, Eg=Y, values=values)

    @classmethod
    def from_mama(cls, path: Pathlike) -> Self:
        ret = mama_read(path)
        if len(ret) == 3:
            values, Y, X = ret
            return cls(Ex=X, Eg=Y, values=values)
        else:
            raise RuntimeError("Wrong format of mama file")

    def from_hdf5(cls, path: Pathlike, **kwargs) -> Self:
        return load_hdf5_2D(path, cls, **kwargs)

    def save(self, path: Pathlike, filetype: Filetype | None = None, **kwargs) -> None:
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
        path = Path(path)
        path, filetype = resolve_filetype(path, filetype)

        match filetype:
            case 'npz':
                self.to_npz(path, **kwargs)
            case 'npy':
                warnings.warn("Saving as numpy is deprecated, use npz instead")
                self.to_numpy(path, **kwargs)
            case 'txt':
                self.to_txt(path, **kwargs)
            case 'tar':
                warnings.warn("Saving to .tar does not preserve metadata. Use .npz instead.")
                save_tar([self.values, self.Y_index.bins, self.X_index.bins], path)
            case 'mama':
                self.to_mama(path, **kwargs)
            case 'hdf5':
                self.to_hdf5(path, **kwargs)
            case _:
                raise ValueError(f"Unknown filetype: {filetype}")

    def to_npz(self, path: Pathlike, **kwargs) -> None:
        """Save matrix to NPZ file format.

        Args:
            path (Pathlike): Path to save the file.
            **kwargs: Additional keyword arguments to pass to save_npz_2D.
        """
        save_npz_2D(path, self, **kwargs)

    def to_hdf5(self, path: Pathlike, **kwargs) -> None:
        """Save matrix to HDF5 file format.

        Args:
            path (Pathlike): Path to save the file.
            **kwargs: Additional keyword arguments to pass to save_hdf5_2D.
        """
        save_hdf5_2D(self, path, **kwargs)

    def to_txt(self, path: Pathlike, **kwargs) -> None:
        """Save matrix to TXT file format.

        Args:
            path (Pathlike): Path to save the file.
            **kwargs: Additional keyword arguments to pass to save_txt_2D.
        """
        X = self.X_index.to_unit('keV').bins
        Y = self.Y_index.to_unit('keV').bins
        save_txt_2D(self.values, Y, X, path, **kwargs)

    def to_numpy(self, path: Pathlike) -> None:
        """Save matrix to NumPy binary file format.

        Args:
            path (Pathlike): Path to save the file.
            **kwargs: Additional keyword arguments to pass to save_numpy_2D.
        """
        X = self.X_index.to_unit('keV').bins
        Y = self.Y_index.to_unit('keV').bins
        save_numpy_2D(self.values, Y, X, path)

    def to_mama(self, path: Pathlike, **kwargs) -> None:
        """Save matrix to MAMA file format.

        Args:
            path (Pathlike): Path to save the file.
            **kwargs: Additional keyword arguments to pass to mama_write.
        """
        mama_write(self, path, comment="Made by OMpy", **kwargs)

    def to_tar(self, path: Pathlike) -> None:
        """Save matrix as tarball

        Args:
             path (Pathlike): Path to save the file
        """
        X = self.X_index.to_unit('keV').bins
        Y = self.Y_index.to_unit('keV').bins
        save_tar([self.values, Y, X])

    def reshape_like(self, other: Matrix,
                     inplace: bool = False) -> Matrix | None:
        """ Cut and rebin a matrix like another matrix (according to energy arrays)

        Args:
            other (Matrix): The other matrix
            inplace (bool, optional): If True make the cut in place. Otherwise
                return a new matrix. Defaults to False.

        Returns:
            Matrix | None: If inplace is False, returns the cut matrix
        """
        if inplace:
            self.rebin(0, bins=other.X_index, inplace=True)
            self.rebin(1, bins=other.Y_index, inplace=True)
        else:
            return self.rebin(1, bins=other.Y_index).rebin(0, bins=other.X_index)

    @overload
    def rebin(self, axis: int | str, *,
              bins: arraylike | Index | None = None,
              factor: float | None = None,
              binwidth: QuantityLike | None = None,
              numbins: int | None = None,
              preserve: Preserve = 'counts',
              inplace: Literal[False] = ...) -> Matrix:
        ...

    @overload
    def rebin(self, axis: int | str, *,
              bins: arraylike | Index | None = None,
              factor: float | None = None,
              binwidth: QuantityLike | None = None,
              numbins: int | None = None,
              preserve: Preserve = 'counts',
              inplace: Literal[True] = ...) -> None:
        ...

    def rebin(self, axis: int | str, *,
              bins: arraylike | Index | None = None,
              factor: float | None = None,
              binwidth: QuantityLike | None = None,
              numbins: int | None = None,
              preserve: Preserve = 'counts',
              inplace: bool = False) -> Self | None:
        """ Rebins one axis of the matrix

        Args:
            axis: the axis to rebin.
            bins: The new mids along the axis. Can not be
                given alongside 'factor' or 'binwidth'.
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
                self.rebin(axis=0, bins=bins, factor=factor,
                           binwidth=binwidth, inplace=True,
                           numbins=numbins)
                self.rebin(axis=1, bins=bins, factor=factor,
                           binwidth=binwidth, inplace=True,
                           numbins=numbins)
                return
            else:
                new = self.rebin(axis=0, bins=bins, factor=factor,
                                 binwidth=binwidth, inplace=False,
                                 numbins=numbins)
                new.rebin(axis=1, bins=bins, factor=factor,
                          binwidth=binwidth, inplace=True,
                          numbins=numbins)
                return new
        elif axis_ == 0:
            index: Index = self.X_index.handle_rebin_arguments(
                bins=bins, factor=factor, binwidth=binwidth, numbins=numbins)
            rebinned = rebin_2D(self.X_index, index.bins,
                                self.values, axis=0, preserve=preserve)
            if inplace:
                self.values = rebinned
                self.X_index = index
            else:
                return self.clone(X=index, values=rebinned)
        else:
            index: Index = self.Y_index.handle_rebin_arguments(
                bins=bins, factor=factor, binwidth=binwidth, numbins=numbins)
            rebinned = rebin_2D(self.Y_index, index.bins,
                                self.values, axis=1, preserve=preserve)
            if inplace:
                self.values = rebinned
                self.Y_index = index
            else:
                return self.clone(Y=index, values=rebinned)

    def index_X(self, x: float) -> int:
        return self.X_index.index_expression(x)

    def index_Y(self, x: float) -> int:
        return self.Y_index.index_expression(x)

    def to_unit(self, unit: Unitlike, axis: str | int = 'both', inplace: bool = False) -> None | Matrix:
        """ Returns a copy with units set to `unit`.

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

    def to_mid(self, axis: int | str = 'both', inplace: bool = False) -> None | Matrix:
        """ Returns a copy with the bins set to the midpoints of the bins.

        Args:
            axis: The axis to transform. Defaults to both.
            inplace: Change the matrix inplace or return a copy.
        Returns:
            A copy of the matrix with the bins set to the midpoints of the bins.
        """
        return self.to_edge('mid', axis=axis, inplace=inplace)

    @overload
    def to_left(self, axis: int | str = ..., inplace: Literal[False] = ...) -> Matrix:
        ...

    @overload
    def to_left(self, axis: int | str = ..., inplace: Literal[True] = ...) -> None:
        ...

    def to_left(self, axis: int | str = 'both', inplace: bool = False) -> None | Matrix:
        """ Returns a copy with the bins set to the left edges of the bins.

        Args:
            axis: The axis to transform. Defaults to both.
            inplace: Change the matrix inplace or return a copy.
        Returns:
            A copy of the matrix with the bins set to the left edges of the bins.
        """
        return self.to_edge('left', axis=axis, inplace=inplace)

    def to_edge(self, edge: Edges, axis: int | str = 'both', inplace: bool = False) -> None | Matrix:
        """ Returns a copy with the bins set to the left or mid edges of the bins.

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
        """ Returns a copy of the matrix with only the rows and columns  where `mask` is True.

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
        s = f"Array type: {type(self.values)}\n"
        s += f'X index:\n{self.X_index.summary()}\n'
        s += f'Y index:\n{self.Y_index.summary()}\n'
        if len(self.metadata.misc) > 0:
            s += 'Metadata:\n'
            for key, val in self.metadata.misc.items():
                s += f'\t{key}: {val}\n'
        s += f'Total counts: {self.sum():.3g}'
        return s

    def summary(self):
        print(self._summary)

    @overload
    def sum(self, axis: Literal['both'] = ...,
            out: np.ndarray | None = ...) -> float:
        ...

    @overload
    def sum(self, axis: Literal[2] = ...,
            out: np.ndarray | None = ...) -> float:
        ...

    @overload
    def sum(self, axis: Literal[0, 1] | str = ...,
            out: np.ndarray | None = ...) -> Vector:
        ...

    def sum(self, axis: int | str = 'both',
            out: np.ndarray | None = None) -> Vector | float:
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

    @property
    def X(self) -> np.ndarray:
        return self.X_index.bins

    @property
    def Y(self) -> np.ndarray:
        return self.Y_index.bins

    def clone(self, X: Index | None = None, Y: Index | None = None,
              values: np.ndarray | None = None,
              metadata: MatrixMetadata | None = None, copy: bool = False,
              dtype: np.dtype | None = None,
              **kwargs) -> Self:
        """ Copies the object.

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
        return type(self)(X=X, Y=Y, values=values, metadata=metadata, copy=copy,
                          dtype=dtype)

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
    def plot(self, ax: Axes, *,
             scale: str | None = None,
             vmin: float | None = None,
             vmax: float | None = None,
             add_cbar: Literal[True] = ...,
             cbarkwargs: dict[str, Any] | None = None,
             bad_map: Callable[[Matrix], ArrayBool | bool] = lambda x: False,
             color_by: ColorBy = ...,
             **kwargs) -> tuple[Axes, tuple[QuadMesh, Colorbar]]:
        ...

    @overload
    def plot(self, ax: Axes, *,
             scale: str | None = None,
             vmin: float | None = None,
             vmax: float | None = None,
             add_cbar: Literal[False] = ...,
             cbarkwargs: dict[str, Any] | None = None,
             bad_map: Callable[[Matrix], ArrayBool | bool] = lambda x: False,
             color_by: ColorBy = ...,
             **kwargs) -> tuple[Axes, tuple[QuadMesh, None]]:
        ...

    def plot(self, ax: Axes | None = None, *,
             scale: str | None = None,
             vmin: float | None = None,
             vmax: float | None = None,
             add_cbar: bool = True,
             cbarkwargs: dict[str, Any] | None = None,
             bad_map: Callable[[Matrix], ArrayBool | bool] = lambda x: False,
             color_by: ColorBy = 'values',
             **kwargs) -> tuple[Axes, tuple[QuadMesh, Colorbar | None]]:
        """ Plots the matrix with the energy along the axis

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
        fig: Figure = ax.figure  # type: ignore

        # In case `values` is on the gpu
        values = np.asarray(self.values)

        # Simple heuristic to determine scale
        if scale is None:
            if np.any(values < 0):
                if np.sum(abs(values)) > 1e3:
                    scale = 'symlog'
                else:
                    scale = 'linear'
            else:
                if np.sum(values) > 1e3:
                    scale = 'log'
                else:
                    scale = 'linear'

        mask = np.isnan(values) | (values == 0) | bad_map(self)
        masked = np.ma.array(values, mask=mask)

        # Try methods to determine colorscale to prevent
        # outliers from skewing the data
        if isinstance(color_by, str):
            color_args = []
        else:
            color_args = color_by[1:] if len(color_by) > 1 else []
            color_by = color_by[0]

        if color_by == 'IQR':
            factor = color_args[0] if color_args else 1.5
            vmin_IQR, vmax_IQR = IQR_range(values[~mask].ravel(), factor)
            if scale == 'log':
                vmin_IQR = max(vmin_IQR, 1e-1)
            vmin = vmin if vmin is not None else vmin_IQR
            vmax = vmax if vmax is not None else vmax_IQR
        elif color_by == 'z-score':
            z = np.zeros_like(values)
            x = values[~mask]
            z[~mask] = robust_z_score(x)
            zmin = color_args[0] if color_args else -2
            zmax = color_args[1] if len(color_args) > 1 else 2
            vmin_z = max(zmin, z.min())
            vmax_z = min(zmax, z.max())

            vmin_z = robust_z_score_i(vmin_z, x)
            vmax_z = robust_z_score_i(vmax_z, x)
            if scale == 'log':
                vmin_z = max(vmin_z, 1e-1)

            vmin = vmin if vmin is not None else vmin_z
            vmax = vmax if vmax is not None else vmax_z
        elif color_by == 'percentile':
            lower: float = color_args[0] if color_args else 0.5
            upper: float = color_args[1] if len(color_args) > 1 else 99.5
            vmin_p = np.percentile(values[~mask], lower)
            vmax_p = np.percentile(values[~mask], upper)

            vmin = vmin if vmin is not None else vmin_p
            vmax = vmax if vmax is not None else vmax_p
        elif color_by == 'values':
            pass
        else:
            raise ValueError(f"Unknown color_by: {color_by}. Supported are"
                             " 'IQR', 'z-score', 'percentile' and 'values'")

        if scale == 'log':
            if vmin is not None and vmin <= 0:
                raise ValueError("`vmin` must be positive for log-scale")
            if vmin is None:
                _max = np.log10(self.max())
                _min = np.log10(values[values > 0].min())
                if _max - _min > 10:
                    vmin = 10 ** (int(_max - 6))
            norm = LogNorm(vmin=vmin, vmax=vmax)
        elif scale == 'symlog':
            lintresh = kwargs.pop('lintresh', 1e-1)
            linscale = kwargs.pop('linscale', 1)
            norm = SymLogNorm(lintresh, linscale, vmin, vmax)
        elif scale == 'linear':
            norm = Normalize(vmin=vmin, vmax=vmax)
        else:
            raise ValueError("Unsupported zscale ", scale)
        norm = kwargs.pop('norm', norm)
        X, Y = self._plot_mesh()

        # Set entries of 0 to white
        current_cmap = copy.copy(cm.get_cmap())
        current_cmap.set_bad(color='white')
        cmap = plt.get_cmap(kwargs.pop('cmap', current_cmap))

        mesh = ax.pcolormesh(Y, X, masked, cmap=cmap, norm=norm, **kwargs)

        # TODO: Let the index handle the ticks?
        if self.Y_index.is_mid():
            ax.xaxis.set_major_locator(MeshLocator(self.Y))
            ax.tick_params(axis='x', rotation=40)
        if self.X_index.is_mid():
            ax.yaxis.set_major_locator(MeshLocator(self.X))
        if hasattr(self.X_index, 'scale'):
            if self.X_index.scale is not None:
                ax.set_yscale(self.X_index.scale)
        if hasattr(self.Y_index, 'scale'):
            if self.Y_index.scale is not None:
                ax.set_xscale(self.Y_index.scale)

        maybe_set(ax, title=self.name)
        maybe_set(ax, ylabel=self.get_xlabel())
        maybe_set(ax, xlabel=self.get_ylabel())

        # show z-value in status bar
        # https://stackoverflow.com/questions/42577204/show-z-value-at-mouse-pointer-position-in-status-line-with-matplotlibs-pcolorme
        def format_coord(x, y):
            xarr = X
            yarr = Y
            if ((x > xarr.min()) & (x <= xarr.max())
                    & (y > yarr.min()) & (y <= yarr.max())):
                col = np.searchsorted(xarr, x) - 1
                row = np.searchsorted(yarr, y) - 1
                z = masked[row, col]
                return f'x={x:1.2f}, y={y:1.2f}, z={z:1.2E}'
                # return f'x={x:1.0f}, y={y:1.0f}, z={z:1.3f}   [{row},{col}]'
            else:
                return f'x={x:1.0f}, y={y:1.0f}'

        # TODO: Takes waaaay to much CPU
        # ax.format_coord = nop

        cbar: Colorbar | None = None
        if add_cbar:
            if cbarkwargs is None:
                cbarkwargs = {}
            kwargs = dict(ax=ax) | cbarkwargs
            cbar = AnnotatedColorbar(mesh,  **kwargs)

        return ax, (mesh, cbar)

    def plot_3d(self, ax: Axes | None = None, vmin: float | None = None,
                vmax: float | None = None, scale='linear',
                add_cbar: bool = True,
                cbarkwargs: dict | None = None, **kwargs):
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
        fig: Figure = ax.figure  # type: ignore

        if scale == 'log':
            if vmin is not None and vmin <= 0:
                raise ValueError("`vmin` must be positive for log-scale")
            if vmin is None:
                _max = np.log10(self.max())
                _min = np.log10(self.values[self.values > 0].min())
                if _max - _min > 10:
                    vmin = 10 ** (int(_max - 6))
            norm = LogNorm(vmin=vmin, vmax=vmax)
        elif scale == 'symlog':
            lintresh = kwargs.pop('lintresh', 1e-1)
            linscale = kwargs.pop('linscale', 1)
            norm = SymLogNorm(lintresh, linscale, vmin, vmax)
        elif scale == 'linear':
            norm = Normalize(vmin=vmin, vmax=vmax)
        else:
            raise ValueError("Unsupported zscale ", scale)
        norm = kwargs.pop('norm', norm)
        cmap = cmaps.get_cmap(kwargs.pop('cmap', cm.get_cmap()))
        X, Y = self.X, self.Y  # self._plot_mesh()
        dx = X[1] - X[0]
        dy = Y[1] - Y[0]
        Y, X = np.meshgrid(Y, X)
        Z = self.values
        colors = cmap(norm(Z.flatten()))

        # Plotting a 3D histogram with color based on height (log normalized)
        mesh = ax.bar3d(Y.flatten(),
                        X.flatten(),
                        np.zeros_like(Z).flatten(),
                        dy, dx, Z.flatten(), shade=True, color=colors)

        if False:
            if self.Y_index.is_mid():
                ax.xaxis.set_major_locator(MeshLocator(self.Y))
                ax.tick_params(axis='x', rotation=40)
            if self.X_index.is_mid():
                ax.yaxis.set_major_locator(MeshLocator(self.X))

        cbar: Colorbar | None = None
        if add_cbar:
            if cbarkwargs is None:
                cbarkwargs = {}
            cbarkwargs.setdefault('fraction', 0.03)
            cbarkwargs.setdefault('pad', 0.04)

            mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
            mappable.set_array(Z)
            if vmin is not None and vmax is not None:
                cbar = fig.colorbar(mappable, ax=ax, extend='both', **cbarkwargs)
            elif vmin is not None:
                cbar = fig.colorbar(mappable, ax=ax, extend='min', **cbarkwargs)
            elif vmax is not None:
                cbar = fig.colorbar(mappable, ax=ax, extend='max', **cbarkwargs)
            else:
                cbar = fig.colorbar(mappable, ax=ax, **cbarkwargs)

            maybe_set(cbar.ax, ylabel=self.vlabel)

        maybe_set(ax, title=self.name)
        maybe_set(ax, ylabel=self.get_xlabel())
        maybe_set(ax, xlabel=self.get_ylabel())

        return ax, (mesh, cbar)

    def get_ylabel(self) -> str:
        unit = f'{self.Y_index.unit:~L}'
        if unit:
            return self.ylabel + f" [${unit}$]"
        return self.ylabel

    def get_xlabel(self) -> str:
        unit = f'{self.X_index.unit:~L}'
        if unit:
            return self.xlabel + f" [${unit}$]"
        return self.xlabel

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
    def axis_to_int(self, axis: int | str,
                    allow_both: Literal[False] = ...) -> AxisEither:
        ...

    @overload
    def axis_to_int(self, axis: int | str,
                    allow_both: Literal[True] = ...) -> AxisBoth:
        ...

    def axis_to_int(self, axis: int | str, allow_both: bool = False) -> Axis:
        match axis:
            case 0 | 1:
                return axis
            case 2:
                if allow_both:
                    return axis
                raise ValueError("Cannot use axis 2 for matrix")
            case 'x' | 'X' | self.xalias:
                return 0
            case 'y' | 'Y' | self.yalias:
                return 1
            case 'both':
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
    def __matmul__(self, other: Matrix) -> Self:
        ...

    @overload
    def __matmul__(self, other: Vector) -> Vector:
        ...

    @overload
    def __matmul__(self, other: np.ndarray) -> np.ndarray:
        ...

    def __matmul__(self, other: Matrix | Vector | np.ndarray) -> Self | Vector | np.ndarray:
        match other:
            case Matrix():
                if self.shape[1] != other.shape[0]:
                    raise ValueError(
                        f"Shape mismatch {self.shape} @ {other.shape}")
                if not self.is_compatible_with_Y(other.X_index):
                    raise ValueError(
                        f"Y index mismatch\n({self.Y_index})\n @\n({other.X_index})")
                return type(self)(X=self.X_index, Y=other.Y_index, values=self.values @ other.values)
            case Vector():
                if self.shape[1] != other.shape[0]:
                    raise ValueError(
                        f"Shape mismatch {self.shape} @ {other.shape}")
                if not self.is_compatible_with_Y(other._index):
                    raise ValueError(
                        f"Y index mismatch between:\n{self.Y_index.summary()}\n\nand:\n{other._index.summary()}")
                return self.meta_into_vector(self.X_index, self.values @ other.values)
            case np.ndarray():
                return self.values @ other
            case _:
                if not hasattr(other, '__rmatmul__'):
                    raise TypeError(
                        f"Cannot multiply {self.__class__.__name__} with {other.__class__.__name__}")
                return other.__rmatmul__(self)

    def last_nonzero(self, i: int, eps: float = 0) -> int:
        """ Returns the index of the last non-zero element """
        j = self.shape[1]
        while (j := j - 1) >= 0:
            if abs(self[i, j]) > eps:
                break
        return j

    def last_nonzeros(self, eps: float = 0.0) -> np.ndarray:
        return last_nonzeros(self.values, eps=eps)

    def to_xarray(self):
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


if XARRAY_AVAILABLE:
    import xarray as xr


    def to_xarray_matrix(mat) -> xr.DataArray:  # type: ignore
        return xr.DataArray(mat.values, coords=[mat.X, mat.Y],
                            dims=[mat.xalias, mat.yalias])
else:
    def to_xarray_matrix(mat) -> Never:
        raise NotImplementedError("xarray not available")

if ROOT_IMPORTED:
    from ROOT import TH2D  # type: ignore


    def to_root_matrix(mat, identifier: str | None = None) -> TH2D:  # type: ignore
        mat = mat.to_left()
        if identifier is None:
            identifier = mat.name
        hist = TH2D(identifier, identifier, len(mat.Y) - 1, mat.Y, len(mat.X) - 1, mat.X)
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


class IndexLocator:
    def __init__(self, matrix: Matrix):
        self.mat = matrix

    @overload
    def __getitem__(self, key: tuple[slice, slice]) -> Matrix:
        ...

    @overload
    def __getitem__(self, key: tuple[slice, int | slice]) -> Vector:
        ...

    @overload
    def __getitem__(self, key: tuple[int | slice, slice]) -> Vector:
        ...

    @overload
    def __getitem__(self, key: tuple[int, int]) -> float:
        ...

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
    def __getitem__(self, key: tuple[slice, slice]) -> Matrix:
        ...

    @overload
    def __getitem__(self, key: tuple[slice, int | float | slice]) -> Vector:
        ...

    @overload
    def __getitem__(self, key: tuple[int | float | slice, slice]) -> Vector:
        ...

    @overload
    def __getitem__(self, key: tuple[int | float, int | float]) -> float:
        ...

    def __getitem__(self, key):
        match key:
            case slice() as x, slice() as y:
                sx: slice = self.mat.X_index.index_slice(x, strict=self.strict)
                sy: slice = self.mat.Y_index.index_slice(y, strict=self.strict)
                xindex = self.mat.X_index[sx]
                yindex = self.mat.Y_index[sy]
                values = self.mat.values.__getitem__((sx, sy))
                return self.mat.clone(values=values, X=xindex, Y=yindex)
            case slice() as x, y:
                sx: slice = self.mat.X_index.index_slice(x, strict=self.strict)
                xindex = self.mat.X_index[sx]
                j: int = self.mat.Y_index.index_expression(
                    y, strict=self.strict)
                values = self.mat.values.__getitem__((sx, j))
                return self.mat.meta_into_vector(xindex, values)
            case x, slice() as y:
                sy: slice = self.mat.Y_index.index_slice(y, strict=self.strict)
                yindex = self.mat.Y_index[sy]
                i: int = self.mat.X_index.index_expression(
                    x, strict=self.strict)
                values = self.mat.values.__getitem__((i, sy))
                return self.mat.meta_into_vector(yindex, values)
            case x, y:
                i: int = self.mat.X_index.index_expression(
                    x, strict=self.strict)
                j: int = self.mat.Y_index.index_expression(
                    y, strict=self.strict)
                return self.mat.values.__getitem__((i, j))

    def __setitem__(self, key, val):
        match key:
            case slice() as x, slice() as y:
                sx: slice = self.mat.X_index.index_slice(x, strict=self.strict)
                sy: slice = self.mat.Y_index.index_slice(y, strict=self.strict)
                self.mat.values.__setitem__((sx, sy), val)
            case slice() as x, y:
                sx: slice = self.mat.X_index.index_slice(x, strict=self.strict)
                j: int = self.mat.Y_index.index_expression(
                    y, strict=self.strict)
                self.mat.values.__setitem__((sx, j), val)
            case x, slice() as y:
                sy: slice = self.mat.Y_index.index_slice(y, strict=self.strict)
                i: int = self.mat.X_index.index_expression(
                    x, strict=self.strict)
                self.mat.values.__setitem__((i, sy), val)
            case x, y:
                i: int = self.mat.X_index.index_expression(
                    x, strict=self.strict)
                j: int = self.mat.Y_index.index_expression(
                    y, strict=self.strict)
                self.mat.values.__setitem__((i, j), val)

    def __call__(self, **mappings: dict[str, slice | int | float | str]) -> Matrix | Vector | None:
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
                    raise ValueError(f"Indices must be different: {index0} maps to {index1}")
                case [False, True], [False, True]:
                    raise ValueError(f"Indices must be different: {index0} maps to {index1}")
                case [False, False], _:
                    raise ValueError(f"Invalid index: {index0}")
                case _, [False, False]:
                    raise ValueError(f"Invalid index: {index1}")

    def is_x_index(self, x) -> bool:
        return x in {'x', 'X', self.mat.xalias}

    def is_y_index(self, x) -> bool:
        return x in {'y', 'y', self.mat.yalias}


class MeshLocator(ticker.Locator):
    # Unrelated to the other locators. Named from matplotlib.ticker.MeshLocator
    def __init__(self, locs, nbins=10):
        'place ticks on the i-th data points where (i-offset)%base==0'
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
        ticks = self.locs[imin:imax + 1:step]
        if vmax - vmin > 0.8 * (dmax - dmin) and imax - imin > 20:
            # Round to the nearest "nicest" number
            # TODO Could be improved by taking vmin into account
            i = min(int(np.log10(abs(self.locs[imax]))), 2)
            i = max(i, 1)
            ticks = np.unique(np.around(ticks, -i))
        return self.raise_if_exceeds(ticks)


@njit
def last_nonzeros(x: np.ndarray, eps: float = 0.0) -> np.ndarray:
    """ Returns a mask with 1 up to the last nonzero value in each row """
    mask = np.zeros_like(x, dtype=np.bool_)
    for i in range(x.shape[0]):
        for j in range(x.shape[1] - 1, -1, -1):
            if abs(x[i, j]) > eps:
                mask[i, :j] = True
                break
    return mask
