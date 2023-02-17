from __future__ import annotations

import copy
import logging
import warnings
from pathlib import Path
from typing import (Sequence, Union, overload, Literal, TypeAlias)

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker
from matplotlib.colors import LogNorm, Normalize, SymLogNorm

from .. import ureg, Unit
from .abstractarray import AbstractArray, to_plot_axis
from .filehandling import (filetype_from_suffix, load_numpy_2D, load_tar,
                           load_txt_2D, mama_read, mama_write, save_numpy_2D,
                           save_tar, save_txt_2D, Filetype, resolve_filetype, load_npz_2D, save_npz_2D)
from ..geometry import Line
from ..library import (diagonal_elements, div0, fill_negative_gauss,
                       handle_rebin_arguments, maybe_set)
from ..stubs import (Unitlike, Pathlike, ArrayKeV, Axes, Figure,
                    Colorbar, QuadMesh, ArrayInt, PointUnit, array, arraylike, ArrayBool, numeric)
from .vector import Vector, maybe_pop_from_kwargs
from .index import Index, Edge, make_or_update_index
from .rebin import rebin_2D
from dataclasses import dataclass, field

LOG = logging.getLogger(__name__)
logging.captureWarnings(True)

#TODO mat*vec[:, None[ doesn't work
AxisEither: TypeAlias = Literal[0, 1]
AxisBoth: TypeAlias = Literal[0, 1, 2]
Axis: TypeAlias = AxisEither | AxisBoth


@dataclass(frozen=True, slots=True)
class MatrixMetadata:
    """Stores metadata for a Matrix.

    """
    valias: str = ''
    vlabel: str = 'Counts'
    name: str = ''
    misc: dict[str, any] = field(default_factory=dict)

    def clone(self, valias: str | None = None, vlabel: str | None = None,
              name: str | None = None, misc: dict[str, any] | None = None) -> MatrixMetadata:
        valias = valias if valias is not None else self.valias
        vlabel = vlabel if vlabel is not None else self.vlabel
        name = name if name is not None else self.name
        misc = misc if misc is not None else self.misc
        return MatrixMetadata(valias, vlabel, name, misc)

    def update(self, **kwargs) -> MatrixMetadata:
        return self.clone(**kwargs)

    def add_comment(self, key: str, value: any) -> MatrixMetadata:
        return self.update(misc=self.misc | {key: value})



class Matrix(AbstractArray):
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
        std: Array of standard deviations
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
    def __init__(self, *,
                 X: arraylike | Index | None = None,
                 Y: arraylike | Index | None = None,
                 values: np.ndarray | None = None,
                 std: np.ndarray | None = None,
                 unit: Unitlike | None = None,
                 edge: Edge = 'left',
                 boundary: bool = False,
                 metadata: MatrixMetadata = MatrixMetadata(),
                 order: Literal['C', 'F'] = 'C',
                 copy: bool = False,
                 indexkwargs: dict[str, any] | None = None,
                 **kwargs):
        #Resolve aliasing
        kwargs, X, xalias = maybe_pop_from_kwargs(kwargs, X, 'X', 'xalias')
        kwargs, Y, yalias = maybe_pop_from_kwargs(kwargs, Y, 'Y', 'yalias')
        kwargs, values, valias = maybe_pop_from_kwargs(kwargs, values, 'values', 'valias')
        xalias = xalias or kwargs.pop('xalias', None)
        yalias = yalias or kwargs.pop('yalias', None)
        if valias is not None:
            kwargs['valias'] = valias

        if copy:
            def fetch(x):
                return np.asarray(x, dtype=float, order=order).copy()
        else:
            def fetch(x):
                return np.asarray(x, dtype=float, order=order)

        self.values = fetch(values)
        if self.values.ndim != 2:
            raise ValueError(f"values must be 2D, not {self.values.ndim}")
        self.std = fetch(std) if std is not None else None
        if self.std is not None and self.std.shape != self.values.shape:
            raise ValueError(f"std must have same shape as values, expected {self.values.shape}, got {self.std.shape}.")

        indexkwargs = indexkwargs or {}
        default_xlabel = 'xlabel' not in kwargs
        xlabel = kwargs.pop('xlabel', r"Excitation energy")
        default_ylabel = 'ylabel' not in kwargs
        ylabel = kwargs.pop('ylabel', r"$\gamma$-energy")
        default_unit = False if unit is not None else True
        unit = 'keV' if default_unit else unit
        self.X_index = make_or_update_index(X, unit=Unit(unit), alias=xalias, label=xlabel,
                                           default_label=default_xlabel,
                                           default_unit=default_unit,
                                           edge=edge, boundary=boundary,
                                           **indexkwargs)
        self.Y_index = make_or_update_index(Y, unit=Unit(unit), alias=yalias, label=ylabel,
                                             default_label=default_ylabel,
                                             default_unit=default_unit,
                                             edge=edge, boundary=boundary,
                                             **indexkwargs)
        if len(self.X_index) != self.values.shape[0]:
            _alias = f' ({xalias})' if xalias else ''
            _valias = f' ({valias})' if valias else ''
            raise ValueError(f"Length of X_index{_alias} must match first dimension of values{_valias}, expected {self.values.shape[0]}, got {len(self.X_index)}")
        if len(self.Y_index) != self.values.shape[1]:
            _alias = f' ({yalias})' if yalias else ''
            _valias = f' ({valias})' if valias else ''
            raise ValueError(f"Length of Y_index{_alias} must match second dimension of values{_valias}, expected {self.values.shape[1]}, got {len(self.Y_index)}")
        wrong_kw = set(kwargs) - set(MatrixMetadata.__slots__)
        if wrong_kw:
            raise ValueError(f"Invalid keyword arguments: {', '.join(wrong_kw)}")
        self.metadata = metadata.update(**kwargs)
        self.iloc = IndexLocator(self)
        self.vloc = ValueLocator(self)
        self.loc = ValueLocator(self, strict=False)


    def __getattr__(self, item) -> any:
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
    def from_path(cls, path: Pathlike, filetype: Filetype | None = None, **kwargs ) -> Matrix:
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
                return load_npz_2D(path, cls, **kwargs)
            case 'npy':
                values, Y, X = load_numpy_2D(path)
            case 'txt':
                values, Y, X = load_txt_2D(path)
            case 'tar':
                values, Y, X = load_tar(path)
            case 'mama':
                values, Y, X = mama_read(path)
            case _:
                raise ValueError(f"Unknown filetype: {filetype}")
        return cls(Ex=X, Eg=Y, values=values)

    def save(self, path: Pathlike, filetype: Filetype | None = None,
             **kwargs) -> None:
        """Save matrix to file

        Args:
            path (str or Path): path to file to save
            filetype (str, optional): Filetype to save. Has an
                auto-recognition. Options: ["numpy", "tar", "mama", "txt"]
            **kwargs: additional keyword arguments
        Raises:
            ValueError: If filetype is unknown
        """
        path = Path(path)
        path, filetype = resolve_filetype(path, filetype)

        X = self.X_index.to_unit('keV').bins
        Y = self.Y_index.to_unit('keV').bins
        # TODO Move this logic to filehandling
        match filetype:
            case 'npz':
                save_npz_2D(path, self, **kwargs)
            case 'npy':
                warnings.warn("Saving as numpy is deprecated, use npz instead")
                save_numpy_2D(self.values, Y, X, path)
            case 'txt':
                warnings.warn("Saving to .txt does not preserve metadata. Use .npz instead.")
                save_txt_2D(self.values, Y, X, path, **kwargs)
            case 'tar':
                warnings.warn("Saving to .tar does not preserve metadata. Use .npz instead.")
                save_tar([self.values, Y, X], path)
            case 'mama':
                warnings.warn("MAMA format does not preserve metadata.")
                mama_write(self, path, comment="Made by OMpy", **kwargs)
            case _:
                raise ValueError(f"Unknown filetype: {filetype}")

    def cut_like(self, other: Matrix,
                 inplace: bool = False) -> Matrix | None:
        """ Cut a matrix like another matrix (according to energy arrays)

        Args:
            other (Matrix): The other matrix
            inplace (bool, optional): If True make the cut in place. Otherwise
                return a new matrix. Defaults to False.

nameReurns:
            Matrix | None: If inplace is False, returns the cut matrix
        """
        if inplace:
            self.cut('Ex', other.Ex.min(), other.Ex.max(), inplace=inplace)
            self.cut('Eg', other.Eg.min(), other.Eg.max(), inplace=inplace)
        else:
            out = self.cut('Ex', other.Ex.min(), other.Ex.max(),
                           inplace=inplace)
            if out is not None:
                out.cut('Eg', other.Eg.min(), other.Eg.max(), inplace=True)
            return out

    def cut_diagonal(self, p1: PointUnit | None = None, p2: PointUnit | None = None,
                     slope: float | None = None,
                     inplace: bool = False) -> Matrix | None:
        """Cut away counts to the right of a diagonal line defined by indices

        Args:
            E1: First point of intercept, ordered as (Eg, Ex)
            E2: Second point of intercept
            inplace: Whether the operation should be applied to the
                current matrix, or to a copy which is then returned.
                Defaults to `False`.

        Returns:
            The matrix with counts above diagonal removed (if inplace is
            False).
        """
        # TODO Fix by using detector resolution
        line = Line(p1=p1, p2=p2, slope=slope)
        mask = line.above(self)

        if inplace:
            self.values[mask] = 0.0
        else:
            matrix = self.clone()
            matrix.values[mask] = 0.0
            return matrix

    def trapezoid(self, Ex_min: float, Ex_max: float,
                  Eg_min: float, Eg_max: float | None = None,
                  inplace: bool = False) -> Matrix | None:
        """Create a trapezoidal cut or mask delimited by the diagonal of the
            matrix

        Args:
            Ex_min: The bottom edge of the trapezoid
            Ex_max: The top edge of the trapezoid
            Eg_min: The left edge of the trapezoid
            Eg_max: The right edge of the trapezoid used for defining the
               diagonal. If not set, the diagonal will be found by
               using the last nonzeros of each row.
        Returns:
            Cut matrix if 'inplace' is True

        TODO:
            -possibility to have inclusive or exclusive cut
            -Remove and move into geometry as Trapezoid
        """
        raise NotImplementedError()
        matrix = self.clone()
        Ex_min = self.to_same_Ex(Ex_min)
        Ex_max = self.to_same_Ex(Ex_max)
        Eg_min = self.to_same_Eg(Eg_min)

        matrix.cut("Ex", Emin=Ex_min, Emax=Ex_max, inplace=True)
        if Eg_max is None:
            lastEx = matrix[-1, :]
            try:
                iEg_max = np.nonzero(lastEx)[0][-1]
            except IndexError():
                raise ValueError("Last Ex column has no non-zero elements")
            Eg_max = matrix.Eg[iEg_max]
        Eg_max = self.to_same_Eg(Eg_max)

        matrix.cut("Eg", Emin=Eg_min, Emax=Eg_max, inplace=True)

        Eg, Ex = np.meshgrid(matrix.Eg, matrix.Ex)
        mask = np.zeros_like(matrix.values, dtype=bool)

        dEg = Eg_max - Ex_max
        if dEg > 0:
            binwidth = Eg[1]-Eg[0]
            dEg = np.ceil(dEg/binwidth) * binwidth
        mask[Eg >= Ex + dEg] = True
        matrix[mask] = 0

        if inplace:
            self.values = matrix.values
            self._Ex = matrix._Ex
            self._Eg = matrix._Eg
        else:
            return matrix

    @overload
    def rebin(self, axis: int | str, *,
              bins: Sequence[float] | None = None,
              factor: float | None = None,
              binwidth: Unitlike | None = None,
              numbins: int | None = None,
              inplace: Literal[False] = ...) -> Matrix: ...

    @overload
    def rebin(self, axis: int | str, *,
              bins: Sequence[float] | None = None,
              factor: float | None = None,
              binwidth: Unitlike | None = None,
              numbins: int | None = None,
              inplace: Literal[True] = ...) -> None: ...

    def rebin(self, axis: int | str, *,
              bins: Sequence[float] | None = None,
              factor: float | None = None,
              binwidth: Unitlike | None = None,
              numbins: int | None = None,
              preserve: str = 'counts',
              inplace: bool = False) -> Matrix | None:
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

        axis: AxisBoth = self.axis_to_int(axis, allow_both=True)
        if axis == 2:
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
        if axis == 0:
            bins: Index = self.X_index.handle_rebin_arguments(bins=bins, factor=factor, binwidth=binwidth, numbins=numbins)
            rebinned = rebin_2D(self.X_index, bins, self.values, axis=0, preserve=preserve)
            if inplace:
                self.values = rebinned
                self.X_index = bins
            else:
                return self.clone(X=bins, values=rebinned)
        else:
            bins: Index = self.Y_index.handle_rebin_arguments(bins=bins, factor=factor, binwidth=binwidth, numbins=numbins)
            rebinned = rebin_2D(self.Y_index, bins, self.values, axis=1, preserve=preserve)
            if inplace:
                self.values = rebinned
                self.Y_index = bins
            else:
                return self.clone(Y=bins, values=rebinned)

    @overload
    def fill_negative(self, window: numeric | array, inplace: bool = Literal[False]) -> Matrix: ...
    @overload
    def fill_negative(self, window: numeric | array, inplace: bool = Literal[True]) -> None: ...

    def fill_negative(self, window: numeric | array, inplace: bool = False) -> Matrix | None:
        """ Wrapper for :func:`ompy.fill_negative_gauss` """
        raise NotImplementedError()
        if not inplace:
            return self.clone(values=fill_negative_gauss(self.values, self.observed, window))
        self.values = fill_negative_gauss(self.values, self.observed, window)

    def remove_negative(self, inplace=False) -> Matrix | None:
        """ Entries with negative values are set to 0 """
        raise DeprecationWarning("Use matrix[matrix < 0] = 0 instead")
        raise NotImplementedError()
        if not inplace:
            return self.clone(values=np.where(self.values > 0, self.values, 0))
        self.values[self.values > 0] = np.where(self.values > 0, self.values, 0)

    @overload
    def fill_and_remove_negative(self, window: numeric | array, inplace=Literal[False]) -> Matrix: ...

    @overload
    def fill_and_remove_negative(self, window: numeric | array, inplace=Literal[True]) -> None: ...

    def fill_and_remove_negative(self, window: numeric | array = 20, inplace=False) -> Matrix | None:
        """ Combination of :meth:`ompy.Matrix.fill_negative` and
        :meth:`ompy.Matrix.remove_negative`

        Args:
            window: See `fill_negative`. Defaults to 20 (arbitrary)!.
            inplace: Whether to change the matrix inplace or return a modified copy.
            """
        if window == 20:
            warnings.warn("Window size 20 is arbitrary. Consider setting it to an informed value > 0")
        if not inplace:
            clone: Matrix = self.fill_negative(window, inplace=False)
            clone[clone < 0] = 0
            return clone
        self.fill_negative(window=window, inplace=True)
        self.values[self.values < 0] = 0

    def index_X(self, x: float) -> int:
        return self.X_index.index(x)

    def index_Y(self, x: float) -> int:
        return self.Y_index.index(x)

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

    def to_left(self, axis: int | str = 'both', inplace: bool = False) -> None | Matrix:
        """ Returns a copy with the bins set to the left edges of the bins.

        Args:
            axis: The axis to transform. Defaults to both.
            inplace: Change the matrix inplace or return a copy.
        Returns:
            A copy of the matrix with the bins set to the left edges of the bins.
        """
        return self.to_edge('left', axis=axis, inplace=inplace)

    def to_edge(self, edge: Literal['left', 'mid'], axis: int | str = 'both', inplace: bool = False) -> None | Matrix:
        """ Returns a copy with the bins set to the left or mid edges of the bins.

        Args:
            edge: The edge to transform to. Either 'left' or 'mid'.
            axis: The axis to transform. Defaults to both.
            inplace: Change the matrix inplace or return a copy.
        Returns:
            A copy of the matrix with the bins set to the left or mid edges of the bins.
        """
        axis: AxisBoth = self.axis_to_int(axis, allow_both=True)
        xindex = self.X_index
        yindex = self.Y_index
        match axis:
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


    def set_order(self, order: str) -> None:
        self.values = self.values.copy(order=order)
        self.X_index = self.X_index.clone(order=order)
        self.Y_index = self.Y_index.clone(order=order)

    @property
    def dX(self) -> float | np.ndarray:
        return self.X_index.steps()

    @property
    def dY(self) -> float | np.ndarray:
        return self.Y_index.steps()

    def from_mask(self, mask: ArrayBool) -> Matrix | Vector:
        """ Returns a copy of the matrix with only the rows and columns  where `mask` is True.

        A Vector is returned if the matrix is 1D.
        """
        raise NotImplementedError("This method is not implemented yet.")
        if mask.shape != self.shape:
            raise ValueError("Mask must have same shape as matrix.")

        # Compute the bounding box of True values in mask
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        if rmin == rmax:
            if cmin == cmax:
                raise ValueError("Mask is a scalar. Use ordinary indexing.")
            return Vector(values = self.values[rmin, cmin:cmax+1], E=self.Ex[rmin:rmax+1])
        elif cmin == cmax:
            return Vector(values = self.values[rmin:rmax+1, cmin], E=self.Eg[cmin:cmax+1])
        values = self.values[rmin:rmax+1, cmin:cmax+1]
        return self.__class__(values, Ex=self.Ex[rmin:rmax+1], Eg=self.Eg[cmin:cmax+1])

    @property
    def T(self) -> Matrix:
        values = self.values.T
        std = None
        if self.std is not None:
            std = self.std.T
        return self.clone(values=values, std=std, X=self.Y_index, Y=self.X_index)

    @property
    def _summary(self) -> str:
        s = f'X index:\n{self.X_index.summary()}\n'
        s += f'Y index:\n{self.Y_index.summary()}\n'
        if len(self.metadata.misc) > 0:
            s += 'Metadata:\n'
            for key, val in self.metadata.misc.items():
                s += f'\t{key}: {val}\n'
        s += f'Total counts: {self.sum():.3g}'
        return s

    def summary(self):
        print(self._summary)

    def sum(self, axis: int | str = 'both') -> Vector | float:
        axis: AxisBoth = self.axis_to_int(axis, allow_both=True)
        if axis == 2:
            return self.values.sum()
        values = self.values.sum(axis=axis)
        index = self.X_index if axis else self.Y_index
        return self.meta_into_vector(index=index, values=values)


    def __str__(self) -> str:
        summary = self._summary
        summary += "\nValues:\n" 
        if self.std is not None:
            return summary+str(self.values)+'\nStd: \n'+str(self.std)
        else:
            return summary+str(self.values)

    @property
    def X(self) -> np.ndarray:
        return self.X_index.bins

    @property
    def Y(self) -> np.ndarray:
        return self.Y_index.bins

    def clone(self, X: Index | None = None, Y: Index | None = None,
              values: np.ndarray | None = None, std: np.ndarray | None = None,
              metadata: MatrixMetadata | None = None, copy: bool = False,
              **kwargs) -> Matrix:
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
        std  = std if std is not None else self.std
        metadata = metadata if metadata is not None else self.metadata
        metadata = metadata.update(**kwargs)
        return Matrix(X=X, Y=Y, values=values, std=std, metadata=metadata, copy=copy)

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

    def normalize(self, axis: str | Literal[0, 1, 2], inplace=False) -> Matrix | None:
        axis: AxisBoth = self.axis_to_int(axis, allow_both=True)
        if not inplace:
            match axis:
                case 0:
                    s = self.values.sum(axis=0)
                    s[s == 0] = 1
                    values = self.values/s[np.newaxis, :]
                case 1:
                    s = self.values.sum(axis=1)
                    s[s == 0] = 1
                    values = self.values/s[:, np.newaxis]
                case 2:
                    s = sum(self.values)
                    values = self.values/s
            return self.clone(values=values)
        else:
            match axis:
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
    def plot(self, *, ax: Axes | None = None,
             scale: str | None = None,
             vmin: float | None = None,
             vmax: float | None = None,
             add_cbar: Literal[True] = ...,
             **kwargs) -> (Axes, (QuadMesh, Colorbar)): ...

    @overload
    def plot(self, *, ax: Axes | None = None,
             scale: str | None = None,
             vmin: float | None = None,
             vmax: float | None = None,
             add_cbar: Literal[False] = ...,
             **kwargs) -> (Axes, (QuadMesh, None)): ...

    def plot(self, *, ax: Axes | None = None,
             scale: str | None = None,
             vmin: float | None = None,
             vmax: float | None = None,
             add_cbar: bool = True,
             **kwargs) -> (Axes, (QuadMesh, Colorbar | None)):
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
        fig, ax = plt.subplots() if ax is None else (ax.figure, ax)
        assert ax is not None
        assert isinstance(fig, Figure)

        if scale is None:
            scale = 'log' if self.sum() > 1000 else 'linear'
        if scale == 'log':
            if vmin is not None and vmin <= 0:
                raise ValueError("`vmin` must be positive for log-scale")
            if vmin is None:
                _max = np.log10(self.max())
                _min = np.log10(self.values[self.values > 0].min())
                if _max - _min > 10:
                    vmin = 10**(int(_max-6))
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
        if self.X_index.is_mid():
            X = self.X_index.to_left().ticks()
        else:
            X = self.X_index.ticks()

        if self.Y_index.is_mid():
            Y = self.Y_index.to_left().ticks()
        else:
            Y = self.Y_index.ticks()

        # Set entries of 0 to white
        current_cmap = copy.copy(cm.get_cmap())
        current_cmap.set_bad(color='white')
        kwargs.setdefault('cmap', current_cmap)
        mask = np.isnan(self.values) | (self.values == 0)
        masked = np.ma.array(self.values, mask=mask)

        mesh = ax.pcolormesh(Y, X, masked, norm=norm, **kwargs)
        if self.Y_index.is_mid():
            ax.xaxis.set_major_locator(MeshLocator(self.Y))
            ax.tick_params(axis='x', rotation=40)
        if self.X_index.is_mid():
            ax.yaxis.set_major_locator(MeshLocator(self.X))
        #if midbin_ticks:
        #    ax.xaxis.set_major_locator(MeshLocator(self.X))
        #    ax.tick_params(axis='x', rotation=40)
        #    ax.yaxis.set_major_locator(MeshLocator(self.Y))
        # ax.xaxis.set_major_locator(ticker.FixedLocator(self.Eg, nbins=10))
        # fix_pcolormesh_ticks(ax, xvalues=self.Eg, yvalues=self.Ex)

        maybe_set(ax, 'title', self.name)
        maybe_set(ax, 'ylabel', self.xlabel + f" [${self.X_index.unit:~L}$]")
        maybe_set(ax, 'xlabel', self.ylabel + f" [${self.Y_index.unit:~L}$]")

        # show z-value in status bar
        # https://stackoverflow.com/questions/42577204/show-z-value-at-mouse-pointer-position-in-status-line-with-matplotlibs-pcolorme
        def format_coord(x, y):
            xarr = X
            yarr = Y
            if ((x > xarr.min()) & (x <= xarr.max())
                    & (y > yarr.min()) & (y <= yarr.max())):
                col = np.searchsorted(xarr, x)-1
                row = np.searchsorted(yarr, y)-1
                z = masked[row, col]
                return f'x={x:1.2f}, y={y:1.2f}, z={z:1.2E}'
                # return f'x={x:1.0f}, y={y:1.0f}, z={z:1.3f}   [{row},{col}]'
            else:
                return f'x={x:1.0f}, y={y:1.0f}'
        # TODO: Takes waaaay to much CPU
        def nop(x, y):
            return ''
        #ax.format_coord = nop

        cbar: Colorbar | None = None
        if add_cbar:
            if vmin is not None and vmax is not None:
                cbar = fig.colorbar(mesh, ax=ax, extend='both')
            elif vmin is not None:
                cbar = fig.colorbar(mesh, ax=ax, extend='min')
            elif vmax is not None:
                cbar = fig.colorbar(mesh, ax=ax, extend='max')
            else:
                cbar = fig.colorbar(mesh, ax=ax)

            maybe_set(cbar.ax, 'ylabel', self.vlabel)
        return ax, (mesh, cbar)

    def meta_into_vector(self, index: np.ndarray | Index, values: np.ndarray) -> Vector:
        return Vector(X=index, values=values, vlabel=self.vlabel, valias=self.valias, name=self.name)

    @overload
    def axis_to_int(self, axis: int | str, allow_both: bool = Literal[True]) -> AxisEither: ...

    @overload
    def axis_to_int(self, axis: int | str, allow_both: bool = Literal[False]) -> AxisBoth: ...

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

    @property
    def ylabel(self) -> str:
        return self.Y_index.label

    @overload
    def __matmul__(self, other: Matrix) -> Matrix: ...
    @overload
    def __matmul__(self, other: Vector) -> Vector: ...
    @overload
    def __matmul__(self, other: np.ndarray) -> np.ndarray: ...

    def __matmul__(self, other: Matrix | Vector | np.ndarray) -> Matrix | Vector | np.ndarray:
        match other:
            case Matrix():
                if self.shape[1] != other.shape[0]:
                    raise ValueError(f"Shape mismatch {self.shape} @ {other.shape}")
                if not self.is_compatible_with_Y(other.X_index):
                    raise ValueError(f"Y index mismatch {self.Y_index} @ {other.X_index}")
                return Matrix(X=self.X_index, Y=other.Y_index, values=self.values @ other.values)
            case Vector():
                if self.shape[1] != other.shape[0]:
                    raise ValueError(f"Shape mismatch {self.shape} @ {other.shape}")
                if not self.is_compatible_with_Y(other._index):
                    raise ValueError(f"Y index mismatch {self.Y_index} @ {other._index}")
                return self.meta_into_vector(self.X_index, self.values @ other.values)
            case _:
                return self.values @ other


class IndexLocator:
    def __init__(self, matrix: Matrix):
        self.mat = matrix

    def __getitem__(self, key) -> Matrix | Vector | float:
        if isinstance(key, array):
            return self.linear_index(key)
        xi, yi = key
        values = self.mat.values.__getitem__(key)
        if isinstance(xi, slice):
            X = self.mat.X.__getitem__(xi)
            if isinstance(yi, slice):
                Y = self.mat.Y.__getitem__(yi)
                return self.mat.clone(values=values, X=X, Y=Y)
            return self.mat.into_vector(X, values)
        elif isinstance(yi, slice):
            Y = self.mat.Y.__getitem__(yi)
            return self.mat.into_vector(Y, values)
        else:
            return values

    def linear_index(self, indices) -> Matrix:
        values = np.where(indices, self.mat.values, 0)
        return self.mat.clone(values=values)


class ValueLocator:
    def __init__(self, matrix: Matrix, strict: bool = True):
        self.mat = matrix
        self.strict = strict

    def __getitem__(self, key) -> Matrix | Vector | float:
        match key:
            case slice() as x, slice() as y:
                sx: slice = self.mat.X_index.index_slice(x, strict=self.strict)
                sy: slice = self.mat.Y_index.index_slice(y, strict=self.strict)
                xindex = self.mat.X_index[sx]
                yindex = self.mat.Y_index[sy]
                values = self.mat.values.__getitem__((sx, sy))
                std = None
                if self.mat.std is not None:
                    std = self.mat.std.__getitem__((sx, sy))
                return self.mat.clone(values=values, X=xindex, Y=yindex, std=std)
            case slice() as x, y:
                sx: slice = self.mat.X_index.index_slice(x, strict=self.strict)
                xindex = self.mat.X_index[sx]
                j: int = self.mat.Y_index.index_expression(y, strict=self.strict)
                values = self.mat.values.__getitem__((sx, j))
                return self.mat.into_vector(xindex, values)
            case x, slice() as y:
                sy: slice = self.mat.Y_index.index_slice(y, strict=self.strict)
                yindex = self.mat.Y_index[sy]
                i: int = self.mat.X_index.index_expression(x, strict=self.strict)
                values = self.mat.values.__getitem__((i, sy))
                return self.mat.into_vector(yindex, values)
            case x, y:
                i: int = self.mat.X_index.index_expression(x, strict=self.strict)
                j: int = self.mat.Y_index.index_expression(y, strict=self.strict)
                return self.mat.values.__getitem__((i, j))

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
        step = max(int(np.ceil((imax-imin) / self.nbins)), 1)
        ticks = self.locs[imin:imax+1:step]
        if vmax - vmin > 0.8*(dmax - dmin) and imax-imin > 20:
            # Round to the nearest "nicest" number
            # TODO Could be improved by taking vmin into account
            i = min(int(np.log10(abs(self.locs[imax]))), 2)
            i = max(i, 1)
            ticks = np.unique(np.around(ticks, -i))
        return self.raise_if_exceeds(ticks)
