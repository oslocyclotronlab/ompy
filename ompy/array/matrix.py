from __future__ import annotations

import copy
import logging
import warnings
from ctypes import ArgumentError
from pathlib import Path
from typing import (Any, Dict, Iterable, Iterator, Sequence, Union, Callable, overload, Literal, Tuple)

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker
from matplotlib.colors import LogNorm, Normalize, SymLogNorm

from .. import ureg
from .abstractarray import AbstractArray, to_plot_axis
from .filehandling import (filetype_from_suffix, load_numpy_2D, load_tar,
                           load_txt_2D, mama_read, mama_write, save_numpy_2D,
                           save_tar, save_txt_2D)
from ..geometry import Line
from ..library import (diagonal_elements, div0, fill_negative_gauss,
                      handle_rebin_arguments)
from .matrixstate import MatrixState
from .rebin import rebin_2D
from ..stubs import (Unitlike, Pathlike, ArrayKeV, Axes, Figure,
                    Colorbar, QuadMesh, ArrayInt, PointUnit, array, arraylike, ArrayBool, numeric)
from .vector import Vector
from .index_fn import index_left as index

LOG = logging.getLogger(__name__)
logging.captureWarnings(True)

#TODO mat*vec[:, None[ doesn't work

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
    def __init__(self,
                 values: np.ndarray | None = None,
                 Eg: arraylike | None = None,
                 Ex: arraylike | None = None,
                 std: np.ndarray | None = None,
                 path: Pathlike | None = None,
                 state: Union[str, MatrixState] | None = None,
                 Ex_units: Unitlike = 'keV',
                 Eg_units: Unitlike = 'keV',
                 copy: bool = True,
                 xlabel: str = r"$\gamma$-energy $E_{\gamma}$",
                 ylabel: str = r"Excitation energy $E_{x}$"):
        """
        There is the option to initialize it in an empty state.
        In that case, all class variables will be None.
        It can be filled later using the load() method.

        For initializing one can give values, [Ex, Eg] arrays,
        a filename for loading a saved matrix or a shape
        for initializing it with empty entries.

        Args:
            values: Set the matrix' values.
            Eg: The gamma ray energies using midbinning.
            Ex: The excitation energies using midbinning.
            std: The standard deviations at each bin of `values`
            path: Load a Matrix from a given path
            state: An enum to keep track of what has been done to the matrix.
                Can also be a str. like in ["raw", "unfolded", ...]
            copy: Whether to copy the arguments or take them as reference.
                Defaults to `True`.

        """
        if copy:
            def fetch(x):
                return np.asarray(x, dtype=float).copy()
        else:
            def fetch(x):
                return np.asarray(x, dtype=float)

        self.std: np.ndarray | None = None
        if path is not None:
            if (values is not None or
                Eg is not None or
                Ex is not None):
                LOG.warning(("Loading from path. Other arguments"
                             "are ignored"))
            self.load(path)
        else:
            if (values is None or
                Eg is None or
                Ex is None):
                raise ValueError("Provide either values and energies, or path.")
            try:
                Eg_unit = Eg.units
                self._Eg: ArrayKeV = np.atleast_1d(fetch(Eg.magnitude))*Eg_unit
            except AttributeError:
                #Eg_unit = ureg.Quantity(Eg_units)
                #Eg_unit = ureg(Eg_units)
                Eg_unit = ureg.Unit(Eg_units)
                self._Eg: ArrayKeV = np.atleast_1d(fetch(Eg))*Eg_unit

            try:
                Ex_unit = Ex.units
                self._Ex: ArrayKeV = np.atleast_1d(fetch(Ex.magnitude))*Ex_unit
            except AttributeError:
                #Ex_unit = ureg(Ex_units)
                #Ex_unit = ureg.Quantity(Ex_units)
                Ex_unit = ureg.Unit(Ex_units)
                self._Ex: ArrayKeV = np.atleast_1d(fetch(Ex)) * Ex_unit

            self.values = np.atleast_2d(fetch(values))

        if std is not None:
            self.std = np.atleast_2d(fetch(std))

        self.state = state
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.loc = ValueLocator(self)
        self.iloc = IndexLocator(self)

        self.verify_integrity()

    def verify_integrity(self, check_equidistant: bool = False):
        """ Runs checks to verify internal structure

        Args:
            check_equidistant (bool, optional): Check whether energy array
                are equidistant spaced. Defaults to False.

        Raises:
            ValueError: If any check fails
        """
        if self.values.ndim != 2:
            raise ValueError(f"Values must be ndim 2, not {self.values.ndim}.")

        # Check shapes:
        shape = self.values.shape
        if self.Ex.ndim == 1:
            if shape[0] != len(self.Ex):
                raise ValueError(("Shape mismatch between matrix and Ex:"
                                  f" (_{shape[0]}_, {shape[1]}) ≠ "
                                  f"{len(self.Ex)}"))
        else:
            raise ValueError(f"Ex array must be ndim 1, not {self.Ex.ndim}")

        if self.Eg.ndim == 1:
            if shape[1] != len(self.Eg):
                raise ValueError(("Shape mismatch between matrix and Eg:"
                                  f" ({shape[0]}, _{shape[1]}_) ≠ "
                                  f"{len(self.Eg)}"))
        else:
            raise ValueError(f"Eg array must be ndim 1, not {self.Eg.ndim}")

        if check_equidistant and not self.is_equidistant():
            raise ValueError("Is not equidistant.")

        if self.std is not None:
            if shape != self.std.shape:
                raise ValueError("Shape mismatch between self.values and std.")

    def is_equidistant(self) -> bool:
        if len(self) <= 1:
            return True

        diff_Ex = (self.Ex - np.roll(self.Ex, 1))[1:]
        diff_Eg = (self.Eg - np.roll(self.Eg, 1))[1:]
        dEx = diff_Ex[0]
        dEg = diff_Eg[0]
        return (np.all(np.isclose(diff_Ex, dEx*np.ones_like(diff_Ex))) and
                np.all(np.isclose(diff_Eg, dEg*np.ones_like(diff_Eg))))

    def load(self, path: Union[str, Path],
             filetype: str | None = None) -> None:
        """ Load matrix from specified format

        Args:
            path (str or Path): path to file to load
            filetype (str, optional): Filetype to load. Has an
                auto-recognition.

        Raises:
            ValueError: If filetype is unknown
        """
        path = Path(path) if isinstance(path, str) else path
        if filetype is None:
            filetype = filetype_from_suffix(path)
        filetype = filetype.lower()

        if filetype == 'numpy':
            self.values, self._Eg, self._Ex = load_numpy_2D(path)
        elif filetype == 'txt':
            self.values, self._Eg, self._Ex = load_txt_2D(path)
        elif filetype == 'tar':
            self.values, self._Eg, self._Ex = load_tar(path)
        elif filetype == 'mama':
            self.values, self._Eg, self._Ex = mama_read(path)
        else:
            try:
                self.values, self._Eg, self._Ex = mama_read(path)
            except ValueError:  # from within mama_read
                raise ValueError(f"Unknown filetype {filetype}")
        self._Eg *= ureg('keV')
        self._Ex *= ureg('keV')
        self.verify_integrity()

    def save(self, path: Union[str, Path], filetype: str | None = None,
             which: str | None = 'values', **kwargs):
        """Save matrix to file

        Args:
            path (str or Path): path to file to save
            filetype (str, optional): Filetype to save. Has an
                auto-recognition. Options: ["numpy", "tar", "mama", "txt"]
            which (str, optional): Which attribute to save. Default is
                'values'. Options: ["values", "std"]
            **kwargs: additional keyword arguments
        Raises:
            ValueError: If filetype is unknown
            RuntimeError: If `std` attribute not set.
            NotImplementedError: If which is unknown
        """
        path = Path(path) if isinstance(path, str) else path
        if filetype is None:
            filetype = filetype_from_suffix(path)
        filetype = filetype.lower()

        values = None
        if which.lower() == 'values':
            values = self.values
            if self.std is not None:
                warnings.warn(UserWarning("The std attribute of Matrix class has to be saved to file 'manually'. Call with which='std'."))  # noqa
        elif which.lower() == 'std':
            if self.std is None:
                raise RuntimeError(f"Attribute `std` not set.")
            values = self.std
        else:
            raise NotImplementedError(
                f"{which} is unsupported: Use 'values' or 'std'")

        Eg = self._Eg.to('keV').magnitude
        Ex = self._Ex.to('keV').magnitude
        if filetype == 'numpy':
            save_numpy_2D(values, Eg, Ex, path)
        elif filetype == 'txt':
            save_txt_2D(values, Eg, Ex, path, **kwargs)
        elif filetype == 'tar':
            save_tar([values, Eg, Ex], path)
        elif filetype == 'mama':
            if which.lower() == 'std':
                warnings.warn(UserWarning(
                    "Cannot write std attrbute to MaMa format."))

            mama_write(self, path, comment="Made by OMpy",
                       **kwargs)
        else:
            raise ValueError(f"Unknown filetype {filetype}")

    def calibration(self) -> Dict[str, np.ndarray]:
        """ Calculates the calibration coefficients of the energy axes

        Returns:
            The calibration coefficients in a dictionary.
        """
        calibration = {
            "a0x": self._Ex.to('keV').magnitude[0],
            "a1x": self._Ex.to('keV').magnitude[1]-self._Ex.to('keV').magnitude[0],
            "a0y": self._Eg.to('keV').magnitude[0],
            "a1y": self._Eg.to('keV').magnitude[1]-self._Eg.to('keV').magnitude[0],
        }
        return calibration

    def same_shape(self, other: array, error: bool = False) -> bool:
        """ Check whether `other` has same shape `self.values`.

        Args:
            other (array-like): Object to compare to.

        Returns:
            Returns `True` if shapes are equal.
        Raises:
            ValueError: if shapes aren't equal and `error` is true.
            TypeError: if `other` lacks `.shape`
        """
        return True  # Bug: Buggy implementation
        try:
            same = other.shape == self.shape
        except AttributeError:
            if error:
                raise TypeError("Other needs to be array-like.")
            else:
                return False

        if error and not same:
            raise ValueError(f"Expected {self.shape}, got {other.shape}.")
        return same


    @overload
    def plot(self, *, ax: Axes | None = None,
             title: str | None = None,
             scale: str | None = None,
             vmin: float | None = None,
             vmax: float | None = None,
             midbin_ticks: bool = False,
             add_cbar: Literal[True] = ...,
             **kwargs) -> (Axes, (QuadMesh, Colorbar)): ...

    @overload
    def plot(self, *, ax: Axes | None = None,
             title: str | None = None,
             scale: str | None = None,
             vmin: float | None = None,
             vmax: float | None = None,
             midbin_ticks: bool = False,
             add_cbar: Literal[False] = ...,
             **kwargs) -> (Axes, (QuadMesh, None)): ...

    def plot(self, *, ax: Axes | None = None,
             title: str | None = None,
             scale: str | None = None,
             vmin: float | None = None,
             vmax: float | None = None,
             midbin_ticks: bool = False,
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

        if len(self.Ex) <= 1 or len(self.Eg) <= 1:
            raise ValueError("Number of bins must be greater than 1")

        if scale is None:
            scale = 'log' if self.counts > 1000 else 'linear'
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
        Eg, Ex = self.plot_bins()

        # Set entries of 0 to white
        current_cmap = copy.copy(cm.get_cmap())
        current_cmap.set_bad(color='white')
        kwargs.setdefault('cmap', current_cmap)
        mask = np.isnan(self.values) | (self.values == 0)
        masked = np.ma.array(self.values, mask=mask)

        mesh = ax.pcolormesh(Eg, Ex, masked, norm=norm, **kwargs)
        if midbin_ticks:
            ax.xaxis.set_major_locator(MeshLocator(self.Eg))
            ax.tick_params(axis='x', rotation=40)
            ax.yaxis.set_major_locator(MeshLocator(self.Ex))
        # ax.xaxis.set_major_locator(ticker.FixedLocator(self.Eg, nbins=10))
        # fix_pcolormesh_ticks(ax, xvalues=self.Eg, yvalues=self.Ex)

        ax.set_title(title if title is not None else self.state)
        ax.set_xlabel(self.xlabel + f" [${self.Eg_units:~L}$]")
        ax.set_ylabel(self.ylabel + f" [${self.Ex_units:~L}$]")

        # show z-value in status bar
        # https://stackoverflow.com/questions/42577204/show-z-value-at-mouse-pointer-position-in-status-line-with-matplotlibs-pcolorme
        def format_coord(x, y):
            xarr = Eg
            yarr = Ex
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
        ax.format_coord = nop

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

            # cbar.ax.set_ylabel("# counts")
            # plt.show()
        return ax, (mesh, cbar)

    def plot_bins(self):
        # Move all bins down to lower bins
        self.to_lower_bin()
        # Must extend it the range to make pcolormesh happy
        dEg = self.Eg[-1] - self.Eg[-2]
        dEx = self.Ex[-1] - self.Ex[-2]
        Eg = np.append(self.Eg, self.Eg[-1] + dEg)
        Ex = np.append(self.Ex, self.Ex[-1] + dEx)
        # Move the bins back up
        self.to_mid_bin()
        return Eg, Ex

    def cut(self, axis: Union[int, str],
            Emin: float | None = None,
            Emax: float | None = None,
            inplace: bool = False,
            Emin_inclusive: bool = True,
            Emax_inclusive: bool = True) -> Matrix | None:
        """Cuts the matrix to the sub-interval limits along given axis.

        Args:
            axis: Which axis to apply the cut to.
                Can be 0, "Eg" or 1, "Ex".
            Emin: Lowest energy to be included. Defaults to
                lowest energy. Inclusive by default.
            Emax: Higest energy to be included. Defaults to
                highest energy. Inclusive by default.
            inplace: If True make the cut in place. Otherwise return a new
                matrix. Defaults to False.
            Emin_inclusive: whether the bin containing the lower bin
                should be included (True) or excluded (False).
                Defaults to True.
            Emax_inclusive: whether the bin containing the higest bin
                should be included (True) or excluded (False).
                Defaults to True.

        Returns:
            None if inplace == False
            cut_matrix (Matrix): The cut version of the matrix
        """
        warnings.warn("To be deprecated in favor of mat[i:j, n:m]", DeprecationWarning)
        axis = to_plot_axis(axis)
        range = self.Eg if axis == 0 else self.Ex
        index = self.index_Eg if axis == 0 else self.index_Ex
        magnitude = self.to_same_Eg if axis == 0 else self.to_same_Ex
        Emin = magnitude(Emin) if Emin is not None else min(range)
        Emax = magnitude(Emax) if Emax is not None else max(range)

        iEmin = index(Emin)
        if range[iEmin] < Emin:
            iEmin += 1
        if not Emin_inclusive:
            iEmin += 1

        iEmax = index(Emax)
        if range[iEmax] > Emax:
            iEmax -= 1
        if not Emax_inclusive:
            iEmax -= 1

        # Fix for boundaries
        if iEmin >= len(range):
            iEmin = len(range)-1
        if iEmax <= 0:
            iEmax = 0
        iEmax += 1  # Because of slicing
        Eslice = slice(iEmin, iEmax)
        Ecut = range[Eslice]

        if axis == 0:
            values_cut = self.values[:, Eslice]
            Eg = Ecut*self.Eg_units
            Ex = self._Ex
        elif axis == 1:
            values_cut = self.values[Eslice, :]
            Ex = Ecut*self.Eg_units
            Eg = self._Eg
        else:
            raise ValueError("`axis` must be 0 or 1.")

        if inplace:
            self.values = values_cut
            self._Ex = Ex
            self._Eg = Eg
        else:
            return self.clone(values=values_cut, Eg=Eg, Ex=Ex)

    def cut_like(self, other: Matrix,
                 inplace: bool = False) -> Matrix | None:
        """ Cut a matrix like another matrix (according to energy arrays)

        Args:
            other (Matrix): The other matrix
            inplace (bool, optional): If True make the cut in place. Otherwise
                return a new matrix. Defaults to False.

        Returns:
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
    def rebin(self, axis: int | str,
              bins: Sequence[float] | None = None,
              factor: float | None = None,
              binwidth: Unitlike | None = None,
              numbins: int | None = None,
              inplace: Literal[False] = ...) -> Matrix: ...

    @overload
    def rebin(self, axis: int | str,
              bins: Sequence[float] | None = None,
              factor: float | None = None,
              binwidth: Unitlike | None = None,
              numbins: int | None = None,
              inplace: Literal[True] = ...) -> None: ...

    def rebin(self, axis: int | str,
              bins: Sequence[float] | None = None,
              factor: float | None = None,
              binwidth: Unitlike | None = None,
              numbins: int | None = None,
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

        axis: int = to_plot_axis(axis)
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
                return new.rebin(axis=1, bins=bins, factor=factor,
                                 binwidth=binwidth, inplace=False,
                                 numbins=numbins)

        oldbins = self.Ex if axis else self.Eg
        transform = self.to_same_Ex if axis else self.to_same_Eg
        unit = self.Ex_units if axis else self.Eg_units

        newbins = handle_rebin_arguments(bins=oldbins, transform=transform,
                                         LOG=LOG, factor=factor, newbins=bins,
                                         binwidth=binwidth, numbins=numbins)

        naxis = (axis + 1) % 2
        rebinned = rebin_2D(self.values, oldbins, newbins, naxis)


        if inplace:
            self.values = rebinned
            if axis:
                self._Ex = newbins * unit
            else:
                self._Eg = newbins * unit
            self.verify_integrity()
        else:
            if naxis:
                return self.clone(Eg=newbins*unit, values=rebinned)
            else:
                return self.clone(Ex=newbins*unit, values=rebinned)

    def diagonal_elements(self) -> Iterator[(int, int)]:
        """ Iterates over the last non-zero elements
        Note:
            Assumes that the matrix is diagonal, i.e. that there are no
            entries with `Eg > Ex + dE`.
        Args:
            mat: The matrix to iterate over
            Iterator[(int, int]): Indicies (i, j) over the last
                non-zero(=diagonal) elements.
        """
        return diagonal_elements(self.values)

    @overload
    def fill_negative(self, window: numeric | array, inplace: bool = Literal[False]) -> Matrix: ...
    @overload
    def fill_negative(self, window: numeric | array, inplace: bool = Literal[True]) -> None: ...

    def fill_negative(self, window: numeric | array, inplace: bool = False) -> Matrix | None:
        """ Wrapper for :func:`ompy.fill_negative_gauss` """
        if not inplace:
            return self.clone(values=fill_negative_gauss(self.values, self.Eg, window))
        self.values = fill_negative_gauss(self.values, self.Eg, window)

    def remove_negative(self, inplace=False) -> Matrix | None:
        """ Entries with negative values are set to 0 """
        raise DeprecationWarning("Use matrix[matrix < 0] = 0 instead")
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

    def index_Eg(self, E: Unitlike) -> int:
        """ Returns the closest index corresponding to the Eg value """
        E = self.to_same_Eg(E)
        # Almost as fast, but numba is much faster(!?)
        return index(self.Eg, E) #np.searchsorted(self.Eg, E, side='left')

    def index_Ex(self, E: Unitlike) -> int:
        """ Returns the closest index corresponding to the Ex value """
        E = self.to_same_Ex(E)
        return index(self.Ex, E) #np.searchsorted(self.Ex, E, side='left')

    def index_Eg_extended(self, E: Unitlike) -> float:
        """ Assumes a continuous mapping R[index] -> R[energy] """
        E = self.to_same_Eg(E)
        return (E-self.Eg[0])/self.dEg

    def index_Ex_extended(self, E: Unitlike) -> float:
        """ Assumes a continuous mapping R[index] -> R[energy] """
        E = self.to_same_Eg(E)
        return (E-self.Ex[0])/self.dEx

    def indices_Eg(self, E: Iterable[Unitlike]) -> ArrayInt:
        """ Returns the closest indices corresponding to the Eg value"""
        e = [self.to_same_Eg(e_) for e_ in E]
        return np.searchsorted(self.Eg, e)

    def indices_Ex(self, E: Iterable[Unitlike]) -> ArrayInt:
        """ Returns the closest indices corresponding to the Ex value"""
        e = [self.to_same_Ex(e_) for e_ in E]
        return np.searchsorted(self.Ex, e)

    @property
    def range_Eg(self) -> np.ndarray:
        """ Returns all indices of Eg """
        return np.arange(0, len(self.Eg), dtype=int)

    @property
    def range_Ex(self) -> np.ndarray:
        """ Returns all indices of Ex """
        return np.arange(0, len(self.Ex), dtype=int)

    @property
    def shape(self) -> (int, int):
        return self.values.shape

    @property
    def counts(self) -> float:
        return self.values.sum()

    @property
    def state(self) -> MatrixState:
        return self._state

    @state.setter
    def state(self, state: Union[str, MatrixState]) -> None:
        if state is None:
            self._state = None
        elif isinstance(state, str):
            self._state = MatrixState.str_to_state(state)
        # Buggy. Impossible to compare type of Enum??
        elif type(state) == type(MatrixState.RAW):
            self._state = state
        else:
            raise ValueError(f"state must be str or MatrixState"
                             f". Got {type(state)}")

    def to_same_Ex(self, E: Unitlike) -> float:
        """ Convert the units of E to the unit of `Ex` and return magnitude.

        Args:
            E: Convert its units to the same units of `Ex`.
               If `E` is dimensionless, assume to be of the same unit
               as `Ex`.
        """
        E = ureg.Quantity(E)
        if not E.dimensionless:
            E = E.to(self.Ex_units)
        return E.magnitude

    def to_same_Eg(self, E: Unitlike) -> float:
        """ Convert the units of E to the unit of `Eg` and return magnitude.

        Args:
            E: Convert its units to the same units of `Eg`.
               If `E` is dimensionless, assume to be of the same unit
               as `Eg`.
        """
        E = ureg.Quantity(E)
        if not E.dimensionless:
            E = E.to(self.Eg_units)
        return E.magnitude

    def to(self, unit: str, inplace: bool = False) -> Matrix:
        """ Returns a copy with units set to `unit`.

        Args:
            unit: The unit to transform to.
        Returns:
            A copy of the matrix with the unit of `Ex` and
            `Eg` set to `unit`.
        """
        if inplace:
            self._Eg = self._Eg.to(unit)
            self._Ex = self._Ex.to(unit)
            return self

        new = self.clone(Ex=self._Ex.to(unit), Eg=self._Eg.to(unit))
        return new

    def to_lower_bin(self):
        """ Transform Eg and Ex from mid bin (=default) to lower bin. """
        dEx = (self._Ex[1] - self._Ex[0])/2
        dEg = (self._Eg[1] - self._Eg[0])/2
        self._Ex -= dEx
        self._Eg -= dEg

    def to_mid_bin(self) -> Matrix:
        """ Transform Eg and Ex from lower bin to mid bin (=default). """
        dEx = (self._Ex[1] - self._Ex[0])/2
        dEg = (self._Eg[1] - self._Eg[0])/2
        self._Ex += dEx
        self._Eg += dEg
        return self

    def set_order(self, order: str) -> Matrix:
        self.values = self.values.copy(order=order)
        self._Ex = self._Ex.copy(order=order)
        self._Eg = self._Eg.copy(order=order)
        return self

    def iter(self) -> Iterator[(int, int)]:
        for row in range(self.shape[0]):
            for col in range(self.shape[1]):
                yield row, col

    def has_equal_binning(self, other, **kwargs) -> bool:
        """ Check whether `other` has equal binning as `self` within precision.
        Args:
            other (Matrix): Matrix to compare to.
            **kwargs: Additional kwargs to `np.allclose`.

        Returns:
            bool (bool): Returns `True` if both arrays are equal  .

        Raises:
            TypeError: If other is not a Matrix
            ValueError: If any of the bins in any of the arrays are not equal.

        """
        return True  # BUG:  Buggy implementatin down to the root
        if not isinstance(other, Matrix):
            raise TypeError("Other must be a Matrix")
        if np.any(self.shape != other.shape):
            raise ValueError("Must have equal number of energy bins.")
        if not np.allclose(self._Ex, other._Ex, **kwargs) \
           or not np.allclose(self._Eg, other._Eg, **kwargs):
            raise ValueError("Must have equal energy binning.")
        else:
            return True

    @property
    def dEx(self) -> float:
        if True: #self.is_equidistant():
            return self.Ex[1] - self.Ex[0]
        else:
            Ex = self.Ex
            return (np.roll(Ex, 1) - self.Ex)[1:]

    @property
    def dEg(self) -> float:
        if True: #self.is_equidistant():
            return self.Eg[1] - self.Eg[0]
        else:
            Eg = self.Eg
            return (np.roll(Eg, 1) - self.Eg)[1:]

    def from_mask(self, mask: ArrayBool) -> Matrix | Vector:
        """ Returns a copy of the matrix with only the rows and columns  where `mask` is True.

        A Vector is returned if the matrix is 1D.
        """
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

    def __matmul__(self, other: Matrix | Vector) -> Matrix | Vector:
        # cannot use has_equal_binning as we don't need the same
        # shape for Ex and Eg.
        # HACK isinstance doesn't work (autoreload?)
        if str(other.__class__.__name__) == 'Matrix':
            if not np.allclose(self.Eg, other.Ex):
                raise ValueError("Incompatible shapes {self.shape}, {other.shape}")
        elif str(other.__class__.__name__) == 'Vector':
            if not np.allclose(self.Eg, other.E):
                raise ValueError("Incompatible shapes {self.shape}, {other.shape}")
            values = self.values @ other.values
            return Vector(values=values, E=other.E)
        else:
            raise NotImplementedError(f"Matrix@{type(other)} not implemented")

        Ex = self.Ex
        Eg = other.Eg
        values = self.values@other.values
        # TODO how to handle labels??
        return Matrix(values=values, Ex=Ex, Eg=Eg)

    def __neg__(self):
        return self.clone(values=-self.values)

    @property
    def T(self) -> 'Matrix':
        values = self.values.T
        assert self.std is None
        return self.clone(values=values, Eg=self.Ex, Ex=self.Eg,
                          xlabel=self.ylabel, ylabel=self.xlabel)

    @property
    def _summary(self) -> str:
        if self.is_equidistant():
            m, n = self.shape
            s = f"Eᵧ: {self.Eg[0]} to {self.Eg[-1]} [{self.Eg_units:~}]\n"
            s += f"{n} bins with dEᵧ: {self.dEg}\n"
            s += f"Eₓ: {self.Ex[0]} to {self.Ex[-1]} [{self.Ex_units:~}]\n"
            s += f"{m} bins with dEₓ: {self.dEx}\n"
            s += f"Total counts: {self.sum():,}\n"
        else:
            s = f"Eᵧ: {self.Eg[0]} to {self.Eg[-1]} [{self.Eg_units:~}]\n"
            s += f"Eₓ: {self.Ex[0]} to {self.Ex[-1]} [{self.Ex_units:~}]\n"
            s += f"Variable bin width.\n"
        return s

    def summary(self):
        print(self._summary)

    def sum(self, axis: int | str = 'both') -> Vector | float:
        axis = to_plot_axis(axis)
        if axis == 2:
            return self.values.sum()

        isEx = axis == 0
        values = self.values.sum(axis=int(not axis))
        label = self.ylabel if isEx else self.xlabel
        E = self.Ex if isEx else self.Eg
        return Vector(E=E, values=values, xlabel=label)


    def __str__(self) -> str:
        summary = self._summary
        summary += "\nValues:\n" 
        if self.std is not None:
            return summary+str(self.values)+'\nStd: \n'+str(self.std)
        else:
            return summary+str(self.values)

    @property
    def Ex(self) -> np.ndarray:
        return self._Ex.magnitude

    @property
    def Eg(self) -> np.ndarray:
        return self._Eg.magnitude

    @property
    def Eg_units(self) -> Any:
        return self._Eg.units

    @property
    def Ex_units(self) -> Any:
        return self._Ex.units

    def clone(self, **kwargs) -> Matrix:
        """ Copies the object.

        Any keyword argument will override the equivalent
        attribute in the copy. For example, matrix.clone(Eg=[1,2,3])
        tries to set the gamma energy to [1,2,3].

        kwargs: Any keyword argument is overwritten
            in the copy.
        Returns:
            The copy
        """
        Eg = kwargs.pop('Eg', self._Eg)
        Ex = kwargs.pop('Ex', self._Ex)
        values = kwargs.pop('values', self.values)
        std = kwargs.pop('std', self.std)
        order = kwargs.pop('order', None)
        Ex_units = kwargs.pop('Ex_units', self.Ex_units)
        Eg_units = kwargs.pop('Eg_units', self.Eg_units)
        xlabel = kwargs.pop('xlabel', self.xlabel)
        ylabel = kwargs.pop('ylabel', self.ylabel)
        for key in kwargs.keys():
            raise RuntimeError(f"Matrix has no setable attribute {key}.")

        matrix = Matrix(Ex=Ex, Eg=Eg, values=values, std=std,
                        Ex_units=Ex_units, Eg_units=Eg_units,
                        xlabel=xlabel, ylabel=ylabel)
        if order is not None:
            matrix.set_order(order)
        return matrix

    def __lt__(self, other):
        match other:
            case Matrix():
                raise NotImplementedError("Matrix <= Matrix not implemented")
            case _:
                return self.values <= other


class IndexLocator:
    def __init__(self, matrix: Matrix):
        self.mat = matrix

    def __getitem__(self, key) -> Matrix | Vector:
        if isinstance(key, array):
            return self.linear_index(key)
        if len(key) != 2:
            raise ValueError("Expected [mask] or two integers [i, j]")

        ex, eg = key
        Eg = self.mat.Eg.__getitem__(eg)
        Ex = self.mat.Ex.__getitem__(ex)
        values = self.mat.values.__getitem__(key)
        if isinstance(eg, (int, np.integer)):
            return Vector(values=values, E=Ex, xlabel=self.mat.ylabel)
        elif isinstance(ex, (int, np.integer)):
            return Vector(values=values, E=Eg, xlabel=self.mat.xlabel)
        return self.mat.clone(Eg=Eg, Ex=Ex, values=values)

    def linear_index(self, indices) -> Matrix:
        values = np.where(indices, self.mat.values, 0)
        return self.mat.clone(values=values)


class ValueLocator:
    def __init__(self, matrix: Matrix):
        self.mat = matrix

    @overload
    def parse_Eg(self, eg: Unitlike) -> int: ...

    @overload
    def parse_Eg(self, eg: slice) -> slice: ...

    def parse_Eg(self, eg: Unitlike | slice) -> int | slice:
        if isinstance(eg, Unitlike):
            return self.mat.index_Eg(eg)
        elif isinstance(eg, slice):
            return parse_unit_slice(eg, self.mat.index_Eg, self.mat.dEg, len(self.mat.Eg))
        else:
            raise ArgumentError(f"Expected slice or Unitlike, got type: {type(eg)}")

    @overload
    def parse_Ex(self, ex: Unitlike) -> int: ...

    @overload
    def parse_Ex(self, ex: slice) -> slice: ...

    def parse_Ex(self, ex: Unitlike | slice) -> int | slice:
        if isinstance(ex, Unitlike):
            return self.mat.index_Ex(ex)
        elif isinstance(ex, slice):
            return parse_unit_slice(ex, self.mat.index_Ex, self.mat.dEx, len(self.mat.Ex))
        else:
            raise ArgumentError(f"Expected slice or Unitlike, got type: {type(ex)}")

    def __getitem__(self, key) -> Matrix | Vector:
        if len(key) == 2:
            ex, eg = key
            iex: int | slice = self.parse_Ex(ex)
            ieg: int | slice = self.parse_Eg(eg)
            if isinstance(iex, int) and isinstance(ieg, int):
                raise ArgumentError("Can not cast scalar to Matrix or Vector")

            Eg = self.mat.Eg.__getitem__(ieg)
            Ex = self.mat.Ex.__getitem__(iex)
            values = self.mat.values.__getitem__((iex, ieg))
            std = None
            if self.mat.std is not None:
                std = self.mat.std.__getitem__((iex, ieg))
            if isinstance(iex, (int, np.integer)):
                return Vector(values=values, E=Eg, xlabel=self.mat.xlabel)
            if isinstance(ieg, (int, np.integer)):
                return Vector(values=values, E=Ex, xlabel=self.mat.ylabel)

            return self.mat.clone(values=values, Ex=Ex, Eg=Eg, std=std)
        else:
            raise ValueError("Expect two indices [x, y]")

    def __setitem__(self, key, val):
        if len(key) == 2:
            ex, eg = key
            iex: int | slice = self.parse_Ex(ex)
            ieg: int | slice = self.parse_Eg(eg)

            self.mat.values[iex, ieg] = val
        else:
            raise ValueError("Expect two indices [x, y]")


@overload
def preparse(s: None) -> (Literal[False], None): ...
@overload
def preparse(s: Unitlike) -> (bool, Unitlike): ...

def preparse(s: Unitlike | None) -> (bool, Unitlike | None):
    inclusive = False
    if isinstance(s, str):
        s = s.strip()
        if s[0] == '<':
            s = s[1:]
        elif s[0] == '>':
            inclusive = True
            s = s[1:]

    return (inclusive, s)

def parse_unit_slice(s: slice, index: Callable[[Unitlike], int],
                     dx: float, length: int) -> slice:
    start = None if s.start is None else index(s.start)
    inclusive, stop_ = preparse(s.stop)
    stop = None if stop_ is None else index(stop_)
    if inclusive and stop < length:
        assert stop is not None
        stop += 1

    step = None if s.step is None else np.ceil(s.step / dx)
    return slice(start, stop, step)


class MeshLocator(ticker.Locator):
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
