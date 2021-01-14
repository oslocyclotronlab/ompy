from __future__ import annotations
import logging
import warnings
import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import ticker
from pathlib import Path
from matplotlib.colors import LogNorm, Normalize
from typing import (Dict, Iterable, Any, Union, Tuple,
                    Sequence, Optional, Iterator)
import warnings
from .abstractarray import AbstractArray, to_plot_axis
from .decomposition import index
from .filehandling import (mama_read, mama_write,
                           save_numpy_2D, load_numpy_2D,
                           save_txt_2D, load_txt_2D, save_tar, load_tar,
                           filetype_from_suffix)
from .library import div0, fill_negative_gauss, diagonal_elements
from .matrixstate import MatrixState
from .rebin import rebin_2D
from .vector import Vector

LOG = logging.getLogger(__name__)
logging.captureWarnings(True)


class Matrix(AbstractArray):
    """Stores 2d array with energy axes (a matrix).

    Stores matrices along with calibration and energy axis arrays. Performs
    several integrity checks to verify that the arrays makes sense in relation
    to each other.


    Note that since a matrix is numbered NxM where N is rows going in the
    y-direction and M is columns going in the x-direction, the "x-dimension"
    of the matrix has the same shape as the Ex array (Excitation axis)

    Note:
        Many functions will implicity assume linear binning.

    .. parsed-literal::
                   Diagonal Ex=Eγ
                                  v
         a y E │██████▓▓██████▓▓▓█░   ░
         x   x │██ █████▓████████░   ░░
         i a   │█████████████▓▓░░░░░
         s x i │███▓████▓████░░░░░ ░░░░
           i n │███████████░░░░   ░░░░░
         1 s d │███▓█████░░   ░░░░ ░░░░ <-- "Folded" counts
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
        - Find a way to handle units
        - Synchronize cuts. When a cut is made along one axis,
          such as values[min:max, :] = 0, make cuts to the
          other relevant variables
        - Make values, Ex and Eg to properties so that
          the integrity of the matrix can be ensured.
    """
    def __init__(self,
                 values: Optional[np.ndarray] = None,
                 Eg: Optional[np.ndarray] = None,
                 Ex: Optional[np.ndarray] = None,
                 std: Optional[np.ndarray] = None,
                 path: Optional[Union[str, Path]] = None,
                 shape: Optional[Tuple[int, int]] = None,
                 state: Union[str, MatrixState] = None):
        """
        There is the option to initialize it in an empty state.
        In that case, all class variables will be None.
        It can be filled later using the load() method.

        For initializing one can give values, [Ex, Eg] arrays,
        a filename for loading a saved matrix or a shape
        for initialzing it with zero entries.

        Args:
            values: Set the matrix' values.
            Eg: The gamma ray energies using midbinning.
            Ex: The excitation energies using midbinning.
            std: The standard deviations at each bin of `values`
            path: Load a Matrix from a given path
            state: An enum to keep track of what has been done to the matrix.
                Can also be a str. like in ["raw", "unfolded", ...]
            shape: Depreciated. Use `ZerosMatrix` instead.

        """
        if shape is not None:
            warnings.warn("Creating a Matrix with zeros as entries by the "
                          "shape argument is depreciated. Use ZerosMatrix "
                          "instead.", DeprecationWarning)
            values = ZerosMatrix(shape=shape, Ex=Ex, Eg=Eg).values

        if values is None and Ex is not None and Eg is not None:
            warnings.warn("Creating a Matrix with zeros as entries only"
                          "providing Ex and Eg is is depreciated. Use "
                          "ZerosMatrix instead.", DeprecationWarning)
            values = ZerosMatrix(Ex=Ex, Eg=Eg).values

        self.values = np.asarray(values, dtype=float).copy()

        if (values is not None) and Ex is None:
            Ex = range(self.values.shape[0])
            Ex = np.asarray(Ex) + 0.5
        if (values is not None) and Eg is None:
            Eg = range(self.values.shape[1])
            Eg = np.asarray(Eg) + 0.5

        self.Eg: np.ndarray = np.asarray(Eg, dtype=float).copy()
        self.Ex: np.ndarray = np.asarray(Ex, dtype=float).copy()
        self.std = std

        if path is not None:
            self.load(path)
        self.verify_integrity()

        self.state = state

    def verify_integrity(self, check_equidistant: bool = False):
        """ Runs checks to verify internal structure

        Args:
            check_equidistant (bool, optional): Check whether energy array
                are equidistant spaced. Defaults to False.

        Raises:
            ValueError: If any check fails
        """
        if self.values is None or self.values.ndim != 2:
            return

        # Check shapes:
        shape = self.values.shape
        if self.Ex is not None and self.Ex.ndim == 1:
            if shape[0] != len(self.Ex):
                raise ValueError(("Shape mismatch between matrix and Ex:"
                                  f" (_{shape[0]}_, {shape[1]}) ≠ "
                                  f"{len(self.Ex)}"))
        if self.Ex is not None and self.Ex.ndim > 1:
            raise ValueError(f"Ex array must be ndim 1, not {self.Ex.ndim}")

        if self.Eg is not None and self.Eg.ndim == 1:
            if shape[1] != len(self.Eg):
                raise ValueError(("Shape mismatch between matrix and Eg:"
                                  f" (_{shape[0]}_, {shape[1]}) ≠ "
                                  f"{len(self.Eg)}"))
        if self.Eg is not None and self.Eg.ndim > 1:
            raise ValueError(f"Eg array must be ndim 1, not {self.Eg.ndim}")

        if check_equidistant:
            self.verify_equdistant("Ex")
            self.verify_equdistant("Eg")

        if self.std is not None:
            if shape != self.std.shape:
                raise ValueError("Shape mismatch between self.values and std.")

    def load(self, path: Union[str, Path],
             filetype: Optional[str] = None) -> None:
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
            self.values, self.Eg, self.Ex = load_numpy_2D(path)
        elif filetype == 'txt':
            self.values, self.Eg, self.Ex = load_txt_2D(path)
        elif filetype == 'tar':
            self.values, self.Eg, self.Ex = load_tar(path)
        elif filetype == 'mama':
            self.values, self.Eg, self.Ex = mama_read(path)
        else:
            try:
                self.values, self.Eg, self.Ex = mama_read(path)
            except ValueError:  # from within mama_read
                raise ValueError(f"Unknown filetype {filetype}")
        self.verify_integrity()

    def save(self, path: Union[str, Path], filetype: Optional[str] = None,
             which: Optional[str] = 'values', **kwargs):
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

        if filetype == 'numpy':
            save_numpy_2D(values, self.Eg, self.Ex, path)
        elif filetype == 'txt':
            save_txt_2D(values, self.Eg, self.Ex, path, **kwargs)
        elif filetype == 'tar':
            save_tar([values, self.Eg, self.Ex], path)
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
            "a0x": self.Ex[0],
            "a1x": self.Ex[1]-self.Ex[0],
            "a0y": self.Eg[0],
            "a1y": self.Eg[1]-self.Eg[0],
        }
        return calibration

    def plot(self, *, ax: Any = None,
             title: Optional[str] = None,
             scale: Optional[str] = None,
             vmin: Optional[float] = None,
             vmax: Optional[float] = None,
             midbin_ticks: bool = False,
             add_cbar: bool = True,
             **kwargs) -> Any:
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
        if len(self.Ex) <= 1 or len(self.Eg) <= 1:
            raise ValueError("Number of bins must be greater than 1")

        if scale is None:
            scale = 'log' if self.counts > 1000 else 'linear'
        if scale == 'log':
            if vmin is not None and vmin <= 0:
                raise ValueError("`vmin` must be positive for log-scale")
            norm = LogNorm(vmin=vmin, vmax=vmax)
        elif scale == 'linear':
            norm = Normalize(vmin=vmin, vmax=vmax)
        else:
            raise ValueError("Unsupported zscale ", scale)
        # Move all bins down to lower bins
        self.to_lower_bin()
        # Must extend it the range to make pcolormesh happy
        dEg = self.Eg[-1] - self.Eg[-2]
        dEx = self.Ex[-1] - self.Ex[-2]
        Eg = np.append(self.Eg, self.Eg[-1] + dEg)
        Ex = np.append(self.Ex, self.Ex[-1] + dEx)
        # Move the bins back up
        self.to_mid_bin()

        # Set entries of 0 to white
        current_cmap = copy.copy(cm.get_cmap())
        current_cmap.set_bad(color='white')
        kwargs.setdefault('cmap', current_cmap)
        mask = np.isnan(self.values) | (self.values == 0)
        masked = np.ma.array(self.values, mask=mask)

        lines = ax.pcolormesh(Eg, Ex, masked, norm=norm, **kwargs)
        if midbin_ticks:
            ax.xaxis.set_major_locator(MeshLocator(self.Eg))
            ax.tick_params(axis='x', rotation=40)
            ax.yaxis.set_major_locator(MeshLocator(self.Ex))
        # ax.xaxis.set_major_locator(ticker.FixedLocator(self.Eg, nbins=10))
        # fix_pcolormesh_ticks(ax, xvalues=self.Eg, yvalues=self.Ex)

        ax.set_title(title if title is not None else self.state)
        ax.set_xlabel(r"$\gamma$-ray energy $E_{\gamma}$")
        ax.set_ylabel(r"Excitation energy $E_{x}$")

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
        ax.format_coord = format_coord

        if add_cbar:
            if vmin is not None and vmax is not None:
                cbar = fig.colorbar(lines, ax=ax, extend='both')
            elif vmin is not None:
                cbar = fig.colorbar(lines, ax=ax, extend='min')
            elif vmax is not None:
                cbar = fig.colorbar(lines, ax=ax, extend='max')
            else:
                cbar = fig.colorbar(lines, ax=ax)

            # cbar.ax.set_ylabel("# counts")
            # plt.show()
        return lines, ax, fig

    def plot_projection(self, axis: Union[int, str],
                        Emin: float = None,
                        Emax: float = None, *, ax: Any = None,
                        normalize: bool = False,
                        scale: str = 'linear', **kwargs) -> Any:
        """ Plots the projection of the matrix along axis

        Args:
            axis: The axis to project onto.
                  Can be either of (0, 'Eg', 'x'), (1, 'Ex', 'y')
            Emin: The minimum energy to be summed over.
            Emax: The maximum energy to be summed over.
            ax: The axes object to plot onto.
            normalize: If True, normalize the counts to 1. Defaults to False.
            scale (optional, str): y-scale, i.e `log` or `linear`. Defaults to
                "linear".
            **kwargs: Additional kwargs to plot command.

        Raises:
            ValueError: If axis is not in [0, 1]

        Returns:
            The ax used for plotting
        """
        if ax is None:
            fig, ax = plt.subplots()

        axis = to_plot_axis(axis)
        is_Ex = axis == 1
        projection, energy = self.projection(axis, Emin, Emax,
                                             normalize=normalize)
        Vector(values=projection, E=energy).plot(ax=ax, **kwargs)

        if is_Ex:
            ax.set_xlabel(r"Excitation energy $E_{x}$")
        else:
            ax.set_xlabel(r"$\gamma$-ray energy $E_{\gamma}$")

        ax.set_yscale(scale)

        return ax

    def projection(self, axis: Union[int, str], Emin: float = None,
                   Emax: float = None,
                   normalize: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """ Returns the projection along the specified axis

        Args:
            axis: The axis to project onto. Can be 0 or 1.
            Emin (optional): The minimum energy to be summed over.
            Emax (optional): The maximum energy to be summed over.
            normalize: If True, normalize the counts to 1. Defaults to False.

        Raises:
            ValueError: If axis is not in [0, 1]

        Returns:
            The projection and the energies summed onto
        """
        axis = to_plot_axis(axis)
        if axis not in (0, 1):
            raise ValueError(f"Axis must be 0 or 1, got: {axis}")

        isEx = axis == 1

        # Determine subset of the other axis to be summed
        indexE = self.index_Eg if isEx else self.index_Ex
        rangeE = self.range_Eg if isEx else self.range_Ex
        imin = indexE(Emin) if Emin is not None else rangeE[0]
        imax = indexE(Emax) if Emax is not None else rangeE[-1]
        subset = slice(imin, imax+1)
        selection = self.values[:, subset] if isEx else self.values[subset, :]
        energy = self.Ex if isEx else self.Eg

        projection = selection.sum(axis=axis)
        if normalize:
            projection = div0(projection, selection.sum(axis=axis).sum())

        return projection, energy

    def ascii_plot(self, shape=(5, 5)):
        """Plots a rebinned ascii version of the matrix

        Args:
            shape (tuple, optional): Shape of the rebinned matrix
        """
        values = np.unique(np.sort(self.values.flatten()))
        values = values[values > 0]
        N = len(values)/4

        def block(count):
            i = np.argmin(np.abs(count - values))
            b = int(i // N)
            return ['░░', '▒▒', '▓▓', '██'][b]

        for row in reversed(range(self.shape[0])):
            print('│', end='')
            for col in range(self.shape[1]):
                elem = self[row, col]
                if elem == 0:
                    print('  ', end='')
                else:
                    print(block(elem), end='')
            print('')
        print('└', end='')
        for col in range(self.shape[1]):
            print('──', end='')
        print('')

    def cut(self, axis: Union[int, str],
            Emin: Optional[float] = None,
            Emax: Optional[float] = None,
            inplace: bool = True,
            Emin_inclusive: bool = True,
            Emax_inclusive: bool = True) -> Optional[Matrix]:
        """Cuts the matrix to the sub-interval limits along given axis.

        Args:
            axis: Which axis to apply the cut to.
                Can be 0, "Eg" or 1, "Ex".
            Emin: Lowest energy to be included. Defaults to
                lowest energy. Inclusive.
            Emax: Higest energy to be included. Defaults to
                highest energy. Inclusive.
            inplace: If True make the cut in place. Otherwise return a new
                matrix. Defaults to True
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
        axis = to_plot_axis(axis)
        range = self.Eg if axis == 0 else self.Ex
        index = self.index_Eg if axis == 0 else self.index_Ex
        Emin = Emin if Emin is not None else min(range)
        Emax = Emax if Emax is not None else max(range)

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
        if Emin >= len(range):
            Emin = len(range)-1
        if Emax <= 0:
            Emax = 0
        iEmax += 1  # Because of slicing
        Eslice = slice(iEmin, iEmax)
        Ecut = range[Eslice]

        if axis == 0:
            values_cut = self.values[:, Eslice]
            Eg = Ecut
            Ex = self.Ex
        elif axis == 1:
            values_cut = self.values[Eslice, :]
            Ex = Ecut
            Eg = self.Eg
        else:
            raise ValueError("Expected axis 0 or 1")

        if inplace:
            self.values = values_cut
            self.Ex = Ex
            self.Eg = Eg
        else:
            return Matrix(values_cut, Eg=Eg, Ex=Ex)

    def cut_like(self, other: Matrix,
                 inplace: bool = True) -> Optional[Matrix]:
        """ Cut a matrix like another matrix (according to energy arrays)

        Args:
            other (Matrix): The other matrix
            inplace (bool, optional): If True make the cut in place. Otherwise
                return a new matrix. Defaults to True

        Returns:
            Optional[Matrix]: If inplace is False, returns the cut matrix
        """
        if inplace:
            self.cut('Ex', other.Ex.min(), other.Ex.max())
            self.cut('Eg', other.Eg.min(), other.Eg.max())
        else:
            out = self.cut('Ex', other.Ex.min(), other.Ex.max(), inplace=False)
            assert out is not None
            out.cut('Eg', other.Eg.min(), other.Eg.max())
            return out

    def cut_diagonal(self, E1: Optional[Iterable[float]] = None,
                     E2: Optional[Iterable[float]] = None,
                     inplace: bool = True) -> Optional[Matrix]:
        """Cut away counts to the right of a diagonal line defined by indices

        Args:
            E1: First point of intercept, ordered as (Eg, Ex)
            E2: Second point of intercept
            inplace: Whether the operation should be applied to the
                current matrix, or to a copy which is then returned.

        Returns:
            The matrix with counts above diagonal removed (if inplace is
            False).
        """
        if E1 is None or E2 is None:
            raise ValueError("If either E1 or E2 is specified, "
                             "both must be specified and have same type")
        else:
            mask = self.line_mask(E1, E2)

        if inplace:
            self.values[mask] = 0.0
        else:
            matrix = copy.deepcopy(self)
            matrix.values[mask] = 0.0
            return matrix

    def line_mask(self, E1: Iterable[float],
                  E2: Iterable[float]) -> np.ndarray:
        """Create a mask for above (True) and below (False) a line

        Args:
            E1: First point of intercept, ordered as Ex, Eg
            E2: Second point of intercept

        Returns:
            The boolean array with counts below the line set to False

        TODO:
            - Write as a property with memonized output for unchanged matrix

        NOTE:
            This method and Jørgen's original method give 2 pixels difference
            Probably because of how the interpolated line is drawn
        """
        # Transform from energy to index basis
        # NOTE: Ex and Ey refers to x- and y-direction
        # not excitation and gamma
        Ex1, Ey1 = E1
        Ex2, Ey2 = E2
        Ix = self.indices_Eg([Ex1, Ex2])
        Iy = self.indices_Ex([Ey1, Ey2])

        # Interpolate between the two points
        assert(Ix[1] != Ix[0])
        a = (Iy[1]-Iy[0])/(Ix[1]-Ix[0])
        b = Iy[0] - a*Ix[0]
        line = lambda x: a*x + b  # NOQA E731

        # Mask all indices below this line to 0
        i_mesh, j_mesh = np.meshgrid(self.range_Eg, self.range_Ex)
        mask = np.where(j_mesh < line(i_mesh), True, False)
        return mask

    def trapezoid(self, Ex_min: float, Ex_max: float,
                  Eg_min: float, Eg_max: Optional[float] = None,
                  inplace: bool = True) -> Optional[Matrix]:
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
        """
        matrix = self.copy()

        matrix.cut("Ex", Emin=Ex_min, Emax=Ex_max)
        if Eg_max is None:
            lastEx = matrix[-1, :]
            try:
                iEg_max = np.nonzero(lastEx)[0][-1]
            except IndexError():
                raise ValueError("Last Ex column has no non-zero elements")
            Eg_max = matrix.Eg[iEg_max]

        matrix.cut("Eg", Emin=Eg_min, Emax=Eg_max)

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
            self.Ex = matrix.Ex
            self.Eg = matrix.Eg
            self.state = matrix.state
        else:
            return matrix

    def rebin(self, axis: Union[int, str],
              mids: Optional[Sequence[float]] = None,
              factor: Optional[float] = None,
              inplace: bool = True) -> Optional[Matrix]:
        """ Rebins one axis of the matrix

        Args:
            axis: the axis to rebin.
            mids: The new mids along the axis. Can not be
                given alongside 'factor'.
            factor: The factor by which the step size shall be
                changed. Can not be given alongside 'mids'.
            inplace: Whether to change the axis and values
                inplace or return the rebinned matrix.
        Returns:
            The rebinned Matrix if inplace is 'False'.
        Raises:
            ValueError if the axis is not a valid axis.
        """

        axis: int = to_plot_axis(axis)
        if not (mids is None) ^ (factor is None):
            raise ValueError("Either 'mids' or 'factor' must be"
                             " specified, but not both.")
        mids_old = self.Ex if axis else self.Eg

        if axis == 2:
            if inplace:
                self.rebin(axis=0, mids=mids, factor=factor, inplace=inplace)
                self.rebin(axis=1, mids=mids, factor=factor, inplace=inplace)
                return None
            else:
                new = self.rebin(axis=0, mids=mids,
                                 factor=factor, inplace=False)
                new.rebin(axis=1, mids=mids, factor=factor, inplace=True)
                return new

        if factor is not None:
            if factor <= 0:
                raise ValueError("'factor' must be positive")
            num_mids = int(len(mids_old)/factor)
            mids, step = np.linspace(mids_old[0], mids_old[-1],
                                     num=num_mids, retstep=True)
            LOG.debug("Rebinning with factor %g, giving %g mids",
                      factor, num_mids)
            LOG.debug("Old step size: %g\nNew step size: %g",
                      mids_old[1] - mids_old[0], step)
            mids = np.asarray(mids, dtype=float)

        naxis = (axis + 1) % 2
        rebinned = rebin_2D(self.values, mids_old, mids, naxis)
        if inplace:
            self.values = rebinned
            if axis:
                self.Ex = mids
            else:
                self.Eg = mids
            self.verify_integrity()
        else:
            if naxis:
                return Matrix(Eg=mids, Ex=self.Ex, values=rebinned)
            else:
                return Matrix(Eg=self.Eg, Ex=mids, values=rebinned)

    def diagonal_elements(self) -> Iterator[Tuple[int, int]]:
        """ Iterates over the last non-zero elements
        Note:
            Assumes that the matrix is diagonal, i.e. that there are no
            entries with `Eg > Ex + dE`.
        Args:
            mat: The matrix to iterate over
            Iterator[Tuple[int, int]]: Indicies (i, j) over the last
                non-zero(=diagonal) elements.
        """
        return diagonal_elements(self.values)

    def fill(self, Eg: float, Ex: float, count: Optional[float] = 1) -> None:
        """ Add counts to the bin containing Eg and Ex.
        Args:
            Eg (float): Eg energy value (x-axis value)
            Ex (float): Ex energy value (y-axis value)
            count (float, otional): Number to add to the bin. Defaults to 1.
        """
        self.values[index(self.Ex, Ex)][index(self.Eg, Eg)] += count

    def fill_negative(self, window_size: int):
        """ Wrapper for :func:`ompy.fill_negative_gauss` """
        self.values = fill_negative_gauss(self.values, self.Eg, window_size)

    def remove_negative(self):
        """ Entries with negative values are set to 0 """
        self.values = np.where(self.values > 0, self.values, 0)

    def fill_and_remove_negative(self,
                                 window_size: Tuple[int, np.ndarray] = 20):
        """ Combination of :meth:`ompy.Matrix.fill_negative` and
        :meth:`ompy.Matrix.remove_negative`

        Args:
            window_size: See `fill_negative`. Defaults to 20 (arbitrary)!.
            """

        self.fill_negative(window_size=window_size)
        self.remove_negative()

    def index_Eg(self, E: float) -> int:
        """ Returns the closest index corresponding to the Eg value """
        return index(self.Eg, E)
        # return np.abs(self.Eg - E).argmin()

    def index_Ex(self, E: float) -> int:
        """ Returns the closest index corresponding to the Ex value """
        return index(self.Ex, E)
        # return np.abs(self.Ex - E).argmin()

    def indices_Eg(self, E: Iterable[float]) -> np.ndarray:
        """ Returns the closest indices corresponding to the Eg value"""
        indices = [self.index_Eg(e) for e in E]
        return np.array(indices)

    def indices_Ex(self, E: Iterable[float]) -> np.ndarray:
        """ Returns the closest indices corresponding to the Ex value"""
        indices = [self.index_Ex(e) for e in E]
        return np.array(indices)

    @property
    def range_Eg(self) -> np.ndarray:
        """ Returns all indices of Eg """
        return np.arange(0, len(self.Eg), dtype=int)

    @property
    def range_Ex(self) -> np.ndarray:
        """ Returns all indices of Ex """
        return np.arange(0, len(self.Ex), dtype=int)

    @property
    def shape(self) -> Tuple[int, int]:
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

    def to_lower_bin(self):
        """ Transform Eg and Ex from mid bin (=default) to lower bin. """
        dEx = (self.Ex[1] - self.Ex[0])/2
        dEg = (self.Eg[1] - self.Eg[0])/2
        self.Ex -= dEx
        self.Eg -= dEg

    def to_mid_bin(self):
        """ Transform Eg and Ex from lower bin to mid bin (=default). """
        dEx = (self.Ex[1] - self.Ex[0])/2
        dEg = (self.Eg[1] - self.Eg[0])/2
        self.Ex += dEx
        self.Eg += dEg

    def iter(self) -> Iterator[Tuple[int, int]]:
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
        if not isinstance(other, Matrix):
            raise TypeError("Other must be a Matrix")
        if np.any(self.shape != other.shape):
            raise ValueError("Must have equal number of energy bins.")
        if not np.allclose(self.Ex, other.Ex, **kwargs) \
           or not np.allclose(self.Eg, other.Eg, **kwargs):
            raise ValueError("Must have equal energy binning.")
        else:
            return True

    def __matmul__(self, other: Matrix) -> Matrix:
        result = self.copy()
        # cannot use has_equal_binning as we don't need the same
        # shape for Ex and Eg.
        if isinstance(other, Matrix):
            if len(self.Eg) != len(self.Ex):
                raise ValueError("Must have equal number of energy bins.")
            if not np.allclose(self.Eg, other.Eg):
                raise ValueError("Must have equal energy binning on Eg.")
        else:
            NotImplementedError("Type not implemented")

        result.values = result.values@other.values
        return result


class ZerosMatrix(Matrix):
    """ Return new Matrix of given shape, filled with zeros.

    Args:
        shape: Shape of the new Matrix as [len(Ex), len(Eg)].
            If Ex and Eg are provided, the shape is inferred.
        Eg: The gamma ray energies using midbinning. Defaults to an array
            with the length inferred from shape, if not provided.
        Ex: The excitation energies using midbinning. Defaults to an array
            with the length inferred from shape, if not provided.
        std: Whether to create an array for the `std`, too
    """
    def __init__(self, shape: Optional[Tuple[int, int]] = None,
                 Ex: Optional[np.ndarray] = None,
                 Eg: Optional[np.ndarray] = None,
                 std: bool = False,
                 state: Union[str, MatrixState] = None):

        # Case if Eg and Ex are given but no shape
        if shape is None:
            if (Eg is not None) and (Ex is not None):
                shape = (len(Ex), len(Eg))
            else:
                raise AssertionError("Shape can only be inferred if"
                                     "*both* Eg and Ex are given.")

        values = np.zeros(shape, dtype=float)
        if std:
            self.std = np.zeros(shape, dtype=float)
        else:
            std = None

        super().__init__(values=values, Ex=Ex, Eg=Eg, std=std, state=state)


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
