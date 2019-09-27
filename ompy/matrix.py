"""
Library of utility classes and functions for the Oslo method.

---

This file is part of oslo_method_python, a python implementation of the
Oslo method.

Copyright (C) 2018 Jørgen Eriksson Midtbø, 2019 Erlend Lima
Oslo Cyclotron Laboratory
jorgenem [0] gmail.com
erlendlima@outlook.com

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""

from __future__ import annotations
import logging
import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import ticker
from pathlib import Path
from matplotlib.colors import LogNorm, Normalize, LinearSegmentedColormap
from typing import (Dict, Iterable, Any, Union, Tuple,
                    List, Sequence, Optional, Iterator)
from .matrixstate import MatrixState
from .library import div0, fill_negative, diagonal_resolution, diagonal_elements
from .filehandling import (mama_read, mama_write, save_numpy_1D, load_numpy_1D,
                           save_numpy_2D, load_numpy_2D, save_tar, load_tar,
                           filetype_from_suffix)
from .constants import DE_PARTICLE, DE_GAMMA_1MEV
from .rebin import rebin_2D
from .decomposition import index

LOG = logging.getLogger(__name__)
logging.captureWarnings(True)


class Matrix():
    """ Class for high level manipulation of counts and energy axes

    Stores matrices along with calibration and energy axis arrays. Performs
    several integrity checks to verify that the arrays makes sense in relation
    to each other.


    Note that since a matrix is numbered NxM where N is rows going in the
    y-direction and M is columns going in the x-direction, the "x-dimension"
    of the matrix has the same shape as the Ex array (Excitation axis)

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
        matrix: 2D matrix storing the counting data
        Eg: The gamma energy along the x-axis
        Ex: The excitation energy along the y-axis
        std: Array of standard deviations
        state: An enum to keep track of what has been done to the matrix


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
                 state: Union[str, MatrixState] = 'raw'):
        """
        There is the option to initialize it in an empty state.
        In that case, all class variables will be None.
        It can be filled later using the load() method.

        For initializing one can give values, [Ex, Eg] arrays,
        a filename for loading a saved matrix or a shape
        for initialzing it with zero entries.

        """

        if shape is not None and values is not None:
            raise ValueError("'shape' and 'values' are exclusive")

        if shape is not None:
            self.values = np.zeros(shape, dtype=float)
        else:
            self.values = np.asarray(values, dtype=float)

        if (values is not None or shape is not None) and Ex is None:
            Ex = range(self.values.shape[0])
            Ex = np.asarray(Ex) + 0.5
        if (values is not None or shape is not None) and Eg is None:
            Eg = range(self.values.shape[1])
            Eg = np.asarray(Eg) + 0.5

        self.Eg: np.ndarray = np.asarray(Eg, dtype=float)
        self.Ex: np.ndarray = np.asarray(Ex, dtype=float)
        self.std = std

        if path is not None:
            self.load(path)
        self.verify_integrity()

        self.state = state

    def verify_integrity(self):
        """ Runs checks to verify internal structure

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
            if len(self.Ex) > 2:
                # Verify equispaced array
                diff = (self.Ex - np.roll(self.Ex, 1))[1:]
                try:
                    diffdiff = diff - diff[1]
                    np.testing.assert_array_almost_equal(diffdiff,
                            np.zeros_like(diff))
                except AssertionError:
                    raise ValueError("Ex array is not equispaced")
        if self.Ex is not None and self.Ex.ndim > 1:
            raise ValueError(f"Ex array must be ndim 1, not {self.Ex.ndim}")

        if self.Eg is not None and self.Eg.ndim == 1:
            if shape[1] != len(self.Eg):
                raise ValueError(("Shape mismatch between matrix and Eg:"
                                  f" (_{shape[0]}_, {shape[1]}) ≠ "
                                  f"{len(self.Eg)}"))
            if len(self.Eg) > 2:
                # Verify equispaced array
                diff = (self.Eg - np.roll(self.Eg, 1))[1:]
                diffdiff = diff - diff[1]
                if not np.allclose(diffdiff, np.zeros_like(diff)):
                    print(self.Eg)
                    raise ValueError("Eg array is not equispaced")
        if self.Eg is not None and self.Eg.ndim > 1:
            raise ValueError(f"Eg array must be ndim 1, not {self.Eg.ndim}")

        if self.std is not None:
            if shape != self.std.shape:
                raise ValueError("Shape mismatch between self.values and std.")

    def load(self, path: Union[str, Path],
             filetype: Optional[str] = None) -> None:
        """ Load vector from specified format
        """
        path = Path(path) if isinstance(path, str) else path
        if filetype is None:
            filetype = filetype_from_suffix(path)
        filetype = filetype.lower()

        if filetype == 'numpy':
            self.values, self.Eg, self.Ex = load_numpy_2D(path)
        elif filetype == 'tar':
            self.values, self.Eg, self.Eg = load_tar(path)
        elif filetype == 'mama':
            matrix, Eg, Ex = mama_read(path)
            self.values = matrix
            self.Eg = Eg
            self.Ex = Ex
        else:
            raise ValueError(f"Unknown filetype {filetype}")
        self.verify_integrity()

        return None

    def save(self, path: Union[str, Path], filetype: Optional[str] = None):
        """Save matrix to mama file
        """
        path = Path(path) if isinstance(path, str) else path
        if filetype is None:
            filetype = filetype_from_suffix(path)
        filetype = filetype.lower()

        if filetype == 'numpy':
            save_numpy_2D(self.values, self.Eg, self.Ex, path)
        elif filetype == 'tar':
            save_tar([self.values, self.Eg, self.Ex], path)
        elif filetype == 'mama':
            mama_write(self, path, comment="Made by Oslo Method Python")
        else:
            raise ValueError(f"Unknown filetype {filetype}")

    def calibration(self) -> Dict[str, np.ndarray]:
        """ Calculates the calibration coefficients of the energy axes

        Returns: The calibration coefficients in a dictionary.
        """
        # Formatted as "a{axis}{power of E}"
        calibration = {
            "a00": self.Ex[0],
            "a01": self.Ex[1]-self.Ex[0],
            "a10": self.Eg[0],
            "a11": self.Eg[1]-self.Eg[0],
        }
        return calibration

    def calibration_array(self) -> np.ndarray:
        """ Calculates the calibration coefficients of the energy axes

        Returns: The calibration coefficients in an array.
        """
        return np.array(list(self.calibration().values()))

    def plot(self, ax: Any = None,
             title: Optional[str] = None,
             scale: Optional[str] = None,
             vmin: Optional[float] = None,
             vmax: Optional[float] = None,
             midbin_ticks: bool = True,
             xlabel: Optional[str] = None,
             ylabel: Optional[str] = None,
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
            xlabel (optional, str): Label on x-axis. Default see source.
            ylabel (optional, str): Label on y-axis. Default see source.
        Returns:
            The ax used for plotting
        Raises:
            ValueError: If scale is unsupported
        """
        fig, ax = plt.subplots() if ax is None else (None, ax)
        if len(self.Ex) <= 1 or len(self.Eg) <= 1:
            raise ValueError("Number of bins must be greater than 1")

        if scale is None:
            scale = 'log' if self.counts > 1000 else 'linear'
        if scale == 'log':
            norm = LogNorm(vmin=vmin, vmax=vmax)
        elif scale == 'linear':
            norm = Normalize(vmin=vmin, vmax=vmax)
        else:
            raise ValueError("Unsupported zscale ", scale)
        # Move all bins down to lower bins
        self.to_lower_bin()
        # Must extend it the range to make pcolormesh happy
        dEg = self.Eg[1] - self.Eg[0]
        dEx = self.Ex[1] - self.Ex[0]
        Eg = np.append(self.Eg, self.Eg[-1] + dEg)
        Ex = np.append(self.Ex, self.Ex[-1] + dEx)
        # Move the bins back up
        self.to_mid_bin()

        # Set entries of 0 to white
        current_cmap = cm.get_cmap()
        current_cmap.set_bad(color='white')
        mask = np.isnan(self.values) | (self.values == 0)
        masked = np.ma.array(self.values, mask=mask)

        lines = ax.pcolormesh(Eg, Ex, masked, norm=norm, **kwargs)
        if midbin_ticks:
            ax.xaxis.set_major_locator(MeshLocator(self.Eg))
            ax.tick_params(axis='x', rotation=40)
            ax.yaxis.set_major_locator(MeshLocator(self.Ex))
        # ax.xaxis.set_major_locator(ticker.FixedLocator(self.Eg, nbins=10))
        #fix_pcolormesh_ticks(ax, xvalues=self.Eg, yvalues=self.Ex)

        ax.set_title(title if title is not None else self.state)
        if xlabel is None:
            ax.set_xlabel(r"$\gamma$-ray energy $E_{\gamma}$ [eV]")
        else:
            ax.set_xlabel(xlabel)
        if ylabel is None:
            ax.set_ylabel(r"Excitation energy $E_{x}$ [eV]")
        else:
            ax.set_ylabel(ylabel)

        if fig is not None:
            if vmin is not None and vmax is not None:
                cbar = fig.colorbar(lines, ax=ax, extend='both')
            elif vmin is not None:
                cbar = fig.colorbar(lines, ax=ax, extend='min')
            elif vmax is not None:
                cbar = fig.colorbar(lines, ax=ax, extend='max')
            else:
                cbar = fig.colorbar(lines, ax=ax)

            # cbar.ax.set_ylabel("# counts")
            plt.show()
        return lines, ax, fig

    def plot_projection(self, axis: int, Emin: float = None,
                        Emax: float = None, *, ax: Any = None,
                        normalize: bool = False,
                        xlabel: Optional[str] = "Energy",
                        ylabel: Optional[str] = None, **kwargs) -> Any:
        """ Plots the projection of the matrix along axis

        Args:
            axis: The axis to project onto.
                  Can be either of (0, 'Eg', 'x'), (1, 'Ex', 'y')
            Emin: The minimum energy to be summed over.
            Emax: The maximum energy to be summed over.
            ax: The axes object to plot onto.
            normalize: Whether or not to normalize the counts.
            xlabel (optional, str): Label on x-axis. See source.
            ylabel (optional, str): Label on y-axis. Default is `None`.
        Raises:
            ValueError: If axis is not in [0, 1]
        Returns:
            The ax used for plotting
        TODO: Fix normalization
        """
        if ax is None:
            fig, ax = plt.subplots()

        axis = to_plot_axis(axis)
        is_Ex = axis == 1
        projection, energy = self.projection(axis, Emin, Emax, normalize=normalize)

        # Shift energy by a half bin to make the steps correct
        #shifted_energy = energy + (energy[1] - energy[0])/2

        if is_Ex:
            ax.step(energy, projection, where='mid', **kwargs)
            ax.set_xlabel(r"Excitation energy $E_{x}$ [eV]")
        else:
            ax.step(energy, projection, where='mid', **kwargs)
            ax.set_xlabel(r"$\gamma$-ray energy $E_{\gamma}$ [eV]")
        if xlabel is not None:  # overwrite the above
            ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        return ax

    def projection(self, axis: Union[int, str], Emin: float = None,
                   Emax: float = None,
                   normalize: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """ Returns the projection along the specified axis

        Args:
            axis: The axis to project onto. Can be 0 or 1.
            Emin (optional): The minimum energy to be summed over.
            Emax (optional): The maximum energy to be summed over.
            normalize: Whether or not to normalize the counts.
                       Defaults to False
        Raises:
            ValueError: If axis is not in [0, 1]
        Returns:
            The projection and the energies summed onto
        TODO: Fix normalization
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
        """ Plots a rebinned ascii version of the matrix

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
            inplace: Whether to make the cut in place or not.
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

    def cut_like(self, other, inplace=True) -> Optional[Matrix]:
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

        If no limits are provided, an automatic cut will be made.
        Args:
            E1: First point of intercept, ordered as Ex, Eg
            E2: Second point of intercept
            inplace: Whether the operation should be applied to the
                current matrix, or to a copy which is then returned.
        Returns:
            The matrix with counts above diagonal removed if not inplace.
        """
        if E1 is None and E2 is None:
            mask = self.diagonal_mask()
        elif E1 is None or E2 is None:
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
        """Create a mask for above (True) and below (False) the line

        Args:
            E1: First point of intercept, ordered as Ex, Eg
            E2: Second point of intercept
        Returns:
            The boolean array with counts below the line set to False
        TODO: Write as a property with memonized output for unchanged matrix

        NOTE: My method and Jørgen's method give 2 pixels difference
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
        """Create a trapezoidal cut or mask delimited by the diagonal of the matrix

        Args:
            Ex_min: The bottom edge of the trapezoid
            Ex_max: The top edge of the trapezoid
            Eg_min: The left edge of the trapezoid
            #Eg_max: The right edge of the trapezoid used for defining the
            #    diagonal. If not set, the diagonal will be found by
            #    using the last nonzeros of each row.
        Returns:
            Cut matrix if 'inplace' is True
        """
        # Transform to index basis
        if Eg_max is not None:
            raise NotImplementedError()

        for i, j in reversed(list(self.diagonal_elements())):
            if self.Ex[i] <= Ex_max:
                Eg_max = self.Eg[j]
                break

        iEx = (Ex_min < self.Ex) & (self.Ex < Ex_max)
        iEg = (Eg_min < self.Eg) & (self.Eg < Eg_max)
        indicies = np.ix_(iEx, iEg)
        #mask = np.zeros_like(self.values, dtype=bool)
        #mask[indicies] = True
        if inplace:
            self.values = self.values[indicies]
            self.Ex = self.Ex[iEx]
            self.Eg = self.Eg[iEg]
        else:
            return Matrix(values=self.values[indicies], Ex=self.Ex[iEx],
                          Eg=self.Eg[iEg])


    def rebin(self, axis: Union[int, str],
              edges: Optional[Sequence[float]] = None,
              factor: Optional[float] = None,
              inplace: bool = True) -> Optional[Matrix]:
        """ Rebins one axis of the matrix

        Args:
            axis: the axis to rebin.
            edges: The new edges along the axis. Can not be
                given alongside 'factor'.
            factor: The factor by which the step size shall be
                changed. Can not be given alongside 'edges'.
            inplace: Whether to change the axis and values
                inplace or return the rebinned matrix.
        Returns:
            The rebinned Matrix if inplace is 'False'.
        Raises:
            ValueError if the axis is not a valid axis.
        """

        axis: int = to_plot_axis(axis)
        if axis not in (0, 1):
            raise ValueError("Axis must be 0 or 1")
        if not (edges is None) ^ (factor is None):
            raise ValueError("Either 'edges' or 'factor' must be"
                             " specified, but not both.")
        edges_old = self.Ex if axis else self.Eg

        if factor is not None:
            if factor <= 0:
                raise ValueError("'factor' must be positive")
            num_edges = int(len(edges_old)/factor)
            old_step = edges_old[1] - edges_old[0]
            step = factor*old_step
            edge = edges_old[0]
            edges = []
            while len(edges) < num_edges:
                edges.append(edge)
                edge += step
            LOG.debug("Rebinning with factor %g, giving %g edges",
                      factor, num_edges)
            LOG.debug("Old step size: %g\nNew step size: %g",
                      old_step, step)
            edges = np.asarray(edges, dtype=float)

        naxis = (axis + 1) % 2
        rebinned = rebin_2D(self.values, edges_old, edges, naxis)
        if inplace:
            self.values = rebinned
            if axis:
                self.Ex = edges
            else:
                self.Eg = edges
            self.verify_integrity()
        else:
            if naxis:
                return Matrix(Eg=edges, Ex=self.Ex, values=rebinned)
            else:
                return Matrix(Eg=self.Eg, Ex=edges, values=rebinned)

    def copy(self) -> Matrix:
        """ Return a copy of the matrix """
        return copy.deepcopy(self)

    def diagonal_elements(self) -> Iterator[Tuple[int, int]]:
        """ Iterates over the last non-zero elements

        Args:
            mat: The matrix to iterate over
        Yields:
            Indicies (i, j) over the last non-zero (=diagonal)
            elements.
        """
        return diagonal_elements(self.values)

    def diagonal_resolution(self) -> np.ndarray:
        return diagonal_resolution(self.Ex)

    def diagonal_mask(self) -> np.ndarray:
        # TODO Implement an arbitrary diagonal mask
        dEg = np.repeat(diagonal_resolution(self.Ex), len(self.Eg))
        dEg = dEg.reshape(self.shape)
        Eg, Ex = np.meshgrid(self.Eg, self.Ex)
        mask = np.zeros_like(self.values, dtype=bool)
        mask[Eg >= Ex + dEg] = True
        return mask

    def fill_negative(self, window_size):
        self.values = fill_negative(self.values, window_size)

    def remove_negative(self):
        self.values = np.where(self.values > 0, self.values, 0)

    def fill_and_remove_negative(self):
        """Temporary function to remove boilerplate"""
        self.fill_negative(window_size=10)
        self.remove_negative()

    def index_Eg(self, E: float) -> int:
        """ Returns the closest index corresponding to the Eg value """
        return index(self.Eg, E)
        # return np.abs(self.Eg - E).argmin()

    def index_Ex(self, E: float) -> int:
        """ Returns the closest index corresponding to the Ex value """
        return index(self.Ex, E)
        #return np.abs(self.Ex - E).argmin()

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
    def counts(self) -> float:
        return self.values.sum()

    @property
    def shape(self) -> Tuple[int, int]:
        return self.values.shape

    @property
    def state(self) -> MatrixState:
        return self._state

    @state.setter
    def state(self, state: Union[str, MatrixState]) -> None:
        if isinstance(state, str):
            self._state = MatrixState.str_to_state(state)
        # Buggy. Impossible to compare type of Enum??
        elif type(state) == type(MatrixState.RAW):
            self._state = state
        else:
            raise ValueError(f"state must be str or MatrixState"
                             f". Got {type(state)}")

    def to_lower_bin(self):
        dEx = (self.Ex[1] - self.Ex[0])/2
        dEg = (self.Eg[1] - self.Eg[0])/2
        self.Ex -= dEx
        self.Eg -= dEg

    def to_mid_bin(self):
        """ Transform Eg and Ex from lower bin to mid bin """
        dEx = (self.Ex[1] - self.Ex[0])/2
        dEg = (self.Eg[1] - self.Eg[0])/2
        self.Ex += dEx
        self.Eg += dEg

    def iter(self) -> Iterator[Tuple[int, int]]:
        for row in range(self.shape[0]):
            for col in range(self.shape[1]):
                yield row, col

    def __getitem__(self, key):
        return self.values.__getitem__(key)

    def __setitem__(self, key, item):
        return self.values.__setitem__(key, item)

    def __sub__(self, other) -> Matrix:
        if not isinstance(other, Matrix):
            raise TypeError("Other must be a Matrix")
        if np.all(self.Ex != other.Ex) or np.all(self.Eg != other.Eg):
            raise NotImplementedError()
            # other = other.rebin('Ex', self.Ex, inplace=False)
            # other = other.rebin('Eg', self.Eg, inplace=False)
        result = copy.deepcopy(self)
        result.values -= other.values
        return result

    def __add__(self, other) -> Matrix:
        if not isinstance(other, Matrix):
            raise TypeError("Other must be a Matrix")
        raise NotImplementedError

    def __rmul__(self, factor) -> Matrix:
        other = copy.deepcopy(self)
        other.values *= factor
        return other

    def __mul__(self, factor) -> Matrix:
        return self.__rmul__(factor)


def to_plot_axis(axis: Any) -> int:
    """Maps axis to 0, 1 or 2 according to which axis is specified

    Args:
        axis: Can be either of (0, 'Eg', 'x'), (1, 'Ex', 'y'), or
              (2, 'both', 'egex', 'exeg', 'xy', 'yx')
    Returns:
        An int describing the axis in the basis of the plot,
        _not_ the values' dimension.
    Raises:
        ValueError if the axis is not supported
    """
    try:
        axis = axis.lower()
    except AttributeError:
        pass

    if axis in (0, 'eg', 'x'):
        return 0
    elif axis in (1, 'ex', 'y'):
        return 1
    elif axis in (2, 'both', 'egex', 'exeg', 'xy', 'yx'):
        return 2
    else:
        raise ValueError(f"Unrecognized axis: {axis}")


def to_values_axis(axis: Any) -> int:
    """Maps axis to 0, 1 or 2 according to which axis is specified

    Args:
        axis: Can be 0, 1, 'Eg', 'Ex', 'both', 2
    Returns:
        An int describing the axis in the basis of values,
        _not_ the plot's dimension.
    Raises:
        ValueError if the axis is not supported
    """
    try:
        axis = axis.lower()
    except AttributeError:
        pass

    axis = to_plot_axis(axis)
    if axis == 2:
        return axis
    return (axis + 1) % 2


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
