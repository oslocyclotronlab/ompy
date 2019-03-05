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

import numpy as np
import matplotlib.pyplot as plt
import warnings
from copy import copy
from matplotlib.colors import LogNorm, Normalize
from typing import Dict, Iterable, Any, Union, Tuple
from enum import Enum, unique
from .library import (mama_read, mama_write, div0, fill_negative)
from .constants import DE_PARTICLE, DE_GAMMA_1MEV, DE_GAMMA_8MEV
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import


@unique
class MatrixState(Enum):
    """ Simple enumeration to keep track of matrix states """
    RAW = 1
    UNFOLDED = 2
    FIRST_GENERATION = 3

    def __str__(self):
        return {1: 'Raw', 2: 'Unfolded', 3: 'First Generation'}[self.value]


class Matrix():
    """ Class for high level manipulation of counts and energy axes

    Stores matrices along with calibration and energy axis arrays. Performs
    several integrity checks to verify that the arrays makes sense in relation
    to each other.


    Note that since a matrix is numbered NxM where N is rows going in the
    y-direction and M is columns going in the x-direction, the "x-dimension"
    of the matrix has the same shape as the Ex array (Excitation axis)

              Diagonal Ex=Eγ │
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
    TODO: Find a way to handle units
    """
    def __init__(self, matrix: np.ndarray = None,
                 Eg: np.ndarray = None,
                 Ex: np.ndarray = None,
                 std: np.ndarray = None,
                 filename: str = None):
        """
        There is the option to initialize it in an empty state.
        In that case, all class variables will be None.
        It can be filled later using the load() method.
        """

        # Fill class variables:
        self.matrix: np.ndarray = matrix
        self.Eg: np.ndarray = Eg
        self.Ex: np.ndarray = Ex
        self.std: np.ndarray = std  # slot for matrix of standard deviations

        if filename is not None:
            self.load(filename)
        self.verify_integrity()

        self.state = MatrixState.RAW

    def verify_integrity(self):
        """ Runs checks to verify internal structure

        Raises:
            ValueError: If any check fails
        """
        if self.matrix is None:
            return

        # Check shapes:
        mshape = self.matrix.shape
        if self.Ex is not None:
            if mshape[0] != len(self.Ex):
                raise ValueError(("Shape mismatch between matrix and Ex:"
                                  f" (_{mshape[0]}_, {mshape[1]}) ≠ "
                                  f"{len(self.Ex)}"))
        if self.Eg is not None:
            if mshape[1] != len(self.Eg):
                raise ValueError(("Shape mismatch between matrix and Eg:"
                                  f" (_{mshape[0]}_, {mshape[1]}) ≠ "
                                  f"{len(self.Eg)}"))
        if self.std is not None:
            if mshape != self.std.shape:
                raise ValueError("Shape mismatch between self.matrix and std.")

    def load(self, filename: str):
        """ Load matrix from mama file

        Args:
            filename: Path to mama file
        """
        if self.matrix is not None:
            warnings.warn("load() called on non-empty matrix")

        # Load matrix from file:
        matrix_object = mama_read(filename)
        self.matrix = matrix_object.matrix
        self.Eg = matrix_object.E1_array
        self.Ex = matrix_object.E0_array

        self.verify_integrity()

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

    def plot(self, ax: Any = None, title: str = None, zscale: str = "log",
             zmin: float = None, zmax: float = None) -> Any:
        """ Plots the matrix with the energy along the axis

        Args:
            ax: A matplotlib axis to plot onto
            title: Defaults to the current matrix state
            zscale: Scale along the z-axis. Defaults to logarithmic
            zmin: Minimum value for coloring in scaling
            zmax Maximum value for coloring in scaling
        Returns:
            The ax used for plotting
        Raises:
            ValueError: If zscale is unsupported
        TODO: Add fancy 3D histogram
        """
        if ax is None:
            fig, ax = plt.subplots()
        if zscale == 'log':
            norm = LogNorm(vmin=zmin, vmax=zmax)
        elif zscale == 'linear':
            norm = Normalize(vmin=zmin, vmax=zmax)
        else:
            raise ValueError("Unsupported zscale ", zscale)
        lines = ax.pcolormesh(self.Eg, self.Ex, self.matrix,
                              norm=norm)
        ax.set_title(title if title is not None else self.state)
        ax.set_xlabel(r"$\gamma$-ray energy $E_{\gamma}$ [eV]")
        ax.set_ylabel(r"Excitation energy $E_{x}$ [eV]")
        cbar = fig.colorbar(lines, ax=ax)
        cbar.ax.set_ylabel("# counts")
        plt.show()
        return ax

    def plot_3d(self, ax: Any = None, zscale: str = "log",
                zmin: float = 1, zmax: float = None) -> Any:
        """ Plots the matrix with the energy along the axis

        Args:
            ax: A matplotlib axis to plot onto
            zscale: Scale along the z-axis. Defaults to logarithmic
            zmin: Minimum value for coloring in scaling
            zmax Maximum value for coloring in scaling
        Returns:
            The ax used for plotting
        Raises:
            ValueError: If zscale is unsupported

        """
        if zmax is None:
            zmax = self.matrix.max()
        if ax is None:
            fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
        if zscale == 'log':
            norm = LogNorm(vmin=zmin, vmax=zmax)
        elif zscale == 'linear':
            norm = Normalize(vmin=zmin, vmax=zmax)
        else:
            raise ValueError("Unsupported zscale ", zscale)
        x, y = np.meshgrid(self.Eg, self.Ex)
        z = np.where(self.matrix > 1, self.matrix, np.nan)
        # palette = copy(plt.cm.viridis)
        # palette.set_bad(color="white")
        surf = ax.plot_surface(x, y, z,
                               cmap=plt.cm.viridis, rstride=1, cstride=1, norm=norm)
        fig.colorbar(surf, shrink=0.5, aspect=5)
        ax.set_xlabel(r"$\gamma$-ray energy $E_{\gamma}$ [eV]")
        ax.set_ylabel(r"Excitation energy $E_{x}$ [eV]")
        # cbar = fig.colorbar(lines, ax=ax)
        # cbar.ax.set_ylabel("# counts")
        plt.show()
        return ax

    def plot_projection(self, axis: int, Emin: float = None,
                        Emax: float = None, ax: Any = None,
                        normalize: bool = False) -> Any:
        """ Plots the projection of the matrix along axis

        Args:
            axis: The axis to project onto. Can be 0 or 1.
            Emin: The minimum energy to be summed over.
            Emax: The maximum energy to be summed over.
            ax: The axes object to plot onto.
            normalize: Whether or not to normalize the counts.
        Raises:
            ValueError: If axis is not in [0, 1]
        Returns:
            The ax used for plotting
        TODO: Fix normalization
        """
        if ax is None:
            fig, ax = plt.subplots()

        axis = self.axis_toint(axis)
        if axis not in (0, 1):
            raise ValueError(f"Axis must be 0 or 1, got: {axis}")

        # Determine subset of the other axis to be summed
        indexE = self.index_Eg if axis else self.index_Ex
        rangeE = self.range_Eg if axis else self.range_Ex
        imin = indexE(Emin) if Emin is not None else rangeE[0]
        imax = indexE(Emax) if Emax is not None else rangeE[-1]
        subset = slice(imin, imax)
        selection = self.matrix[subset, :] if axis else self.matrix[:, subset]

        naxis = 0 if axis else 1

        projection = selection.sum(axis=naxis)
        if normalize:
            # Don't know what calibration does, so using specialized code here
            if axis:
                projection = div0(selection, selection.sum(axis=axis))
                projection = projection.mean(axis=naxis)
            else:
                calibration = self.calibration()["a01"]
                calibrated_sum = selection.sum(axis=axis)*calibration
                calibrated = div0(selection, calibrated_sum[:, None])
                projection = calibrated.mean(axis=naxis)

        if not axis:
            ax.step(self.Ex, projection)
            ax.set_xlabel(r"$\gamma$-ray energy $E_{\gamma}$ [eV]")
        else:
            ax.step(self.Eg, projection)
            ax.set_xlabel(r"Excitation energy $E_{x}$ [eV]")
        if normalize:
            ax.set_ylabel(r"$\# counts/\Sigma \# counts $")
        else:
            ax.set_ylabel(r"$ \# counts $")

        return ax

    def save(self, fname):
        """Save matrix to mama file
        """
        mama_write(self, fname, comment="Made by Oslo Method Python")

    def cut_rect(self, axis: Union[int, str],
                 limits: Iterable[float],
                 inplace: bool = True) -> Any:
        """Cuts the matrix to the sub-interval limits along given axis.

        Args:
            axis: Which axis to apply the cut to.
                Can be 0, "Eg" or 1, "Ex", or "both".
            limits: [E_min, E_max, (E_min, E_max)], where
                E_min, E_max: Upper and lower energy limits for cut.
                Supply 4 numbers if 'axis' is 'both'.
            inplace: Whether to make the cut in place or not.

        Returns:
            None if inplace==False
            cut_matrix (Matrix): The cut version of the matrix
        """
        assert(limits[1] >= limits[0])  # Sanity check
        axis = self.axis_toint(axis)
        out = None
        if axis == 0:
            iE_min, iE_max = self.indices_Eg(limits)
            matrix_cut = self.matrix[:, iE_min:iE_max]
            E_cut = self.Eg[iE_min:iE_max]
            if inplace:
                self.matrix = matrix_cut
                self.Eg = E_cut
            else:
                out = Matrix(matrix_cut, E_cut, self.Ex)

        elif axis == 1:
            iE_min, iE_max = self.indices_Ex(limits)
            matrix_cut = self.matrix[iE_min:iE_max, :]
            E_cut = self.Ex[iE_min:iE_max]
            if inplace:
                self.matrix = matrix_cut
                self.Ex = E_cut
            else:
                out = Matrix(matrix_cut, self.Eg, E_cut)
        elif axis == 2:
            iEg_min, iEg_max = self.indicies_Eg(limits[:2])
            iEx_min, iEx_max = self.indicies_Ex(limits[2:])
            matrix_cut = self.matrix[iEg_min:iEg_max, iEx_min:iEx_max]
            Eg_cut = self.Eg[iEg_min:iEg_max]
            Ex_cut = self.Ex[iEx_min:iEx_max]
            if inplace:
                self.matrix = matrix_cut
                self.Eg = Eg_cut
                self.Ex = Ex_cut
            else:
                out = Matrix(matrix_cut, Eg_cut, Ex_cut)

        return out

    def cut_diagonal(self, E1: Iterable[float],
                     E2: Iterable[float]):
        """Cut away counts to the right of a diagonal line defined by indices

        Args:
            E1: First point of intercept, ordered as Ex, Eg
            E2: Second point of intercept
        Returns:
            The matrix with counts above diagonal removed
        """
        self.matrix[self.line_mask(E1, E2)] = 0

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
        line = lambda x: a*x + b

        # Mask all indices below this line to 0
        i_mesh, j_mesh = np.meshgrid(self.range_Eg, self.range_Ex)
        # indexing='ij')
        mask = np.where(j_mesh < line(i_mesh), True, False)
        return mask

    def diagonal_mask(self, Ex_min: float, Ex_max: float,
                      Eg_min: float) -> np.ndarray:
        """Create a trapezoidal mask delimited by the diagonal of the matrix

        Args:
            Ex_min: The bottom edge of the trapezoid
            Ex_max: The top edge of the trapezoid
            Eg_min: The left edge of the trapezoid
        Returns:
            The boolean array with counts below the line set to False
        TODO: Doesn't work. Use Fabio's method
        """
        # Transform to index basis
        iEx_min, iEx_max = self.indices_Eg([Ex_min, Ex_max])
        iEg_min = self.index_Ex(Eg_min)

        mask = np.zeros_like(self.matrix, dtype=bool)
        for iEx in range(iEx_min, iEx_max+1):
            # Loop over rows in array and fill with ones up to sliding Eg
            # threshold
            Ex = self.matrix[iEx]
            # Assume constant particle resolution
            dE_particle = DE_PARTICLE
            dE_gamma = ((DE_GAMMA_8MEV - DE_GAMMA_1MEV) / (8000 - 1000)
                        * (Ex - 1000))
            Eg_max = Ex + np.sqrt(dE_particle**2 + dE_gamma**2)
            iEg_max = self.index_Eg(Eg_max)
            if iEg_max < iEg_min:
                continue

            mask[iEx, iEg_min:iEg_max+1] = True

        return mask

    def fill_negative(self, window_size):
        self.matrix = fill_negative(self.matrix, window_size)

    def remove_negative(self):
        self.matrix = np.where(self.matrix > 0, self.matrix, 0)

    def index_Eg(self, E: float) -> int:
        """ Returns the closest index corresponding to the Eg value """
        return np.abs(self.Eg - E).argmin()

    def index_Ex(self, E: float) -> int:
        """ Returns the closest index corresponding to the Ex value """
        return np.abs(self.Ex - E).argmin()

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
        return self.matrix.sum()

    @property
    def shape(self) -> Tuple[int]:
        return self.matrix.shape

    def axis_toint(self, axis: Any) -> int:
        """Maps axis to 0, 1 or 2 according to which axis is specified

        Args:
            axis: Can be 0, 1, 'Eg', 'Ex', 'both', 2
        Returns:
            An int describing the axis in the basis of the plot,
            _not_ the matrix' dimension.
        Raises:
            AttributeError if the axis is not supported
        """
        try:
            axis = axis.lower()
        except AttributeError:
            pass

        if axis in (0, 'eg'):
            return 0
        elif axis in (1, 'ex'):
            return 1
        elif axis in (2, 'both', 'egex', 'exeg'):
            return 2
        else:
            raise ValueError(f"Unrecognized axis: {axis}")


class Vector():
    def __init__(self, vector=None, E_array=None):
        self.vector = vector
        self.E_array = E_array

    def calibration(self):
        """Calculate and return the calibration coefficients of the energy axes
        """
        calibration = None
        if (self.vector is not None and self.E_array is not None):
            calibration = {
                           # Formatted as "a{axis}{power of E}"
                           "a0": self.E_array[0],
                           "a1": self.E_array[1]-self.E_array[0],
                          }
        else:
            raise RuntimeError("calibration() called on empty Vector instance")
        return calibration

    def plot(self, ax=None, yscale="linear", ylim=None, xlim=None,
             title=None, label=None):
        if ax is None:
            f, ax = plt.subplots(1, 1)

        # Plot with middle-bin energy values:
        E_array_midbin = self.E_array + self.calibration()["a1"]/2
        if label is None:
            ax.plot(E_array_midbin, self.vector)
        elif isinstance(label, str):
            ax.plot(E_array_midbin, self.vector, label=label)
        else:
            raise ValueError("Keyword label must be None or string, but is",
                             label)

        ax.set_yscale(yscale)
        if ylim is not None:
            ax.set_ylim(ylim)
        if xlim is not None:
            ax.set_xlim(xlim)
        if title is not None:
            ax.set_title(title)
        if ax is None:
            plt.show()
        return True

    def save(self, fname):
        """
        Save vector to mama file
        """
        raise NotImplementedError("Not implemented yet")

        return None

    def load(self, fname):
        """
        Load vector from mama file
        """
        raise NotImplementedError("Not implemented yet")

        return None
