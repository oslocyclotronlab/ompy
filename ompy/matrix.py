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
from matplotlib.colors import LogNorm, Normalize
from typing import Dict, Iterable, Any, Tuple
from enum import Enum, unique
from .library import (mama_read, mama_write, i_from_E,
                      div0, fill_negative)


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

    Attributes:
        matrix: 2D matrix storing the counting data
        E0_array: The gamma energy along the x-axis, Eγ
        E1_array: The excitation energy along the y-axis, Ex
        std: Array of standard deviations
        state: An enum to keep track of what has been done to the matrix
    """
    def __init__(self, matrix: np.ndarray = None,
                 E0_array: np.ndarray = None,
                 E1_array: np.ndarray = None,
                 std: np.ndarray = None,
                 filename: str = None):
        """
        There is the option to initialize it in an empty state.
        In that case, all class variables will be None.
        It can be filled later using the load() method.
        """

        # Fill class variables:
        self.matrix: np.ndarray = matrix
        self.E0_array: np.ndarray = E0_array
        self.E1_array: np.ndarray = E1_array
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
        # Check shapes:
        if self.E0_array is not None:
            if self.matrix.shape[0] != len(self.E0_array):
                raise ValueError("Shape mismatch between matrix and E0_array.")
        if self.E1_array is not None:
            if self.matrix.shape[1] != len(self.E1_array):
                raise ValueError("Shape mismatch between matrix and E1_array.")
        if self.std is not None:
            if self.matrix.shape != self.std.shape:
                raise ValueError("Shape mismatch between self.matrix and std.")

    def load(self, fname: str):
        """ Load matrix from mama file

        Args:
            fname: Path to mama file
        """
        if self.matrix is not None:
            warnings.warn("load() called on non-empty matrix")

        # Load matrix from file:
        matrix_object = mama_read(fname)
        self.matrix = matrix_object.matrix
        self.E0_array = matrix_object.E0_array
        self.E1_array = matrix_object.E1_array

        self.verify_integrity()

    def calibration(self) -> Dict[str, np.ndarray]:
        """ Calculates the calibration coefficients of the energy axes

        Returns: The calibration coefficients in a dictionary.
        """
        # Formatted as "a{axis}{power of E}"
        calibration = {
            "a00": self.E0_array[0],
            "a01": self.E0_array[1]-self.E0_array[0],
            "a10": self.E1_array[0],
            "a11": self.E1_array[1]-self.E1_array[0],
        }
        return calibration

    def plot(self, ax: Any = None, title: str = None, zscale: str = "log",
             zmin: float = None, zmax: float = None) -> Any:
        """ Plots the matrix with the energy along the axis

        Args:
            ax: A matplotlib axis to plot onto
            title: Defaults to the current matrix state
            zscale: Scale along the z-axis. Defaults to logarithmic
            vmin: Minimum value for coloring in scaling
            vmax Maximum value for coloring in scaling
        Returns:
            The ax used for plotting
        Raises:
            ValueError: If zscale is unsupported
        """
        if ax is None:
            fig, ax = plt.subplots()
        if zscale == 'log':
            norm = LogNorm(vmin=zmin, vmax=zmax)
        elif zscale == 'linear':
            norm = Normalize(vmin=zmin, vmax=zmax)
        else:
            raise ValueError("Unsupported zscale ", zscale)

        lines = ax.pcolormesh(self.E1_array, self.E0_array, self.matrix,
                              norm=norm)
        ax.set_title(title if title is not None else self.state)
        ax.set_xlabel(r"$\gamma$-ray energy $E_{\gamma}$ [eV]")
        ax.set_ylabel(r"Excitation energy $E_{x}$ [eV]")
        cbar = fig.colorbar(lines, ax=ax)
        cbar.ax.set_ylabel("# counts")
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
        """
        if ax is None:
            fig, ax = plt.subplots()
        if axis not in (0, 1):
            raise ValueError(f"Axis must be 0 or 1, got {axis}.")

        indexE = self.index_E0 if axis else self.index_E1
        rangeE = self.range_E0 if axis else self.range_E1
        imin = indexE(Emin) if Emin is not None else rangeE[0]
        imax = indexE(Emax) if Emax is not None else rangeE[-1]
        subset = slice(imin, imax)
        selection = self.matrix[subset, :] if axis else self.matrix[:, subset]

        naxis = 0 if axis else 1

        projection = selection.sum(axis=naxis)
        if normalize:
            # Don't know what calibration does, so using specialized code here
            if not axis:
                projection = div0(selection, selection.sum(axis=axis))
                projection = projection.mean(axis=naxis)
            else:
                calibration = self.calibration()["a01"]
                calibrated_sum = selection.sum(axis=axis)*calibration
                calibrated = div0(selection, calibrated_sum[:, None])
                projection = calibrated.mean(axis=naxis)

        if axis:
            ax.plot(self.E1_array, projection)
            ax.set_xlabel(r"$\gamma$-ray energy $E_{\gamma}$ [eV]")
        else:
            ax.plot(self.E0_array, projection)
            ax.set_xlabel(r"Excitation energy $E_{x}$ [eV]")

        if normalize:
            ax.set_ylabel(r"$\# counts/\Sigma \# counts $")
        else:
            ax.set_ylabel(r"$ \# counts $")

        return ax

    def save(self, fname):
        """ Save matrix to mama file
        """
        mama_write(self, fname, comment="Made by Oslo Method Python")

    def cut_rect(self, axis, E_limits, inplace=True):
        """
        Cuts the matrix (and std, if present) to the sub-interval E_limits.

        Args:
            axis (int): Which axis to apply the cut to.
            E_limits (list): [E_min, E_max], where
                E_min, E_max (float): Upper and lower energy limits for cut
            inplace (bool): Whether to make the cut in place or not

        Returns:
            None if inplace==False
            cut_matrix (Matrix): The cut version of the matrix
        """
        assert(E_limits[1] >= E_limits[0])  # Sanity check
        matrix_cut = None
        std_cut = None
        out = None
        if axis == 0:
            i_E_min = np.argmin(np.abs(self.E0_array-E_limits[0]))
            i_E_max = np.argmin(np.abs(self.E0_array-E_limits[1]))
            matrix_cut = self.matrix[i_E_min:i_E_max, :]
            E0_array_cut = self.E0_array[i_E_min:i_E_max]
            if inplace:
                self.matrix = matrix_cut
                self.E0_array = E0_array_cut
            else:
                out = Matrix(matrix_cut, E0_array_cut, E1_array)

        elif axis == 1:
            i_E_min = np.argmin(np.abs(self.E1_array-E_limits[0]))
            i_E_max = np.argmin(np.abs(self.E1_array-E_limits[1]))
            matrix_cut = self.matrix[:, i_E_min:i_E_max]
            E1_array_cut = self.E1_array[i_E_min:i_E_max]
            if inplace:
                self.matrix = matrix_cut
                self.E1_array = E1_array_cut
            else:
                out = Matrix(matrix_cut, E0_array, E1_array_cut)
        else:
            raise ValueError("Axis must be one of (0, 1), but is", axis)

        return out

    def cut_diagonal(self, E1: Iterable, E2: Iterable):
        """
        Cut away counts to the right of a diagonal line defined by indices

        Args:
            E1: First point of intercept, ordered as Ex, Eg
            E2: Second point of intercept
        Returns:
            The matrix with counts above diagonal removed
        """
        # Transform from energy to index basis
        Ex1, Ey1 = E1
        Ex2, Ey2 = E2
        Ix = self.indices_E0([Ex1, Ex2])
        Iy = self.indices_E1([Ey1, Ey2])

        # Interpolate between the two points
        a = (Iy[1]-Iy[0])/(Ix[1]-Ix[0])
        b = Iy[0] - a*Ix[0]
        line = lambda x: a*x + b

        # Mask all indices below this line to 0
        i_mesh, j_mesh = np.meshgrid(self.range_E0, self.range_E1,
                                     indexing='ij')
        mask = np.where(j_mesh > line(i_mesh), True, False)

        self.matrix[mask] = 0

    def fill_negative(self, window_size):
        self.matrix = fill_negative(self.matrix, window_size)

    def remove_negative(self):
        self.matrix = np.where(self.matrix > 0, self.matrix, 0)

    def index_E0(self, E: float) -> int:
        """ Returns the closest index corresponding to the E0 value """
        return np.abs(self.E0_array - E).argmin()

    def index_E1(self, E: float) -> int:
        """ Returns the closest index corresponding to the E1 value """
        return np.abs(self.E1_array - E).argmin()

    def indices_E0(self, E: Iterable) -> np.ndarray:
        """ Returns the closest indices corresponding to the E0 value"""
        indices = [self.index_E0(e) for e in E]
        return np.array(indices)

    def indices_E1(self, E: Iterable) -> np.ndarray:
        """ Returns the closest indices corresponding to the E1 value"""
        indices = [self.index_E1(e) for e in E]
        return np.array(indices)

    @property
    def range_E0(self) -> np.ndarray:
        """ Returns all indices of E0_array """
        return np.arange(0, len(self.E0_array), dtype=int)

    @property
    def range_E1(self) -> np.ndarray:
        """ Returns all indices of E1_array """
        return np.arange(0, len(self.E1_array), dtype=int)

    @property
    def counts(self) -> float:
        return self.matrix.sum()


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
