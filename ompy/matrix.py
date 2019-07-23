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
import copy
from matplotlib.colors import LogNorm, Normalize
from typing import Dict, Iterable, Any, Union, Tuple
from enum import IntEnum
from .matrixstate import MatrixState
from .library import mama_read, mama_write, div0, fill_negative, diagonal_resolution
from .constants import DE_PARTICLE, DE_GAMMA_1MEV, DE_GAMMA_8MEV


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
        mask: A boolean array for cutting the array
    TODO:
        * Find a way to handle units
        * Synchronize cuts. When a cut is made along one axis,
          such as values[min:max, :] = 0, make cuts to the
          other relevant variables
    """
    def __init__(self, values: np.ndarray = None,
                 Eg: np.ndarray = None,
                 Ex: np.ndarray = None,
                 std: np.ndarray = None,
                 filename: str = None,
                 state: Union[str, MatrixState] = 'raw'):
        """
        There is the option to initialize it in an empty state.
        In that case, all class variables will be None.
        It can be filled later using the load() method.
        """

        self.values: np.ndarray = values
        self.Eg: np.ndarray = np.array(Eg, dtype=float)
        self.Ex: np.ndarray = np.array(Ex, dtype=float)
        self.std: np.ndarray = std

        if filename is not None:
            self.load(filename)
        self.verify_integrity()

        self.state = state
        self.mask = None

    def verify_integrity(self):
        """ Runs checks to verify internal structure

        Raises:
            ValueError: If any check fails
        """
        if self.values is None:
            return

        # Check shapes:
        shape = self.values.shape
        if self.Ex is not None:
            if shape[0] != len(self.Ex):
                raise ValueError(("Shape mismatch between matrix and Ex:"
                                  f" (_{shape[0]}_, {shape[1]}) ≠ "
                                  f"{len(self.Ex)}"))
        if self.Eg is not None:
            if shape[1] != len(self.Eg):
                raise ValueError(("Shape mismatch between matrix and Eg:"
                                  f" (_{shape[0]}_, {shape[1]}) ≠ "
                                  f"{len(self.Eg)}"))
        if self.std is not None:
            if shape != self.std.shape:
                raise ValueError("Shape mismatch between self.values and std.")

    def load(self, filename: str):
        """ Load matrix from mama file

        Args:
            filename: Path to mama file
        """
        if self.values is not None:
            warnings.warn("load() called on non-empty matrix")

        # Load matrix from file:
        matrix, Eg, Ex = mama_read(filename)
        self.values = matrix
        self.Eg = Eg
        self.Ex = Ex

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
             zmin: float = None, zmax: float = None, **kwargs) -> Any:
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
        """
        fig, ax = plt.subplots() if ax is None else (None, ax)
        if zscale == 'log':
            norm = LogNorm(vmin=zmin, vmax=zmax)
        elif zscale == 'linear':
            norm = Normalize(vmin=zmin, vmax=zmax)
        else:
            raise ValueError("Unsupported zscale ", zscale)
        # TODO: Pcolormesh ignores the last row of self.values if
        # len(self.Eg) < len(self.values) + 1
        # Must extend it
        Eg = np.append(self.Eg, self.Eg[-1] + self.Eg[1] - self.Eg[0])
        Ex = np.append(self.Ex, self.Ex[-1] + self.Ex[1] - self.Ex[0])
        lines = ax.pcolormesh(Eg, Ex, self.values, norm=norm, **kwargs)
        ax.set_title(title if title is not None else self.state)
        ax.set_xlabel(r"$\gamma$-ray energy $E_{\gamma}$ [eV]")
        ax.set_ylabel(r"Excitation energy $E_{x}$ [eV]")
        if fig is not None:
            cbar = fig.colorbar(lines, ax=ax)
            cbar.ax.set_ylabel("# counts")
            plt.show()
        return ax

    def plot_projection(self, axis: int, Emin: float = None,
                        Emax: float = None, ax: Any = None,
                        normalize: bool = False, **kwargs) -> Any:
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

        axis = axis_toint(axis)
        is_Ex = axis == 1
        projection, energy = self.projection(axis, Emin, Emax, normalize=normalize)

        # Shift energy by a half bin to make the steps correct
        shifted_energy = energy + (energy[1] - energy[0])/2

        if is_Ex:
            ax.step(shifted_energy, projection, where='mid', **kwargs)
            ax.set_xlabel(r"Excitation energy $E_{x}$ [eV]")
        else:
            ax.step(energy, projection, where='mid', **kwargs)
            ax.set_xlabel(r"$\gamma$-ray energy $E_{\gamma}$ [eV]")
        if normalize:
            ax.set_ylabel(r"$\# counts/\Sigma \# counts $")
        else:
            ax.set_ylabel(r"$ \# counts $")

        return ax

    def projection(self, axis: Union[int, str], Emin: float = None,
                   Emax: float = None,
                   normalize: bool = False) -> Tuple[np.ndarray]:
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
        axis = axis_toint(axis)
        if axis not in (0, 1):
            raise ValueError(f"Axis must be 0 or 1, got: {axis}")

        isEx = axis == 1

        # Determine subset of the other axis to be summed
        indexE = self.index_Ex if isEx else self.index_Eg
        rangeE = self.range_Ex if isEx else self.range_Eg
        imin = indexE(Emin) if Emin is not None else rangeE[0]
        imax = indexE(Emax) if Emax is not None else rangeE[-1]
        subset = slice(imin, imax+1)
        selection = self.values[subset, :] if isEx else self.values[:, subset]
        energy = self.Ex[subset] if isEx else self.Eg[subset]

        projection = selection.sum(axis=axis)
        if normalize:
            projection = div0(projection, selection.sum(axis=axis).sum())

        return projection, energy

    def save(self, fname):
        """Save matrix to mama file
        """
        mama_write(self, fname, comment="Made by Oslo Method Python")

    def cut(self, axis: Union[int, str],
            Emin: Union[None, float] = None,
            Emax: Union[None, float] = None,
            inplace: bool = True) -> Any:
        """Cuts the matrix to the sub-interval limits along given axis.

        Args:
            axis: Which axis to apply the cut to.
                Can be 0, "Eg" or 1, "Ex".
            Emin: lower energy limit for cut. Defaults to
                lowest energy.
            Emax: upper energy limit for cut. Defaults to
                highest energy.
            inplace: Whether to make the cut in place or not.

        Returns:
            None if inplace==False
            cut_matrix (Matrix): The cut version of the matrix
        """
        axis = axis_toint(axis)
        range = self.Eg if axis == 0 else self.Ex
        indices = self.indices_Eg if axis == 0 else self.indices_Ex
        Emin = Emin if Emin is not None else min(range)
        Emax = Emax if Emax is not None else max(range)
        iEmin, iEmax = indices((Emin, Emax))
        Ecut = range[iEmin:iEmax]

        if axis == 0:
            values_cut = self.values[:, iEmin:iEmax]
            Eg = Ecut
            Ex = self.Ex
        elif axis == 1:
            values_cut = self.values[iEmin:iEmax, :]
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

    def cut_diagonal(self, E1: Iterable[float] = None,
                     E2: Iterable[float] = None,
                     inplace: bool = True) -> Union[None, Any]:
        """Cut away counts to the right of a diagonal line defined by indices

        If no limits are provided, an automatic cut will be made.
        Args:
            E1 (optional): First point of intercept, ordered as Ex, Eg
            E2 (optional): Second point of intercept
            inplace (optional): Whether the operation should be applied to the
                current matrix, or to a copy which is then returned.
        Returns:
            The matrix with counts above diagonal removed if not inplace.
        """
        if E1 is None and E2 is None:
            dEg = np.repeat(diagonal_resolution(self.Ex), len(self.Eg)).reshape(self.shape)
            Eg, Ex = np.meshgrid(self.Eg, self.Ex)
            mask = np.zeros_like(self.values, dtype=bool)
            mask[Eg >= Ex + dEg] = True
        elif E1 is None or E2 is None:
            raise ValueError("If either E1 or E2 is specified, "
                             "both must be specified and have same type")
        else:
            mask = self.line_mask(E1, E2)

        if inplace:
            self.mask = mask
            self.values[mask] = 0.0
        else:
            matrix = copy.deepcopy(self)
            matrix.mask = mask
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

    def diagonal_mask(self, Ex_min: float, Ex_max: float,
                      Eg_min: float) -> np.ndarray:
        """Create a trapezoidal mask delimited by the diagonal of the matrix

        Args:
            Ex_min: The bottom edge of the trapezoid
            Ex_max: The top edge of the trapezoid
            Eg_min: The left edge of the trapezoid
        Returns:
            The boolean array with counts below the line set to False
        TODO: Is this function necessary?
        """
        # Transform to index basis
        iEx_min, iEx_max = self.indices_Eg([Ex_min, Ex_max])
        iEg_min = self.index_Ex(Eg_min)

        Eg, Ex = np.meshgrid(self.Eg, self.Ex)
        Ex_cut = self.Ex[(self.Ex > Ex_min) & (self.Ex < Ex_max)]
        dEg = np.repeat(diagonal_resolution, len(self.Ex)).reshape(self.shape)
        mask = np.zeros_like(self.values, dtype=bool)
        mask[(Ex_min < Ex) & (Ex < Ex_max) & (Eg < Ex + dEg)] = True

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
        #TODO FIX
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
        return self.values.sum()

    @property
    def shape(self) -> Tuple[int]:
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

    def __getitem__(self, key):
        return self.values.__getitem__(key)

    def __setitem__(self, key, item):
        return self.values.__setitem__(key, item)


def axis_toint(axis: Any) -> int:
    """Maps axis to 0, 1 or 2 according to which axis is specified

    Args:
        axis: Can be 0, 1, 'Eg', 'Ex', 'both', 2
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


class Vector():
    def __init__(self, values=None, E=None):
        self.values = values
        self.E = E

    def calibration(self):
        """Calculate and return the calibration coefficients of the energy axes
        """
        calibration = None
        if (self.values is not None and self.E is not None):
            calibration = {
                           # Formatted as "a{axis}{power of E}"
                           "a0": self.E[0],
                           "a1": self.E[1]-self.E[0],
                          }
        else:
            raise RuntimeError("calibration() called on empty Vector instance")
        return calibration

    def plot(self, ax=None, yscale="linear", ylim=None, xlim=None,
             title=None, label=None):
        if ax is None:
            f, ax = plt.subplots(1, 1)

        # Plot with middle-bin energy values:
        E_midbin = self.E + self.calibration()["a1"]/2
        if label is None:
            ax.plot(E_midbin, self.values)
        elif isinstance(label, str):
            ax.plot(E_midbin, self.values, label=label)
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

    def transform(self, const=1, alpha=0, implicit=False):
        """
        Return a transformed version of the vector:
        vector -> const * vector * exp(alpha*E_array)
        """
        E_array_midbin = self.E + self.calibration()["a1"]/2
        vector_transformed = (const * self.values
                              * np.exp(alpha*E_array_midbin)
                              )
        if implicit:
            self.values= vector_transformed
        else:
            return Vector(vector_transformed, E=self.E)
