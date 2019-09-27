from __future__ import annotations
import copy
from typing import Optional, Iterable, Union, Any, Tuple
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray
from .filehandling import (load_numpy_1D, save_numpy_1D, filetype_from_suffix,
                           load_tar, save_tar)
from .matrix import MeshLocator
from .decomposition import index


class Vector():
    def __init__(self, values: Optional[Iterable[float]] = None,
                 E: Optional[Iterable[float]] = None,
                 path: Optional[Union[str, Path]] = None,
                 std: Optional[Iterable[float]] = None):
        """ Create a vector

        An empty initalization defaults to one bin with zero values.
        An initalization with only `values` creates an energy binning
        of 1 MeV starting at 0.5 (midbin).
        An initialization with only `E`, or energy, defaults to
        zero values.
        An initialization using a path loads a vector from said path.

        If no `std` is given, it will default to None

        Args:
            values: The values at each bin.
            E: The energy of each bin.
            path: The path to load a saved vector from
            std: The standard deviation of the counts

        Raises:
           ValueError if the given arrays are of differing lenghts.
        """
        if values is None and E is not None:
            self.E = np.asarray(E, dtype=float)
            self.values = np.zeros_like(E)
        elif values is not None and E is None:
            self.values = np.asarray(values, dtype=float)
            self.E = np.arange(0.5, len(self.values), 1)
        elif values is None and E is None:
            self.values = np.zeros(1)
            self.E = np.array([0.5])
        else:
            self.values = np.asarray(values, dtype=float)
            self.E = np.asarray(E, dtype=float)

        if std is not None:
            std = np.asarray(std, dtype=float)
        self.std: Optional[ndarray] = std

        if path is not None:
            self.load(path)
        self.verify_integrity()

    def verify_integrity(self):
        """ Verify the internal consistency of the vector

        Raises:
            AssertionError or ValueError if any test fails
        """
        assert self.values is not None
        assert self.E is not None
        if self.E.shape != self.values.shape:
            raise ValueError("Energy and values must have same shape")
        if self.std is not None:
            if self.std.shape != self.values.shape:
                raise ValueError("std and values must have same shape")

    def calibration(self):
        """Calculate and return the calibration coefficients of the energy axes
        """
        #  Formatted as "a{axis}{power of E}"
        calibration = {"a0": self.E[0],
                       "a1": self.E[1]-self.E[0]}
        return calibration

    def plot(self, ax: Optional[Any] = None,
             scale: str = 'linear',
             xlabel: Optional[str] = "Energy",
             ylabel: Optional[str] = None, **kwargs) -> Tuple[Any, Any]:
        """ Plots the vector as a step graph

        Args:
            ax: The axis to plot onto.
            scale: The scale to use. Can be `linear`, `log`
                `symlog` or `logit`.
            xlabel (optional, str): Label on x-axis. Default is `"Energy"`.
            ylabel (optional, str): Label on y-axis. Default is `None`.
        Returns:
            The figure and axis used.
        """
        fig, ax = plt.subplots() if ax is None else (None, ax)

        ax.step(self.E, self.values, where='mid', **kwargs)
        #ax.xaxis.set_major_locator(MeshLocator(self.E))
        ax.set_yscale(scale)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        if self.std is not None:
            # TODO: Fix color
            ax.errorbar(self.E, self.values, yerr=self.std,
                        fmt='o', ms=1, lw=1, color='k')
        return fig, ax

    def save(self, path: Union[str, Path],
             filetype: Optional[str] = None) -> None:
        """ Save to a file of specified format

        Raises:
            ValueError if the filetype is not supported
        """
        path = Path(path) if isinstance(path, str) else path
        if filetype is None:
            filetype = filetype_from_suffix(path)
        filetype = filetype.lower()
        if filetype == 'numpy':
            save_numpy_1D(self.values, self.E, path)
        elif filetype == 'tar':
            save_tar([self.values, self.E], path)
        else:
            raise ValueError(f"Unknown filetype {filetype}")

    def load(self, path: Union[str, Path],
             filetype: Optional[str] = None) -> None:
        """ Load vector from specified format
        """
        path = Path(path) if isinstance(path, str) else path
        if filetype is None:
            filetype = filetype_from_suffix(path)
        filetype = filetype.lower()

        if filetype == 'numpy':
            self.values, self.E = load_numpy_1D(path)
        elif filetype == 'tar':
            self.values, self.E = load_tar(path)
        else:
            raise ValueError(f"Unknown filetype {filetype}")
        self.verify_integrity()

        return None

    def transform(self, const=1, alpha=0, inplace=True) -> Optional[Vector]:
        """ Apply a normalization transformation

        vector -> const * vector * exp(alpha*energy)

        If the vector has `std`, the `std` will be transformed
        as well.

        Args:
            const: The constant. Defaults to 1.
            alpha: The exponential coefficient. Defaults to 0.
            inplace: Whether to apply the transformation inplace.
                If False, returns the transformed vector.
        Returns:
            The transformed vector if `inplace` is False.
        """
        if self.std is not None:
            relative_uncertainty = self.std / self.values
        transformed = const*self.values*np.exp(alpha*self.E)

        if self.std is not None:
            std = relative_uncertainty * transformed
        if not inplace:
            if self.std is not None:
                return Vector(transformed, E=self.E, std=std)
            return Vector(transformed, E=self.E)

        self.values = transformed
        if self.std is not None:
            self.std = std

    def error(self, other: Union[Vector, ndarray],
              std: Optional[ndarray] = None) -> float:
        """ Computes the χ² error or MSE

        Args:
            other: The reference to compare itself to. If an
                array, assumes it has the same energy binning
                as itself.
        """
        # Hack since something is screwy with the import
        if hasattr(other, 'E'):
            if np.all(other.E != self.E):
                raise ValueError("Must have equal energy binning")
            other = other.values
        squared_error = (self.values - other)**2
        if self.std is not None:
            error = squared_error / self.std**2
        else:
            error = squared_error

        if std is not None:
            # The std overwrites the std of self.
            error = squared_error / std**2

        return error.sum()

    def cut(self, Emin: Optional[float] = None,
            Emax: Optional[float] = None,
            inplace: bool = True) -> Optional[Vector]:
        """ Cut the vector at the energy limits

        Args:
            Emin: The lower energy limit
            Emax: The higher energy limit
            inplace: Whether to perform the cut on this vector
                or (False) return a copy.
        Returns:
            The cut vector if `inplace` is True.
        """
        Emin = Emin if Emin is not None else self.E.min()
        Emax = Emax if Emax is not None else self.E.max()
        imin = self.index(Emin)
        imax = self.index(Emax)
        cut = slice(imin, imax + 1)

        values = self.values[cut]
        E = self.E[cut]
        if self.std is not None:
            std = self.std[cut]
        else:
            std = None

        if inplace:
            self.E = E
            self.values = values
            self.std = std
        else:
            return Vector(values=values, E=E, std=std)

    def copy(self) -> Vector:
        """ Return a copy of the matrix """
        return copy.deepcopy(self)

    def index(self, E) -> int:
        """ Returns the closest index corresponding to the E value """
        return index(self.E, E)
