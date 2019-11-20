from __future__ import annotations
from typing import Optional, Iterable, Union, Any, Tuple, Dict
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray
from .filehandling import (load_numpy_1D, save_numpy_1D,
                           mama_read, mama_write,
                           filetype_from_suffix,
                           load_tar, save_tar)
from .decomposition import index
from .library import div0
from .abstractarray import AbstractArray


class Vector(AbstractArray):
    """ Stores 1d array with energy axes (a vector)

    Attributes:
        values (np.ndarray): The values at each bin.
        E (np.ndarray): The energy of each bin. (mid-bin calibration)
        path (str or Path): The path to load a saved vector from
        std (np.ndarray): The standard deviation of the counts
        unit (str): Unit of the energies. Can be "keV" or "MeV".
            Defaults to "keV".
    """

    def __init__(self, values: Optional[Iterable[float]] = None,
                 E: Optional[Iterable[float]] = None,
                 path: Optional[Union[str, Path]] = None,
                 std: Optional[Iterable[float]] = None,
                 units: Optional[str] = "keV"):
        """
        There are several ways to initialize

        - An initialization with only `E`, or energy, defaults to
          zero values.
        - An initialization using a path loads a vector from said path.

        If no `std` is given, it will default to None

        Args:
            values: see above
            E: see above
            path: see above
            std: see above
            unit: see above

        Raises:
           ValueError if the given arrays are of differing lenghts.

        ToDo:
            - Crean up initialization.
        """
        if values is not None and E is not None:
            self.values = np.asarray(values, dtype=float).copy()
            self.E = np.asarray(E, dtype=float).copy()
        elif values is None and E is not None:
            self.E = np.asarray(E, dtype=float).copy()
            self.values = np.zeros_like(E)
        else:
            self.values = None
            self.E = None

        if std is not None:
            std = np.asarray(std, dtype=float).copy()
        self.std: Optional[ndarray] = std

        self.units = units

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

    def calibration(self) -> Dict[str, float]:
        """Calculate and return the calibration coefficients of the energy axes

        Formatted as "a{axis}{power of E}"
        """

        calibration = {"a0": self.E[0],
                       "a1": self.E[1]-self.E[0]}
        return calibration

    def plot(self, ax: Optional[Any] = None,
             scale: str = 'linear',
             kind: str = 'line',
             **kwargs) -> Tuple[Any, Any]:
        """ Plots the vector

        Args:
            ax (matplotlib axis, optional): The axis to plot onto. If not
                provided, a new figure is created
            scale (str, optional): The scale to use. Can be `linear`
                (default), `log`, `symlog` or `logit`.
            kind (str, optional):
                - 'line' : line plot (default) evokes `ax.plot`
                - 'plot' : same as 'line'
                - 'step' : step plot
                - 'bar' : vertical bar plot
            kwargs (optional): Additional kwargs to plot command.

        Returns:
            The figure and axis used.
        """
        fig, ax = plt.subplots() if ax is None else (ax.figure, ax)

        if kind in ["plot", "line"]:
            kwargs.setdefault("markersize", "2")
            ax.plot(self.E, self.values, "o-", **kwargs)
        elif kind == "step":
            kwargs.setdefault("where", "mid")
            ax.step(self.E, self.values, **kwargs)
        else:
            raise NotImplementedError()
        # ax.xaxis.set_major_locator(MeshLocator(self.E))
        ax.set_yscale(scale)
        ax.set_xlabel("Energy")

        if self.std is not None:
            # TODO: Fix color
            ax.errorbar(self.E, self.values, yerr=self.std,
                        fmt='o', ms=1, lw=1, color='k')
        return fig, ax

    def save(self, path: Union[str, Path],
             filetype: Optional[str] = None) -> None:
        """Save to a file of specified format

        Args:
            path (str or Path): Path to save
            filetype (str, optional): Filetype. Default uses
                auto-recognition from suffix.

        Raises:
            ValueError: Filetype is not supported
        """
        vector = self.copy()
        vector.to_keV()
        path = Path(path) if isinstance(path, str) else path
        if filetype is None:
            filetype = filetype_from_suffix(path)
        filetype = filetype.lower()
        if filetype == 'numpy':
            save_numpy_1D(vector.values, vector.E, path)
        elif filetype == 'tar':
            save_tar([vector.values, vector.E], path)
        elif filetype == 'mama':
            mama_write(self, path)
        else:
            raise ValueError(f"Unknown filetype {filetype}")

    def load(self, path: Union[str, Path],
             filetype: Optional[str] = None) -> None:
        """Load to a file of specified format

        Args:
            path (str or Path): Path to Load
            filetype (str, optional): Filetype. Default uses
                auto-recognition from suffix.

        Raises:
            ValueError: Filetype is not supported
        """
        path = Path(path) if isinstance(path, str) else path
        if filetype is None:
            filetype = filetype_from_suffix(path)
        filetype = filetype.lower()

        if filetype == 'numpy':
            self.values, self.E = load_numpy_1D(path)
        elif filetype == 'tar':
            self.values, self.E = load_tar(path)
        elif filetype == 'mama':
            self.values, self.E = mama_read(path)
        else:
            try:
                self.values, self.E = mama_read(path)
            except ValueError:  # from within ValueError
                raise ValueError(f"Unknown filetype {filetype}")
        self.verify_integrity()

        return None

    def transform(self, const: float = 1,
                  alpha: float = 0, inplace: bool = True) -> Optional[Vector]:
        """Apply a normalization transformation::

            vector -> const * vector * exp(alpha*energy)

        If the vector has `std`, the `std` will be transformed
        as well.

        Args:
            const (float, optional): The constant. Defaults to 1.
            alpha (float, optional): The exponential coefficient.
                Defaults to 0.
            inplace (bool, optional): Whether to apply the transformation
                inplace. If False, returns the transformed vector.

        Returns:
            Optional[Vector]
        """
        if self.std is not None:
            relative_uncertainty = self.std / self.values
        transformed = const*self.values*np.exp(alpha*self.E)

        if self.std is not None:
            std = relative_uncertainty * transformed
        if not inplace:
            units = self.units
            if self.std is not None:
                return Vector(transformed, E=self.E, std=std, units=units)
            return Vector(transformed, E=self.E, units=units)

        self.values = transformed
        if self.std is not None:
            self.std = std

    def error(self, other: Union[Vector, ndarray],
              std: Optional[ndarray] = None) -> float:
        """Computes the (weighted) χ²

        Args:
            other (Vector or ndarray]): The reference to compare itself to. If
                an array, assumes it has the same energy binning as itself.
            std (Optional[ndarray], optional): Standard deviations to use as
                inverse of the weights.

        Returns:
            float: χ²

        """
        # Hack since something is screwy with the import
        # |-> is this comment still up to date?
        try:
            self.has_equal_binning(other)
            other = other.values
        except TypeError:  # already an array
            pass
        squared_error = (self.values - other)**2
        if self.std is not None:
            if std is not None:
                sigmasq = self.std**2 + std**2
            else:
                sigmasq = self.std**2
        else:
            if std is not None:
                sigmasq = std**2
            else:
                sigmasq = 1

        error = div0(squared_error, sigmasq)
        return error.sum()

    def cut(self, Emin: Optional[float] = None,
            Emax: Optional[float] = None,
            inplace: bool = True) -> Optional[Vector]:
        """ Cut the vector at the energy limits

        Args:
            Emin (float, optional): The lower energy limit
            Emax (float, optional): The higher energy limit
            inplace (bool, optional): If `True` perform the cut on this vector
                or if `False` returns a copy. Defaults to True
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
            units = self.units
            return Vector(values=values, E=E, std=std, units=units)

    def cut_nan(self, inplace: bool = True) -> Vector:
        """ Cut the vector where elements are `np.nan`

        Args:
            inplace (bool, optional): If `True` perform the cut on this vector
                or if `False` returns a copy. Defaults to True
        Returns:
            The cut vector if `inplace` is True.
        """
        inan = np.argwhere(np.isnan(self.values))

        values = np.delete(self.values, inan)
        E = np.delete(self.E, inan)
        if inplace:
            self.values = values
            self.E = E
        else:
            return Vector(values=values, E=E, units=self.units)

    def to_MeV(self) -> Vector:
        """ Convert E from keV to MeV if necessary """
        if self.units == "MeV":
            pass
        elif self.units == "keV":
            self.E /= 1e3
            self.units = "MeV"
        else:
            raise NotImplementedError("Units must be keV or MeV")

    def to_keV(self) -> Vector:
        """ Convert E from MeV to keV if necessary """
        if self.units == "keV":
            pass
        elif self.units == "MeV":
            self.E *= 1e3
            self.units = "keV"
        else:
            raise NotImplementedError("Units must be keV or MeV")

    def has_equal_binning(self, other: Vector, **kwargs) -> bool:
        """ Check whether `other` has equal_binning as `self` within precision.

        Args:
            other (Vector): Vector to compare to.
            kwargs: Additional kwargs to `np.allclose`.

        Returns:
            Returns `True` if both arrays are equal  .

        Raises:
            TypeError: If other is not a Vector.
            ValueError: If any of the bins in any of the arrays are not equal.

        """
        if not isinstance(other, Vector):
            raise TypeError("Other must be a Vector")
        if self.shape != other.shape:
            raise ValueError("Must have equal number of energy bins.")
        if not np.allclose(self.E, other.E, **kwargs):
            raise ValueError("Must have equal energy binning.")
        else:
            return True

    def index(self, E: float) -> int:
        """ Returns the closest index corresponding to the E value """
        return index(self.E, E)

    def __matmul__(self, other: Vector) -> Vector:
        result = self.copy()
        if isinstance(other, Vector):
            self.has_equal_binning(other)
        else:
            NotImplementedError("Type not implemented")

        result.values = result.values@other.values
        return result
