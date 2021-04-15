from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray

from . import ureg
from .abstractarray import AbstractArray
from .decomposition import index
from .filehandling import (filetype_from_suffix, load_csv_1D, load_numpy_1D,
                           load_tar, load_txt_1D, mama_read, mama_write,
                           save_csv_1D, save_numpy_1D, save_tar, save_txt_1D)
from .library import div0
from .rebin import rebin_1D

LOG = logging.getLogger(__name__)
logging.captureWarnings(True)


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
                 *,
                 path: Optional[Union[str, Path]] = None,
                 std: Optional[Iterable[float]] = None,
                 units: Optional[str] = "keV"):
        """
        There are several ways to initialize

        - Providing only the energy `E` will
            create values of same size with random values.
        - Providing only a path loads a vector from said path.
        - Any other combination will yield an error.

        If no `std` is given, it will default to None

        Args:
            values: see above
            E: see above
            path: see above
            std: see above
            unit: see above

        Raises:
           ValueError if the runtime lengths of the arrays are different.
           ValueError if incompatible arguments are provided.

        #TODO
            - Clean up initialization.
        """
        try:
            unit = E.units
        except AttributeError:
            unit = ureg(units)

        if path is not None:
            self.load(path)
        else:
            if values is not None and E is not None:
                self.values = np.asarray(values, dtype=float).copy()
                try:
                    self.E = np.asarray(E.magnitude, dtype=float).copy() * unit
                except AttributeError:
                    self.E = np.asarray(E, dtype=float).copy() * unit
            elif values is None and E is not None:
                try:
                    self.E = np.asarray(E.magnitude, dtype=float).copy() * unit
                except AttributeError:
                    self.E = np.asarray(E, dtype=float).copy() * unit
                self.values = np.empty_like(E)
            elif values is not None and E is None:
                raise ValueError("The values can not provided without the energy.")
                # Alternative solution:
                # self.values = np.asarray(values, dtype=float)
                # self.E = np.arange(0, len(self.values)) * unit
            else:
                raise ValueError('Can not construct zero-size vector')

        if std is not None:
            std = np.asarray(std, dtype=float).copy()
        self.std: Optional[ndarray] = std

        if path is not None:
            self.load(path)
        self.verify_integrity()

    def verify_integrity(self, check_equidistant: bool = False):
        """ Verify the internal consistency of the vector

        Args:
            check_equidistant (bool, optional): Check whether energy array
                are equidistant spaced. Defaults to False.

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

        if check_equidistant:
            if not self.is_equidistant():
                raise ValueError("Is not equidistant.")

    def is_equidistant(self) -> bool:
        """ Check if the width of energy bins are all equal."""
        if len(self) == 1:
            return True

        diff = (self.E.magnitude - np.roll(self.E.magnitude, 1))[1:]
        de = diff[0]
        return np.all(np.isclose(diff, de*np.ones(diff.shape)))

    def calibration(self) -> Dict[str, float]:
        """Calculate and return the calibration coefficients of the energy axes

        Formatted as "a{axis}{power of E}"
        """

        calibration = {"a0": self.E[0].to('keV').magnitude,
                       "a1": (self.E[1]-self.E[0]).to('keV').magnitude}
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
        units = self.E.units

        if self.std is None:
            if kind in ["plot", "line"]:
                kwargs.setdefault("markersize", 3)
                kwargs.setdefault("marker", "o")
                kwargs.setdefault("linestyle", "-")
                ax.plot(self.E, self.values, "o-", **kwargs)
            elif kind == "step":
                kwargs.setdefault("where", "mid")
                ax.step(self.E, self.values, **kwargs)
            else:
                raise NotImplementedError()
        else:
            kwargs.setdefault("markersize", 3)
            kwargs.setdefault("linewidth", 2)
            kwargs.setdefault("fmt", "o")
            ax.errorbar(self.E, self.values, yerr=self.std, **kwargs)
        ax.set_yscale(scale)
        ax.set_xlabel("Energy" + f" [${units:~L}$]")
        return fig, ax

    def save(self, path: Union[str, Path],
             filetype: Optional[str] = None,
             **kwargs) -> None:
        """Save to a file of specified format

        Args:
            path (str or Path): Path to save
            filetype (str, optional): Filetype. Default uses
                auto-recognition from suffix.
                Options: ["numpy", "txt", "tar", "mama", "csv"]
            **kwargs: additional keyword arguments

        Raises:
            ValueError: Filetype is not supported
        """
        vector = self.copy()
        path = Path(path) if isinstance(path, str) else path
        if filetype is None:
            filetype = filetype_from_suffix(path)
        filetype = filetype.lower()

        E = vector.E.to('keV').magnitude
        if filetype == 'numpy':
            save_numpy_1D(vector.values, E, vector.std, path)
        elif filetype == 'txt':
            save_txt_1D(vector.values, E, vector.std, path, **kwargs)
        elif filetype == 'tar':
            if vector.std is not None:
                save_tar([vector.values, E, vector.std], path)
            else:
                save_tar([vector.values, E], path)
        elif filetype == 'mama':
            mama_write(self, path, comment="Made by OMpy", **kwargs)
            if self.std is not None:
                warnings.warn("MaMa cannot store std. "
                              "Consider using another format")
        elif filetype == 'csv':
            save_csv_1D(vector.values, E, vector.std, path)
        else:
            raise ValueError(f"Unknown filetype {filetype}")

    def load(self, path: Union[str, Path],
             filetype: Optional[str] = None) -> None:
        """Load to a file of specified format

        Units assumed to be keV.

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
            self.values, self.E, self.std = load_numpy_1D(path)
        elif filetype == 'txt':
            self.values, self.E, self.std = load_txt_1D(path)
        elif filetype == 'tar':
            from_file = load_tar(path)
            if len(from_file) == 3:
                self.values, self.E, self.std = from_file
            elif len(from_file) == 2:
                self.values, self.E = from_file
                self.std = None
            else:
                raise ValueError(f"Expected two or three columns\
                 in file '{path}', got {len(from_file)}")
        elif filetype == 'mama':
            self.values, self.E = mama_read(path)
            self.std = None
        elif filetype == 'csv':
            self.values, self.E, self.std = load_csv_1D(path)
        else:
            try:
                self.values, self.E = mama_read(path)
            except ValueError:  # from within ValueError
                raise ValueError(f"Unknown filetype {filetype}")
        self.E *= ureg.keV
        self.verify_integrity()

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
        std = None if self.std is None else np.delete(self.std, inan)
        if inplace:
            self.values = values
            self.E = E
            self.std = std
        else:
            return Vector(values=values, E=E, std=std, units=self.units)

    def rebin(self, mids: Optional[Sequence[float]] = None,
              factor: Optional[float] = None,
              inplace: bool = True) -> Optional[Vector]:
        """ Rebins vector, assuming equidistant binning

        Args:
            mids: The new energy mids. Can not be
                given alongside 'factor'.
            factor: The factor by which the step size shall be
                changed. Can not be given alongside 'mids'.
            inplace: Whether to change E and values
                inplace or return the rebinned vector.
        Returns:
            The rebinned vector if inplace is 'False'.
        """
        if not (mids is None) ^ (factor is None):
            raise ValueError("Either 'mids' or 'factor' must be"
                             " specified, but not both.")
        mids_old = self.E.magnitude
        unit = self.E.units

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
        else:
            try:
                mids = mids.to(unit).magnitude
            except AttributeError:
                pass

        rebinned = rebin_1D(self.values, mids_old, mids)
        mids *= unit

        if inplace:
            self.values = rebinned
            self.E = mids
            self.verify_integrity()
        else:
            return Vector(E=mids, values=rebinned)

    def rebin_like(self, other: Vector, inplace: bool = False) -> Optional[Vector]:
        """ Rebin to match the binning of `other`.

        Args:
            other: Rebin to the bin width of the provided vector.
            inplace: Whether to rebin inplace or return a copy.
                Defaults to `False`.
        """
        if not other.is_equidistant():
            warnings.warn("Non-equidistant rebinning is untested.")

        #dE = (other.E[1] - other.E[0]).magnitude
        #mids = np.arange(self.E.magnitude[0], self.E.magnitude[-1], dE)
        mids = other.E.to(self.E.units).magnitude
        return self.rebin(mids=mids, inplace=inplace)

    def extend_like(self, other: Vector, inplace: bool = False) -> Optional[Vector]:
        """ Resize to match the energy span of `other`.

        Args:
            other: Resize to the energy of the given vector.
            inplace: Whether to perform change inplace or return a copy.
                Defaults to `False`.
        """
        emin = min(other.E[0], self.E[0])
        emax = max(other.E[-1], self.E[-1])
        dE = self.E[1] - self.E[0]
        E = np.arange(emin, emax, dE)
        values = np.zeros_like(E)
        ilow = index(E, self.E[0])
        ihigh = index(E, self.E[-1])
        values[ilow:ihigh+1] = self.values
        if inplace:
            self.values = values
            self.E = E
            self.verify_integrity()
        else:
            return Vector(E=E, values=values)

    def closest(self, E: ndarray, side: Optional[str] = 'right',
                inplace=False) -> Optional[Vector]:
        """ Re-bin the vector without merging bins.

            The resulting vector will have E as the x-axis while
            the jth y-value will be given by the ith value of the original
            y-values where E[i] < E_new[j] <= E[i+1] or
            E[i] <= E_new[j] < E[i+1].

            If E is dimensionless, it is assumed to be in the same unit
            as `Vector.E`

            Args:
                E: Bin value to find. Value or array.
                side: 'left': E[i] < E[j] <= E[i+1],
                      'right': E[i] <= E[j] <= E[i+1]
                inplace: Whether to make the change inplace or not.
            Returns:
                Vector with the new E axis and the bin content of the bins
                that contains E.
            Raises:
                RuntimeError if the x-axis of the original vector is
                not sorted.
        """

        if not np.all(self.E.magnitude[:-1] <= self.E.magnitude[1:]):
            raise RuntimeError("x-axis not sorted.")

        # Convert to same units at strip
        E_old = self.E.magnitude
        unit = self.E.units
        try:
            E = E.to(unit).magntiude
        except AttributeError:
            pass

        indices = np.searchsorted(E_old, E, side=side)

        # Ensure that any element outside the range of E will get index
        # -1.
        indices[indices >= len(self.E.magnitude)] = 0
        indices -= 1

        # We need to append 0 to the end to ensure that we fill 0 if any
        # element E_new is outside of the bounds of self.E
        values = np.append(self.values, [0])
        values = values[indices]

        std = None
        if self.std is not None:
            std = np.append(self.std, [0])
            std = std[indices]

        E *= unit
        if inplace:
            self.E = E
            self.values = values
            self.std = std
        else:
            return Vector(values=values, E=E, std=std)

    def cumsum(self, factor: Union[float, str] = 1.0,
               inplace=False) -> Optional[Vector]:
        """ Cumulative sum of the vector.

            Args:
                factor: A factor to multiply to the resulting vector. Possible
                values are a float or string 'de'. If 'de' the
                factor will be calculated by E[1] - E[0]. The default is 1.0.
                inplace: Whether to make the change inplace or not.
            Returns:
                The cumulative sum vector if inplace is 'False'
            Raises:
                RuntimeError if elements in self.E are not equidistant
                and factor='de'.
                ValueError if factor is a string other than 'de'.
        """

        if isinstance(factor, str):
            if factor.lower() != 'de':
                raise ValueError(f"Unkown option for factor {factor}")
            E = self.E.magnitude
            factor = E[1] - E[0]
            if not np.all(np.diff(E) == factor):
                raise RuntimeError("Vector x-axis isn't equidistant.")

        cumsum = factor*self.values.cumsum()
        cumerr = None
        if self.std is not None:
            cumerr = np.sqrt(np.cumsum(self.std**2))*factor

        if inplace:
            self.values = cumsum
            self.std = cumerr
        else:
            return self.copy(values=cumsum, std=cumerr)

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

    def same_shape(self, other: Sequence[float], error: bool = False) -> bool:
        """ Check whether `other` has same shape `self.values`.

        Args:
            other (array-like): Object to compare to.
            kwargs: Additional kwargs to `np.allclose`.

        Returns:
            Returns `True` if shapes are equal.
        Raises:
            ValueError: if lengths aren't equal and `error` is true.
        """
        same = len(other) == len(self)
        if error and not same:
            raise ValueError("Expected {len(self)}, got {len(other)}.")
        return same

    def to_same(self, E: float) -> float:
        """ Convert the units of E to the unit of `Vector.E` and return magn.

        Args:
            E: Convert its units to the same units of `Vector.E`.
               If `E` is dimensionless, assume to be of the same unit
               as `Vector.E`.
        """
        try:
            return E.to(self.E.units).magnitude
        except AttributeError:
            return E

    def to(self, unit: str) -> Vector:
        new = self.copy()
        new.E = new.E.to(unit)
        return new

    def index(self, E: float) -> int:
        """ Returns the closest index corresponding to the E value

        Args:
            E: The value which index to find. If dimensionless,
               assumes the same units as `Vector.E`
        """
        return index(self.E.magnitude, self.to_same(E))

    def set_order(self, order: str) -> None:
        """ Wrapper around numpy to set the alignment """
        self.values = self.values.copy(order=order)
        self.E = self.E.copy(order=order)

    def __matmul__(self, other: Vector) -> Vector:
        result = self.copy()
        if isinstance(other, Vector):
            self.has_equal_binning(other)
        else:
            NotImplementedError("Type not implemented")

        result.values = result.values@other.values
        return result

    def __len__(self) -> int:
        return len(self.values)

    def __str__(self) -> str:
        return str(self.values) + "\n" + str(self.E)
