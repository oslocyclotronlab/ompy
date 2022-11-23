from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple, Union, Literal, overload

import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray

from . import ureg, DimensionalityError
from .abstractarray import AbstractArray
from .decomposition import index
from .filehandling import (filetype_from_suffix, load_csv_1D, load_numpy_1D,
                           load_tar, load_txt_1D, mama_read, mama_write,
                           save_csv_1D, save_numpy_1D, save_tar, save_txt_1D)
from .library import div0, handle_rebin_arguments, only_one_not_none
from .rebin import rebin_1D
from .stubs import Unitlike, arraylike, Axes

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

    def __init__(self, values: Iterable[float] | None = None,
                 E: Iterable[float] | None = None,
                 *,
                 path: Union[str, Path] | None = None,
                 std: Iterable[float] | None = None,
                 copy: bool = True,
                 units: str | None = "keV",
                 E_label: str = 'Energy'):
        """
        There are two ways to initialize:

        - Providing both `values` and `energy`.
        - Providing only a path loads a vector from said path.

        If no `std` is given, it will default to None

        Args:
            values: see above
            E: see above
            path: see above
            std: see above
            copy: Whether to copy `values` and `E` or by reference.
                Defaults to True.
            unit: see above
            E_label: The default x-label for plotting.

        Raises:
           ValueError if the runtime lengths of the arrays are different.
           ValueError if incompatible arguments are provided.

        """
        try:
            unit = E.units
        except AttributeError:
            unit = ureg.Unit(units)

        if copy:
            def fetch(x):
                return np.atleast_1d(np.asarray(x, dtype=float).copy())
        else:
            def fetch(x):
                return np.atleast_1d(np.asarray(x, dtype=float))

        self.std: ndarray | None = None

        if path is not None:
            if values is not None or E is not None:
                warnings.warn("Loading from path. Other arguments are ignored.")
            self.load(path)
            # Always loads as [KeV]
            self.to(unit, inplace=True)
        else:
            if values is None or E is None:
                raise ValueError("Values and energy must both be given.")
            else:
                self.values = fetch(values)
                try:
                    self._E = fetch(E.magnitude) * unit
                except AttributeError:
                    self._E = fetch(E) * unit

        if std is not None:
            self.std = fetch(std)

        self.loc: ValueLocator = ValueLocator(self)
        self.iloc: IndexLocator = IndexLocator(self)
        self.E_label: str = str(E_label)

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

        if check_equidistant and not self.is_equidistant():
            raise ValueError("Is not equidistant.")

    def is_equidistant(self) -> bool:
        """ Check if the width of energy bins are all equal."""
        if len(self) <= 1:
            return True

        diff = (self.E - np.roll(self.E, 1))[1:]
        de = diff[0]
        return np.all(np.isclose(diff, de * np.ones(diff.shape)))

    def calibration(self) -> Dict[str, float]:
        """Calculate and return the calibration coefficients of the energy axes

        Formatted as "a{axis}{power of E}"
        """

        calibration = {"a0": self._E[0].to('keV').magnitude,
                       "a1": (self._E[1] - self._E[0]).to('keV').magnitude}
        return calibration

    def plot(self, ax: Axes | None = None,
             scale: str = 'linear',
             kind: str = 'step',
             **kwargs) -> Axes:
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
        assert ax is not None

        if kind in ["plot", "line"]:
            kwargs.setdefault("markersize", 3)
            kwargs.setdefault("marker", ".")
            kwargs.setdefault("linestyle", "-")
            if self.std is not None:
                ax.errorbar(self.E, self.values, yerr=self.std,
                            **kwargs)
            else:
                ax.plot(self.E, self.values, "o-", **kwargs)
        elif kind == "step":
            kwargs.setdefault("where", "mid")
            ax.step(self.E, self.values, **kwargs)
        elif kind == 'bar':
            kwargs.setdefault("align", "center")
            kwargs.setdefault('width', self.de)
            kwargs.setdefault('yerr', self.std)
            ax.bar(self.E, self.values, **kwargs)
        elif kind == 'dot':
            kwargs.setdefault("marker", ".")
            ax.scatter(self.E, self.values, **kwargs)
        else:
            raise NotImplementedError()
        ax.set_yscale(scale)
        ax.set_xlabel(self.E_label + f" [${self.units:~L}$]")
        return ax

    def save(self, path: Union[str, Path],
             filetype: str | None = None,
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
        vector = self.clone()
        path = Path(path) if isinstance(path, str) else path
        if filetype is None:
            filetype = filetype_from_suffix(path)
        filetype = filetype.lower()

        E = vector._E.to('keV').magnitude
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
             filetype: str | None = None) -> None:
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
        LOG.debug(f"Loading {path} as {filetype}")

        if filetype == 'numpy':
            self.values, self._E, self.std = load_numpy_1D(path)
        elif filetype == 'txt':
            self.values, self._E, self.std = load_txt_1D(path)
        elif filetype == 'tar':
            from_file = load_tar(path)
            if len(from_file) == 3:
                self.values, self._E, self.std = from_file
            elif len(from_file) == 2:
                self.values, self._E = from_file
                self.std = None
            else:
                raise ValueError(f"Expected two or three columns\
                 in file '{path}', got {len(from_file)}")
        elif filetype == 'mama':
            self.values, self._E = mama_read(path)
            self.std = None
        elif filetype == 'csv':
            self.values, self._E, self.std = load_csv_1D(path)
        else:
            try:
                self.values, self._E = mama_read(path)
            except ValueError:  # from within ValueError
                raise ValueError(f"Unknown filetype {filetype}")
        self._E *= ureg.keV
        self.verify_integrity()

    def transform(self, const: float = 1,
                  alpha: float = 0, inplace: bool = True) -> Vector | None:
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
            Vector | None
        """
        if self.std is not None:
            relative_uncertainty = self.std / self.values
        transformed = const * self.values * np.exp(alpha * self.E)

        if self.std is not None:
            std = relative_uncertainty * transformed
        if not inplace:
            if self.std is not None:
                return self.clone(values=transformed, std=std)
            return self.clone(values=transformed)
        else:
            self.values = transformed
            if self.std is not None:
                self.std = std

    def error(self, other: Vector | ndarray,
              std: ndarray | None = None) -> float:
        """Computes the (weighted) χ²

        Args:
            other (Vector or ndarray]): The reference to compare itself to. If
                an array, assumes it has the same energy binning as itself.
            std (ndarray | None, optional): Standard deviations to use as
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
        squared_error = (self.values - other) ** 2
        if self.std is not None:
            if std is not None:
                sigmasq = self.std ** 2 + std ** 2
            else:
                sigmasq = self.std ** 2
        else:
            if std is not None:
                sigmasq = std ** 2
            else:
                sigmasq = 1

        error = div0(squared_error, sigmasq)
        return error.sum()

    def cut(self, Emin: float | None = None,
            Emax: float | None = None,
            inplace: bool = False) -> Vector | None:
        """ Cut the vector at the energy limits

        Args:
            Emin (float, optional): The lower energy limit
            Emax (float, optional): The higher energy limit
            inplace (bool, optional): If `True` perform the cut on this vector
                or if `False` returns a copy. Defaults to `False`.
        Returns:
            The cut vector if `inplace` is True.
        """
        warnings.warn("cut is being deprecation in favor of vec.iloc[j:k]", DeprecationWarning)
        Emin = Emin if Emin is not None else self.E.min()
        Emax = Emax if Emax is not None else self.E.max()
        imin = self.index(Emin)
        imax = self.index(Emax)
        cut = slice(imin, imax + 1)

        values = self.values[cut]
        E = self.E[cut] * self.units
        if self.std is not None:
            std = self.std[cut]
        else:
            std = None

        if inplace:
            self._E = E
            self.values = values
            self.std = std
        else:
            return self.clone(values=values, E=E, std=std)

    def drop_nan(self, inplace: bool = False) -> Vector:
        """ Drop the elements that are `np.nan`

        Args:
            inplace (bool, optional): If `True` perform the cut on this vector
                or if `False` returns a copy. Defaults to True
        Returns:
            The cut vector if `inplace` is True.
        """
        inan = np.argwhere(np.isnan(self.values))

        values = np.delete(self.values, inan)
        E = np.delete(self.E, inan) * self.units
        std = None if self.std is None else np.delete(self.std, inan)
        if inplace:
            self.values = values
            self._E = E
            self.std = std
        else:
            return self.clone(values=values, E=E, std=std)

    @overload
    def rebin(self, bins: arraylike | None = None,
              factor: float | None = None,
              binwidth: Unitlike | None = None,
              numbins: int | None = None,
              inplace: Literal[False] = ...) -> Vector:
        ...

    @overload
    def rebin(self, bins: arraylike | None = None,
              factor: float | None = None,
              binwidth: Unitlike | None = None,
              numbins: int | None = None,
              inplace: Literal[True] = ...) -> None:
        ...

    def rebin(self, bins: arraylike | None = None,
              factor: float | None = None,
              binwidth: Unitlike | None = None,
              numbins: int | None = None,
              inplace: bool = False) -> Vector | None:
        """ Rebins vector, assuming equidistant binning

        Args:
            bins: The new energy bins. Can not be
                given alongside 'factor' or `binwidth`.
            factor: The factor by which the step size shall be
                changed. E.g `factor=2.0` yields twice as large
                bins. Can not be given alongside 'bins' or `binwidth`.
            binwidth: The new bin width. Can not be given
                alongside `factor` or `bins`.
            numbins: The new number of bins. Must be fewer than before.
            inplace: Whether to change E and values
                inplace or return the rebinned vector.
                Defaults to `false`.
        Returns:
            The rebinned vector if inplace is 'False'.
        """
        oldbins = self.E
        unit = self.units

        newbins = handle_rebin_arguments(bins=oldbins, transform=self.to_same, LOG=LOG,
                                         newbins=bins, factor=factor, numbins=numbins,
                                         binwidth=binwidth)

        rebinned = rebin_1D(self.values, oldbins, newbins)
        newbins *= unit

        if inplace:
            self.values = rebinned
            self._E = newbins
            self.verify_integrity()
        else:
            return self.clone(E=newbins, values=rebinned)

    @overload
    def rebin_like(self, other: Vector, inplace: Literal[False] = ...) -> Vector:
        ...

    @overload
    def rebin_like(self, other: Vector, inplace: Literal[True] = ...) -> None:
        ...

    def rebin_like(self, other: Vector, inplace: bool = False) -> Vector | None:
        """ Rebin to match the binning of `other`.

        Args:
            other: Rebin to the bin width of the provided vector.
            inplace: Whether to rebin inplace or return a copy.
                Defaults to `False`.
        """
        bins = self.to_same(other._E)
        return self.rebin(bins=bins, inplace=inplace)

    # def extend_like(self, other: Vector, inplace: bool = False) -> Vector | None:
    #     """ Resize to match the energy span of `other`.

    #     Args:
    #         other: Resize to the energy of the given vector.
    #         inplace: Whether to perform change inplace or return a copy.
    #             Defaults to `False`.
    #     """
    #     emin = min(other.E[0], self.E[0])
    #     emax = max(other.E[-1], self.E[-1])
    #     dE = self.E[1] - self.E[0]
    #     E = np.arange(emin, emax, dE)
    #     values = np.zeros_like(E)
    #     ilow = index(E, self.E[0])
    #     ihigh = index(E, self.E[-1])
    #     values[ilow:ihigh+1] = self.values
    #     if inplace:
    #         self.values = values
    #         self._E = E
    #         self.verify_integrity()
    #     else:
    #         return self.clone(E=E, values=values)

    def closest(self, E: ndarray, side: str | None = 'right',
                inplace=False) -> Vector | None:
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

        if not np.all(self.E[:-1] <= self.E[1:]):
            raise RuntimeError("x-axis not sorted.")

        # Convert to same units at strip
        E_old = self.E
        E = self.to_same(E)
        indices = np.searchsorted(E_old, E, side=side)

        # Ensure that any element outside the range of E will get index
        # -1.
        indices[indices >= len(self.E)] = 0
        indices -= 1

        # We need to append 0 to the end to ensure that we fill 0 if any
        # element E_new is outside of the bounds of self.E
        values = np.append(self.values, [0])
        values = values[indices]

        std = None
        if self.std is not None:
            std = np.append(self.std, [0])
            std = std[indices]

        E *= self.units
        if inplace:
            self._E = E
            self.values = values
            self.std = std
        else:
            return self.clone(values=values, E=E, std=std)

    def cumulative(self, factor: Union[float, str] = 1.0,
                   inplace: bool = False) -> Vector | None:
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

            if not self.is_equidistant():
                raise RuntimeError("Vector x-axis isn't equidistant.")
            factor = self.de

        cumsum = factor * self.values.cumsum()
        cumerr = None
        if self.std is not None:
            cumerr = np.sqrt(np.cumsum(self.std ** 2)) * factor

        if inplace:
            self.values = cumsum
            self.std = cumerr
        else:
            return self.clone(values=cumsum, std=cumerr)

    def has_equal_binning(self, other: Vector,
                          error: bool = True, **kwargs) -> bool:
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
            if not error:
                return False
            raise TypeError("Other must be a Vector")
        if self.shape != other.shape:
            if not error:
                return False
            raise ValueError("Must have equal number of energy bins.")
        # NOTE: This automatically compares with the correct units.
        # This does not mean that self.E == other.E in magnitude.
        if not np.allclose(self._E, other._E, **kwargs):
            if not error:
                return False
            raise ValueError("Must have equal energy binning.")
        else:
            return True

    def same_shape(self, other: Sequence[float], error: bool = False) -> bool:
        """ Check whether `other` has same shape `self.values`.

        Args:
            other (array-like): Object to compare to.

        Returns:
            Returns `True` if shapes are equal.
        Raises:
            ValueError: if lengths aren't equal and `error` is true.
        """
        same = len(other) == len(self)
        if error and not same:
            raise ValueError(f"Expected {len(self)}, got {len(other)}.")
        return same

    def to_same(self, E: float) -> float:
        """ Convert the units of E to the unit of `Vector.E` and return magn.

        Args:
            E: Convert its units to the same units of `Vector.E`.
               If `E` is dimensionless, assume to be of the same unit
               as `Vector.E`.
        """
        E = ureg.Quantity(E)
        if not E.dimensionless:
            E = E.to(self.units)
        return E.magnitude

    def to(self, unit: str, inplace: bool = False) -> Vector:
        if inplace:
            self._E = self._E.to(unit)
            return self

        new = self.clone(E=self._E.to(unit))
        return new

    def index(self, E: Unitlike) -> int:
        """ Returns the closest index corresponding to the E value

        Args:
            E: The value which index to find. If dimensionless,
               assumes the same units as `Vector.E`
        """
        return np.searchsorted(self.E, self.to_same(E))

    def set_order(self, order: str) -> None:
        """ Wrapper around numpy to set the alignment """
        self.values = self.values.copy(order=order)
        self._E = self._E.copy(order=order)

    @property
    def de(self) -> float:
        if len(self) <= 1:
            raise NotImplementedError()

        if self.is_equidistant():
            return self.E[1] - self.E[0]
        else:
            return (self.E - np.roll(self.E, 1))[1:]

    def last_nonzero(self) -> int:
        """ Returns the index of the last nonzero value """
        j = len(self)
        while (j := j - 1) >= 0:
            if self[j] != 0:
                break
        return j

    def __matmul__(self, other: Vector) -> Vector:
        result = self.clone()
        if isinstance(other, Vector):
            self.has_equal_binning(other)
        else:
            NotImplementedError("Type not implemented")

        result.values = result.values @ other.values
        return result

    def __str__(self) -> str:
        if self.is_equidistant():
            emin = self.E[0]
            emax = self.E[-1]
            de = self.de
            unit = f"{self.units:~}"
            s = f"Energy: {emin} to {emax} [{unit}]\n"
            s += f"{len(self.E)} bins with dE: {de}\n"
            s += f"Total counts: {self.sum()}\n"
            if self.std is not None:
                return s + str(self.values) + '\n' + str(self.std)
            else:
                return s + str(self.values)
        else:
            if self.std is not None:
                return (f"{self.values}\n{self.std}\n{self.E} "
                        f"[{self.units:~}]")
            else:
                return (f"{self.values}\n{self.E} "
                        f"[{self.units:~}]")

    def clone(self, **kwargs) -> Vector:
        """ Copies the object.

        Any keyword argument will override the equivalent
        attribute in the copy. For example, vector.clone(E=[1,2,3])
        tries to set the energy to [1,2,3].

        kwargs: Any keyword argument is overwritten
            in the copy.
        Returns:
            The copy
        """
        E = kwargs.pop('E', self._E)
        values = kwargs.pop('values', self.values)
        std = kwargs.pop('std', self.std)
        units = kwargs.pop('units', self.units)
        for key in kwargs.keys():
            raise RuntimeError(f"Vector has no setable attribute {key}.")
        return Vector(E=E, values=values, std=std, units=units)

    @property
    def E(self) -> np.ndarray:
        return self._E.magnitude

    @property
    def units(self) -> Any:
        return self._E.units


# class ValueLocator:
#     def __init__(self, vector: Vector):
#         self.vec = vector

#     def __getitem__(self, value):
#         if not isinstance(value, slice):
#             indices = self.vec.index(value)
#         else:
#             start = None if value.start is None else self.vec.index(value.start)
#             stop = None if value.stop is None else self.vec.index(value.stop)
#             if value.step is not None:
#                 dx = self.vec[1] - self.vec[0]
#                 step = np.ceil(value.step / dx)
#             else:
#                 step = None
#             indices = slice(start, stop, step)

#         E = self.vec.E.__getitem__(indices)
#         values = self.vec.values.__getitem__(indices)
#         if self.vec.std is not None:
#             std = self.vec.std.__getitem__(indices)
#         else:
#             std = None
#         return self.vec.clone(E=E, values=values, std=std)


class ValueLocator:
    def __init__(self, vector: Vector):
        self.vec = vector

    @overload
    def parse(self, e: Unitlike) -> int: ...

    @overload
    def parse(self, e: slice) -> slice: ...

    def parse(self, e: Unitlike | slice) -> int | slice:
        match e:
            case int() | float() | str() | ureg.Unit():
                return self.vec.index(e)
            case slice():
                return parse_unit_slice(e, self.vec.index, self.vec.dE, len(self.vec.E))
            case _:
                raise ArgumentError(f"Expected slice or Unitlike, got type: {type(e)}")

    def __getitem__(self, key) -> Vector | float:
        i: int | slice = self.parse(key)
        match i:
            case int():
                return self.vec.values.__getitem__((i,))
            case slice():
                E = self.vec.E.__getitem__(i)
                values = self.vec.values.__getitem__((i,))
                std = None
                if self.vec.std is not None:
                    std = self.vec.std.__getitem__((i,))
                return Vector(values=values, E=E, E_label=self.vec.label, std=std)

    def __setitem__(self, key, val):
        i: int | slice = self.parse(key)
        self.vec.values[i] = val

class IndexLocator:
    def __init__(self, vector: Vector):
        self.vector = vector

    def __getitem__(self, key):
        if True:  # len(key) == 1:
            E = self.vector.E.__getitem__(key)
            values = self.vector.values.__getitem__(key)
            return Vector(E=E, values=values)
        else:
            raise ValueError("Expect one integer index [i].")
