from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple, Union, Literal, overload

import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray

from .. import ureg, DimensionalityError, Unit
from .index import index
from .abstractarray import AbstractArray
from .filehandling import (filetype_from_suffix, load_csv_1D, load_numpy_1D,
                           load_tar, load_txt_1D, mama_read, mama_write,
                           save_csv_1D, save_numpy_1D, save_tar, save_txt_1D,
                           save_npz_1D, load_npz_1D)
from ..library import div0, handle_rebin_arguments, only_one_not_none, maybe_set
from .rebin import rebin_1D
from ..stubs import Unitlike, arraylike, Axes, Pathlike
from dataclasses import dataclass, field

LOG = logging.getLogger(__name__)
logging.captureWarnings(True)


@dataclass(frozen=True)
class VectorMetadata:
    xalias: str = 'E'
    valias: str = ''
    xlabel: str = 'Energy'
    ylabel: str = 'Counts'
    name: str = ''
    misc: dict[str, any] = field(default_factory=dict)

    def clone(self, **kwargs) -> VectorMetadata:
        return VectorMetadata(**(self.__dict__ | kwargs))

    def update(self, **kwargs) -> VectorMetadata:
        return self.clone(**kwargs)


class Vector(AbstractArray):
    """ Stores 1d array with energy axes (a vector)

    Attributes:
        values (np.ndarray): The values at each bin.
        X (np.ndarray): The energy of each bin. (mid-bin calibration)
        std (np.ndarray): The standard deviation of the counts
    """
    # HACK: Descriptors really don't work well with %autoreload.
    # comment / uncomment this to silence the errors when developing
    #__slots__ = ('_X', 'values', 'std', 'loc', 'iloc', 'metadata')

    def __init__(self, *, X: Iterable[float] | None = None,
                 values: Iterable[float] | None = None,
                 std: Iterable[float] | None = None,
                 copy: bool = True,
                 units: Unitlike = "keV",
                 order: Literal['C', 'F'] = 'C',
                 metadata: VectorMetadata = VectorMetadata(),
                 **kwargs):
        """
        If no `std` is given, it will default to None

        Args:
            values: see above
            E: see above
            std: see above
            copy: Whether to copy `values` and `E` or by reference.
                Defaults to True.

        Raises:
           ValueError if the runtime lengths of the arrays are different.
           ValueError if incompatible arguments are provided.

        """
        # Resolve aliasing
        # First keyword argument is the alias
        kwiter = kwargs.items().__iter__()
        xalias = valias = None
        if X is None:
            try:
                xalias, X = next(kwiter)
            except StopIteration:
                raise ValueError("No X array provided")
            if 'xalias' in kwargs:
                raise ValueError("Cannot set xalias as a keyword argument")
        # Next keyword is values
        if values is None:
            try:
                valias, values = next(kwiter)
            except StopIteration:
                raise ValueError("No values array provided")
            if 'valias' in kwargs:
                raise ValueError("Cannot set valias as a keyword argument")
        # Rest is passed to the metadata
        kwargs = dict(kwiter)
        if xalias is not None:
            kwargs['xalias'] = xalias
        if valias is not None:
            kwargs['valias'] = valias

        try:
            unit = X.units
        except AttributeError:
            unit = ureg.Unit(units)

        if copy:
            def fetch(x):
                return np.atleast_1d(np.asarray(x, dtype=float, order=order).copy())
        else:
            def fetch(x):
                return np.atleast_1d(np.asarray(x, dtype=float, order=order))

        self.values = fetch(values)
        try:
            self._X = fetch(X.magnitude) * unit
        except AttributeError:
            self._X = fetch(X) * unit

        self.std = fetch(std) if std is not None else None
        self.loc: ValueLocator = ValueLocator(self)
        self.iloc: IndexLocator = IndexLocator(self)
        self.metadata = metadata.update(**kwargs)

        self.verify_integrity()

    def is_equidistant(self) -> bool:
        """ Returns True if the vector is equidistant """
        return np.allclose(np.diff(self._X), np.diff(self._X)[0])

    def __getattr__(self, item):
        meta = self.__dict__['metadata']
        if item == meta.xalias:
            x = self.X
        elif item == meta.valias:
            x = self.__dict__['values']
        elif item == 'd' + meta.xalias:
            x = self.dX
        else:
            x = super().__getattr__(item)
        return x

    def verify_integrity(self, check_equidistant: bool = False):
        """ Verify the internal consistency of the vector

        Args:
            check_equidistant (bool, optional): Check whether energy array
                are equidistant spaced. Defaults to False.

        Raises:
            AssertionError or ValueError if any test fails
        """
        if self.X.shape != self.values.shape:
            raise ValueError(f"Energy and values must have same shape. Got {self.X.shape} and {self.values.shape}.")
        if self.std is not None:
            if self.std.shape != self.values.shape:
                raise ValueError("std and values must have same shape")

        #if check_equidistant and not self.is_equidistant():
        #    raise ValueError("Is not equidistant.")


    def calibration(self) -> Dict[str, float]:
        """Calculate and return the calibration coefficients of the energy axes

        Formatted as "a{axis}{power of E}"
        """

        calibration = {"a0": self._X[0].to('keV').magnitude,
                       "a1": (self._X[1] - self._X[0]).to('keV').magnitude}
        return calibration

    def plot(self, ax: Axes | None = None,
             scale: str = 'linear',
             kind: str = 'step',
             **kwargs) -> Axes:
        """ Plots the vector

        Args:
            ax (matplotlib axis, optional): The axis to plot onto. If not
                provided, a new figure is created
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
                ax.errorbar(self.X, self.values, yerr=self.std,
                            **kwargs)
            else:
                ax.plot(self.X, self.values, "o-", **kwargs)
        elif kind == "step":
            kwargs.setdefault("where", "mid")
            ax.step(self.X, self.values, **kwargs)
        elif kind == 'bar':
            kwargs.setdefault("align", "center")
            kwargs.setdefault('width', self.de)
            kwargs.setdefault('yerr', self.std)
            ax.bar(self.X, self.values, **kwargs)
        elif kind == 'dot':
            kwargs.setdefault("marker", ".")
            ax.scatter(self.X, self.values, **kwargs)
        else:
            raise NotImplementedError()
        maybe_set(ax, 'xlabel', self.xlabel + f" [${self.units:~L}$]")
        maybe_set(ax, 'ylabel', self.ylabel)
        maybe_set(ax, 'title', self.name)
        return ax

    def save(self, path: Pathlike,
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
        path = Path(path)
        if filetype is None:
            filetype = filetype_from_suffix(path)
            if filetype is None:
                raise ValueError("Filetype could not be determined from suffix: {path}")
            # Fallback case
            if filetype == '':
                filetype = 'npz'
        filetype = filetype.lower()

        E = self._X.to('keV').magnitude
        match filetype:
            case "npy":
                warnings.warn("Saving as .npy is deprecated. Use .npz instead.")
                save_numpy_1D(self.values, E, self.std, path)
            case 'npz':
                save_npz_1D(path, self)
            case "txt":
                warnings.warn("Saving to .txt does not preserve metadata. Use .npz instead.")
                save_txt_1D(self.values, E, self.std, path, **kwargs)
            case 'tar':
                warnings.warn("Saving to .tar does not preserve metadata. Use .npz instead.")
                if self.std is not None:
                    save_tar([self.values, E, self.std], path)
                else:
                    save_tar([self.values, E], path)
            case 'mama':
                warnings.warn("MAMA format does not preserve metadata.")
                mama_write(self, path, comment="Made by OMpy", **kwargs)
                if self.std is not None:
                    warnings.warn("MaMa cannot store std. "
                                  "Consider using another format")
            case 'csv':
                warnings.warn("CSV format does not preserve metadata.")
                save_csv_1D(self.values, E, self.std, path)
            case _:
                raise ValueError(f"Unknown filetype {filetype}")

    @classmethod
    def from_path(cls, path: Pathlike, filetype: str | None = None) -> Vector:
        """Load to a file of specified format

        Units assumed to be keV.

        Args:
            path (str or Path): Path to Load
            filetype (str, optional): Filetype. Default uses
                auto-recognition from suffix.

        Raises:
            ValueError: Filetype is not supported
        """
        path = Path(path)
        if filetype is None:
            filetype = filetype_from_suffix(path)
            if filetype is None:
                raise ValueError("Filetype could not be determined from suffix: {path}")
            # Fallback case
            if filetype == '':
                filetype = 'npz'
                path = path.with_suffix('.npz')
        filetype = filetype.lower()
        LOG.debug(f"Loading {path} as {filetype}")

        match filetype:
            case 'npy':
                values, E, std = load_numpy_1D(path)
            case 'npz':
                return load_npz_1D(path, Vector)
            case 'txt':
                values, E, std = load_txt_1D(path)
            case 'tar':
                from_file = load_tar(path)
                if len(from_file) == 3:
                    values, E, std = from_file
                elif len(from_file) == 2:
                    values, E = from_file
                    std = None
                else:
                    raise ValueError(f"Expected two or three columns\
                     in file '{path}', got {len(from_file)}")
            case 'mama':
                values, E = mama_read(path)
                std = None
            case 'csv':
                values, E, std = load_csv_1D(path)
            case _:
                try:
                    values, E = mama_read(path)
                except ValueError:  # from within ValueError
                    raise ValueError(f"Unknown filetype {filetype}")
        E *= ureg.keV
        return Vector(E=E, values=values, std=std)

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
        transformed = const * self.values * np.exp(alpha * self.X)

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
        Emin = Emin if Emin is not None else self.X.min()
        Emax = Emax if Emax is not None else self.X.max()
        imin = self.index(Emin)
        imax = self.index(Emax)
        cut = slice(imin, imax + 1)

        values = self.values[cut]
        E = self.X[cut] * self.units
        if self.std is not None:
            std = self.std[cut]
        else:
            std = None

        if inplace:
            self._X = E
            self.values = values
            self.std = std
        else:
            return self.clone(values=values, X=E, std=std)

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
        E = np.delete(self.X, inan) * self.units
        std = None if self.std is None else np.delete(self.std, inan)
        if inplace:
            self.values = values
            self._X = E
            self.std = std
        else:
            return self.clone(values=values, X=E, std=std)

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
        oldbins = self.X
        unit = self.units

        newbins = handle_rebin_arguments(bins=oldbins, transform=self.to_same, LOG=LOG,
                                         newbins=bins, factor=factor, numbins=numbins,
                                         binwidth=binwidth)

        rebinned = rebin_1D(self.values, oldbins, newbins)
        newbins *= unit

        if inplace:
            self.values = rebinned
            self._X = newbins
            self.verify_integrity()
        else:
            return self.clone(X=newbins, values=rebinned)

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
        warnings.warn("This gives wrong result when the start and end bins are different")
        bins = self.to_same(other._X)
        return self.rebin(bins=bins, inplace=inplace)


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

        if not np.all(self.X[:-1] <= self.X[1:]):
            raise RuntimeError("x-axis not sorted.")

        # Convert to same units at strip
        E_old = self.X
        E = self.to_same(E)
        indices = np.searchsorted(E_old, E, side=side)

        # Ensure that any element outside the range of E will get index
        # -1.
        indices[indices >= len(self.X)] = 0
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
            self._X = E
            self.values = values
            self.std = std
        else:
            return self.clone(values=values, X=E, std=std)

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
        if not np.allclose(self._X, other._X, **kwargs):
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
            self._X = self._X.to(unit)
            return self

        new = self.clone(X=self._X.to(unit))
        return new

    def index(self, X: Unitlike) -> int:
        """ Returns the closest index corresponding to the E value

        Args:
            E: The value which index to find. If dimensionless,
               assumes the same units as `Vector.E`
        """
        # TODO Replace with numba
        warnings.warn("Probably buggy")
        return np.searchsorted(self.X, self.to_same(X))

    def set_order(self, order: str) -> None:
        """ Wrapper around numpy to set the alignment """
        self.values = self.values.copy(order=order)
        self._X = self._X.copy(order=order)

    @property
    def dX(self) -> float:
        if len(self) <= 1:
            raise NotImplementedError()

        if self.is_equidistant():
            return self.X[1] - self.X[0]
        else:
            return (self.X - np.roll(self.X, 1))[1:]

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

    @property
    def _summary(self) -> str:
        emin = self.X[0]
        emax = self.X[-1]
        de = self.dX
        unit = f"{self.units:~}"
        s = ''
        s += f'Index alias: {self.metadata.xalias}\tValue alias: {self.metadata.valias}\n'
        s += f'xlabel: {self.metadata.xlabel}\t ylabel: {self.metadata.ylabel}\n'
        if len(self.metadata.misc) > 0:
            s += "Metadata:\n"
            for key, val in self.metadata.misc.items():
                s += f'\t{key}: {val}\n'
        s += f"Index: {emin} to {emax} [{unit}]\n"
        s += f"{len(self.X)} bins with step: {de} {unit}\n"
        s += f"Total counts: {self.sum()}\n"
        return s

    def summary(self) -> None:
        print(self._summary)

    def __str__(self) -> str:
        summary = self._summary
        summary += "\nValues:\n"
        if self.is_equidistant():
            if self.std is not None:
                return summary + str(self.values) + '\n' + str(self.std)
            else:
                return summary + str(self.values)
        else:
            if self.std is not None:
                return (f"{self.values}\n{self.std}\n{self.X} "
                        f"[{self.units:~}]")
            else:
                return (f"{self.values}\n{self.X} "
                        f"[{self.units:~}]")

    def clone(self, X=None, values=None, std=None, units=None, order='C',
              metadata=None, **kwargs) -> Vector:
        """ Copies the object.

        Any keyword argument will override the equivalent
        attribute in the copy. For example, vector.clone(E=[1,2,3])
        tries to set the energy to [1,2,3].

        kwargs: Any keyword argument is overwritten
            in the copy.
        Returns:
            The copy
        """
        X = X if X is not None else self._X
        values = values if values is not None else self.values
        std = std if std is not None else self.std
        units = units if units is not None else self.units
        metadata = metadata if metadata is not None else self.metadata
        metadata = metadata.update(**kwargs)
        return Vector(X=X, values=values, std=std, units=units, order=order,
                      metadata=metadata)

    @property
    def units(self) -> Any:
        return self._X.units

    @property
    def xlabel(self) -> str:
        return self.metadata.xlabel

    @property
    def ylabel(self) -> str:
        return self.metadata.ylabel

    @property
    def name(self) -> str:
        return self.metadata.name

    @property
    def X(self) -> np.ndarray:
        return self._X.magnitude

    @property
    def __values(self) -> np.ndarray:
        return self.values

    def enumerate(self) -> Iterable[Tuple[int, float, float]]:
        """ Returns an iterator over the indices and values """
        for i, x in enumerate(self.X):
            yield i, x, self.values[i]

    def unpack(self) -> tuple[np.ndarray, np.ndarray]:
        """ Returns the energy and values as separate arrays """
        return self.X, self.values


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
                return int(self.vec.index(e))
            case slice():
                return parse_unit_slice(e, self.vec.index, self.vec.dX, len(self.vec.X))
            case _:
                raise ValueError(f"Expected slice or Unitlike, got type: {type(e)}")

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
                return self.vec.clone(values=values, X=E)
            case _:
                raise RuntimeError("Should not happen")

    def __setitem__(self, key, val):
        i: int | slice = self.parse(key)
        self.vec.values[i] = val

class IndexLocator:
    def __init__(self, vector: Vector):
        self.vector = vector

    def __getitem__(self, key):
        if True:  # len(key) == 1:
            X = self.vector.X.__getitem__(key)
            values = self.vector.values.__getitem__(key)
            return self.vec.clone(X=X, values=values)
        else:
            raise ValueError("Expect one integer index [i].")


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
