from __future__ import annotations

import copy
from typing import Union, Tuple, Sequence
import numpy as np


class AbstractArray:
    def __init__(self):
        """ Abstract class for Matrix and Vector.

        Do not initialize itself.
        """
        raise NotImplementedError()

    def has_equal_binning(other: AbstractArray) -> bool:
        """ Raise error as it is implemented in subclass only """
        raise NotImplementedError()

    def same_shape(self, other: Sequence[float], error: bool = False) -> bool:
        return self.has_equal_binning()

    def copy(self, **kwargs) -> AbstractArray:
        """ Return a deepcopy of the class

        Args:
            kwargs: Overwrite attributes of the copied object
        """
        new = copy.deepcopy(self)
        for attr, val in kwargs.items():
            setattr(new, attr, val)
        return new

    def verify_equdistant(self, axis: Union[int, str]):
        """ Runs checks to verify if energy arrays are equidistant

        axis: The axis to project onto.
                  Can be either of (0, 'Eg', 'x'), (1, 'Ex', 'y')
        Raises:
            ValueError: If any check fails
        """
        axis = to_plot_axis(axis)
        isEx = (axis == 1)
        try:  # better with isinstance, but good for now
            energy = self.Ex if isEx else self.Eg
            name = "Ex" if isEx else "Eg"
        except AttributeError:
            energy = self.E
            name = "E"

        # Check shapes:
        if len(energy) > 2:
            diff = (energy - np.roll(energy, 1))[1:]  # E_{i} - E_{i-1}
            try:
                diffdiff = diff - diff[1]
                np.testing.assert_array_almost_equal(diffdiff,
                                                     np.zeros_like(diff))
            except AssertionError:
                raise ValueError(f"{name} array is not equispaced")

    def __eq__(self, other) -> None:
        if self.__class__ != other.__class__:
            return False
        else:
            dicother = other.__dict__
            truth = []
            for key, value in self.__dict__.items():
                if isinstance(value, np.ndarray):
                    test = np.allclose(value, dicother[key])
                else:
                    test = (value == dicother[key])
                truth.append(test)
            return all(truth)

    def __sub__(self, other) -> AbstractArray:
        result = self.copy()
        if isinstance(other, (int, float)):
            result.values -= other
        else:
            if isinstance(other, np.ndarray):
                self.same_shape(other, error=True)
                result.values -= other
            elif self.has_equal_binning(other):
                result.values -= other.values
        return result

    def __rsub__(self, other) -> AbstractArray:
        result = self.copy()
        if isinstance(other, (int, float)):
            result.values = other - result.values
        else:
            if isinstance(other, np.ndarray):
                self.same_shape(other, error=True)
                result.values = other - result.values
            elif self.has_equal_binning(other):
                result.values = other.values - result.values
        return result

    def __add__(self, other) -> AbstractArray:
        result = self.copy()
        if isinstance(other, (int, float)):
            result.values += other
        else:
            if isinstance(other, np.ndarray):
                self.same_shape(other, error=True)
                result.values += other
            elif self.has_equal_binning(other):
                result.values += other.values
        return result

    def __radd__(self, other) -> AbstractArray:
        return self.__add__(other)

    def __mul__(self, other) -> AbstractArray:
        result = self.copy()
        if isinstance(other, (int, float)):
            result.values *= other
        else:
            if isinstance(other, np.ndarray):
                self.same_shape(other, error=True)
                result.values *= other
            elif self.has_equal_binning(other):
                result.values *= other.values
        return result

    def __rmul__(self, other) -> AbstractArray:
        return self.__mul__(other)

    def __truediv__(self, other) -> AbstractArray:
        result = self.copy()
        if isinstance(other, (int, float)):
            result.values /= other
        else:
            if isinstance(other, np.ndarray):
                self.same_shape(other, error=True)
                result.values /= other
            elif self.has_equal_binning(other):
                result.values /= other.values
        return result

    def __rtruediv__(self, other) -> AbstractArray:
        result = self.copy()
        if isinstance(other, (int, float)):
            result.values = other / result.values
        else:
            if isinstance(other, np.ndarray):
                self.same_shape(other, error=True)
                result.values = other / result.values
            elif self.has_equal_binning(other):
                result.values = other.values / result.values
        return result

    def __matmul__(self, other) -> AbstractArray:
        """
        Implemented in subclasses
        """
        raise NotImplementedError()

    @property
    def shape(self) -> Union[Tuple[int], Tuple[int, int]]:
        return self.values.shape

    def __getitem__(self, key):
        return self.values.__getitem__(key)

    def __setitem__(self, key, item):
        return self.values.__setitem__(key, item)


def to_plot_axis(axis: Union[int, str]) -> int:
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
