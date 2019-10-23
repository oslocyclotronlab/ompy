from __future__ import annotations

import copy
from typing import Union, Tuple


class AbstractArray:
    def __init__(self):
        raise NotImplementedError()

    def has_equal_binning(other: AbstractArray) -> bool:
        """ Raise error as it is implemented in subclass only """
        raise NotImplementedError()

    def copy(self) -> AbstractArray:
        """ Return a deepcopy of the class"""
        return copy.deepcopy(self)

    def __sub__(self, other) -> AbstractArray:
        result = self.copy()
        if isinstance(other, (int, float)):
            result.values -= other
        else:
            self.has_equal_binning(other)
            result.values -= other.values
        return result

    def __rsub__(self, other) -> AbstractArray:
        result = self.copy()
        if isinstance(other, (int, float)):
            result.values = other - result.values
        else:
            self.has_equal_binning(other)
            result.values = other.values - result.values
        return result

    def __add__(self, other) -> AbstractArray:
        result = self.copy()
        if isinstance(other, (int, float)):
            result.values += other
        else:
            self.has_equal_binning(other)
            result.values += other.values
        return result

    def __radd__(self, other) -> AbstractArray:
        return self.__add__(other)

    def __mul__(self, other) -> AbstractArray:
        result = self.copy()
        if isinstance(other, (int, float)):
            result.values *= other
        else:
            self.has_equal_binning(other)
            result.values *= other.values
        return result

    def __rmul__(self, other) -> AbstractArray:
        return self.__mul__(other)

    def __truediv__(self, other) -> AbstractArray:
        result = self.copy()
        if isinstance(other, (int, float)):
            result.values /= other
        else:
            self.has_equal_binning(other)
            result.values /= other.values
        return result

    def __rtruediv__(self, other) -> AbstractArray:
        result = self.copy()
        if isinstance(other, (int, float)):
            result.values = other / result.values
        else:
            self.has_equal_binning(other)
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

if __name__ == "__main__":
    print("hey")
