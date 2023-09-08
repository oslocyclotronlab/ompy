from __future__ import annotations

import logging
import numpy as np
from abc import ABC, abstractmethod
from .index import Index
from .. import Unit
from .abstractarrayprotocol import AbstractArrayProtocol
from nptyping import NDArray, Shape, Floating
from typing import Self, Literal


LOG = logging.getLogger(__name__)
logging.captureWarnings(True)

#TODO Implement all of the i-methods and logical methods
# [ ] __imatmul__


class AbstractArray(AbstractArrayProtocol, ABC):
    __default_unit: Unit = Unit('keV')

    def __init__(self, values: np.ndarray):
        self.values: NDArray[Shape['*', ...], Floating] = values

    @property
    def __array_interface__(self):
        return self.values.__array_interface__

    def __array__(self, dtype=None) -> np.ndarray:
        return np.asarray(self.values, dtype=dtype)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        # TODO Untested. Might summon demons.
        cls = type(self)
        # Replace ArrayWrapper instances with their .values attribute
        inputs = tuple(i.values if isinstance(i, AbstractArray) else i for i in inputs)

        # Perform the operation on the underlying numpy arrays
        result = getattr(ufunc, method)(*inputs, **kwargs)

        # Wrap the result back in an ArrayWrapper (or handle other result types as needed)
        if method == 'at':
            # In-place method, no return value
            return None
        elif isinstance(result, tuple):
            # Multiple return values
            return tuple(self.clone(values=x) for x in result)
        elif method == 'reduceat':
            # reduceat returns a single array
            return self.clone(values=result)
        else:
            # Standard ufunc, single return value
            return self.clone(values=result)

    @abstractmethod
    def is_compatible_with(self, other: AbstractArray | Index) -> bool: ...

    @abstractmethod
    def clone(self, **kwargs) -> Self: ...

    def copy(self, **kwargs) -> Self:
        return self.clone(copy=True, **kwargs)

    def check_or_assert(self, other) -> np.ndarray | float:
        if isinstance(other, AbstractArray):
            if not self.shape == other.shape:
                raise ValueError(f"Incompatible shapes. {self.shape} != {other.shape}")
            if not self.is_compatible_with(other):
                raise ValueError("Incompatible binning.")
            other = other.values
        return other

    def __sub__(self, other) -> Self:
        other = self.check_or_assert(other)
        return self.clone(values = self.values - other)

    def __rsub__(self, other) -> Self:
        result = self.clone(values = other - self.values)
        return result

    def __add__(self, other) -> Self:
        other = self.check_or_assert(other)
        return self.clone(values = self.values + other)

    def __radd__(self, other) -> Self:
        x = self.__add__(other)
        return x

    def __mul__(self, other) -> Self:
        other = self.check_or_assert(other)
        return self.clone(values = self.values * other)

    def __rmul__(self, other) -> Self:
        other = self.check_or_assert(other)
        return self.clone(values = other * self.values)

    def __truediv__(self, other) -> Self:
        other = self.check_or_assert(other)
        return self.clone(values = self.values / other)

    def __rtruediv__(self, other) -> Self:
        other = self.check_or_assert(other)
        return self.clone(values = other / self.values)

    def __pow__(self, val: float) -> Self:
        return self.clone(values = self.values ** val)

    def __iand__(self, other) -> Self:
        other = self.check_or_assert(other)
        self.values &= other
        return self

    def __and__(self, other) -> Self:
        other = self.check_or_assert(other)
        return self.clone(values = self.values & other)

    def __or__(self, other) -> Self:
        other = self.check_or_assert(other)
        return self.clone(values = self.values | other)

    def __ior__(self, other) -> Self:
        other = self.check_or_assert(other)
        self.values |= other
        return self

    def __ixor__(self, other) -> Self:
        other = self.check_or_assert(other)
        self.values ^= other
        return self

    def __xor__(self, other: AbstractArrayProtocol | np.ndarray) -> Self:
        other = self.check_or_assert(other)
        return self.clone(values = self.values ^ other)

    def __lshift__(self, other) -> Self:
        other = self.check_or_assert(other)
        return self.clone(values=self.values << other)

    def __rshift__(self, other) -> Self:
        other = self.check_or_assert(other)
        return self.clone(values=self.values >> other)

    def __ilshift__(self, other) -> Self:
        other = self.check_or_assert(other)
        self.values <<= other
        return self

    def __irshift__(self, other) -> Self:
        other = self.check_or_assert(other)
        self.values >>= other
        return self

    def __iadd__(self, other) -> Self:
        other = self.check_or_assert(other)
        self.values += other
        return self

    def __isub__(self, other) -> Self:
        other = self.check_or_assert(other)
        self.values -= other
        return self

    def __imul__(self, other) -> Self:
        other = self.check_or_assert(other)
        self.values *= other
        return self

    def __itruediv__(self, other) -> Self:
        other = self.check_or_assert(other)
        self.values /= other
        return self

    def __invert__(self):
        return self.clone(values=~self.values)

    @abstractmethod
    def __matmul__(self, other) -> AbstractArray: ...

    #def __rmatmul__(self, other) -> AbstractArray:

    @property
    def shape(self) -> tuple[int, ...]:
        return self.values.shape

    def __getitem__(self, key):
        return self.values.__getitem__(key)

    def __setitem__(self, key, item):
        return self.values.__setitem__(key, item)

    def __getattr__(self, attr):
        """ Fallback to numpy if AbstractArray does not have the attribute
        """
        name = self.__class__.__name__
        if attr.startswith("_"):
            raise AttributeError(f"'{name}' object has no attribute {attr}")
        res = getattr(self.__dict__['values'], attr, None)
        if res is not None:
            return res
        # Can't use AttributeError as that is handled exceptionally and
        # causes a really, reeally wierd bug
        raise Exception(f"Neither {name} nor {name}.values has '{attr}'")

    def __len__(self) -> int:
        return len(self.values)

    def __neg__(self):
        return self.clone(values=-self.values)

    def __lt__(self, other):
        return self.values < other

    def __gt__(self, other):
        return self.values > other

    def __le__(self, other):
        return self.values <= other

    def __ge__(self, other):
        return self.values >= other

    def __abs__(self):
        return self.clone(values=np.abs(self.values))

    @property
    def vlabel(self) -> str:
        return self.metadata.vlabel

    @property
    def valias(self) -> str:
        return self.metadata.valias

    @property
    def name(self) -> str:
        return self.metadata.name

    @name.setter
    def name(self, name: str):
        self.metadata = self.metadata.update(name=name)

def to_plot_axis(axis: int | str) -> Literal[1,2,3]:
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
