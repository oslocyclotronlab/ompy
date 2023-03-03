from __future__ import annotations

import logging
import numpy as np
from ..stubs import arraylike
from abc import ABC, abstractmethod
from .index import Index
from .. import Unit

LOG = logging.getLogger(__name__)
logging.captureWarnings(True)


class AbstractArray(ABC):
    __default_unit: Unit = Unit('keV')
    @abstractmethod
    def is_compatible_with(self, other: AbstractArray | Index) -> bool: ...

    @abstractmethod
    def clone(self, **kwargs) -> AbstractArray: ...

    def check_or_assert(self, other) -> np.ndarray | float:
        if isinstance(other, AbstractArray):
            if not self.shape == other.shape:
                raise ValueError(f"Incompatible shapes. {self.shape} != {other.shape}")
            if not self.is_compatible_with(other):
                raise ValueError("Incompatible binning.")
            if self.std is not None or other.std is not None:
                raise NotImplementedError("Cannot handle arrays with uncertainties")
            other = other.values
        return other

    def __sub__(self, other) -> AbstractArray:
        other = self.check_or_assert(other)
        return self.clone(values = self.values - other)

    def __rsub__(self, other) -> AbstractArray:
        result = self.clone(values = other - self.values)
        return result

    def __add__(self, other) -> AbstractArray:
        other = self.check_or_assert(other)
        return self.clone(values = self.values + other)

    def __radd__(self, other) -> AbstractArray:
        print(type(other))
        x = self.__add__(other)
        print("X!!", type(x))
        return x

    def __mul__(self, other) -> AbstractArray:
        other = self.check_or_assert(other)
        return self.clone(values = self.values * other)

    def __rmul__(self, other) -> AbstractArray:
        other = self.check_or_assert(other)
        return self.clone(values = other * self.values)

    def __truediv__(self, other) -> AbstractArray:
        other = self.check_or_assert(other)
        return self.clone(values = self.values / other)

    def __rtruediv__(self, other) -> AbstractArray:
        other = self.check_or_assert(other)
        return self.clone(values = other / self.values)

    def __pow__(self, val: float) -> AbstractArray:
        return self.clone(values = self.values ** val)

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

def to_plot_axis(axis: int | str) -> int:
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
