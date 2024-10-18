from __future__ import annotations

import logging
import numpy as np
from abc import abstractmethod
from .index import Index
from .. import Unit
from typing import Protocol, Self
#from nptyping import NDArray, Floating, Shape

LOG = logging.getLogger(__name__)
logging.captureWarnings(True)

#TODO Implement all of the i-methods and logical methods
# [ ] __imatmul__


class AbstractArrayProtocol(Protocol):
    __default_unit: Unit
    #values: NDArray[Shape['*', ...], Floating]
    values: np.ndarray

    def __init__(self, values: np.ndarray): ...

    @property
    def __array_interface__(self): ...

    def __array__(self, dtype=None) -> np.ndarray: ...

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs): ...

    @abstractmethod
    def is_compatible_with(self, other: AbstractArrayProtocol | Index) -> bool: ...

    @abstractmethod
    def clone(self, **kwargs) -> Self: ...

    def copy(self, **kwargs) -> Self: ...

    def check_or_assert(self, other) -> np.ndarray | float: ...

    def __sub__(self, other) -> Self: ...

    def __rsub__(self, other) -> Self: ...

    def __add__(self, other) -> Self: ...

    def __radd__(self, other) -> Self: ...

    def __mul__(self, other) -> Self: ...

    def __rmul__(self, other) -> Self: ...

    def __truediv__(self, other) -> Self: ...

    def __rtruediv__(self, other) -> Self: ...

    def __pow__(self, val: float) -> Self: ...

    def __iand__(self, other) -> Self: ...

    def __and__(self, other) -> Self: ...

    def __or__(self, other) -> Self: ...

    def __ior__(self, other) -> Self: ...

    def __ixor__(self, other) -> Self: ...

    def __xor__(self, other) -> Self: ...

    def __lshift__(self, other) -> Self: ...

    def __rshift__(self, other) -> Self: ...

    def __ilshift__(self, other) -> Self: ...

    def __irshift__(self, other) -> Self: ...

    def __iadd__(self, other) -> Self: ...

    def __isub__(self, other) -> Self: ...

    def __imul__(self, other) -> Self: ...

    def __itruediv__(self, other) -> Self: ...

    def __invert__(self): ...

    @abstractmethod
    def __matmul__(self, other) -> AbstractArrayProtocol: ...

    #def __rmatmul__(self, other) -> AbstractArrayProtocol:

    @property
    def shape(self) -> tuple[int, ...]: ...

    def __getitem__(self, key): ...

    def __setitem__(self, key, item): ...

    def __getattr__(self, attr): ...

    def __len__(self) -> int: ...

    def __neg__(self): ...

    def __lt__(self, other): ...

    def __gt__(self, other): ...

    def __le__(self, other): ...

    def __ge__(self, other): ...

    def __abs__(self): ...

    @property
    def vlabel(self) -> str: ...

    @property
    def valias(self) -> str: ...

    @property
    def name(self) -> str: ...

    @name.setter
    def name(self, name: str): ...
