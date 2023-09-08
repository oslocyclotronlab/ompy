
from __future__ import annotations

import logging
from typing import Any, Iterable, Literal, overload, TypeAlias
from typing import TYPE_CHECKING, Protocol, Self

import numpy as np
from numpy import ndarray

from .abstractarray import AbstractArray
from .index import Edges, Index
from .. import make_axes
from ..stubs import Unitlike, arraylike, Axes, Pathlike, Plot1D, QuantityLike, array1D
from typing import TypeVar
from .vectormetadata import VectorMetadata
from .abstractarrayprotocol import AbstractArrayProtocol
from .rebin import Preserve

if TYPE_CHECKING:
    from .matrixprotocol import MatrixProtocol

LOG = logging.getLogger(__name__)
logging.captureWarnings(True)

KwargsDict: TypeAlias = dict[str, Any]
T = TypeVar('T', bound='VectorProtocol')


class VectorProtocol(AbstractArrayProtocol, Protocol):
    """ Stores 1d array with energy axes (a vector)

    Attributes:
    """
    values: array1D

    def __init__(self, *, X:  arraylike | Index | None = None,
                 values: arraylike | None = None,
                 copy: bool = False,
                 unit: Unitlike | None = None,
                 order: Literal['C', 'F'] = 'C',
                 edge: Edges = 'left',
                 boundary: bool = False,
                 metadata: VectorMetadata = VectorMetadata(),
                 indexkwargs: dict[str, Any] | None = None,
                 dtype: type = float,
                 **kwargs): ...

    def __getattr__(self, item) -> Any: ...


    def save(self, path: Pathlike,
             filetype: str | None = None,
             exist_ok: bool = True, **kwargs) -> None: ...

    @classmethod
    def from_path(cls: type[T], path: Pathlike, filetype: str | None = None) -> T: ...

    def transform(self, const: float = 1,
                  alpha: float = 0, inplace: bool = True) -> Self | None: ...

    def error(self, other: T | ndarray,
              std: ndarray | None = None) -> float: ...

    def drop_nan(self, inplace: bool = False) -> Self: ...

    @overload
    def rebin(self, bins: arraylike | None = None,
              factor: float | None = None,
              binwidth: QuantityLike | None = None,
              numbins: int | None = None,
              preserve: Preserve = 'counts',
              inplace: Literal[False] = ...) -> Self:
        ...

    @overload
    def rebin(self, bins: arraylike | None = None,
              factor: float | None = None,
              binwidth: QuantityLike | None = None,
              numbins: int | None = None,
              preserve: Preserve = 'counts',
              inplace: Literal[True] = ...) -> None:
        ...

    def rebin(self, bins: arraylike | None = None,
              factor: float | None = None,
              binwidth: QuantityLike | None = None,
              numbins: int | None = None,
              preserve: Preserve = 'counts',
              inplace: bool = False) -> Self | None: ...

    @overload
    def rebin_like(self, other: VectorProtocol, inplace: Literal[False] = ..., preserve: Preserve = ...) -> Self:
        ...

    @overload
    def rebin_like(self, other: VectorProtocol, inplace: Literal[True] = ..., preserve: Preserve = ...) -> None:
        ...

    def rebin_like(self, other: VectorProtocol | Index, inplace: bool = False, preserve: Preserve = 'counts') -> Self | None: ...

    def closest(self, E: ndarray, side: str | None = 'right',
                inplace=False) -> VectorProtocol | None: ...

    def cumulative(self, factor: float | Literal['de'] = 1.0,
                   inplace: bool = False) -> Self | None: ...

    def set_order(self, order: str) -> None: ...

    @property
    def dX(self) -> float | np.ndarray: ...

    def last_nonzero(self) -> int: ...

    def update(self, xlabel: str | None = None, vlabel: str | None = None,
               name: str | None = None, misc: dict[str, Any] | None = None,
               inplace: bool = False) -> None | Self: ...

    def add_comment(self, key: str, comment: Any, inplace: bool = False) -> None | Self: ...

    @property
    def _summary(self) -> str: ...

    def summary(self) -> None: ...

    def __str__(self) -> str: ...

    def clone(self, X=None, values=None, std=None, order: np._OrderKAFC ='C',
              metadata=None, copy=False, **kwargs) -> Self: ...

    def copy(self, **kwargs) -> Self: ...

    @property
    def unit(self) -> Any: ...

    @property
    def xlabel(self) -> str: ...

    @property
    def ylabel(self) -> str: ...

    @property
    def alias(self) -> str: ...

    @property
    def X(self) -> np.ndarray: ...

    @property
    def X_index(self) -> Index: ...

    def enumerate(self) -> Iterable[tuple[int, float, float]]: ...

    def unpack(self) -> tuple[np.ndarray, np.ndarray]: ...

    def index(self, x: float) -> int: ...

    def is_compatible_with(self, other: AbstractArray | Index) -> bool: ...

    def to_unit(self, unit: Unitlike, inplace: bool = False) -> None | Self: ...

    def to_edge(self, edge: Edges, inplace: bool = False) -> None | Self: ...

    def to_left(self, inplace: bool = False) -> None | Self: ...

    def to_mid(self, inplace: bool = False) -> None | Self: ...

    @make_axes
    def plot(self, ax: Axes,
             kind: str = 'step',
             **kwargs) -> Plot1D: ...

    def integrate(self) -> float: ...

    @overload
    def __matmul__(self, other: MatrixProtocol) -> Self: ...
    @overload
    def __matmul__(self, other: VectorProtocol) -> float: ...
    @overload
    def __matmul__(self, other: np.ndarray) -> np.ndarray | float: ...

    def __matmul__(self, other: MatrixProtocol | VectorProtocol | np.ndarray) -> Self | float | np.ndarray: ...

    def from_mask(self, mask: np.ndarray) -> Self: ...
