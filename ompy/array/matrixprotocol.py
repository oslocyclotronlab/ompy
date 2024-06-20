from __future__ import annotations

import logging
from typing import (Any, Callable, Literal, Protocol, Sequence, TypeAlias,
                    TypeVar, overload, Self)

import numpy as np
from nptyping import Floating, NDArray, Shape

from ..stubs import (ArrayBool, Axes, Colorbar, Pathlike, QuadMesh, Unitlike,
                     array, arraylike, numeric)
from .abstractarray import AbstractArray
from .abstractarrayprotocol import AbstractArrayProtocol
from .filehandling import Filetype
from .index import Edges, Index
from .matrixmetadata import MatrixMetadata
from .vector import Vector

LOG = logging.getLogger(__name__)
logging.captureWarnings(True)

# TODO mat*vec[:, None[ doesn't work
AxisEither: TypeAlias = Literal[0, 1]
AxisBoth: TypeAlias = Literal[0, 1, 2]
Axis: TypeAlias = AxisEither | AxisBoth

T = TypeVar('T', bound='MatrixProtocol')

class MatrixProtocol(AbstractArrayProtocol, Protocol):
    values: NDArray[Shape['*', '*'], Floating]

    def __init__(self, *,
                 X: arraylike | Index | None = None,
                 Y: arraylike | Index | None = None,
                 values: np.ndarray | None = None,
                 unit: Unitlike | None = None,
                 edge: Edges = 'left',
                 boundary: bool = False,
                 metadata: MatrixMetadata = MatrixMetadata(),
                 order: Literal['C', 'F'] = 'C',
                 copy: bool = False,
                 indexkwargs: dict[str, Any] | None = None,
                 **kwargs): ...

    def __getattr__(self, item) -> Any: ...

    @classmethod
    def from_path(cls, path: Pathlike, filetype: Filetype | None = None, **kwargs) -> MatrixProtocol: ...

    def save(self, path: Pathlike, filetype: Filetype | None = None,
            **kwargs) -> None: ...

    def reshape_like(self, other: MatrixProtocol,
                     inplace: bool = False) -> MatrixProtocol | None: ...

    @overload
    def rebin(self, axis: int | str, *,
              bins: Sequence[float] | Index | None = None,
              factor: float | None = None,
              binwidth: Unitlike | None = None,
              numbins: int | None = None,
              inplace: Literal[False] = ...) -> MatrixProtocol: ...

    @overload
    def rebin(self, axis: int | str, *,
              bins: Sequence[float] | Index | None = None,
              factor: float | None = None,
              binwidth: Unitlike | None = None,
              numbins: int | None = None,
              inplace: Literal[True] = ...) -> None: ...

    def rebin(self, axis: int | str, *,
              bins: Sequence[float] | Index | None = None,
              factor: float | None = None,
              binwidth: Unitlike | None = None,
              numbins: int | None = None,
              preserve: str = 'counts',
              inplace: bool = False) -> MatrixProtocol | None: ...

    def index_X(self, x: float) -> int: ...

    def index_Y(self, x: float) -> int: ...

    def to_unit(self, unit: Unitlike, axis: str | int = 'both', inplace: bool = False) -> None | Self: ...

    def to_mid(self, axis: int | str = 'both', inplace: bool = False) -> None | Self: ...

    @overload
    def to_left(self, axis: int | str = 'both', inplace: Literal[False] = ...) -> Self: ...

    @overload
    def to_left(self, axis: int | str = 'both', inplace: Literal[True] = ...) -> None: ...

    def to_left(self, axis: int | str = 'both', inplace: bool = False) -> None | Self: ...

    def to_edge(self, edge: Literal['left', 'mid'], axis: int | str = 'both', inplace: bool = False) -> None | Self: ...

    def set_order(self, order: str) -> None: ...

    @property
    def dX(self) -> float | np.ndarray: ...

    @property
    def dY(self) -> float | np.ndarray: ...

    def from_mask(self, mask: ArrayBool) -> MatrixProtocol | Vector: ...

    @property
    def T(self) -> Self: ...

    @property
    def _summary(self) -> str: ...

    def summary(self): ...

    def sum(self, axis: int | str = 'both') -> Vector | float: ...

    def __str__(self) -> str: ...

    @property
    def X(self) -> np.ndarray: ...

    @property
    def Y(self) -> np.ndarray: ...

    def clone(self, X: Index | None = None, Y: Index | None = None,
              values: np.ndarray | None = None,
              metadata: MatrixMetadata | None = None, copy: bool = False,
              **kwargs) -> MatrixProtocol: ...

    def is_compatible_with(self, other: AbstractArray | Index) -> bool: ...

    def is_compatible_with_X(self, other: AbstractArray | Index) -> bool: ...

    def is_compatible_with_Y(self, other: AbstractArray | Index) -> bool: ...

    def normalize(self, axis: str | Literal[0, 1, 2], inplace=False) -> MatrixProtocol | None: ...

    @overload
    def plot(self, ax: Axes, *,
             scale: str | None = None,
             vmin: float | None = None,
             vmax: float | None = None,
             add_cbar: Literal[True] = ...,
             **kwargs) -> tuple[Axes, tuple[QuadMesh, Colorbar]]: ...

    @overload
    def plot(self, ax: Axes, *,
             scale: str | None = None,
             vmin: float | None = None,
             vmax: float | None = None,
             add_cbar: Literal[False] = ...,
             **kwargs) -> tuple[Axes, tuple[QuadMesh, None]]: ...

    def plot(self, ax: Axes, *,
             scale: str | None = None,
             vmin: float | None = None,
             vmax: float | None = None,
             add_cbar: bool = True,
             cbarkwargs: dict[str, Any] | None = None,
             bad_map: Callable[[MatrixProtocol], ArrayBool | bool] = lambda x: False,
             **kwargs) -> tuple[Axes, tuple[QuadMesh, Colorbar | None]]: ...

    def _plot_mesh(self) -> tuple[np.ndarray, np.ndarray]: ...

    def meta_into_vector(self, index: np.ndarray | Index, values: np.ndarray) -> Vector: ...

    @overload
    def axis_to_int(self, axis: int | str,
                    allow_both: Literal[False]) -> AxisEither: ...

    @overload
    def axis_to_int(self, axis: int | str,
                    allow_both: Literal[True]) -> AxisBoth: ...

    def axis_to_int(self, axis: int | str, allow_both: bool = False) -> Axis: ...

    @property
    def xalias(self) -> str: ...

    @property
    def yalias(self) -> str: ...

    @property
    def xlabel(self) -> str: ...

    @xlabel.setter
    def xlabel(self, value: str) -> None: ...

    @property
    def ylabel(self) -> str: ...

    @ylabel.setter
    def ylabel(self, value: str) -> None: ...

    @overload
    def __matmul__(self: T, other: MatrixProtocol) -> T: ...
    @overload
    def __matmul__(self, other: Vector) -> Vector: ...
    @overload
    def __matmul__(self, other: np.ndarray) -> np.ndarray: ...

    def __matmul__(self: T, other: MatrixProtocol | Vector | np.ndarray) -> T | Vector | np.ndarray: ...

    def last_nonzero(self, i: int) -> int: ...

    def last_nonzeros(self) -> np.ndarray: ...

    def copy(self: T, **kwargs) -> T: ...
