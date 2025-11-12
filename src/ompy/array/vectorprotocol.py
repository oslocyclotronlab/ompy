from __future__ import annotations

from typing import Any, Iterable, Literal, Protocol, Self, runtime_checkable

import numpy as np

from ..stubs import Axes, Pathlike, Plot1D, QuantityLike, Unitlike, array1D, arraylike
from .abstractarrayprotocol import AbstractArrayProtocol
from .index import Edges, Index
from .vectormetadata import VectorMetadata
from .rebin import Preserve


@runtime_checkable
class VectorProtocol(AbstractArrayProtocol[array1D, np.generic], Protocol):
    values: array1D
    metadata: VectorMetadata
    loc: Any
    vloc: Any
    iloc: Any

    def __init__(
        self,
        *,
        X: arraylike | Index | None = None,
        values: arraylike | None = None,
        copy: bool = False,
        unit: Unitlike | None = None,
        order: Literal["C", "F"] | None = None,
        edge: Edges = "left",
        boundary: bool = False,
        metadata: VectorMetadata = VectorMetadata(),
        indexkwargs: dict[str, Any] | None = None,
        dtype: Any = np.float32,
        **kwargs: Any,
    ) -> None: ...

    def save(
        self,
        path: Pathlike,
        filetype: str | None = None,
        exist_ok: bool = True,
        **kwargs: Any,
    ) -> None: ...

    @classmethod
    def from_path(cls, path: Pathlike, filetype: str | None = None, **kwargs: Any) -> Self: ...

    def drop_nan(self, inplace: bool = False) -> Self | None: ...

    def rebin(
        self,
        bins: arraylike | Index | None = None,
        factor: float | None = None,
        binwidth: QuantityLike | None = None,
        numbins: int | None = None,
        preserve: Preserve = "counts",
        inplace: bool = False,
    ) -> Self | None: ...

    def rebin_like(self, other: VectorProtocol | Index, inplace: bool = False, preserve: Preserve = "counts") -> Self | None: ...

    def set_order(self, order: Literal["C", "F"]) -> None: ...

    @property
    def dX(self) -> float | np.ndarray: ...

    def last_nonzero(self, eps: float = 0) -> int: ...

    def update(
        self,
        xlabel: str | None = None,
        vlabel: str | None = None,
        name: str | None = None,
        misc: dict[str, Any] | None = None,
        inplace: bool = False,
        title: str | None = None,
    ) -> Self | None: ...

    def add_comment(self, key: str, comment: Any, inplace: bool = False) -> Self | None: ...

    def clone(
        self,
        X: Index | arraylike | None = None,
        values: arraylike | None = None,
        order: Any = None,
        metadata: VectorMetadata | None = None,
        copy: bool = False,
        dtype: Any = None,
        **kwargs: Any,
    ) -> Self: ...

    def copy(self, **kwargs: Any) -> Self: ...

    @property
    def unit(self) -> Unitlike: ...

    @property
    def xlabel(self) -> str: ...

    @xlabel.setter
    def xlabel(self, value: str) -> None: ...

    @property
    def ylabel(self) -> str: ...

    @ylabel.setter
    def ylabel(self, value: str) -> None: ...

    @property
    def alias(self) -> str: ...

    @property
    def X(self) -> np.ndarray: ...

    @property
    def X_index(self) -> Index: ...

    def enumerate(self) -> Iterable[tuple[int, float, float]]: ...

    def unpack(self) -> tuple[np.ndarray, np.ndarray]: ...

    def index(self, x: QuantityLike) -> int: ...

    def to_unit(self, unit: Unitlike, inplace: bool = False) -> Self | None: ...

    def to_edge(self, edge: Edges, inplace: bool = False) -> Self | None: ...

    def to_left(self, inplace: bool = False) -> Self | None: ...

    def to_mid(self, inplace: bool = False) -> Self | None: ...

    def to_same_edge(self, other: VectorProtocol, inplace: bool = False) -> Self | None: ...

    def to_same(self, other: VectorProtocol) -> Self: ...

    def shift_index(self, offset: QuantityLike, inplace: bool = False) -> Self | None: ...

    def integrate(self, method: Any = ...) -> Any: ...

    def plot(self, ax: Axes | None = None, kind: str = "step", **kwargs: Any) -> Plot1D: ...

    def from_mask(self, mask: np.ndarray, inplace: bool = False) -> Self | None: ...

    def clone_from_slice(self, slice_: slice) -> Self: ...
