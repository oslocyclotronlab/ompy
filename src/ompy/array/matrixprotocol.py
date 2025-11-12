from __future__ import annotations

from typing import Any, Callable, Literal, Protocol, Sequence, Self, runtime_checkable

import numpy as np

from ..stubs import ArrayBool, Axes, Colorbar, Pathlike, QuadMesh, Unitlike, arraylike
from .abstractarrayprotocol import AbstractArrayProtocol
from .filehandling import Filetype
from .index import Edges, Index
from .matrixmetadata import MatrixMetadata


@runtime_checkable
class MatrixProtocol(AbstractArrayProtocol[np.ndarray, np.generic], Protocol):
    values: np.ndarray
    metadata: MatrixMetadata
    loc: Any
    iloc: Any

    def __init__(
        self,
        *,
        X: arraylike | Index | None = None,
        Y: arraylike | Index | None = None,
        values: np.ndarray | None = None,
        X_unit: Unitlike | None = None,
        Y_unit: Unitlike | None = None,
        edge: Edges = "left",
        boundary: bool = False,
        metadata: MatrixMetadata = MatrixMetadata(),
        order: Any = None,
        copy: bool = False,
        indexkwargs: dict[str, Any] | None = None,
        dtype: Any = np.float32,
        **kwargs: Any,
    ) -> None: ...

    def save(self, path: Pathlike, filetype: Filetype | None = None, **kwargs: Any) -> None: ...

    @classmethod
    def from_path(cls, path: Pathlike, filetype: Filetype | None = None, **kwargs: Any) -> Self: ...

    def reshape_like(self, other: MatrixProtocol, inplace: bool = False) -> MatrixProtocol | None: ...

    def rebin(
        self,
        axis: int | str,
        *,
        bins: Sequence[float] | Index | None = None,
        factor: float | None = None,
        binwidth: Unitlike | None = None,
        numbins: int | None = None,
        preserve: str = "counts",
        inplace: bool = False,
    ) -> MatrixProtocol | None: ...

    def rebin_coarsest(self, inplace: bool = False, preserve: str = "counts") -> MatrixProtocol | None: ...

    def rebin_identical(self, inplace: bool = False, preserve: str = "counts") -> MatrixProtocol | None: ...

    def index_X(self, x: float) -> int: ...

    def index_Y(self, x: float) -> int: ...

    def to_unit(self, unit: Unitlike, axis: int | str = "both", inplace: bool = False) -> MatrixProtocol | None: ...

    def to_mid(self, axis: int | str = "both", inplace: bool = False) -> MatrixProtocol | None: ...

    def to_left(self, axis: int | str = "both", inplace: bool = False) -> MatrixProtocol | None: ...

    def to_edge(self, edge: Literal["left", "mid"], axis: int | str = "both", inplace: bool = False) -> MatrixProtocol | None: ...

    def shift_index(self, which: int | str, offset: Any, inplace: bool = False) -> MatrixProtocol | None: ...

    def set_order(self, order: Literal["C", "F"]) -> None: ...

    @property
    def dX(self) -> float | np.ndarray: ...

    @property
    def dY(self) -> float | np.ndarray: ...

    def from_mask(self, mask: ArrayBool, inplace: bool = False) -> MatrixProtocol | None: ...

    @property
    def T(self) -> MatrixProtocol: ...

    @property
    def _summary(self) -> str: ...

    def summary(self) -> None: ...

    def sum(self, axis: int | str = "both") -> Any: ...

    @property
    def X(self) -> np.ndarray: ...

    @property
    def Y(self) -> np.ndarray: ...

    def clone(
        self,
        X: Index | None = None,
        Y: Index | None = None,
        values: np.ndarray | None = None,
        metadata: MatrixMetadata | None = None,
        copy: bool = False,
        **kwargs: Any,
    ) -> MatrixProtocol: ...

    def is_compatible_with_X(self, other: MatrixProtocol | Index) -> bool: ...

    def is_compatible_with_Y(self, other: MatrixProtocol | Index) -> bool: ...

    def normalize(self, axis: int | str = "both", inplace: bool = False) -> MatrixProtocol | None: ...

    def plot(
        self,
        ax: Axes,
        *,
        scale: str | None = None,
        vmin: float | None = None,
        vmax: float | None = None,
        add_cbar: bool = True,
        cbarkwargs: dict[str, Any] | None = None,
        bad_map: Callable[[MatrixProtocol], ArrayBool | bool] = lambda _: False,
        **kwargs: Any,
    ) -> tuple[Axes, tuple[QuadMesh, Colorbar | None]]: ...

    def _plot_mesh(self) -> tuple[np.ndarray, np.ndarray]: ...

    def meta_into_vector(self, index: np.ndarray | Index, values: np.ndarray) -> Any: ...

    def axis_to_int(self, axis: int | str, allow_both: bool = False) -> int: ...

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
