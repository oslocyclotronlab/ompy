
@runtime_checkable
class IndexProtocol(Protocol):
    def __init__(self, bins: np.ndarray, boundary: float,
                 metadata: IndexMetadata = IndexMetadata(),
                 dtype: type = np.float64, **kwargs): ...

    @property
    def bins(self) -> NDArray[np.number]: ...

    @property
    def meta(self) -> IndexMetadata: ...

    @property
    def boundary(self) -> float: ...

    @property
    def unit(self) -> Unit: ...

    @property
    def label(self) -> str: ...

    @property
    def alias(self) -> str: ...

    def steps(self) -> np.ndarray: ...

    def step(self, i: int) -> float: ...

    def is_uniform(self) -> bool: ...

    @property
    def leftmost(self) -> float: ...

    @property
    def rightmost(self) -> float: ...

    def is_left(self) -> bool: ...

    def is_mid(self) -> bool: ...

    def to_left(self) -> IndexProtocol: ...

    def to_mid(self) -> IndexProtocol: ...

    def _rebin(self, other: IndexProtocol, values: np.ndarray, preserve: Preserve = 'counts') -> tuple[IndexProtocol, np.ndarray]: ...

    def rebin(self: IndexProtocol, other: IndexProtocol | np.ndarray, values: np.ndarray, preserve: Preserve = 'counts') -> tuple[
            IndexProtocol, np.ndarray]: ...

    def to_same(self: IndexProtocol, other: IndexProtocol) -> IndexProtocol: ...

    def to_same_edge(self, other: IndexProtocol) -> IndexProtocol: ...

    def to_same_unit(self, x: Unitful) -> float: ...

    def index(self, x: QuantityLike) -> int: ...

    def _index(self, x: float) -> int: ...

    @classmethod
    def from_extrapolate(cls: type[IndexProtocol], bins: np.ndarray, n: int = 1, direction: Direction = 'auto',
                         extrapolate_boundary: bool = True, **kwargs) -> IndexProtocol: ...

    @classmethod
    def extrapolate(cls, bins: np.ndarray, direction: Direction, n: int = 1) -> np.ndarray: ...


    @staticmethod
    def extrapolate_boundary(bins: np.ndarray, direction: DirectionLR = 'left') -> float: ...

    @classmethod
    def from_array(cls: type[IndexProtocol], bins: np.ndarray, extrapolate_boundary: bool = True, **kwargs) -> IndexProtocol: ...

    @staticmethod
    def split_bins_boundary(bins: np.ndarray) -> tuple[float, np.ndarray]: ...

    @overload
    def __getitem__(self, key: int) -> float: ...

    @overload
    def __getitem__(self, key: slice) -> IndexProtocol: ...

    def __getitem__(self, key: int | slice) -> float | IndexProtocol: ...

    def __len__(self) -> int: ...

    def summary(self: IndexProtocol) -> str: ...

    def __str__(self: IndexProtocol) -> str: ...

    def __eq__(self: IndexProtocol, other: IndexProtocol) -> bool: ...

    def update_metadata(self, **kwargs) -> IndexProtocol: ...

    def to_unit(self: IndexProtocol, unit: Unitlike | IndexProtocol) -> IndexProtocol: ...


    @overload
    def index_expression(self, expr: QuantityLike | str, strict: bool = True) -> int:
        ...

    @overload
    def index_expression(self, expr: None, strict: bool = True) -> None:
        ...

    def index_expression(self, expr: QuantityLike | str | None, strict: bool = True) -> int | None: ...

    def index_slice(self, s: slice, strict: bool = True) -> slice: ...

    @classmethod
    def from_calibration(cls: type[IndexProtocol], calibration: Calibration, n: int, **kwargs) -> IndexProtocol: ...

    def to_dict(self: IndexProtocol) -> dict[str, Any]: ...

    @classmethod
    def from_dict(cls, d: IndexDict) -> Self: ...

    @classmethod
    def _assert_cls(cls, d: IndexDict): ...

    def is_compatible_with(self: IndexProtocol, other: IndexProtocol, rtol: float = 1e-4, **kwargs) -> bool: ...


    def __clone(self: IndexProtocol, bins: np.ndarray | None = None,
                boundary: float | None = None,
                meta: IndexMetadata | None = None,
                order: np._OrderKACF | None = None,
                **kwargs) -> IndexProtocol: ...

    def update(self: IndexProtocol, alias: str | None = None, label: str | None = None,
               unit: Unitlike | None = None) -> IndexProtocol:
        meta = self.meta.clone(alias=alias, label=label, unit=unit)
        return self.__clone(meta=meta)

    def handle_rebin_arguments(self: IndexProtocol, *,
                               bins: arraylike | IndexProtocol | None = None,
                               factor: float | None = None,
                               binwidth: QuantityLike | None = None,
                               numbins: int | None = None) -> IndexProtocol: ...

    @classmethod
    def uniform_cls(cls: type[IndexProtocol]) -> IndexProtocol: ...


    def is_content_close(self, other: IndexProtocol) -> bool: ...

    def is_metadata_equal(self, other: IndexProtocol) -> bool: ...

    def other_edge_cls(self) -> Type[IndexProtocol]: ...

    def assert_inbounds(self, x: float): ...

    def ticks(self: IndexProtocol) -> np.ndarray: ...


    def steps(self) -> np.ndarray:
        ...

    def step(self, i: int) -> float:
        ...

    def is_uniform(self) -> bool: ...

    def to_calibration(self: IndexProtocol) -> Calibration: ...

class NonUniformProtocol(Protocol):
    dX: NDArray[np.number]

class NonUniformIndex(NonUniformProtocol, Index, Protocol): ...


@runtime_checkable
class UniformProtocol(Protocol):
    __hash: int | None
    @property
    def dX(self) -> float: ...

def has_dX(x) -> TypeGuard[UniformIndex]:
    return hasattr(x, 'dX')

@runtime_checkable
class UniformIndex(UniformProtocol, Index, Protocol): ...