from typing import TypeAlias, Literal, Any
from numpy.typing import NDArray, DTypeLike

Array: TypeAlias = NDArray[Any]
NPOrder: TypeAlias = Literal['K', 'A', 'C', 'F']  # Copy of np._OrderKACF
