from __future__ import annotations

from .resampling import Resampling
from .resample1d import resample_vector
from .resample2d import resample_matrix
from typing import Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from ..result import Result


def resample(
    res: Result, N: int, base: Literal["raw", "nu"] = "raw", **kwargs
) -> Resampling:
    match res.ndim:
        case 1:
            return resample_vector(res, N, base=base, **kwargs)
        case 2:
            return resample_matrix(res, N, base=base, **kwargs)
        case _:
            raise ValueError(f"Unknown result type {res.__class__.__name__}")