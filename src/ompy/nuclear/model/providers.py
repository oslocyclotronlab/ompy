from __future__ import annotations

from typing import Callable, TypeVar

from ..readers.ripl3.reader import GammaRecord
from .levels import GammaBranch

F = TypeVar("F", bound=Callable[..., object])

try:
    from .registry import register_provider  # type: ignore
except ImportError:  # pragma: no cover - fallback for legacy layouts
    def register_provider(*_args, **_kwargs):  # type: ignore
        def decorator(func: F) -> F:
            return func

        return decorator


@register_provider(
    "from_ripl3_gamma_record",
    **{"from": GammaRecord, "to": GammaBranch},
)
def gamma_record_provider(record: GammaRecord) -> GammaBranch:
    """Convert a RIPL-3 gamma record into a :class:`GammaBranch`."""

    return GammaBranch(
        final=record.Nf,
        Eg=record.Eg,
        Pg=record.Pg,
        Pem=record.Pe,
        ICC=record.ICC,
    )
