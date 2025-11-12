from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

from . import numpy_backend

try:  # optional dependency
    from . import numba_backend
except Exception:  # pragma: no cover
    numba_backend = None

try:  # optional dependency
    from . import jax_backend
except Exception:  # pragma: no cover
    jax_backend = None


@dataclass(frozen=True)
class Backend:
    run_vector: Callable[..., tuple]
    run_matrix: Callable[..., tuple]
    name: str


class BackendNotAvailableError(RuntimeError):
    """Raised when a requested Richardson-Lucy backend is not available."""


_BACKENDS: Dict[str, Backend] = {
    "numpy": Backend(
        run_vector=numpy_backend.run_vector,
        run_matrix=numpy_backend.run_matrix,
        name="numpy",
    )
}

if numba_backend is not None and getattr(numba_backend, "AVAILABLE", False):
    _BACKENDS["numba"] = Backend(
        run_vector=numba_backend.run_vector,
        run_matrix=numba_backend.run_matrix,
        name="numba",
    )

if jax_backend is not None and getattr(jax_backend, "AVAILABLE", False):
    _BACKENDS["jax"] = Backend(
        run_vector=jax_backend.run_vector,
        run_matrix=jax_backend.run_matrix,
        name="jax",
    )

PREFERRED_ORDER = ("jax", "numba", "numpy")


def is_backend_available(name: str) -> bool:
    return name in _BACKENDS


def best_available_backend(order: tuple[str, ...] = PREFERRED_ORDER) -> Backend:
    for candidate in order:
        if candidate in _BACKENDS:
            return _BACKENDS[candidate]
    available = ", ".join(sorted(_BACKENDS))
    raise BackendNotAvailableError(
        "No Richardson-Lucy backend is available. "
        f"Detected backends: {available if available else 'none'}"
    )


def get_backend(name: str) -> Backend:
    try:
        return _BACKENDS[name]
    except KeyError as exc:  # pragma: no cover - simple guard
        available = ", ".join(sorted(_BACKENDS))
        raise BackendNotAvailableError(
            f"Unknown Richardson-Lucy backend '{name}'. "
            f"Available backends: {available}"
        ) from exc


__all__ = [
    "Backend",
    "BackendNotAvailableError",
    "get_backend",
    "is_backend_available",
    "best_available_backend",
    "PREFERRED_ORDER",
]
