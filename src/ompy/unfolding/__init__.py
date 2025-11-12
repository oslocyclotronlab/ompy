from typing import TYPE_CHECKING
import importlib

from .unfolder import Unfolder
from .result1d import UnfoldedResult1D


__all__ = [
    "Unfolder",
    "UnfoldedResult1D",
    "fics",
    "rmle",
    "resampling",
    "richardsonlucy",
    "FICS",
    "RMLE",
    "RichardsonLucy",
]  

_LAZY_MODULES = {"fics", "rmle", "resampling", "richardsonlucy"}


_LAZY_SYMBOLS = {
    "FICS": ("fics", "FICS"),
    "RMLE": ("rmle", "RMLE"),
    "RichardsonLucy": ("richardsonlucy", "RichardsonLucy"),
}

def __getattr__(name: str):
    if name in _LAZY_MODULES:
        mod = importlib.import_module(f".{name}", __package__)
        globals()[name] = mod               # cache for subsequent lookups
        return mod

    # Lazy symbols: om.unfolding.FICS
    if name in _LAZY_SYMBOLS:
        modname, attr = _LAZY_SYMBOLS[name]
        mod = importlib.import_module(f".{modname}", __package__)
        obj = getattr(mod, attr)
        globals()[name] = obj               # cache symbol
        return obj

    raise AttributeError(f"module {__name__} has no attribute {name}")

def __dir__():
    # helps tab-completion
    return sorted(set(globals().keys()) | _LAZY_MODULES | _LAZY_SYMBOLS.keys())

# Help static type checkers (mypy/pyright) without paying runtime import cost
if TYPE_CHECKING:
    from .fics import FICS as FICS  # noqa: F401
    from .rmle import RMLE as RMLE  # noqa: F401
    from . import fics as fics      # noqa: F401
    from . import rmle as rmle      # noqa: F401
    from . import resampling as resampling  # noqa: F401
    from . import richardsonlucy as richardsonlucy  # noqa: F401
    from .richardsonlucy import RichardsonLucy as RichardsonLucy  # noqa: F401
