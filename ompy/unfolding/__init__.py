import importlib
from .unfolder import Unfolder
from .result1d import UnfoldedResult1D
import typing

if typing.TYPE_CHECKING:
    from . import rmle

def __getattr__(name):
    def _import(name):
        return importlib.import_module("." + name, __package__)
    match name:
        case "rmle":
            return _import("rmle")
        case "resampling":
            return _import("resampling")
        case "fics":
            return _import("fics")
        case _:
            raise AttributeError(f"module {__name__} has no attribute {name}")



__dir__ = ["resampling", "rmle", "fics"]
__all__ = ["resampling", "rmle", "fics", "Unfolder"]