from .stubs import Array
from .index import Index, to_index
from .abstractarray import AbstractArray, on_device
from .vector import Vector
from .matrix import Matrix
from .covariance import CorrelationMatrix
from . import ops
from .arraylist import ArrayList
from .error_vector import AsymmetricVector
from .error_matrix import ErrorMatrix, AsymmetricMatrix
from .plotsettings import PlotSettings
from .vectorpreset import VectorPreset, register_preset, get_preset, list_presets

__all__ = [
    # Core types
    "Index",
    "AbstractArray",
    "Vector",
    "Matrix",
    # Convenience types
    "CorrelationMatrix",
    "ArrayList",
    "AsymmetricVector",
    "ErrorMatrix",
    "AsymmetricMatrix",
    # Plot configuration
    "PlotSettings",
    "VectorPreset",
    # Preset functions
    "register_preset",
    "get_preset",
    "list_presets",
    # Functions
    'on_device',
    'ops',
    'to_index',
]
