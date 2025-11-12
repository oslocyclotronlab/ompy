from __future__ import annotations

# ---- Version (setuptools-scm writes src/ompy/_version.py at build time) ----
try:  # during development (no build yet), fall back cleanly
    from ._version import version as __version__
except Exception:  # pragma: no cover
    __version__ = "0+unknown"

# ---- Very lightweight utilities kept at top-level ----
from importlib import import_module as _import_module, util as _util


# Optional: expose config helpers without importing NumPy at import-time.
# (These are tiny and only import NumPy inside the functions if needed.)
from . import config as _config  # lightweight module
from .accel import (
    gpu_available as _gpu_available,
    h5py_available as _h5py_available,
    jax_available as _jax_available,
    jax_working as _jax_working,
    numba_available as _numba_available,
    numba_cuda_available as _numba_cuda_available,
    uproot_available as _uproot_available,
    ROOT_imported as _root_imported,
    xarray_available as _xarray_available,
)

from .array import Vector, Matrix, Index
set_global_dtype = _config.set_global_dtype
get_global_dtype = _config.get_global_dtype
get_global_dtype_name = _config.get_global_dtype_name
__all__ = ["set_global_dtype", "get_global_dtype", "get_global_dtype_name"]

# ---- Backwards compatibility flags for optional accelerators ----

def _has(pkg: str) -> bool:
    try:
        return _util.find_spec(pkg) is not None
    except Exception:
        return False


_NUMBA_AVAIL = _numba_available()
_NUMBA_CUDA_AVAIL = _numba_cuda_available()
_JAX_AVAIL = _jax_available()
_JAX_WORKING = _jax_working()
_GPU_AVAIL = _gpu_available()
_H5PY_AVAIL = _h5py_available()
_UPROOT_AVAIL = _uproot_available()
_XARRAY_AVAIL = _xarray_available()

NUMBA_AVAILABLE = _NUMBA_AVAIL
NUMBA_CUDA_AVAILABLE = _NUMBA_CUDA_AVAIL
# Keep mutable container for legacy code expecting to flip flag at runtime.
NUMBA_CUDA_WORKING = [_NUMBA_CUDA_AVAIL]
JAX_AVAILABLE = _JAX_AVAIL
JAX_WORKING = _JAX_WORKING
GPU_AVAILABLE = _GPU_AVAIL
H5PY_AVAILABLE = _H5PY_AVAIL
UPROOT_AVAILABLE = _UPROOT_AVAIL
ROOT_AVAILABLE = _has("ROOT")
ROOT_IMPORTED = _root_imported()
XARRAY_AVAILABLE = _XARRAY_AVAIL
PYMC_AVAILABLE = _has("pymc")
PYRO_AVAILABLE = _has("pyro")
SKLEARN_AVAILABLE = _has("sklearn")
OPTAX_AVAILABLE = _has("optax")

# ---- Lazy access to big subpackages (imported on first attribute access) ----
# Add here any top-level submodules you want importable as `ompy.<name>`
# without importing them eagerly.
_lazy_modules = {
    "status",
    "array",
    "ensemble",
    "nuclear",
    "external",
    "response",
    "detector",
    "unfolding",
    "firstgeneration",
    "decomposition",
    "normalization",
    "examples",
}

def __getattr__(name: str):
    if name in _lazy_modules:
        return _import_module("." + name, __package__)
    raise AttributeError(f"module {__name__} has no attribute {name!r}")

def __dir__():
    return sorted(list(globals().keys()) + list(_lazy_modules))


__all__ += [
    'Vector',
    'Matrix',
    'status',
    'NUMBA_AVAILABLE',
    'NUMBA_CUDA_AVAILABLE',
    'NUMBA_CUDA_WORKING',
    'JAX_AVAILABLE',
    'JAX_WORKING',
    'GPU_AVAILABLE',
    'H5PY_AVAILABLE',
    'UPROOT_AVAILABLE',
    'ROOT_AVAILABLE',
    'ROOT_IMPORTED',
    'XARRAY_AVAILABLE',
    'PYMC_AVAILABLE',
    'PYRO_AVAILABLE',
    'SKLEARN_AVAILABLE',
    'OPTAX_AVAILABLE',
]
