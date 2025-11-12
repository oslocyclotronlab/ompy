from __future__ import annotations

# Store dtype as a *name* to avoid importing NumPy at module import time.
# Users can set to "float32", "float64", etc. Default favours float64 for precision.
_DTYPE_NAME = "float32"

def set_global_dtype(dtype) -> None:
    """
    Accepts either a NumPy dtype object (np.float32) or a string ("float32").
    Does not import NumPy unless you pass a NumPy dtype.
    """
    global _DTYPE_NAME
    if hasattr(dtype, "name"):  # likely a NumPy dtype
        try:
            # avoid importing numpy unless really a numpy dtype
            name = getattr(dtype, "name")
        except Exception:
            raise TypeError("Unsupported dtype object; pass a NumPy dtype or a string name.")
        _DTYPE_NAME = str(name)
    elif isinstance(dtype, str):
        _DTYPE_NAME = dtype
    else:
        raise TypeError("dtype must be a NumPy dtype or a string like 'float32'.")

def get_global_dtype_name() -> str:
    """Return the configured dtype name (string), e.g., 'float32'."""
    return _DTYPE_NAME

def get_global_dtype():
    """
    Return a NumPy dtype object *if NumPy is available*; otherwise raise ImportError.
    Keep call-sites honest: only call this inside code paths that actually use NumPy.
    """
    try:
        import numpy as _np  # local import to avoid hard dependency at import time
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "NumPy is required to materialize the global dtype. "
            "Install with `pip install 'ompy[recommended]'` or `pip install numpy`."
        ) from e
    return getattr(_np, _DTYPE_NAME) if hasattr(_np, _DTYPE_NAME) else _np.dtype(_DTYPE_NAME)
