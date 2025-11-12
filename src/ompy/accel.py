from __future__ import annotations
from importlib import util as _util
from types import SimpleNamespace
import numpy as _np

def _has(pkg: str) -> bool:
    try:
        return _util.find_spec(pkg) is not None
    except Exception:
        return False

# -------- Availability (no heavy imports at import-time) --------

def numba_available() -> bool:
    return _has("numba")

def numba_cuda_available(verbose: bool = False):
    """
    Return True if numba_array is importable AND CUDA runtime looks available.
    If verbose=True, return (bool, message).
    """
    ok, msg = False, ""
    if not numba_available():
        msg = "Numba not installed"
        return (ok, msg) if verbose else ok
    try:
        from numba import cuda  # local import
        # cuda.is_available() checks driver/runtime presence reasonably well
        ok = bool(cuda.is_available())
        msg = "CUDA available" if ok else "CUDA not available"
        @cuda.jit(device=True)
        def cuda_test(x, y):
            i = cuda.grid(1)
            y[i] = x[i] + 1
        import numpy as np
        x = np.arange(10)
        y = np.zeros_like(x)
        cuda_test[1, 10](x, y)
    except Exception as e:  # driver issues, permissions, etc.
        ok = False
        msg = f"CUDA probe failed: {e.__class__.__name__}: {e}"
    return (ok, msg) if verbose else ok

def jax_available() -> bool:
    return _has("jax")

def jax_devices(verbose: bool = False):
    """
    Try to import jax and list devices.
    Returns a list (possibly empty). If verbose=True, returns (list, message).
    """
    if not jax_available():
        msg = "JAX not installed"
        return ([], msg) if verbose else []
    try:
        import jax  # local import
        devs = list(jax.devices())
        msg = f"{len(devs)} device(s) detected"
        return (devs, msg) if verbose else devs
    except Exception as e:
        msg = f"JAX devices() failed: {e.__class__.__name__}: {e}"
        return ([], msg) if verbose else []

def jax_working(verbose: bool = False, require_accel: bool = True):
    """
    Heuristic: JAX considered 'working' if it imports AND devices() succeeds.
    If require_accel=True, at least one accelerator (GPU/TPU) must be present.
    Otherwise, any backend (including CPU) counts as working.

    Returns bool, or (bool, message) if verbose=True.
    """
    if not jax_available():
        msg = "JAX not installed"
        return (False, msg) if verbose else False
    devs, dmsg = jax_devices(verbose=True)
    if not devs:
        msg = f"No JAX devices visible ({dmsg})"
        return (False, msg) if verbose else False

    # Classify devices; jax reports platform in .platform or .device_kind
    accel = []
    for d in devs:
        plat = getattr(d, "platform", "") or getattr(d, "device_kind", "")
        plat = str(plat).lower()
        if any(k in plat for k in ("gpu", "cuda", "tpu", "rocm")):
            accel.append(d)

    if require_accel:
        ok = len(accel) > 0
        msg = "Accelerator present" if ok else "Only CPU devices visible"
    else:
        ok = True
        msg = "JAX devices OK"

    return (ok, msg) if verbose else ok

def gpu_available(verbose: bool = False):
    """
    Composite check: True if either Numba-CUDA or JAX has an accelerator.
    """
    n_ok, n_msg = numba_cuda_available(verbose=True)
    j_ok, j_msg = jax_working(verbose=True, require_accel=True)
    ok = bool(n_ok or j_ok)
    if verbose:
        return ok, {"numba_cuda": n_msg, "jax": j_msg}
    return ok


# -------- Custom exceptions for optional accelerators --------

class JAXNotInstalledError(ImportError):
    """Raised when JAX-dependent functionality is requested but JAX is not installed."""

    def __init__(self, details: str | None = None):
        msg = "JAX is required but not installed. Install the `jax` package to continue."
        if details:
            msg = f"{msg} {details}"
        super().__init__(msg)


class JAXNotWorkingError(RuntimeError):
    """Raised when JAX is installed but not usable (e.g., missing accelerator device)."""

    def __init__(self, details: str | None = None):
        msg = (
            "JAX is installed but not currently usable for this operation "
            "(no compatible devices detected or initialization failed)."
        )
        if details:
            msg = f"{msg} {details}"
        super().__init__(msg)

def xarray_available() -> bool:
    return _has("xarray")

def h5py_available() -> bool:
    return _has("h5py")

def uproot_available() -> bool:
    return _has("uproot")

def ROOT_imported() -> bool:
    # Not fixed yet
    return False

def root_available() -> bool:
    return _has("ROOT")

def pymc_available() -> bool:
    return _has("pymc")

def pyro_available() -> bool:
    return _has("pyro")

def sklearn_available() -> bool:
    return _has("sklearn")

def optax_available() -> bool:
    return _has("optax")

# -------- Safe shims (only import numba_array when asked) --------

def _noop_decorator(*args, **kwargs):
    """
    Return a decorator that leaves the target function unchanged.

    Works for both `@decorator` and `@decorator(...)` usage patterns.
    """
    if args and callable(args[0]) and len(args) == 1 and not kwargs:
        return args[0]

    def decorator(fn):
        return fn

    return decorator

noop_decorator = _noop_decorator


def _python_fori_loop(lower, upper, body_fun, state):
    result = state
    for i in range(lower, upper):
        result = body_fun(i, result)
    return result


def _make_jax_shim():
    shim = SimpleNamespace(
        jit=noop_decorator,
        lax=SimpleNamespace(fori_loop=_python_fori_loop),
    )
    return shim, _np


def configure_jax_dtype(enable_x64: bool = False, dtype_bits: str | None = None):
    """
    Configure the default dtype for JAX operations.

    Args:
        enable_x64: If True, enable 64-bit (float64/int64) precision.
                   If False (default), use 32-bit (float32/int32) precision.
        dtype_bits: Optional string to directly set jax_default_dtype_bits.
                   Valid values: '32' or '64'. Overrides enable_x64 if provided.

    Raises:
        JAXNotInstalledError: If JAX is not installed.
        
    Examples:
        >>> configure_jax_dtype()  # Use 32-bit precision (default)
        >>> configure_jax_dtype(enable_x64=True)  # Use 64-bit precision
        >>> configure_jax_dtype(dtype_bits='64')  # Explicitly set to 64-bit
    """
    if not jax_available():
        raise JAXNotInstalledError("Cannot configure JAX dtype: JAX not installed")
    
    try:
        import jax  # type: ignore
        
        if dtype_bits is not None:
            if dtype_bits not in ('32', '64'):
                raise ValueError(f"dtype_bits must be '32' or '64', got {dtype_bits}")
            jax.config.update("jax_default_dtype_bits", dtype_bits)
        else:
            jax.config.update("jax_enable_x64", enable_x64)
            
    except Exception as e:
        raise RuntimeError(f"Failed to configure JAX dtype: {e}") from e


def get_jax_dtype_config():
    """
    Get the current JAX dtype configuration.

    Returns:
        dict: Dictionary with keys 'x64_enabled' (bool) and 'dtype_bits' (str or None).
              Returns None if JAX is not installed.
              
    Examples:
        >>> config = get_jax_dtype_config()
        >>> print(config['x64_enabled'])  # True or False
        >>> print(config['dtype_bits'])   # '32', '64', or None
    """
    if not jax_available():
        return None
    
    try:
        import jax  # type: ignore
        
        config = {
            'x64_enabled': jax.config.jax_enable_x64,
        }
        
        # Try to get dtype_bits if available
        try:
            config['dtype_bits'] = jax.config.jax_default_dtype_bits
        except AttributeError:
            config['dtype_bits'] = '64' if config['x64_enabled'] else '32'
            
        return config
        
    except Exception:
        return None


def get_jax(
    require_working: bool = True,
    allow_fallback: bool = False,
    prefer_accel: bool = True,
):
    """
    Attempt to import JAX and return (jax_module, jax_numpy, working, available).

    Args:
        require_working: If True, insist on `jax_working(require_accel=True)` being True.
        allow_fallback: If True, return a lightweight shim (jit no-op, lax.fori_loop python loop)
            when JAX is unavailable or not considered working.
        prefer_accel: If False and allow_fallback=True, return the shim even if JAX is working.

    Returns:
        Tuple[jax_like, jax_numpy_like, bool working, bool available]
        - working indicates we intend to use the real JAX backend.
        - available indicates whether the JAX package itself is importable.
    """

    available = jax_available()
    if not available:
        if allow_fallback:
            shim, shim_jnp = _make_jax_shim()
            return shim, shim_jnp, False, False
        return None, None, False, False

    try:
        import jax as _jax  # type: ignore
        import jax.numpy as _jnp  # type: ignore
    except Exception:
        if allow_fallback:
            shim, shim_jnp = _make_jax_shim()
            return shim, shim_jnp, False, False
        return None, None, False, False

    try:
        working = bool(jax_working(require_accel=require_working))
    except Exception:
        working = False

    if working and prefer_accel:
        return _jax, _jnp, True, True

    if (not working or not prefer_accel) and allow_fallback:
        shim, shim_jnp = _make_jax_shim()
        return shim, shim_jnp, False, True

    return _jax, _jnp, working, True


def jit(*args, **kwargs):
    """Return numba.jit if available; otherwise a no-op decorator."""
    if numba_available():
        from numba import jit as _jit  # local import
        return _jit(*args, **kwargs)
    return noop_decorator(*args, **kwargs)

def njit(*args, **kwargs):
    if numba_available():
        from numba import njit as _njit  # local import
        return _njit(*args, **kwargs)
    return noop_decorator(*args, **kwargs)

prange = range
if numba_available():
    from numba import prange as _prange  # local import
    prange = _prange

# Dtype aliases: prefer numba_array types if present; fall back to Python types.
try:
    if numba_available():
        from numba import int32, float32, float64  # type: ignore
    else:  # fallbacks
        int32 = int
        float32 = float
        float64 = float
except Exception:
    int32 = int
    float32 = float
    float64 = float
