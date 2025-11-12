from ...accel import numba_available

if numba_available():
    from .vector import Vector
