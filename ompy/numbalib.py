import numpy as np
import warnings

def nop_decorator(func, *aargs, **kkwargs):
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

def nop_nop(*aargs, **kkwargs):
    def decorator(func):
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator

try:
    from numba import jit, njit, int32, float32, float64, prange
    from numba.experimental import jitclass
    from numba.typed import List as NList
    from numba.types import ListType
    from numba.types.npytypes import Array as NumpyArray
    NUMPY = True
except ImportError as e:
    NUMPY = False
    warnings.warn("Numba could not be imported. Falling back to non-jiting which will be much slower")
    int32 = np.int32
    float32 = np.float32
    float64 = np.float64
    prange = range
    NumpyArray = np.ndarray

    njit = nop_nop #nop_decorator
    jit = nop_nop
    jitclass = nop_nop
    NList = list
    ListType = list
