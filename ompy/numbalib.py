import numpy as np
import warnings
from . import NUMBA_AVAILABLE, NUMBA_CUDA_AVAILABLE, NUMBA_CUDA_WORKING
import logging

LOG = logging.getLogger(__name__)

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
    from numba import jit, njit, int32, float32, float64, prange, objmode
    from numba.experimental import jitclass
    from numba.typed import List as NList
    from numba.types import ListType
    from numba.types.npytypes import Array as NumpyArray
    NUMPY = True
except ImportError as e:
    LOG.debug(f"Could not import numba: {e}")
    NUMPY = False
    #warnings.warn("Numba could not be imported. Falling back to non-jiting which will be much slower")
    int32 = np.int32
    float32 = np.float32
    float64 = np.float64
    prange = range
    NumpyArray = np.ndarray
    objmode = None

    njit = nop_nop #nop_decorator
    jit = nop_nop
    jitclass = nop_nop
    NList = list
    ListType = list

if NUMBA_CUDA_AVAILABLE:
    from numba import cuda
    from numba.cuda.cudadrv.driver import CudaAPIError, CudaSupportError, LinkerError
    from numba.core.errors import NumbaPerformanceWarning
    HAIL_MARY = False

    if HAIL_MARY:
        LOG.warning("Hail Mary: forcing cuda to work")
        NUMBA_CUDA_WORKING[0] = True
    else:
        LOG.info("Testing if cuda is working...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=NumbaPerformanceWarning)
            try:
                # Test if cuda is working
                @cuda.jit
                def cuda_test(x, y):
                    i = cuda.grid(1)
                    y[i] = x[i] + 1
                x = np.arange(10)
                y = np.zeros_like(x)
                cuda_test[1, 10](x, y)
                NUMBA_CUDA_WORKING[0] = True
            except (CudaAPIError, CudaSupportError, LinkerError) as e:
                warnings.warn("Numba CUDA is available but not working.\n"
                    f"{e}")
                NUMBA_CUDA_WORKING[0] = False

