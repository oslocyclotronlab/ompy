from numba.cuda.cudadrv.driver import CudaAPIError, CudaSupportError
from .numbalib import *
from .response_old import *
from .calibrator import *
from .io import save, load
from .responsedata import *
from .interpolation import *
from .gf3 import GF3Interpolator, GF3Interpolation
from .interpolations import *
from .discreteinterpolation import *
from .compton import *
from .. import NUMBA_CUDA_AVAILABLE, NUMBA_CUDA_WORKING
from warnings import warn

if NUMBA_CUDA_AVAILABLE:
    if False:
        from .comptongpu import *
        NUMBA_CUDA_WORKING[0] = True
    else:
        try:
            from .comptongpu import *
            NUMBA_CUDA_WORKING[0] = True
        except (CudaAPIError, CudaSupportError) as e:
            warn("Numba CUDA is available but not working.\n"
                f"{e}")
            NUMBA_CUDA_WORKING[0] = False
            #raise e


from .response import Response as Response
from .perturb import *
