import numpy as np
import warnings
from collections import OrderedDict
from .. import get_global_dtype

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

    njit = nop_nop #nop_decorator
    jit = nop_nop
    jitclass = nop_nop
    NList = list
    ListType = list

LOCAL_DTYPE = get_global_dtype()
match LOCAL_DTYPE:
    case np.float32:
        LOCAL_DTYPE = float32
    case np.float64:
        LOCAL_DTYPE = float64
    case _:
        raise ValueError(f"Unsupported dtype {LOCAL_DTYPE}")
def set_local_dtype(dtype):
    global LOCAL_DTYPE
    LOCAL_DTYPE = dtype

def get_local_dtype():
    return LOCAL_DTYPE

@njit
def get_local_dtype_njit():
    return LOCAL_DTYPE


@njit
def index(E: np.ndarray, e: LOCAL_DTYPE) -> int:
    if e < E[0]:
        raise IndexError("Energy below bounds.")
    i = 0
    while i < len(E):
        if E[i] > e:
            return i - 1
        i += 1
    #if e > E[-1] + (E[-1] - E[-2]):
    #    raise IndexError("Energy above bounds.")
    return i - 1


# @njit
# def index(E: np.ndarray, e: LOCAL_DTYPE) -> int:
#     if e < E[0]:
#         return 0
#         #raise IndexError("Energy below bounds.")
#     i = 0
#     while i < len(E):
#         if E[i] > e:
#             return i
#         i += 1
#
#     return i-1

@njit
def index_mid(E: np.ndarray, e: LOCAL_DTYPE) -> int:
    """ Index for mid-binning """
    i = 0
    while i < len(E):
        if E[i] > e:
            a = abs(E[i-1] - e)
            b = abs(E[i] - e)
            if a < b:
                return i - 1
            else:
                if i == len(E) - 1:
                    return i
                else:
                    c = abs(E[i+1] - e)
                    if b < c:
                        return i
                    else:
                        return i + 1
        i += 1
    return i - 1


@njit
def rebin_1D(counts, mids_in, mids_out):
    """Rebin an array of counts from binning mids_in to binning mids_out

    Assumes equidistant binning.

    Args:
        counts: Array of counts to be rebinned
        mids_in: Array of mid-bins energies giving
             the calibration of counts_in
        mids_out: Array of mid-bins energies of the
              counts array after rebin
    Returns:
        counts_out: Array of rebinned counts with calibration
             given by mids_out
    """

    # Get calibration coefficients and number of elements from array:
    Nin = mids_in.shape[0]
    Emin_in, dE_in = mids_in[0], mids_in[1] - mids_in[0]
    Nout = mids_out.shape[0]
    Emin_out, dE_out = mids_out[0], mids_out[1] - mids_out[0]

    # convert to lower-bin edges
    Emin_in -= dE_in / 2
    Emin_out -= dE_out / 2

    # Allocate rebinned array to fill:
    counts_out = np.zeros(Nout, dtype=DTYPE)
    counts_out_view = counts_out
    for i in range(Nout):
        # Only loop over the relevant subset of j indices where there may be
        # overlap:
        jmin = max(0, int((Emin_out + dE_out * (i - 1) - Emin_in) / dE_in))
        jmax = min(Nin - 1, int((Emin_out + dE_out * (i + 1) - Emin_in) / dE_in))
        # Calculate the bin edge energies manually for speed:
        Eout_i = Emin_out + dE_out * i
        for j in range(jmin, jmax + 1):
            # Calculate proportionality factor based on current overlap:
            Ein_j = Emin_in + dE_in * j
            bins_overlap = overlap(Ein_j, Ein_j + dE_in,
                                   Eout_i, Eout_i + dE_out)
            counts_out_view[i] += counts[j] * bins_overlap / dE_in

    return counts_out


@njit
def overlap(edge_in_l, edge_in_u,
            edge_out_l, edge_out_u):
    """ Calculate overlap between energy intervals

       1
    |_____|_____|_____| Binning A
    |___|___|___|___|__ Binning B
      2   3
    Overlap of bins A1 and B2 is 3_
    Overlap of bins A1 and B3 is 1.5_

    Args:
        edge_in_l: Lower edge of input interval
        edge_in_u: Upper edge of input interval
        edge_out_l: Lower edge of output interval
        edge_out_u: Upper edge of output interval
    Returns:
        overlap of the two bins
    """
    overlap = max(0,
                  min(edge_out_u, edge_in_u) -
                  max(edge_out_l, edge_in_l)
                  )
    return overlap


@njit
def div0_1(a, b):
    """ division function designed to ignore / 0, i.e. div0([-1, 0, 1], 0 ) -> [0, 0, 0] """
    # with np.errstate(divide='ignore', invalid='ignore'):
    # c = np.true_divide(a, b)
    # c[~ np.isfinite(c)] = 0.0  # -inf inf NaN
    c = a / b
    for i in range(len(c)):
        if not np.isfinite(c[i]):
            c[i] = 0.0
    return c


@njit
def div0_2(a, b):
    """ division function designed to ignore / 0, i.e. div0([-1, 0, 1], 0 ) -> [0, 0, 0] """
    # with np.errstate(divide='ignore', invalid='ignore'):
    # c = np.true_divide(a, b)
    # c[~ np.isfinite(c)] = 0.0  # -inf inf NaN
    c = a / b
    if not np.isfinite(c):
        return 0.0
    return c


@njit
def normalize(R):
    for j in range(R.shape[0]):
        R[j, :] = div0_1(R[j, :], np.sum(R[j, :]))


spec = OrderedDict()
if NUMPY:
    spec['_index'] = LOCAL_DTYPE[::1]
    spec['values'] = LOCAL_DTYPE[::1]

@jitclass(spec)
class NVector:
    """ A minimalistic vector class that can be used in numba """
    def __init__(self, index: np.ndarray, values: np.ndarray):
        self._index = index
        self.values = values

    def __getitem__(self, i):
        return self.values[i]

    def __setitem__(self, key, value):
        self.values[key] = value

    def index(self, e: float) -> int:
        return index(self._index, e)

    def at(self, e: float) -> float:
        """ Get value at energy e. Note that indices out of bound are silently accepted """
        return self.values[self.index(e)]

    @property
    def E(self) -> np.ndarray:
        return self._index

    def __len__(self) -> int:
        return len(self._index)

    @property
    def dtype(self):
        return self.values.dtype


@njit
def empty_nvector(E: np.ndarray, dtype=LOCAL_DTYPE):
    return NVector(E, np.empty(len(E), dtype=dtype))


@njit
def lerp(x, x0, x1, y0, y1):
    t = (x - x0) / (x1 - x0)
    return (1 - t) * y0 + t * y1
