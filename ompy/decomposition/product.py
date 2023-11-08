from .. import Matrix, Vector
from ..numbalib import njit, prange
from ..stubs import array1D, array2D
import numpy as np
from typing import Literal, TypeAlias

ExOption: TypeAlias = Literal['auto']


def nld_T_product(nld: Vector, gsf: Vector, Ex_max: ExOption = 'auto',
                  Ex: array1D | None = None,
                  normalize: bool = False):
    """ Compute the product of the NLD and GSF.

    Arguments:
        nld: The NLD.
        gsf: The GSF.
        Ex_max: How to compute the maximum excitation energy.
        normalize: Whether to normalize the rows.
    Returns:
        The product, a trapezoid cutout of the first generation matrix
    """
    P, Ef, Eg, Ex = setup(nld, gsf, Ex_max, Ex)
    # Precompute the index map (Ex, Eg) -> (Ef)
    imap = index_map(Ex, Eg, Ef)
    nld_T_product_(P, nld.values, gsf.values, imap, normalize)
    return Matrix(Ex=Ex, Eg=Eg, values=P)


def setup(nld: Vector, gsf: Vector, Ex_max: ExOption, Ex: array1D | None = None) -> tuple[array2D, array1D, array1D, array1D]:
    """ Set up arrays for the computation.

    The intent is to make it easier to split up the setup from the computation,
    so that the setup can be done once and the computation can be done multiple
    times.

    Arguments:
        nld: The NLD.
        gsf: The GSF.
        Ex_max: How to compute the maximum excitation energy.
    Returns:
        P: The output array.
        Ef: The final energy.
        Eg: The gamma ray energy.
        Ex: The excitation energy.
    """
    Ef = nld.X
    Eg = gsf.X
    if Ex is None:
        dEf = Ef[1] - Ef[0]
        dEg = Eg[1] - Eg[0]
        dEx = min(dEf, dEg)
        Ex_min = Ef[0] + Eg[0]
        if Ex_max == 'auto':
            #Ex_max = Ef[-1] + Eg[-1]
            Ex_max = Eg[-1]
        Ex = np.arange(Ex_min, Ex_max, dEx, dtype=Ef.dtype)
    P = np.zeros((len(Ex), len(Eg)), dtype=Ef.dtype)
    return P, Ef, Eg, Ex


@njit(parallel=True)
def nld_T_product_(P: array2D, nld: array1D, gsf: array1D, map: array2D,
                   normalize: bool = False) -> None:
    """ Compute the product of the NLD and GSF inplace.

    Gets a 10x speedup from numba.

    Arguments:
        P: The output array.
        nld: The NLD.
        gsf: The GSF.
        map: The index map mapping (Ex, Eg) -> (Ef).
        normalize: Whether to normalize the rows.
    Returns:
        None
    """
    for i in prange(P.shape[0]):
        s = 0.0
        for j in range(P.shape[1]):
            k = map[i, j]
            if k < 0:
                continue
            p = nld[k]*gsf[j]
            P[i, j] = p
            s += p

        if normalize and abs(s) > 1e-10:
            P[i, :] /= s


def index_map(Ex: array1D, Eg: array1D, Ef: array1D) -> array2D:
    """ Compute the index map (Ex, Eg) -> (Ef).
    
    Arguments:
        Ex: The excitation energy.
        Eg: The gamma ray energy.
        Ef: The final energy.
    Returns:
        The index map.
    """
    map = np.empty((len(Ex), len(Eg)), dtype=int)
    map[:] = -1
    for i in prange(len(Ex)):
        for j in range(len(Eg)):
            ef = Ex[i] - Eg[j]
            k = index(Ef, ef)
            map[i, j] = k
    return map
            

@njit
def index(X, x):
    low = 0
    high = len(X) - 1

    while low <= high:
        mid = (low + high) // 2
        if X[mid] <= x:
            if mid == len(X) - 1 or x < X[mid + 1]:  # check boundaries
                return mid
            low = mid + 1
        else:
            high = mid - 1

    return -1  # if we reach here, the element was not found 
