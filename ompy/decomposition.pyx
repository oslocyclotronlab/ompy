cimport cython
cimport numpy as np
import numpy as np

IF OPENMP:
    from cython.parallel import prange
ELSE:
    pass


ctypedef np.float64_t DTYPE_t
DTYPE=np.float64

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.embedsignature(True)
def nld_T_product(double[::1] nld, double[::1] T, double[::1] resolution,
                  double[::1] E_nld, double[::1] Eg, double[::1] Ex):
    """ Computes first generation matrix from nld and gSF

    Uses the equation
           P(Ex, Eγ) ∝ ρ(Ex-Eγ)·T(Eγ)

    The energy resolution is exploited to only perform the product up to
    the diagonal, and employs multithreading along each
    row.

    Args:
        nld: Nuclear level density
        T: Transmission coefficients
        resolution: The resolution at each bin of Eg
        E_nld: The energy array for the nuclear level density
        Eg: The gamma energy
        Ex: The excitation energy

    Returns:
        The first generation matrix
    TODO: Is the indexing optimal?
    """
    cdef:
        Py_ssize_t num_Ex = len(Ex)
        Py_ssize_t num_Eg = len(Eg)
        Py_ssize_t i_Ex
        int i_Eg, i_E_nld
        double Eg_max, E_f
    firstgen = np.zeros((num_Ex, num_Eg), dtype=DTYPE)
    cdef double[:, ::1] firstgen_view = firstgen

    # Remember to change both loops simultaneously
    IF OPENMP:
        for i_Ex in prange(num_Ex, nogil=True, schedule='static'):
            Eg_max = Ex[i_Ex] + resolution[i_Ex]
            i_Eg = 0
            while i_Eg < num_Eg and Eg[i_Eg] <= Eg_max:
                E_f = Ex[i_Ex] - Eg[i_Eg]
                i_E_nld = _index(E_nld, E_f)
                firstgen_view[i_Ex, i_Eg] = nld[i_E_nld] * T[i_Eg]
                i_Eg = i_Eg + 1
    ELSE:
        for i_Ex in range(num_Ex):
            Eg_max = Ex[i_Ex] + resolution[i_Ex]
            i_Eg = 0
            while i_Eg < num_Eg and Eg[i_Eg] <= Eg_max:
                E_f = Ex[i_Ex] - Eg[i_Eg]
                i_E_nld = _index(E_nld, E_f)
                firstgen_view[i_Ex, i_Eg] = nld[i_E_nld] * T[i_Eg]
                i_Eg = i_Eg + 1

    # We want probabilities
    normalize(firstgen)
    return firstgen

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.embedsignature(True)
cdef int _index(double[:] array, double element) nogil:
    """ Finds the index of the closest element in the array

    Unsafe.
    9 times faster than np.abs(array - element).argmin()

    Args:
        array: The array to index
        element: The element to find
    Returns:
        The index (int) to the closest element in the array.
    """
    cdef:
        int i = 0
        double distance
        double prev_distance = (array[0] - element)**2

    for i in range(1, len(array)):
        distance = (array[i] - element)**2
        if distance > prev_distance:
            return i - 1
        else:
            prev_distance = distance
    return i

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.embedsignature(True)
def index(double[:] array, double element):
    """ Finds the index of the closest element in the array

    Unsafe.
    9 times faster than np.abs(array - element).argmin()

    Args:
        array: The array to index
        element: The element to find
    Returns:
        The index (int) to the closest element in the array.
    """
    cdef:
        int i = 0
        double distance
        double prev_distance = (array[0] - element)**2

    for i in range(1, len(array)):
        distance = (array[i] - element)**2
        if distance > prev_distance:
            return i - 1
        else:
            prev_distance = distance
    return i

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.embedsignature(True)
def chisquare_diagonal_with_zero(double[:, ::1] fact, double[:, ::1] fit,
              double[:, ::1] std, double[::1] resolution,
              double[::1] Eg, double[::1] Ex):
    """ Computes χ² of two matrices

    Exploits the diagonal resolution to do less computation

    Args:
        fact: The true matrix
        fit: The candidate matrix
        std: The standard deviation of the fit/fact
        resolution: Array describing the resolution at
            each element of the gamma energy Eg
        Eg: The gamma energy
        Ex: The excitation energy
    Returns:
        The value of χ²
    """

    cdef:
        double chi = 0.0
        double Eg_max
        Py_ssize_t num_Eg = len(Eg)
        Py_ssize_t num_Ex = len(Ex)
        int i, j

    for i in range(num_Ex):# prange(num_Ex, nogil=True, schedule='static'):
        Eg_max = Ex[i] + resolution[i]
        for j in range(num_Eg):
            if Eg[j] > Eg_max:
                break
            if std[i, j] == 0:
                chi = chi + (fact[i, j] - fit[i, j])**2
            else:
                chi = chi + (fact[i, j] - fit[i, j])**2/std[i, j]**2
    return chi


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.embedsignature(True)
def chisquare_diagonal(double[:, ::1] fact, double[:, ::1] fit,
              double[:, ::1] std, double[::1] resolution,
              double[::1] Eg, double[::1] Ex):
    """ Computes χ² of two matrices

    Exploits the diagonal resolution to do less computation

    Args:
        fact: The true matrix
        fit: The candidate matrix
        std: The standard deviation of the fit/fact
        resolution: Array describing the resolution at
            each element of the gamma energy Eg
        Eg: The gamma energy
        Ex: The excitation energy
    Returns:
        The value of χ²
    """

    cdef:
        double chi = 0.0
        double Eg_max
        Py_ssize_t num_Eg = len(Eg)
        Py_ssize_t num_Ex = len(Ex)
        int i, j

    for i in range(num_Ex):# prange(num_Ex, nogil=True, schedule='static'):
        Eg_max = Ex[i] + resolution[i]
        for j in range(num_Eg):
            if Eg[j] > Eg_max:
                break
            if std[i, j] == 0:
                continue
            chi = chi + (fact[i, j] - fit[i, j])**2/std[i, j]**2
    return chi

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.embedsignature(True)
def chisquare(double[:, ::1] fact, double[:, ::1] fit,
              double[:, ::1] std):
    """ Computes χ² of two matrices

    Exploits the diagonal resolution to do less computation

    Args:
        fact: The true matrix
        fit: The candidate matrix
        std: The standard deviation of the fit/fact
    Returns:
        The value of χ²
    """
    cdef:
        double chi = 0.0
        Py_ssize_t num_Eg = fact.shape[1]
        Py_ssize_t num_Ex = fact.shape[0]
        int i, j

    for i in range(num_Ex):#prange(num_Ex, nogil=True, schedule='static'):
        for j in range(num_Eg):
            if std[i, j] == 0:
                continue
            chi = chi + (fact[i, j] - fit[i, j])**2/std[i, j]**2
    return chi

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.embedsignature(True)
def normalize(np.ndarray[DTYPE_t, ndim=2] matrix):
    """ Row-normalizes the matrix

    Args:
        matrix: The matrix to normalize
    """
    matrix /= matrix.sum(axis=1)[:, np.newaxis]
