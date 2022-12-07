cimport cython

ctypedef fused number:
    cython.short
    cython.int
    cython.long
    cython.float
    cython.double

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.embedsignature(True)
def index(number[:] array, number element):
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
def index_2(number[:] array, number element):
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
        double d1
        double d2
        int N = len(array)
    if array[0] >= element:
        return 0
    if array[N-1] < element:
        return len(array)

    for i in range(1, N):
        if array[i] >= element:
            break
    d1 = (array[i-1] - element)**2
    d2 = (array[i] - element)**2
    if d1 < d2:
        return i - 1
    return i

