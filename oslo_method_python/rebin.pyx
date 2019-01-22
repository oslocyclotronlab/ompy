# Function to rebin spectra
# Implemented using Cython, with inspiration from here:
# https://cython.readthedocs.io/en/latest/src/userguide/numpy_tutorial.html
import numpy as np
cimport cython


# We now need to fix a datatype for our arrays. I've used the variable
# DTYPE for this, which is assigned to the usual NumPy runtime
# type info object.
DTYPE = np.float64


cdef double calc_overlap(double Ein_l, double Ein_h,
                         double Eout_l, double Eout_h):
    """Calculate overlap between energy intervals

    It is made for use in a rebin function, hence the names "in" and "out"
    for the energy intervals.
    It is implemented as a pure C function to be as fast as possible.

    Args:
        Ein_l (double): Lower edge of input interval
        Ein_h (double): Upper edge of input interval
        Eout_l (double): Lower edge of output interval
        Eout_h (double): Upper edge of output interval
    Returns:
        overlap
    """
    cdef double overlap
    overlap = max(0,
                  min(Eout_h, Ein_h)
                  - max(Eout_l, Ein_l)
                  )
    return overlap


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
@cython.cdivision(True)
def rebin(double[:] counts_in, double[:] E_array_in,
                 double[:] E_array_out):
    """Rebin an array of counts from binning E_array_in to binning E_array_out
    """

    cdef int Nin, Nout
    cdef int jmin, jmax  # To select subset in inner loop below
    cdef double a0_in, a1_in, a0_out, a1_out

    # Get calibration coefficients and number of elements from array:
    Nin = E_array_in.shape[0]
    a0_in, a1_in = E_array_in[0], E_array_in[1]-E_array_in[0]
    Nout = E_array_out.shape[0]
    a0_out, a1_out = E_array_out[0], E_array_out[1]-E_array_out[0]

    # Allocate rebinned array to fill:
    counts_out = np.zeros(Nout, dtype=DTYPE)
    cdef double[:] counts_out_view = counts_out
    cdef int i, j
    cdef double Eout_i, Ein_j
    for i in range(Nout):
        # Only loop over the relevant subset of j indices where there may be 
        # overlap:
        jmin = max(0, int((a0_out + a1_out*(i-1) - a0_in)/a1_in))
        jmax = min(Nin-1, int((a0_out + a1_out*(i+1) - a0_in)/a1_in))
        # Calculate the bin edge energies manually for speed:
        Eout_i = a0_out + a1_out*i
        for j in range(jmin, jmax+1):
            # Calculate proportionality factor based on current overlap:
            Ein_j = a0_in + a1_in*j
            overlap = calc_overlap(Ein_j, Ein_j+a1_in,
                                   Eout_i, Eout_i+a1_out)
            counts_out_view[i] += counts_in[j] * overlap / a1_in

    return counts_out
