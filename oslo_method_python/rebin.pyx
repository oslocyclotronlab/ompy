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
    overlap = max(0,
                  min(Eout_h, Ein_h)
                  - max(Eout_l, Ein_l)
                  )
    return overlap


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
@cython.cdivision(True)
def rebin_cython(double[:] counts_in, double[:] E_array_in,
                 double[:] E_array_out):
    """Rebin an array of counts from binning E_array_in to binning E_array_out
    """

    # assert counts_in.dtype == DTYPE
    # assert E_array_in.dtype == DTYPE
    # assert E_array_out.dtype == DTYPE

    cdef int Nin, Nout
    cdef int jmin, jmax  # To select subset in inner loop below
    cdef double a0_in, a1_in, a0_out, a1_out

    # Get calibration coefficients and number of elements from array:
    Nin = E_array_in.shape[0]
    a0_in, a1_in = E_array_in[0], E_array_in[1]-E_array_in[0]
    Nout = E_array_out.shape[0]
    a0_out, a1_out = E_array_out[0], E_array_out[1]-E_array_out[0]

    a1_in = E_array_in[1]-E_array_in[0]

    # For the overlap calculation, it is convenient to have energy arrays with
    # one extra bin (the upper edge of the last bin):
    # E_array_in_ext = np.append(E_array_in, E_array_in[Nin]+a1_in)
    # E_array_out_ext = np.append(E_array_out,
                                # E_array_out[Nout]
                                # + (E_array_out[1]-E_array_out[0]))

    # Allocate rebinned array to fill:
    counts_out = np.zeros(Nout, dtype=DTYPE)
    cdef double[:] counts_out_view = counts_out
    cdef int i, j
    cdef double Eout_i, Ein_j
    for i in range(Nout):
        # for j in range(Nin):
        # Can we make it faster by looping j over a subset? What is a
        # sufficient subset? Yes!
        jmin = max(0, <int>((a0_out + a1_out*(i-1) - a0_in)/a1_in))
        jmax = min(Nin-1, <int>((a0_out + a1_out*(i+1) - a0_in)/a1_in))
        # Calculate the bin edge energies manually for speed:
        Eout_i = a0_in + a1_in*i
        for j in range(jmin, jmax+1):
            # Calculate proportionality factor based on current overlap:
            Ein_j = a0_out + a1_out*j
            overlap = calc_overlap(Ein_j, Ein_j+a1_in,
                                   Eout_i, Eout_i+a1_out)
            # overlap = max(0, (
            #                     min(E_array_out_ext[i+1], E_array_in_ext[j+1])
            #                     - max(E_array_out_ext[i], E_array_in_ext[j]))
            #               )
            counts_out_view[i] += counts_in[j] * overlap / a1_in

    return counts_out


# Look at this for inspiration, delete after:
def rebin_python(counts_in, Ein_array, Eout_array):
    """
    Rebins, i.e. redistributes the numbers from vector counts
    (which is assumed to have calibration given by Ein_array) 
    into a new vector, which is returned. 
    The total number of counts is preserved.
    In case of upsampling (smaller binsize out than in), 
    the distribution of counts between bins is done by simple 
    proportionality.
    Inputs:
    counts: Numpy array of counts in each bin
    Ein_array: Numpy array of energies, assumed to be linearly spaced, 
               corresponding to the middle-bin energies of counts
    Eout_array: Numpy array of energies corresponding to the desired
                rebinning of the output vector, also middle-bin
    """

    # Get calibration coefficients and number of elements from array:
    Nin = len(Ein_array)
    a0_in, a1_in = Ein_array[0], Ein_array[1]-Ein_array[0]
    Nout = len(Eout_array)
    a0_out, a1_out = Eout_array[0], Eout_array[1]-Eout_array[0]

    # Replace the arrays by bin-edge energy arrays of length N+1 
    # (so that all bins are covered on both sides).
    Ein_array = np.linspace(a0_in - a1_in/2, a0_in - a1_in/2 + a1_in*Nin, Nin+1)
    Eout_array = np.linspace(a0_out - a1_out/2, a0_out - a1_out/2 + a1_out*Nout, Nout+1)




    # Allocate vector to fill with rebinned counts
    counts_out = np.zeros(Nout)
    # Loop over all indices in both arrays. Maybe this can be speeded up?
    for i in range(Nout):
        # for j in range(Nin):
        # Can we make it faster by looping j over a subset? What is a
        # sufficient subset?
        jmin = max(0, int((a0_out + a1_out*(i-1) - a0_in)/a1_in))
        jmax = min(Nin-1, int((a0_out + a1_out*(i+1) - a0_in)/a1_in))
        for j in range(jmin, jmax+1):
            # Calculate proportionality factor based on current overlap:
            overlap = np.minimum(Eout_array[i+1], Ein_array[j+1]) - np.maximum(Eout_array[i], Ein_array[j])
            overlap = overlap if overlap > 0 else 0
            counts_out[i] += counts_in[j] * overlap / a1_in

    return counts_out