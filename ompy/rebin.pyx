import numpy as np
cimport cython
cimport numpy as np


# We now need to fix a datatype for our arrays. I've used the variable
# DTYPE for this, which is assigned to the usual NumPy runtime
# type info object.
DTYPE = np.float64


cdef double overlap(double edge_in_l, double edge_in_u,
                    double edge_out_l, double edge_out_u):
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
    cdef double overlap
    overlap = max(0,
                  min(edge_out_u, edge_in_u) -
                  max(edge_out_l, edge_in_l)
                  )
    return overlap


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
@cython.cdivision(True)
def rebin_1D(np.ndarray counts, np.ndarray mids_in, np.ndarray mids_out):
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

    cdef int Nin, Nout
    cdef int jmin, jmax  # To select subset in inner loop below
    cdef double Emin_in, dE_in, Emin_out, dE_out

    # Get calibration coefficients and number of elements from array:
    Nin = mids_in.shape[0]
    Emin_in, dE_in = mids_in[0], mids_in[1]-mids_in[0]
    Nout = mids_out.shape[0]
    Emin_out, dE_out = mids_out[0], mids_out[1]-mids_out[0]

    # convert to lower-bin edges
    Emin_in -= dE_in/2
    Emin_out -= dE_out/2

    # Allocate rebinned array to fill:
    counts_out = np.zeros(Nout, dtype=DTYPE)
    cdef double[:] counts_out_view = counts_out
    cdef int i, j
    cdef double Eout_i, Ein_j
    cdef double bins_overlap
    for i in range(Nout):
        # Only loop over the relevant subset of j indices where there may be
        # overlap:
        jmin = max(0, int((Emin_out + dE_out*(i-1) - Emin_in)/dE_in))
        jmax = min(Nin-1, int((Emin_out + dE_out*(i+1) - Emin_in)/dE_in))
        # Calculate the bin edge energies manually for speed:
        Eout_i = Emin_out + dE_out*i
        for j in range(jmin, jmax+1):
            # Calculate proportionality factor based on current overlap:
            Ein_j = Emin_in + dE_in*j
            bins_overlap = overlap(Ein_j, Ein_j+dE_in,
                                   Eout_i, Eout_i+dE_out)
            counts_out_view[i] += counts[j] * bins_overlap / dE_in

    return counts_out


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
@cython.cdivision(True)
def rebin_2D(np.ndarray counts, np.ndarray mids_in,
             np.ndarray mids_out, int axis=0):
    """Rebin a matrix of counts from binning mids_in to binning mids_out

    This is a currently just a wrapper for rebin() to handle the logistics
    of getting a matrix as input.
    Todo: It is unnecessary to calculate the overlap for each bin along
    the axis that is not being rebinned.

    Args:
        counts: (N,M) Array of counts  to rebin
        mids_in: Array of mid-bins energies of input
            matrix along rebin axis
        mids_out: Array of mid-bins energies of output
            matrix along rebin axis
        axis: Axis to rebin

    Returns:
        counts_out: Matrix of rebinned counts
    """

    # Define variables for Cython:
    cdef int other_axis, N_loop, i

    # Axis number of non-rebin axis (Z2 group, fancy!):
    if axis not in (0, 1):
        raise ValueError("Axis must be either 0 or 1, got %i" % axis)

    other_axis = (axis + 1) % 2

    # Number of bins along that axis:
    N_loop = counts.shape[other_axis]

    # Calculate shape of rebinned matrix and allocate it:
    shape = np.array([counts.shape[0], counts.shape[1]],
                     dtype=int)
    shape[axis] = len(mids_out)
    counts_out = np.zeros(shape, dtype=DTYPE)

    if axis == 0:
        # TODO figure out how to best put arrays into counts_out.
        # Use memoryview or no?
        #
        for i in range(N_loop):
            counts_out[:, i] = rebin_1D(counts[:, i], mids_in, mids_out)
    else:
        for i in range(N_loop):
            counts_out[i, :] = rebin_1D(counts[i, :], mids_in, mids_out)
    return counts_out
