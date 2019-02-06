"""
Function to rebin spectra.
Implemented using Cython, with inspiration from here:
https://cython.readthedocs.io/en/latest/src/userguide/numpy_tutorial.html
Compile by running
   python setup.py build_ext --inplace
in the top directory of the git repo

---

This file is part of oslo_method_python, a python implementation of the
Oslo method.

Copyright (C) 2018 Jørgen Eriksson Midtbø
Oslo Cyclotron Laboratory
jorgenem [0] gmail.com

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""

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

    Args:
        counts_in (np.ndarray): Array of counts to be rebinned
        E_array_in (np.ndarray): Array of lower-bin-edge energies giving
                                 the calibration of counts_in
        E_array_out (np.ndarray): Array of lower-bin-edge energies of the
                                  counts array after rebin
    Returns:
        counts_out (np.ndarray): Array of rebinned counts with calibration
                                 given by E_array_out
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


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
@cython.cdivision(True)
def rebin_matrix(double[:, :] mat_counts_in, double[:] E_array_in,
                 double[:] E_array_out, int rebin_axis=0):
    """Rebin a matrix of counts from binning E_array_in to binning E_array_out

    This is a currently just a wrapper for rebin() to handle the logistics
    of getting a matrix as input.
    Todo: It is unnecessary to calculate the overlap for each bin along
    the axis that is not being rebinned.

    Args:
        mat_counts_in (np.ndarray): Matrix of counts to rebin
        E_array_in (np.ndarray): Lower-bin-edge energy calibration of input
                                 matrix along rebin axis
        E_array_out (np.ndarray): Lower-bin-edge energy calibration of output
                                  matrix along rebin axis
        rebin_axis (int): Axis to rebin
    Returns:
        mat_counts_out (np.ndarray): Matrix of rebinned counts



    """

    # Define variables for Cython:
    cdef int other_axis, N_loop, i
    # cdef int[:] shape_out

    # Axis number of non-rebin axis (Z2 group, fancy!):
    assert (rebin_axis == 0 or rebin_axis == 1)
    other_axis = (rebin_axis + 1) % 2
    # Number of bins along that axis:
    N_loop = mat_counts_in.shape[other_axis]

    # Calculate shape of rebinned matrix and allocate it:
    shape_out = np.array([mat_counts_in.shape[0], mat_counts_in.shape[1]],
                         dtype=int)
    shape_out[rebin_axis] = len(E_array_out)
    mat_counts_out = np.zeros(shape_out, dtype=DTYPE)

    # For simplicity I use an if test to know axis ordering. Can probably
    # be done smarter later:
    # cdef double[:, :] mat_counts_out_view = mat_counts_out
    if rebin_axis == 0:
        # TODO figure out how to best put arrays into mat_counts_out. 
        # Use memoryview or no?
        for i in range(N_loop):
            mat_counts_out[:, i] = rebin(mat_counts_in[:, i],
                                         E_array_in, E_array_out)
    else:
        for i in range(N_loop):
            counts_out = rebin(mat_counts_in[i, :],
                               E_array_in, E_array_out)
            mat_counts_out[i, :] = counts_out
    return mat_counts_out
