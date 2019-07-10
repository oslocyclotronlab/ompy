import os
import numpy as np
from scipy.interpolate import interp1d#, interp2d

# from .firstgen import *
# from .unfold import *
from .rebin import *
from .library import *

DTYPE = np.float64


def gaussian(double[:] E_array, double mu, double sigma):
    """
    Returns a normalized Gaussian supported on E_array.

    NB! All arguments (E_array, mu and sigma) must have the
    same units. In OMpy the default unit is keV.

    Args:
        E_array (array, double): Array of energies to evaluate
        mu (double): Centroid
        sigma (double): Standard deviation
    Returns:
        gaussian_array (array, double): Array of gaussian
            distribution values matching E_array.
    """
    gaussian_array = np.zeros(len(E_array), dtype=DTYPE)
    cdef double[:] gaussian_array_view = gaussian_array
    cdef double prefactor, eps

    eps = 1e-6  # Avoid zero division
    sigma += eps

    prefactor = (1/(sigma*np.sqrt(2*np.pi)))
    cdef int i
    for i in range(len(E_array)):
        gaussian_array_view[i] = (prefactor
                                  * np.exp(
                                    -(E_array[i]-mu)
                                    * (E_array[i]-mu)/(2*sigma*sigma))
                                  )

    return gaussian_array


def gauss_smoothing(double[:] vector_in, double[:] E_array,
                    double[:] fwhm_array):
    """
    Function which smooths an array of counts by a Gaussian
    of full-width-half-maximum FWHM. Preserves number of counts.
    Args:
        vector_in (array, double): Array of inbound counts to be smoothed
        E_array (array, double): Array with energy calibration of vector_in
        fwhm (double): The full-width-half-maximum value to smooth by, in
                       percent.

    Returns:
        vector_out: Array of smoothed counts

    """
    if not len(vector_in) == len(E_array):
        raise ValueError("Length mismatch between vector_in and E_array")
    if not len(vector_in) == len(fwhm_array):
        raise ValueError("Length mismatch between vector_in and fwhm_array")

    cdef double[:] vector_in_view = vector_in
    cdef double bin_width

    bin_width = E_array[1] - E_array[0]

    vector_out = np.zeros(len(vector_in), dtype=DTYPE)
    # cdef double[:] vector_out_view = vector_out

    cdef int i
    for i in range(len(vector_out)):
        pdf = gaussian(E_array, mu=E_array[i],
                       sigma=fwhm_array[i]/(2.355*100)*E_array[i]
                       )
        pdf = pdf / (np.sum(pdf))
        vector_out += (vector_in_view[i]
                       * pdf)

    return vector_out


def gauss_smoothing_matrix(matrix_in, E_array,
                           fwhm_array):
    cdef int i
    matrix_out = np.zeros(matrix_in.shape, dtype=DTYPE)

    for i in range(matrix_in.shape[0]):
        matrix_out[i, :] = gauss_smoothing(matrix_in[i, :],
                                           E_array, fwhm_array)

    return matrix_out
