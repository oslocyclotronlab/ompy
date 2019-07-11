import os
import numpy as np
from scipy.interpolate import interp1d#, interp2d

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
                    double[:] fwhm_divE_array,
                    double cut_width=3):
    """
    Function which smooths an array of counts by a Gaussian
    of full-width-half-maximum FWHM. Preserves number of counts.
    Args:
        vector_in (array, double): Array of inbound counts to be smoothed
        E_array (array, double): Array with energy calibration of vector_in
        fwhm_divE_array (array, double): The full-width-half-maximum value to smooth
                                  by, in percent of the energy. Note well that
                                  this means that
                                  fwhm = fwhm_divE/100 * E_array
                                  gives you the absolute FWHM.
        cut_width (double, optional): The window width of the Gaussian that is used to
                            smoothe, in units of sigma. Defaults to 3.

    Returns:
        vector_out: Array of smoothed counts

    """
    if not len(vector_in) == len(E_array):
        raise ValueError("Length mismatch between vector_in and E_array")
    if not len(vector_in) == len(fwhm_divE_array):
        raise ValueError("Length mismatch between vector_in and fwhm_divE_array")

    cdef double[:] vector_in_view = vector_in
    cdef double a0, a1

    a0 = E_array[0]
    a1 = E_array[1] - E_array[0]

    vector_out = np.zeros(len(vector_in), dtype=DTYPE)
    # cdef double[:] vector_out_view = vector_out

    cdef int i
    for i in range(len(vector_out)):
        counts = vector_in_view[i]
        if counts > 0:
            E_centroid_current = E_array[i]
            sigma_current = fwhm_divE_array[i]/(2.355*100)*E_array[i]
            E_cut_low = E_centroid_current - cut_width * sigma_current
            i_cut_low = int((E_cut_low - a0) / a1)
            i_cut_low = max(0, i_cut_low)
            E_cut_high = E_centroid_current + cut_width * sigma_current
            i_cut_high = int((E_cut_high - a0) / a1)
            i_cut_high = max(min(len(vector_in), i_cut_high), i_cut_low+1)
            pdf = np.zeros(len(vector_in), dtype=DTYPE)
            pdf[i_cut_low:i_cut_high] =\
                gaussian(E_array[i_cut_low:i_cut_high],
                         mu=E_centroid_current,
                         sigma=sigma_current
                         )
            pdf = pdf / np.sum(pdf)
            vector_out += counts * pdf

    return vector_out


def gauss_smoothing_matrix(matrix_in, E_array,
                           fwhm_array):
    cdef int i
    matrix_out = np.zeros(matrix_in.shape, dtype=DTYPE)

    for i in range(matrix_in.shape[0]):
        matrix_out[i, :] = gauss_smoothing(matrix_in[i, :],
                                           E_array, fwhm_array)

    return matrix_out
