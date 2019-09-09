import os
import numpy as np
cimport numpy as np
from scipy.interpolate import interp1d#, interp2d

from .rebin import *
from .library import *
from .matrix import to_plot_axis

DTYPE = np.float64


def gaussian(double[:] Emids, double mu, double sigma):
    """
    Returns a normalized Gaussian supported on Emids.

    NB! All arguments (Emids, mu and sigma) must have the
    same units. In OMpy the default unit is keV.

    Args:
        Emids (array, double): Array of energies to evaluate
                               (center bin calibration)
        mu (double): Centroid
        sigma (double): Standard deviation
    Returns:
        gaussian_array (array, double): Array of gaussian
        distribution values matching Emids.
    """
    gaussian_array = np.zeros(len(Emids), dtype=DTYPE)
    cdef double[:] gaussian_array_view = gaussian_array
    cdef double prefactor, eps

    eps = 1e-6  # Avoid zero division
    sigma += eps

    prefactor = (1/(sigma*np.sqrt(2*np.pi)))
    cdef int i
    for i in range(len(Emids)):
        gaussian_array_view[i] = (prefactor
                                  * np.exp(
                                    -(Emids[i]-mu)
                                    * (Emids[i]-mu)/(2*sigma*sigma))
                                  )

    return gaussian_array


def gauss_smoothing(double[:] array_in, double[:] E_array,
                    double[:] fwhm,
                    double truncate=3):
    """
    Function which smooths an array of counts by a Gaussian
    of full-width-half-maximum FWHM. Preserves number of counts.
    Args:
        array_in (array, double): Array of inbound counts to be smoothed
        E_array (array, double): Array with energy calibration of array_in, in
                                 lower-bin-edge calibration
        fwhm (array, double): The full-width-half-maximums. Need to be
                              same size as array_in
        truncate (double, optional): The window width of the Gaussian that is
                                     used to smoothe, in units of sigma.
                                     Defaults to 3.

    Returns:
        array_out: Array of smoothed counts

    """
    if not len(array_in) == len(E_array):
        raise ValueError("Length mismatch between array_in and E_array")
    if not len(array_in) == len(fwhm):
        raise ValueError("Length mismatch between array_in and fwhm")

    cdef double[:] array_in_view = array_in
    cdef double a0, a1

    # a0_lower_bin_edge = E_array[0]
    a0 = E_array[0]
    a1 = E_array[1] - E_array[0]

    # # Convert from lower bin edge to middle-bin energy:
    # E_array = E_array + a1/2
    # a0 = E_array[0]

    array_out = np.zeros(len(array_in), dtype=DTYPE)
    # cdef double[:] array_out_view = array_out

    def find_truncation_indices(double E_centroid_current,
                                double sigma_current,
                                double truncate=truncate):
        cdef int i_cut_low, i_cut_high
        E_cut_low = E_centroid_current - truncate * sigma_current
        i_cut_low = int((E_cut_low - a0) / a1)
        i_cut_low = max(0, i_cut_low)
        E_cut_high = E_centroid_current + truncate * sigma_current
        i_cut_high = int((E_cut_high - a0) / a1)
        i_cut_high = max(min(len(array_in), i_cut_high), i_cut_low+1)
        return i_cut_low, i_cut_high

    cdef int i
    for i in range(len(array_out)):
        counts = array_in_view[i]
        if counts > 0:
            E_centroid_current = E_array[i] + a1/2
            sigma_current = fwhm[i]/2.355
            i_cut_low, i_cut_high = find_truncation_indices(E_centroid_current,
                                                            sigma_current)
            pdf = np.zeros(len(array_in), dtype=DTYPE)
            # using lower bin instead of center bin in both E_mid and mu
            # below-> canceles out
            pdf[i_cut_low:i_cut_high] =\
                gaussian(E_array[i_cut_low:i_cut_high],
                         mu=E_array[i],
                         sigma=sigma_current
                         )
            pdf = pdf / np.sum(pdf)
            array_out += counts * pdf

    return array_out


def gauss_smoothing_matrix_1D(matrix_in, E_array,
                              fwhm,
                              axis="Eg"):
    """ Smooth a matrix with a Gaussian

    Function which smooths an array of counts by a Gaussian
    of full-width-half-maximum FWHM. Preserves number of counts.

    Args:
        matrix_in (array, double): Array of inbound counts to be smoothed
        E_array (array, double): Array with energy calibration of matrix_in, in
                                 lower-bin-edge calibration
        fwhm (double or array of doubles): The full-width-half-maximums
        abs_or_rel (str): fhwm given absolute, or relative in %
                          relative: fwhm = fwhm_divE/100 * E_array
        axis: The axis along which smoothing should happen.
              Can be either of (0, 'Eg', 'x'), (1, 'Ex', 'y')
    """
    cdef int i
    matrix_out = np.zeros(matrix_in.shape, dtype=DTYPE)

    if len(fwhm) == 1:
        fwhm = np.full_like(matrix_in, fwhm)

    axis = to_plot_axis(axis)
    is_Eg = axis == 0

    if is_Eg:
        for i in range(matrix_in.shape[0]):
            matrix_out[i, :] = gauss_smoothing(matrix_in[i, :],
                                               E_array, fwhm)
    else:
        for i in range(matrix_in.shape[1]):
            matrix_out[:, i] = gauss_smoothing(matrix_in[:, i],
                                               E_array, fwhm)

    return matrix_out
