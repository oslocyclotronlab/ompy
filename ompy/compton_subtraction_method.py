import numpy as np

from .library import *
from .rebin import *
from .constants import *


def rebin_and_shift(counts_in, E_array_in, E_array_out, energy_shift=0):
    """
    Rebin an array from calibration E_array_in to E_array_out, while
    also shifting all counts by amount energy_shift.

    The function is actually a wrapper for the rebin() function that
    "fakes" the input energy calibration to give a shift.

    Args:
        counts_in (numpy array, float): Array of counts
        E_array_in (numpy array, float): Energy calibration (lower bin edge)
                                         of input counts
        E_array_out (numpy array, float): Desired energy calibration of output
                                          array
        energy_shift (float): Amount to shift the counts by. Negative means
                              shift to lower energies. Default is 0.
    """
    E_array_in_shifted = E_array_in + energy_shift
    counts_out = rebin(counts_in, E_array_in_shifted, E_array_out)
    return counts_out


def shift(counts_in, E_array_in, energy_shift):
    """
    Shift the counts_in array by amount energy_shift.

    The function is actually a wrapper for the rebin() function that
    "fakes" the input energy calibration to give a shift. It is similar to
    the rebin_and_shift() function defined above, but even simpler.

    Args:
        counts_in (numpy array, float): Array of counts
        E_array_in (numpy array, float): Energy calibration (lower bin edge)
                                         of input counts
        energy_shift (float): Amount to shift the counts by. Negative means
                              shift to lower energies. Default is 0.
    """
    E_array_in_shifted = E_array_in + energy_shift
    counts_out = rebin(counts_in, E_array_in_shifted, E_array_in)
    return counts_out


def shift_matrix(counts_in_matrix, E_array_in, energy_shift):
    """
    Function which takes a matrix of counts and shifts it
    along axis 1.
    """
    counts_out_matrix = np.zeros(counts_in_matrix.shape)
    for i in range(counts_in_matrix.shape[0]):
        counts_out_matrix[i, :] = shift(counts_in_matrix[i, :], E_array_in,
                                        energy_shift=energy_shift)
    return counts_out_matrix





def shift_and_smooth3D(array, Eg_array, FWHM, p, shift, smoothing=True):
    # Update 20190708: In the process of replacing this function by a better
    # implementation.
    # Updated 201807: Trying to vectorize so all Ex bins are handled simultaneously.
    # Takes a 2D array of counts, shifts it (downward only!) with energy 'shift'
    # and smooths it with a gaussian of specified 'FWHM'.
    # This version is vectorized to shift, smooth and scale all points
    # of 'array' individually, and then sum together and return.

    # TODO: FIX ME! There is a bug here, it does not do Compton subtraction right.

    # The arrays from resp.dat are missing the first channel.
    p = np.append(0, p) 
    FWHM = np.append(0, FWHM)

    a1_Eg = (Eg_array[1]-Eg_array[0]) # bin width
    N_Ex, N_Eg = array.shape

    # Shift is the same for all energies 
    if shift == "annihilation":
        # For the annihilation peak, all channels should be mapped on E = 511 keV. Of course, gamma channels below 511 keV,
        # and even well above that, cannot produce annihilation counts, but this is taken into account by the fact that p
        # is zero for these channels. Thus, we set i_shift=0 and make a special dimensions_shifted array to map all channels of
        # original array to i(511). 
        i_shift = 0 
    else:
        i_shift = i_from_E(shift, Eg_array) - i_from_E(0, Eg_array) # The number of indices to shift by


    N_Eg_sh = N_Eg - i_shift
    indices_original = np.linspace(i_shift, N_Eg-1, N_Eg-i_shift).astype(int) # Index array for original array, truncated to shifted array length
    if shift == "annihilation": # If this is the annihilation peak then all counts should end up with their centroid at E = 511 keV
        # indices_shifted = (np.ones(N_Eg-i_from_E(511, Eg_array))*i_from_E(511, Eg_array)).astype(int)
        indices_shifted = (np.ones(N_Eg)*i_from_E(511, Eg_array)).astype(int)
    else:
        indices_shifted = np.linspace(0,N_Eg-i_shift-1,N_Eg-i_shift).astype(int) # Index array for shifted array


    if smoothing:
        # Scale each Eg count by the corresponding probability
        # Do this for all Ex bins at once:
        array = array * p[0:N_Eg].reshape(1,N_Eg)
        # Shift array down in energy by i_shift indices,
        # so that index i_shift of array is index 0 of array_shifted.
        # Also flatten array along Ex axis to facilitate multiplication.
        array_shifted_flattened = array[:,indices_original].ravel()
        # Make an array of N_Eg_sh x N_Eg_sh containing gaussian distributions 
        # to multiply each Eg channel by. This array is the same for all Ex bins,
        # so it will be repeated N_Ex times and stacked for multiplication
        # To get correct normalization we multiply by bin width
        pdfarray = a1_Eg* norm.pdf(
                            np.tile(Eg_array[0:N_Eg_sh], N_Eg_sh).reshape((N_Eg_sh, N_Eg_sh)),
                            loc=Eg_array[indices_shifted].reshape(N_Eg_sh,1),
                            scale=FWHM[indices_shifted].reshape(N_Eg_sh,1)/2.355
                        )
                        
        # Remove eventual NaN values:
        pdfarray = np.nan_to_num(pdfarray, copy=False)
        # print("Eg_array[indices_shifted] =", Eg_array[indices_shifted], flush=True)
        # print("pdfarray =", pdfarray, flush=True)
        # Repeat and stack:
        pdfarray_repeated_stacked = np.tile(pdfarray, (N_Ex,1))

        # Multiply array of counts with pdfarray:
        multiplied = pdfarray_repeated_stacked*array_shifted_flattened.reshape(N_Ex*N_Eg_sh,1)

        # Finally, for each Ex bin, we now need to sum the contributions from the smoothing
        # of each Eg bin to get a total Eg spectrum containing the entire smoothed spectrum:
        # Do this by reshaping into 3-dimensional array where each Eg bin (axis 0) contains a 
        # N_Eg_sh x N_Eg_sh matrix, where each row is the smoothed contribution from one 
        # original Eg pixel. We sum the columns of each of these matrices:
        array_out = multiplied.reshape((N_Ex, N_Eg_sh, N_Eg_sh)).sum(axis=1)
        # print("array_out.shape =", array_out.shape)
        # print("array.shape[0],array.shape[1]-N_Eg_sh =", array.shape[0],array.shape[1]-N_Eg_sh)

    else:
        # array_out = np.zeros(N)
        # for i in range(N):
        #     try:
        #         array_out[i-i_shift] = array[i] #* p[i+1]
        #     except IndexError:
        #         pass

        # Instead of above, vectorizing:
        array_out = p[indices_original].reshape(1,N_Eg_sh)*array[:,indices_original]

    # Append zeros to the end of Eg axis so we match the length of the original array:
    if i_shift > 0:
        array_out = np.concatenate((array_out, np.zeros((N_Ex, N_Eg-N_Eg_sh))),axis=1)
    return array_out