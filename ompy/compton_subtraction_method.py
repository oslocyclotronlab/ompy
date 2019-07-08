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
