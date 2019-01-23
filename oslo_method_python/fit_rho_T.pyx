# Function to decompose a first-generation matrix
# into rho and T by a fit
# Based on and contains code by Fabio Zeiser

import numpy as np
from .library import *
from .rebin import *


def fit_rho_T(firstgen, calib_out,
              Ex_min, Ex_max, Eg_min):
    """Fits the firstgen spectrum to the product of transmission coeff T and
    level density rho

    Args:
        firstgen (Matrix): The first-generation matrix. The variable
                           firstgen.std must be filled with standard dev.
        calib_out (dict): Desired commoncalibration of output rho and T on the
                          form {"a0": a0, "a1": a1}
        Ex_min (float): Minimum excitation energy for the fit
        Ex_max (float): Maximum excitation energy for the fit
        Eg_min (float): Minimum gamma-ray energy for the fit
                        (Eg_max is equal to Ex_max)

    Returns:
        rho, T (tuple):
            rho (Vector): Fit result for rho
            T (Vector): Fit result for T
    Todo:
        If firstgen.std=None, it could call a function to estimate the standard
        deviation using the rhosigchi techniques.
    """

    # Set up the energy array common to rho and T
    E_array = E_array_from_calibration(a0=calib_out["a0"],
                                       a1=calib_out["a1"],
                                       E_max=Ex_max)
    Nbins = len(E_array)

    # Rebin firstgen to calib_out along both axes
    # axis 0:
    firstgen_recalib = rebin_matrix(firstgen.matrix, firstgen.E0_array,
                                    E_array, rebin_axis=0)
    # axis 1:
    firstgen_recalib = rebin_matrix(firstgen.matrix, firstgen.E1_array,
                                    E_array, rebin_axis=1)

    # TODO:
    # 1. Rebin firstgen to calib_out

    # - rho and T should be Vector() instances

    # Dummy values to return for testing:
    rho = Vector(np.random.normal(loc=0.1*E_array, size=Nbins), E_array)
    T = Vector(np.random.normal(loc=0.1*E_array, size=Nbins), E_array)

    return rho, T


# @cython.boundscheck(True)  # turn off bounds-checking for entire function
# @cython.wraparound(True)  # turn off negative index wrapping
# def PfromRhoT(np.ndarray rho, np.ndarray T, int Nbins_Ex,
#               np.ndarray Emid_Eg, np.ndarray Emid_nld,
#               np.ndarray Emid_Ex, type="transCoeff"):
#     """ Generate a first gernation matrix P from given nld and T (or gsf)

#     Parameters:
#     -----------
#     rho: ndarray
#         nld
#     T: ndarray, optional
#         transmission coefficient; either this or gsf must be specified
#     gsf: ndarray, optional
#         gamma-ray strength function; either this or gsf must be specified
#     type: string, optional
#         chosen by type= "transCoeff" /or/ "gsfL1"
#     Nbins_Ex, Emid_Eg, Emid_nld, Emid_Ex:
#         bin number and bin center values
#     Note: rho and T must have the same bin width

#     Returns:
#     --------
#     P: ndarray
#         normalized first generations matrix (sum of each Ex bin = 1)
#     """

#     cdef int Nbins_T = len(T)
#     cdef int i_Ex, i_Eg, i_Ef, Nbins
#     cdef double Ef ,Ex
#     cdef double Eg
#     global Emid_Eg
#     cdef np.ndarray P = np.zeros((Nbins_Ex,Nbins_T))
#     # for i_Ex in range(Nbins_Ex):
#     #     for i_Eg in range(Nbins_T):
#     for i_Ex in range(Nbins_Ex):
#         Ex = Emid_Ex[i_Ex]
#         Nbins = (np.abs(Emid_Eg-Ex)).argmin() + 1
#         for i_Eg in range(Nbins):
#             Ef = Emid_Ex[i_Ex] - Emid_Eg[i_Eg]
#             i_Ef = (np.abs(Emid_nld-Ef)).argmin()
#             if i_Ef>=0: # no gamma's with higher energy then the excitation energy
#                 P[i_Ex,i_Eg] = rho[i_Ef] * T[i_Eg]
#                 if type=="gsfL1": # if input T was a gsf, not transmission coeff: * E^(2L+1)
#                     Eg = Emid_Eg[i_Eg]
#                     P[i_Ex,i_Eg] *= np.power(Eg,3.)
#     # normalize each Ex row to 1 (-> get decay probability)
#     for i, normalization in enumerate(np.sum(P,axis=1)):
#         P[i,:] /= normalization
#     return P