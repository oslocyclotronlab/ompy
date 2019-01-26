# Functions to decompose a first-generation matrix
# into rho and T by a fit
# This version uses numpy arrays to do the heavy lifting as much as possible

import numpy as np
from .library import *
from .rebin import *
from scipy.optimize import minimize
import copy


def fit_rho_T(firstgen_in, bin_width_out,
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

    # Cut the firstgen.matrix and firstgen.std to shape [Ex_min:Ex_max,
    # Eg_min:Eg_max].
    # Protect the input matrix:
    firstgen_in = copy.deepcopy(firstgen_in)
    firstgen_in.cut_rect(axis=0, E_limits=[Ex_min, Ex_max], inplace=True)
    firstgen_in.cut_rect(axis=1, E_limits=[Eg_min, Ex_max], inplace=True)

    firstgen_in.plot()

    # Set the calibration for the output result.
    # TODO think about and test proper a0 value. Check rhosigchi.f.
    calib_out = {"a0": -500, "a1": bin_width_out}
    # Set up the energy array common to rho and T
    E_array = E_array_from_calibration(a0=calib_out["a0"],
                                       a1=calib_out["a1"],
                                       E_max=Ex_max)
    Nbins = len(E_array)

    # Rebin firstgen to calib_out along both axes and store it in a new Matrix:
    firstgen = Matrix()
    # axis 0:
    firstgen.matrix = rebin_matrix(firstgen_in.matrix, firstgen_in.E0_array,
                                   E_array, rebin_axis=0)
    # TODO figure out how to properly "rebin" standard deviations
    firstgen.std = rebin_matrix(firstgen_in.std, firstgen_in.E0_array,
                                E_array, rebin_axis=0)
    # axis 1:
    firstgen.matrix = rebin_matrix(firstgen.matrix, firstgen_in.E1_array,
                                   E_array, rebin_axis=1)
    firstgen.std = rebin_matrix(firstgen.std, firstgen_in.E1_array,
                                E_array, rebin_axis=1)
    # Set energy axes accordingly
    firstgen.E0_array = E_array
    firstgen.E1_array = E_array
    # Verify that it got rebinned and assigned correctly:
    calib_firstgen = firstgen.calibration()
    assert(
            calib_firstgen["a00"] == calib_out["a0"] and
            calib_firstgen["a01"] == calib_out["a1"] and
            calib_firstgen["a10"] == calib_out["a0"] and
            calib_firstgen["a11"] == calib_out["a1"]
           )

    # Normalize the firstgen matrix for each Ex bin:
    P_exp = div0(firstgen.matrix, firstgen.matrix.sum(axis=1))
    P_err = firstgen.std
    # TODO should the std matrix be scaled accordingly? Check rhosigchi.f

    # === Testing things borrowed from Fabio: ===

    Nbins_Ex = Nbins
    Nbins_T = Nbins
    Nbins_rho = Nbins

    # initial guesses
    # initial guess for rho is a box:
    rho0 = np.ones(Nbins_rho)
    # initial guess for T following Eq. (6) in Schiller2000:
    T0 = np.zeros(Nbins_T)
    for i_Eg in range(Nbins_T):
        # no need for i_start; we trimmed the matrix already:
        # TODO double check if this is still true in my implementation
        T0[i_Eg] = np.sum(firstgen.matrix[:, i_Eg])

    p0 = np.append(rho0, T0)  # create 1D array of the initial guesses

    # minimization
    # print("firstgen.std =", firstgen.std)
    # res = chisquare_1D(p0, firstgen.matrix, firstgen.std, E_array)
    # print("chisq(p0) = ", res)

    res = minimize(chisquare_1D, x0=p0,
                   args=(firstgen.matrix, firstgen.std,
                         E_array),
                   method="Powell",
                   options={'disp': True})

    # Unpack the fit values:
    print("res =", res)
    x = res.x
    rho_array, T_array = x[0:int(len(x)/2)], x[int(len(x)/2):]

    # TODO:
    # 1. Rebin firstgen to calib_out X
    # 2. Normalize firstgen (and stddev?)
    # 3. Write function that makes probability matrix from rho and T
    # 4. Write chisquare function. Interface to standard minimizers.


    # Dummy values to return for testing:
    rho = Vector(rho_array, E_array)
    T = Vector(T_array, E_array)

    # - rho and T shall be Vector() instances
    return rho, T


def chisquare_1D(x, *args):
    """
    1D version of the chi2 function (needed for minimize function)
    so x has one dimension only, but may be nested to contain rho and T

    Parameters:
    ----------
    x: ndarray
        workaround: 1D representation of the parameters rho and T
    args: tuple
        tuple of the fixed parameters needed to completely specify the function

    """

    P_exp, P_err, E_array = args
    # Are the following things needed?
    # P_exp = np.asarray(P_exp)
    # P_exp_err = np.asarray(P_exp_err)
    # E_array = np.asarray(E_array)
    # P_exp = P_exp.reshape(-1, Pexp.shape[-1])

    # Split the x vector into rho and T:
    rho_array, T_array = x[0:int(len(x)/2)], x[int(len(x)/2):]
    assert(len(rho_array) == len(T_array))  # TODO remove after testing
    assert(len(rho_array) == int(len(x)/2))  # TODO remove after testing

    P_fit = construct_P(rho_array, T_array, E_array)

    return chisquare(rho_array, T_array, P_exp, P_err, P_fit, E_array)


def chisquare(rho, T, P_exp, P_err, P_fit, E_array):
    """ Chi^2 between experimental and fitted first genration matrix"""
    # if np.any(rho<0) or np.any(T<0): # hack to implement lower boundary
        # chi2 = 1e20
    # else:
    # chi^2 = (data - fit)^2 / unc.^2, where unc.^2 = #cnt for Poisson dist.
    chi2 = np.sum(div0((P_exp - P_fit)**2, P_err**2))
    return chi2


def construct_P(rho_array, T_array, E_array):
    """
    Constructs a "theoretical" first generation matrix P from rho and T

    This function is called in each chisquare evaluation,
    and must be as fast as possible.

    Args:
        rho_array (np.ndarray): The input level density rho
        T_array (np.ndarray): The input transmission coefficient T
        E_array (np.ndarray): The common lower-bin-edge energy calibration
                              of rho and T
    returns:
        P (np.ndarray, two-dimensional): The product of rho and T,
            constructed as P(Ex, Eg) = N*rho(Ex-Eg)*T(Eg)
            with N such that sum_Eg(P) = 1 for each Ex bin.
    """
    Nbins = len(E_array)
    # We construct a "meshgrid" of T and rho, then multiply.
    # T_grid is just a repeat of T along each row:
    T_grid = np.tile(T_array, (Nbins, 1))
    # rho_grid is basically a repeat of rho along each column, but
    # because of the (Ex-Eg) argument, rho(0) always starts on the diagonal
    rho_grid = np.zeros((Nbins, Nbins))
    for j in range(Nbins):
        rho_grid[j:Nbins, j] = rho_array[0:Nbins-j]
    return rho_grid*T_grid
