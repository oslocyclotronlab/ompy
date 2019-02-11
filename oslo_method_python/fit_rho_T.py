# -*- coding: utf-8 -*-
"""
Functions to decompose a first-generation matrix
into rho and T by a fit.
Parts of the code is borrowed from Fabio Zeiser's code rhosig.py:
https://github.com/oslocyclotronlab/rhosig.py
This version uses numpy arrays to do the heavy lifting as much as possible

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
from .library import *
from .rebin import *
from .constants import *
from scipy.optimize import minimize
import copy

# Define constants from constants.py as global vars
global DE_PARTICLE
global DE_GAMMA_1MEV
global DE_GAMMA_8MEV


def fit_rho_T(firstgen_in, bin_width_out,
              Ex_min, Ex_max, Eg_min,
              method="Powell",
              verbose=True):
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
        method (str): Method to use for minimization.
                      Must be one of the methods available
                      in scipy.optimize.minimize.
        verbose (bool): Whether to print information
                        from the fitting routine.

    Returns:
        rho, T (tuple):
            rho (Vector): Fit result for rho
            T (Vector): Fit result for T
    Todo:
        - Change input so firstgen_std is its own Matrix() instance + check
          that calibrations are identical between the two
        - If firstgen.std=None, it could call a function to estimate the standard
          deviation using the rhosigchi techniques.
    """
    # Check that Ex_min >= Eg_min:
    if Ex_min < Eg_min:
        raise ValueError("Ex_min must be >= Eg_min.")

    # Cut the firstgen.matrix and firstgen.std to shape [Ex_min:Ex_max,
    # Eg_min:Eg_max].
    # Protect the input matrix:
    firstgen_in = copy.deepcopy(firstgen_in)
    # TODO: Figure out what the bug in cut_rect is. If I only do axis=0,
    # it still seems to cut both axes with the same lower limit. axis=1
    # does nothing at all. But also, it might not be so important since
    # the masking array in the chisquare takes care of things.
    # The only thing to be aware of is normalization of P_exp, whether
    # to include only the Eg bins masked by masking_array or the whole
    # firstgen_in.
    # firstgen_in.cut_rect(axis=0, E_limits=[Ex_min, Ex_max], inplace=True)
    # firstgen_in.cut_rect(axis=1, E_limits=[Eg_min, Ex_max], inplace=True)

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
    # TODO figure out how to properly "rebin" standard deviations.
    # Should maybe just be interpolated instead.
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
    assert (
            calib_firstgen["a00"] == calib_out["a0"] and
            calib_firstgen["a01"] == calib_out["a1"] and
            calib_firstgen["a10"] == calib_out["a0"] and
            calib_firstgen["a11"] == calib_out["a1"]
           ), "Matrix does not have correct calibration."

    # Normalize the firstgen matrix for each Ex bin:
    P_exp = div0(firstgen.matrix, firstgen.matrix.sum(axis=1))
    P_err = firstgen.std
    # TODO should the std matrix be scaled accordingly? Check rhosigchi.f

    # DEBUG: Plot the cut and normalized version of firstgen_in
    # from matplotlib.colors import LogNorm
    # f, (axdebug1, axdebug2, axdebug3) = plt.subplots(1, 3)
    # axdebug1.pcolormesh(E_array, E_array, P_exp,
                        # norm=LogNorm())
    # axdebug1.set_title("P_exp")
    # END DEBUG

    # === Testing things borrowed from Fabio: ===

    Nbins_Ex = Nbins
    Nbins_T = Nbins
    Nbins_rho = Nbins

    # Starting vectors for rho, T minimization:
    # initial guess for rho is a box:
    rho0 = np.ones(Nbins_rho)
    # initial guess for T following Eq. (6) in Schiller2000:
    T0 = np.zeros(Nbins_T)
    i_Ex_min = np.argmin(np.abs(Ex_min - E_array))
    i_Ex_max = np.argmin(np.abs(Ex_max - E_array))
    for i_Eg in range(Nbins_T):
        T0[i_Eg] = np.sum(firstgen.matrix[i_Ex_min:i_Ex_max, i_Eg])
    # DEBUG: Testing T0 as unit box. It makes a difference even on a smooth
    # synthetic spectrum.
    # T0 = np.ones(Nbins_T)
    # END DEBUG

    # # DEBUG: Plot P_fit constructed from initial guess
    # from matplotlib.colors import LogNorm
    # P_fit_0 = construct_P(rho0, T0, E_array)
    # axdebug2.pcolormesh(E_array, E_array, P_fit_0, norm=LogNorm())
    # axdebug2.set_title("P_fit_0")
    # # END DEBUG

    p0 = np.append(rho0, T0)  # create 1D array of the initial guesses

    # minimization
    # print("firstgen.std =", firstgen.std)
    # res = chisquare_1D(p0, firstgen.matrix, firstgen.std, E_array)
    # print("chisq(p0) = ", res)

    # Set up the masking array which defines the area to include in the
    # chisquare fit:
    # TODO move out to its own function
    masking_array = make_masking_array(shape=firstgen.matrix.shape,
                                       Ex_min=Ex_min,
                                       Ex_max=Ex_max,
                                       Eg_min=Eg_min,
                                       E_array=E_array
                                       )
    # # DEBUG:
    # # masking_array = np.ones(firstgen.matrix.shape, dtype=bool)
    # axdebug3.pcolormesh(E_array, E_array, masking_array)
    # axdebug3.set_title("masking array")
    # # END DEBUG

    # # DEBUG
    # from matplotlib.colors import LogNorm
    # fdebug, axdebug = plt.subplots(1, 1)
    # axdebug.pcolormesh(firstgen.E1_array, firstgen.E0_array,
    #                    firstgen.matrix,
    #                    norm=LogNorm())
    # plt.show()
    # import sys
    # sys.exit(0)
    # # END DEBUG

    res = minimize(chisquare_1D, x0=p0,
                   args=(firstgen.matrix, firstgen.std,
                         E_array, masking_array),
                   method=method,
                   options={'disp': verbose})

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
    Wrapper for chisquare() which can be called from a 1D minimizer.

    Args:
    ----------
    x: ndarray
        workaround: 1D representation of the parameters rho and T, concatenated
        as x = np.concatenate(rho, T)
    args: tuple
        tuple of the fixed parameters needed to completely specify the problem:
        args = (P_exp, P_err, E_array, masking_array)
            P_exp (np.ndarray): The probability matrix to fit to
            P_err (np.ndarray): The matrix of statistical errors on P_exp
            E_array (np.ndarray): Array of lower-bin-edge energies calibrating
                                  both axes of the P matrices as well as rho, T
            masking_array (np.ndarray, dtype=bool): Array specifying the index
                                                    range of fit
    returns:
        The chi-squared value (not reduced chi-square), thorough a call
        to chisquare()
    
    """

    P_exp, P_err, E_array, masking_array = args

    # Split the x vector into rho and T:
    rho_array, T_array = x[0:int(len(x)/2)], x[int(len(x)/2):]
    assert(len(rho_array) == len(T_array))  # TODO remove after testing
    assert(len(rho_array) == int(len(x)/2))  # TODO remove after testing

    return chisquare(P_exp, P_err, E_array, masking_array, rho_array,
                     T_array)


def chisquare(P_exp, P_err, E_array,
              masking_array, rho_array, T_array,
              regularizer=0, negatives_penalty=0):
    """ Chi-square of the difference between P_exp and P_err.


    Args:
        P_exp (np.ndarray): The probability matrix to fit to
        P_err (np.ndarray): The matrix of statistical errors on P_exp
        E_array (np.ndarray): Array of lower-bin-edge energies calibrating
                              both axes of the P matrices as well as rho, T
        masking_array (np.ndarray, dtype=bool): Array specifying the index
                                                range of fit
        regularizer (float, optional): A Tikhonov L2 regularization term
            on rho and T which can be added to the chisquare sum. Does not
            seem to help. Defaults to 0.
        negatives_penalty (float, optional): A penalty term on negative rho
                                             and T values
    returns:
        The chi-squared value (not reduced chi-square).
    
    """

    P_fit = construct_P(rho_array, T_array, E_array)

    chi2_matrix = div0((P_exp - P_fit)**2, P_err**2)
    # DEBUG: No division, just mean square error:
    # chi2_matrix = (P_exp - P_fit)**2
    # END DEBUG
    chi2 = (np.sum(chi2_matrix[masking_array])
            # Penalty term to avoid negative rho & T, made to be smooth:
            # Is it needed?
            + negatives_penalty*(np.sum(rho_array[rho_array < 0]**2)
                                 + np.sum(T_array[T_array < 0]**2))
            # + negatives_penalty*(np.sum(rho_array < 0)
                                 # + np.sum(T_array < 0))
            # Optional Tikhonov L2 regularization term:
            + regularizer*(np.sum(rho_array**2) + np.sum(T_array**2)))
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

    P = rho_grid*T_grid
    # Normalize each Ex bin:
    P = div0(P, np.sum(P, axis=1)[:, None])
    # # DEBUG
    # # This is neat, it plots P_fit in every
    # # iteration of the minimization. Gives valuable insight!
    # f, ax = plt.subplots(1, 1)
    # ax.pcolormesh(E_array, E_array, rho_grid*T_grid)
    # plt.show()
    # # END DEBUG

    return P


def make_masking_array(shape, Ex_min, Ex_max, Eg_min, E_array):
    """
    Returns a boolean index matrix of dimension "shape".

    The mask is a trapezoid limited on three sides by Ex_min, Ex_max and
    Eg_min. The right edge is the diagonal defined by Eg = Ex + dE, where dE
    is given by the resolution of the detectors.
    """
    masking_array = np.zeros(shape, dtype=bool)
    i_Ex_min = i_from_E(Ex_min, E_array)
    i_Ex_max = i_from_E(Ex_max, E_array)
    i_Eg_min = i_from_E(Eg_min, E_array)
    # print("i_Ex_min =", i_Ex_min)
    # print("i_Ex_max =", i_Ex_max)
    # print("i_Eg_min =", i_Eg_min)
    for i_Ex in range(i_Ex_min, i_Ex_max+1):
        # Loop over rows in array and fill with ones up to sliding Eg
        # threshold
        Ex = E_array[i_Ex]
        # Assume constant particle resolution:
        dE_particle = DE_PARTICLE
        # Interpolate the gamma resolution linearly:
        dE_gamma = ((DE_GAMMA_8MEV - DE_GAMMA_1MEV) / (8000 - 1000)
                    * (Ex - 1000))
        Eg_max = Ex + np.sqrt(dE_particle**2 + dE_gamma**2)
        i_Eg_max = i_from_E(Eg_max, E_array)
        if i_Eg_max < i_Eg_min:
            continue
        # print("Ex =", Ex, "Eg_max =", Eg_max, flush=True)
        # print("i_Eg_max =", i_Eg_max)
        masking_array[i_Ex, i_Eg_min:i_Eg_max] = 1

    # print("masking_array.shape =", masking_array.shape)

    return masking_array
