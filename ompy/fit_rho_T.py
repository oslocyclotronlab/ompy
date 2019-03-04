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
from .rhosig import *
from .matrix import Matrix, Vector

from scipy.optimize import minimize
import copy

# Define constants from constants.py as global vars
global DE_PARTICLE
global DE_GAMMA_1MEV
global DE_GAMMA_8MEV


def fit_rho_T(firstgen_in, firstgen_std_in, bin_width_out,
              Ex_min, Ex_max, Eg_min,
              method="Powell",
              verbose=True,
              negatives_penalty=0,
              regularizer=0):
    """Fits the firstgen spectrum to the product of transmission coeff T and
    level density rho

    Args:
        firstgen (Matrix): The first-generation matrix.
        firstgen_std (Matrix): The standard deviations in the first-gen matrix
        calib_out (dict): Desired common calibration of output rho and T on the
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
    # Check that firstgen and firstgen_std match each other:
    if not (firstgen_in.calibration() == firstgen_std_in.calibration()):
        raise Exception("Calibration mismatch between firstgen_in and"
                        " firstgen_std_in.")
    if not (firstgen_in.matrix.shape == firstgen_std_in.matrix.shape):
        raise Exception("Shape mismatch between firstgen_in and"
                        " firstgen_std_in")

    # Check that Ex_min >= Eg_min:
    if Ex_min < Eg_min:
        raise ValueError("Ex_min must be >= Eg_min.")


    assert firstgen_in.matrix.min() >= 0, "no negative counts"
    assert firstgen_std_in.matrix.min() >= 0, "no negative stddevs"

    # Cut the firstgen.matrix and firstgen.std to shape [Ex_min:Ex_max,
    # Eg_min:Eg_max].
    # Protect the input matrix:
    firstgen_in = copy.deepcopy(firstgen_in)
    firstgen_std_in = copy.deepcopy(firstgen_in)
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
    # axis 1:
    firstgen.matrix = rebin_matrix(firstgen.matrix, firstgen_in.E1_array,
                                   E_array, rebin_axis=1)
    # Set energy axes accordingly
    firstgen.E0_array = E_array
    firstgen.E1_array = E_array
    # Update 20190212: Interpolate the std matrix instead of rebinning:
    firstgen_std = Matrix()
    firstgen_std.matrix = interpolate_matrix_2D(firstgen_std_in.matrix,
                                         firstgen_std_in.E0_array,
                                         firstgen_std_in.E1_array,
                                         E_array,
                                         E_array
                                         )
    # Set energy axes accordingly
    firstgen_std.E0_array = E_array
    firstgen_std.E1_array = E_array
    # Verify that it got rebinned and assigned correctly:
    calib_firstgen = firstgen.calibration()
    assert (
            calib_firstgen["a00"] == calib_out["a0"] and
            calib_firstgen["a01"] == calib_out["a1"] and
            calib_firstgen["a10"] == calib_out["a0"] and
            calib_firstgen["a11"] == calib_out["a1"]
           ), "firstgen does not have correct calibration."
    calib_firstgen_std = firstgen_std.calibration()
    assert (
            calib_firstgen_std["a00"] == calib_out["a0"] and
            calib_firstgen_std["a01"] == calib_out["a1"] and
            calib_firstgen_std["a10"] == calib_out["a0"] and
            calib_firstgen_std["a11"] == calib_out["a1"]
           ), "firstgen_std does not have correct calibration."

    # Make cuts to the matrices using Fabio's utility:
    pars_fg = {"Egmin" : Eg_min,
               "Exmin" : Ex_min,
               "Emax" : Ex_max}
    E_array_midbin = E_array + calib_out["a1"]/2
    firstgen_matrix, Emid_Eg, Emid_Ex, Emid_nld = fg_cut_matrix(firstgen.matrix,
                                                            E_array_midbin, **pars_fg)
    firstgen_std_matrix, Emid_Eg, Emid_Ex, Emid_nld = fg_cut_matrix(firstgen_std.matrix,
                                                            E_array_midbin, **pars_fg)

    rho_fit, T_fit = decompose_matrix(firstgen_matrix, firstgen_std_matrix,
                                         Emid_Eg=Emid_Eg,
                                         Emid_nld=Emid_nld,
                                         Emid_Ex=Emid_Ex,
                                         method="Powell")

    rho = Vector(rho_fit, Emid_nld-calib_out["a1"]/2)
    T = Vector(T_fit, Emid_Eg-calib_out["a1"]/2)

    # - rho and T shall be Vector() instances
    return rho, T


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
        masking_array[i_Ex, i_Eg_min:i_Eg_max+1] = 1

    # print("masking_array.shape =", masking_array.shape)
    print("masking_array =", masking_array)

    return masking_array

def fg_cut_matrix(array, Emid, Egmin, Exmin, Emax, **kwargs):
    """ Make the first generation cuts to the matrix
    Parameters:
    -----------
    array : ndarray
        2D Array that will be sliced
    Emid : ndarray
        Array of bin center energies [Note: up to here assumed symetrix for
        both axes]
    Egmin, Exmin, Emax : doubles
        Lower and higher cuts for the gamma-ray and excitation energy axis
    kwargs: optional
        Will be ignored, just for compatibility;
    Returns:
    --------
    array : ndarray
        Sliced array
    Emid_Eg : ndarray
        Bin center energies of the gamma-ray axis
    Emid_Ex : ndarray
        Bin center energies of the excitation energy axis
    Emid_nld : ndarray
        Bin center energies of the nld once extracted
    """

    np.copy(array)

    # Eg
    i_Egmin = (np.abs(Emid-Egmin)).argmin()
    i_Emax = (np.abs(Emid-Emax)).argmin()
    # Ex
    i_Exmin = (np.abs(Emid-Exmin)).argmin()

    array = array[i_Exmin:i_Emax,i_Egmin:i_Emax]
    Emid_Ex = Emid[i_Exmin:i_Emax]
    Emid_Eg = Emid[i_Egmin:i_Emax]
    Emid_nld = Emid[:i_Emax-i_Egmin]

    return array, Emid_Eg, Emid_Ex, Emid_nld
