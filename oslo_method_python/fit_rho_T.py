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
# import oslo_method_python.library as lib
# # from oslo_method_python.library import calibration
# import oslo_method_python.rebin as rebin
# from oslo_method_python.rebin import rebin_matrix
# from oslo_method_python.constants import *
# import oslo_method_python.rhosig as rsg

from .library import *
from .rebin import *
from .constants import *
from .rhosig import *

from scipy.optimize import minimize
import copy

# Define constants from constants.py as global vars
global DE_PARTICLE
global DE_GAMMA_1MEV
global DE_GAMMA_8MEV

class FitRhoT:
    def __init__(self,
                 firstgen_in, firstgen_std_in,
                 bin_width_out,
                 Ex_min, Ex_max, Eg_min,
                 method="Powell",
                 verbose=True,
                 negatives_penalty=0,
                 regularizer=0,
                 options={'disp': True}):
        """
        Class Wrapper for fit of the firstgen spectrum to the product of transmission coeff T and level density rho

        Args:
            firstgen (Matrix): The first-generation matrix.
            firstgen_std (Matrix): The standard deviations in the first-gen matrix
            bin_width_out (float): Bin-width of the output rho and T
            Ex_min (float): Minimum excitation energy for the fit
            Ex_max (float): Maximum excitation energy for the fit
            Eg_min (float): Minimum gamma-ray energy for the fit
                            (Eg_max is equal to Ex_max)
            method (str): Method to use for minimization.
                          Must be one of the methods available
                          in scipy.optimize.minimize.
            verbose (bool): Whether to print information
                            from the fitting routine.
            negatives_penalty : For Chi2 # TODO: send to fit
            regularizer : For Chi2 # TODO: send to fit
            options (dict): Options to the minimization routine. See docs of
                scipy.optimize.minimize

        TODO:
            - Implement negatives_penalty and regularizer
            - If firstgen.std=None, it could call a function to estimate
                  the standard deviation using the rhosigchi techniques.
        """
        # Protect the input matrix:
        self.firstgen_in = copy.deepcopy(firstgen_in)
        self.firstgen_std_in = copy.deepcopy(firstgen_in)
        self.bin_width_out = bin_width_out
        self.Ex_min = Ex_min
        self.Ex_max = Ex_max
        self.Eg_min = Eg_min
        self.method = method
        self.verbose = verbose
        self.negatives_penalty = negatives_penalty # TODO: send to fit
        self.regularizer = regularizer # TODO: send to fit
        self.options = options
        self.check_input()

        # To be filled later
        self.firstgen = None
        self.firstgen_std = None
        self.Pfit = None

        self.dE_max_res = None
        self.dE_resolution = None

        # Prepare fit
        # need to calculate detector resolution for recalibration
        self.calc_resolution(Ex_array=self.firstgen_in.E0_array)
        self.recalibrate_and_cut()
        # resolution might change slightly after rebinning
        self.calc_resolution(Ex_array=self.firstgen.E0_array)


    def fit(self, use_z_correction=False):
        # Perform the actual fit
        self.send_to_fit(use_z_correction=use_z_correction)


    def check_input(self):
        """Checks input

        Raises:
            Exception: If there is a mismatch in shape or calibration
            ValueError: If input values are unphysical
        """
        # Check that firstgen and firstgen_std calibration match each other
        firstgen_in = self.firstgen_in
        firstgen_std_in = self.firstgen_std_in
        if not (firstgen_in.calibration() == firstgen_std_in.calibration()):
            raise Exception("Calibration mismatch between firstgen_in and"
                            " firstgen_std_in.")
        if not (firstgen_in.matrix.shape == firstgen_std_in.matrix.shape):
            raise Exception("Shape mismatch between firstgen_in and"
                            " firstgen_std_in")

        # Check that there are no negative counts
        assert firstgen_in.matrix.min() >= 0, "no negative counts"
        assert firstgen_std_in.matrix.min() >= 0, "no negative stddevs"

        # Check that Ex_min >= Eg_min:
        Ex_min = self.Ex_min
        Eg_min = self.Eg_min
        if Ex_min < Eg_min:
            raise ValueError("Ex_min must be >= Eg_min.")


    # def fit_rho_T(self):
    #     """Performs the fit

    #     Returns:
    #         rho, T (tuple):
    #             rho (Vector): Fit result for rho
    #             T (Vector): Fit result for T
    #     Todo:
    #         - Check rebinning/interpolation of 1Gen uncertainty
    #     """



    #     firstgen_in, firstgen_std_in, bin_width_out,
    #               Ex_min, Ex_max, Eg_min,
    #               method="Powell",
    #               verbose=True,
    #               negatives_penalty=0,
    #               regularizer=0

    def recalibrate_and_cut(self):
        """
        Set calibration & cuts [Ex_min:Ex_max, Eg_min:Eg_max] for input matrix.
        """
        Eg_min = self.Eg_min
        Ex_min = self.Ex_min
        bin_width_out = self.bin_width_out
        firstgen_in = self.firstgen_in
        firstgen_std_in = self.firstgen_std_in
        Ex_max = self.Ex_max

        E_min = min(Eg_min, Ex_min)
        # TODO: SHOULD THIS BE + +bin_width_out/2 (?)
        Eg_max = Ex_max + self.dE_max_res
        calib_fit = {"a0": E_min, "a1": bin_width_out}
        # Set up the energy arrays for recalibration of 1Gen matrix
        Eg_array = E_array_from_calibration(a0=calib_fit["a0"],
                                           a1=calib_fit["a1"],
                                           E_max=Eg_max)
        # Cut away the "top" of the Ex-array, which is too large
        # when we have a bad detector resolution
        i_Exmax = (np.abs(Eg_array-Ex_max)).argmin()+1
        Ex_array = Eg_array[:i_Exmax]

        # Rebin firstgen to calib_fit along both axes and store it in a new Matrix:
        firstgen = Matrix()
        firstgen.matrix = rebin_matrix(firstgen_in.matrix,
                                       firstgen_in.E0_array,
                                       Ex_array, rebin_axis=0)
        firstgen.matrix = rebin_matrix(firstgen.matrix,
                                       firstgen_in.E1_array,
                                       Eg_array, rebin_axis=1)
        # Set energy axes accordingly
        firstgen.E0_array = Ex_array
        firstgen.E1_array = Eg_array
        # Update 20190212: Interpolate the std matrix instead of rebinning:
        # TODO: Is this the propper way to get the uncertainties
        firstgen_std = Matrix()
        firstgen_std.matrix = interpolate_matrix_2D(firstgen_std_in.matrix,
                                             firstgen_std_in.E0_array,
                                             firstgen_std_in.E1_array,
                                             Ex_array,
                                             Eg_array
                                             )
        # Set energy axes accordingly
        firstgen_std.E0_array = Ex_array
        firstgen_std.E1_array = Eg_array
        # Verify that it got rebinned and assigned correctly:
        calib_firstgen = firstgen.calibration()
        assert (
                calib_firstgen["a00"] == calib_fit["a0"] and
                calib_firstgen["a01"] == calib_fit["a1"] and
                calib_firstgen["a10"] == calib_fit["a0"] and
                calib_firstgen["a11"] == calib_fit["a1"]
               ), "firstgen does not have correct calibration."
        calib_firstgen_std = firstgen_std.calibration()
        assert (
                calib_firstgen_std["a00"] == calib_fit["a0"] and
                calib_firstgen_std["a01"] == calib_fit["a1"] and
                calib_firstgen_std["a10"] == calib_fit["a0"] and
                calib_firstgen_std["a11"] == calib_fit["a1"]
               ), "firstgen_std does not have correct calibration."

        self.firstgen = firstgen
        self.firstgen_std = firstgen_std

    def send_to_fit(self, use_z_correction=False):

        """ Helper class just for now: sends FG to fit and get rho and T """
        Eg_min = self.Eg_min
        Ex_min = self.Ex_min
        Ex_max = self.Ex_max
        bin_width_out = self.bin_width_out

        pars_fg = {"Egmin" : Eg_min,
                   "Exmin" : Ex_min,
                   "Emax" : Ex_max}

        Enld_array = E_array_from_calibration(a0=-self.dE_max_res,
                                              a1=bin_width_out,
                                              E_max=Ex_max-Eg_min)
        Enld_array -= bin_width_out/2
        Ex_array = self.firstgen.E0_array
        Eg_array = self.firstgen.E1_array

        rho_fit, T_fit = decompose_matrix(self.firstgen.matrix,
                                          self.firstgen_std.matrix,
                                          Emid_Eg=Eg_array+bin_width_out/2,
                                          Emid_nld=Enld_array+bin_width_out/2,
                                          Emid_Ex=Ex_array+bin_width_out/2,
                                          dE_resolution = self.dE_resolution,
                                          method=self.method,
                                          options=self.options,
                                          use_z_correction=use_z_correction)

        rho = Vector(rho_fit, Enld_array)
        T = Vector(T_fit, Eg_array)

        # - rho and T shall be Vector() instances
        self.rho = rho
        self.T = T

        # save "bestfit"
        z_array = None
        if use_z_correction:
            z_array = z(Ex_array+bin_width_out/2, Eg_array+bin_width_out/2)
        else:
            z_array = np.ones((len(Ex_array), len(Eg_array)))
        Pfit = PfromRhoT(rho_fit, T_fit,
                             len(Ex_array),
                             Emid_Eg=Eg_array+bin_width_out/2,
                             Emid_nld=Enld_array+bin_width_out/2,
                             Emid_Ex=Ex_array+bin_width_out/2,
                             dE_resolution=self.dE_resolution,
                             z_array_in=z_array
                             )
        self.Pfit = Matrix(Pfit, Ex_array, Eg_array)


    def calc_resolution(self, Ex_array):
        """ Calculate Ex-dependent detector resolution (sum of sqroot)

        Args:
            Ex_array (np.array): Excitation energy bin array
        """
        # Assume constant particle resolution:
        dE_particle = DE_PARTICLE
        # Interpolate the gamma resolution linearly:
        Eg_array = Ex_array + self.bin_width_out/2
        dE_gamma = ((DE_GAMMA_8MEV - DE_GAMMA_1MEV) / (8000 - 1000)
                    * (Ex_array - 1000))  + DE_GAMMA_1MEV

        dE_resolution = np.sqrt(dE_particle**2 + dE_gamma**2)

        self.dE_max_res = np.max(dE_resolution)
        self.dE_resolution = dE_resolution
