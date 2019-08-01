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

from . import library as lib
from . import rebin as rebin
from . import rhosig as rsg
from .rebin import rebin_2D
from .constants import *
from .matrix import Matrix
from .matrix import Vector
import copy
import numpy as np
from scipy.optimize import minimize
from typing import Optional, Any, Union, Dict, Tuple
import logging


LOG = logging.getLogger(__name__)
logging.captureWarnings(True)

# Define constants from constants.py as global vars
global DE_PARTICLE
global DE_GAMMA_1MEV
global DE_GAMMA_8MEV


class FitRhoT:
    def __init__(self,
                 firstgen: Matrix,
                 firstgen_std: Matrix,
                 bin_width_out: float,
                 Ex_min: float, Ex_max: float,
                 Eg_min: float,
                 method: str = "Powell",
                 verbose: bool = True,
                 negatives_penalty: float = 0,
                 regularizer: float = 0,
                 options={'disp': True}):
        """
        Class Wrapper for fit of the firstgen spectrum to the product of transmission coeff T and level density rho

        Args:
            firstgen: The first-generation matrix.
            firstgen_std: The standard deviations in the first-gen matrix
            bin_width_out: Bin-width of the output rho and T
            Ex_min: Minimum excitation energy for the fit
            Ex_max: Maximum excitation energy for the fit
            Eg_min: Minimum gamma-ray energy for the fit
                            (Eg_max is equal to Ex_max)
            method: Method to use for minimization.
                          Must be one of the methods available
                          in scipy.optimize.minimize.
            verbose: Whether to print information
                            from the fitting routine.
            negatives_penalty: For Chi2 # TODO: send to fit
            regularizer: For Chi2 # TODO: send to fit
            options: Options to the minimization routine. See docs of
                scipy.optimize.minimize

        TODO:
            - Implement negatives_penalty and regularizer
            - If firstgen.std=None, it could call a function to estimate
                  the standard deviation using the rhosigchi techniques.
        """
        # Protect the input matrix:
        matrix: Matrix = copy.deepcopy(firstgen)
        std: Matrix = copy.deepcopy(firstgen_std)
        self.bin_width_out = bin_width_out
        self.Ex_min = Ex_min
        self.Ex_max = Ex_max
        self.Eg_min = Eg_min
        self.method = method
        self.verbose = verbose
        self.negatives_penalty = negatives_penalty  # TODO: send to fit
        self.regularizer = regularizer  # TODO: send to fit
        self.options = options
        self.check_input(matrix, std, Ex_min, Eg_min)

        # To be filled later
        self.firstgen: Optional[Matrix] = None
        self.firstgen_std: Optional[Matrix] = None
        self.Pfit = None

        self.dE_max_res = None
        self.dE_resolution = None

        # Prepare fit
        # need to calculate detector resolution for recalibration
        self.calc_resolution(Ex_array=firstgen.Ex)
        self.firstgen, self.firstgen_std = self.recalibrate_and_cut(matrix, std)
        return
        # resolution might change slightly after rebinning
        self.calc_resolution(Ex_array=self.firstgen.Ex)

    def check_input(self, matrix: Matrix, std: Matrix, Ex_min: float, Eg_min: float):
        """Checks input

        Raises:
            ValueError: If input values are incompatible or are unphysical
        """
        # Check that firstgen and firstgen_std calibration match each other
        if not matrix.calibration() == std.calibration():
            raise ValueError("Calibration mismatch between matrix and std.")
        if not matrix.values.shape == std.values.shape:
            raise ValueError("Shape mismatch between matrix and std")

        # Check that there are no negative counts
        assert matrix.values.min() >= 0, "no negative counts allowed"
        assert std.values.min() >= 0, "no negative stddevs allowed"

        # Check that Ex_min >= Eg_min:
        if Ex_min < Eg_min:
            raise ValueError("Ex_min must be >= Eg_min.")

    def recalibrate_and_cut(self, matrix: Matrix, std: Matrix) -> Tuple[Matrix, Matrix]:
        """
        Set calibration & cuts [Ex_min:Ex_max, Eg_min:Eg_max] for input matrix.
        """
        E_min = min(self.Eg_min, self.Ex_min)
        dE_resolution = lib.diagonal_resolution(matrix.Ex)
        Eg_max = self.Ex_max + dE_resolution.max()
        a0 = E_min
        a1 = self.bin_width_out
        # Set up the energy arrays for recalibration of 1Gen matrix
        Eg = lib.E_array_from_calibration(a0=a0, a1=a1, E_max=Eg_max)
        # Cut away the "top" of the Ex-array, which is too large
        # when we have a bad detector resolution
        i_Exmax = (np.abs(Eg-self.Ex_max)).argmin()+1
        Ex = Eg[:i_Exmax]

        # Rebin firstgen to calib_fit along both axes
        recalibrated = matrix.rebin('Ex', Ex, inplace=False)
        assert recalibrated is not None
        recalibrated.rebin('Eg', Eg)

        # TODO: Is this the propper way to get the uncertainties
        interpolated_std = Matrix(Ex=Ex, Eg=Eg)
        interpolated_std.values = lib.interpolate_matrix_2D(
            std.values, std.Ex, std.Eg,
            Ex, Eg
        )
        # Verify that it got rebinned and assigned correctly:
        calib = recalibrated.calibration()
        assert (
            calib["a00"] == a0 and
            calib["a01"] == a1 and
            calib["a10"] == a0 and
            calib["a11"] == a1
        ), "recalibrated does not have correct calibration."
        assert calib == interpolated_std.calibration(), "interpolated_std does not have correct calibration."

        return recalibrated, interpolated_std

    def fit(self, p0: Optional[np.ndarray] = None,
            use_z: Optional[Union[bool, np.ndarray]] = False,
            spin_dist_par: Optional[Dict[str, Any]] = None):
        """Helper class just for now: sends FG to fit and get rho and T

        Args:
            p0:
                Initial guess for nld and transmission coefficient, given as a
                1D array (rho0, T0). Defaults to the choice desribed in
                Schiller2000
            use_z:
                If `bool`, it either uses the additional "z-factor" in the
                decomposition, which is a spin-dependent factor *potentially*
                missing in the previous implementations of the Oslo Method. By
                default, it will be ignored. If `True`, then an array will be
                created matching the specified spin-parity distribution
                `spin_dist_par`. If the type is `np.ndarray(shape=(Nbins_Ex,
                Nbins_T))`, it will use the specified array directly.
            spin_dist_par:
                Dict of spin-parity paramters to create the `z-factor`, see
                `use_z`.
        """
        Eg_min = self.Eg_min
        Ex_max = self.Ex_max
        bin_width_out = self.bin_width_out

        Enld_array = lib.E_array_from_calibration(a0=-self.dE_max_res,
                                                  a1=bin_width_out,
                                                  E_max=Ex_max-Eg_min)
        Enld_array -= bin_width_out/2
        Ex_array = self.firstgen.Ex
        Eg_array = self.firstgen.Eg

        result = rsg.decompose_matrix(self.firstgen.values,
                                      self.firstgen_std.values,
                                      Emid_Eg=Eg_array+bin_width_out/2,
                                      Emid_nld=Enld_array+bin_width_out/2,
                                      Emid_Ex=Ex_array+bin_width_out/2,
                                      dE_resolution=self.dE_resolution,
                                      p0=p0,
                                      method=self.method,
                                      options=self.options,
                                      use_z=use_z,
                                      spin_dist_par=spin_dist_par
                                      )
        if use_z is False:
            rho_fit, T_fit = result
        else:
            rho_fit, T_fit, z_array = result

        rho = Vector(rho_fit, Enld_array)
        T = Vector(T_fit, Eg_array)

        # - rho and T shall be Vector() instances
        self.rho = rho
        self.T = T

        # save "bestfit"
        z_array = None
        if use_z:
            z_array = z(Ex_array+bin_width_out/2, Eg_array+bin_width_out/2,
                        # TODO implement custom spin_dist_par here:
                        spin_dist_par=spin_dist_par)
        else:
            z_array = np.ones((len(Ex_array), len(Eg_array)))
        Pfit = rsg.PfromRhoT(rho_fit, T_fit,
                             len(Ex_array),
                             Emid_Eg=Eg_array+bin_width_out/2,
                             Emid_nld=Enld_array+bin_width_out/2,
                             Emid_Ex=Ex_array+bin_width_out/2,
                             dE_resolution=self.dE_resolution,
                             z_array_in=z_array)
        self.Pfit = Matrix(values=Pfit, Ex=Ex_array, Eg=Eg_array)

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
                    * (Ex_array - 1000)) + DE_GAMMA_1MEV

        dE_resolution = np.sqrt(dE_particle**2 + dE_gamma**2)

        self.dE_max_res = np.max(dE_resolution)
        self.dE_resolution = dE_resolution
        return dE_resolution
