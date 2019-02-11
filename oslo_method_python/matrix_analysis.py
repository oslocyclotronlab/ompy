# -*- coding: utf-8 -*-
"""
Class MatrixAnalysis. It is a convenience wrapper for the "core" matrix
manipulation functions of oslo_method_python.
It handles unfolding and the first-generation method on Ex-Eg matrices.

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
import matplotlib.pyplot as plt
import numpy as np
from .library import *
from .rebin import *
from .unfold import unfold
from .first_generation_method import first_generation_method

# Set seed for reproducibility:
np.random.seed(1256770)


class MatrixAnalysis():

    def __init__(self, fname_raw=None):
        # self.fname_raw = fname_raw # File name of raw spectrum

        # Allocate matrices to be filled by functions in class later:
        self.raw = Matrix()
        # If fname_raw is specified, load the file into self.raw:
        if fname_raw is not None:
            self.raw.load(fname_raw)
        self.unfolded = Matrix()
        self.firstgen = Matrix()
        # self.var_firstgen = Matrix() # variance matrix of first-generation
        # matrix
        # self.response = Matrix()  # response matrix

        # Allocate variable names for parameters to be filled by class
        # functions. This is intended for the error_propagation module.
        self.unfold_fname_resp_mat = None
        self.unfold_fname_resp_dat = None
        self.unfold_Ex_min = None
        self.unfold_Ex_max = None
        self.unfold_Eg_min = None
        self.unfold_Eg_max = None
        self.unfold_diag_cut = None
        self.unfold_verbose = None
        self.unfold_plot = None
        self.unfold_use_comptonsubtraction = None
        self.fg_Ex_max = None
        self.fg_dE_gamma = None
        self.fg_N_iterations = None
        self.fg_statistical_or_total = None
        self.fg_area_correction = None
        self.fg_fill_and_remove_negative = None

    def unfold(self, fname_resp_mat=None, fname_resp_dat=None,
               Ex_min=None, Ex_max=None, Eg_min=None, Eg_max=None,
               diag_cut=None,
               verbose=False, plot=False, use_comptonsubtraction=False,
               fill_and_remove_negative=False):
        # = Check that raw matrix is present
        if self.raw.matrix is None:
            raise Exception("Error: No raw matrix is loaded.")

        if fname_resp_mat is None or fname_resp_dat is None:
            # if self.response.matrix is None:
            if (self.unfold_fname_resp_mat is None
                or self.unfold_fname_resp_dat is None):

                raise Exception(
                    ("fname_resp_mat and/or fname_resp_dat not given, and"
                     # " no response matrix is previously loaded.")
                     " they are not previously loaded.")
                    )
            elif (isinstance(self.unfold_fname_resp_mat, str)
                  and isinstance(self.unfold_fname_resp_mat, str)):
                fname_resp_mat = self.unfold_fname_resp_mat
                fname_resp_dat = self.unfold_fname_resp_dat
            else:
                raise Exception("Something is wrong with the response matrix")


        # Copy input parameters to class parameters:
        # fname_resp_mat:
        if fname_resp_mat is not None:
            self.unfold_fname_resp_mat = fname_resp_mat
        else:
            if isinstance(self.unfold_fname_resp_mat, str):
                fname_resp_mat = self.unfold_fname_resp_mat
        # fname_resp_dat:
        if fname_resp_dat is not None:
            self.unfold_fname_resp_dat = fname_resp_dat
        else:
            if isinstance(self.unfold_fname_resp_dat, str):
                fname_resp_dat = self.unfold_fname_resp_dat
        # Ex_min:
        if Ex_min is not None:
            self.unfold_Ex_min = Ex_min
        else:
            if isinstance(self.unfold_Ex_min, str):
                Ex_min = self.unfold_Ex_min
        # Ex_max:
        if Ex_max is not None:
            self.unfold_Ex_max = Ex_max
        else:
            if isinstance(self.unfold_Ex_max, str):
                Ex_max = self.unfold_Ex_max
        # Eg_min:
        if Eg_min is not None:
            self.unfold_Eg_min = Eg_min
        else:
            if isinstance(self.unfold_Eg_min, str):
                Eg_min = self.unfold_Eg_min
        # Eg_max:
        if Eg_max is not None:
            self.unfold_Eg_max = Eg_max
        else:
            if isinstance(self.unfold_Eg_max, str):
                Eg_max = self.unfold_Eg_max
        # diag_cut:
        if diag_cut is not None:
            self.unfold_diag_cut = diag_cut
        else:
            if isinstance(self.unfold_diag_cut, str):
                diag_cut = self.unfold_diag_cut
        # verbose:
        if verbose is not None:
            self.unfold_verbose = verbose
        else:
            if isinstance(self.unfold_verbose, str):
                verbose = self.unfold_verbose
        # plot:
        if plot is not None:
            self.unfold_plot = plot
        else:
            if isinstance(self.unfold_plot, str):
                plot = self.unfold_plot
        # unfold_use_comptonsubtraction:
        if use_comptonsubtraction is not None:
            self.unfold_use_comptonsubtraction = use_comptonsubtraction
        else:
            if isinstance(self.unfold_use_comptonsubtraction, str):
                use_comptonsubtraction = self.unfold_use_comptonsubtraction
        # unfold_fill_and_remove_negative:
        if fill_and_remove_negative is not None:
            self.unfold_fill_and_remove_negative = fill_and_remove_negative
        else:
            if isinstance(self.unfold_fill_and_remove_negative, str):
                fill_and_remove_negative = self.unfold_fill_and_remove_negative

        # Call unfolding function
        self.unfolded = unfold(
            raw=self.raw, fname_resp_mat=fname_resp_mat,
            fname_resp_dat=fname_resp_dat,
            Ex_min=Ex_min, Ex_max=Ex_max, Eg_min=Eg_min,
            diag_cut=diag_cut,
            Eg_max=Eg_max, verbose=verbose, plot=plot,
            use_comptonsubtraction=use_comptonsubtraction
        )

        # Fill and remove negative:
        if fill_and_remove_negative:
            # TODO fix fill_negative function, maybe remove window_size
            # argument
            self.unfolded.fill_negative(window_size=10)
            self.unfolded.remove_negative()

    def first_generation_method(self, Ex_max, dE_gamma,
                                N_iterations=10,
                                multiplicity_estimation="statistical",
                                apply_area_correction=False,
                                verbose=False,
                                fill_and_remove_negative=False):
        # = Check that unfolded matrix is present:
        if self.unfolded.matrix is None:
            raise Exception("Error: No unfolded matrix is loaded.")

        # Copy input parameters to class parameters:
        self.fg_Ex_max = Ex_max
        self.fg_dE_gamma = dE_gamma
        self.fg_N_iterations = N_iterations
        self.fg_multiplicity_estimation = multiplicity_estimation
        self.fg_apply_area_correction = apply_area_correction
        self.fg_verbose = verbose
        self.fg_fill_and_remove_negative = fill_and_remove_negative

        # Call first generation method:
        self.firstgen = first_generation_method(matrix_in=self.unfolded,
                                                Ex_max=Ex_max,
                                                dE_gamma=dE_gamma,
                                                N_iterations=N_iterations,
                                                multiplicity_estimation=multiplicity_estimation,
                                                apply_area_correction=apply_area_correction,
                                                verbose=verbose
                                                )
        # Fill and remove negative:
        if fill_and_remove_negative:
            # TODO fix fill_negative function, maybe remove window_size
            # argument
            self.firstgen.fill_negative(window_size=10)
            self.firstgen.remove_negative()
