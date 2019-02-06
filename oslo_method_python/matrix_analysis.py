# -*- coding: utf-8 -*-
"""
Class matrix_analysis(), the "core" matrix manipulation module of pyma.
It handles unfolding and the first-generation method on Ex-Eg matrices.

---

This is a python implementation of the Oslo method.
It handles two-dimensional matrices of event count spectra, and
implements detector response unfolding, first generation method
and other manipulation of the spectra.

It is heavily inspired by MAMA, written by Magne Guttormsen and others,
available at https://github.com/oslocyclotronlab/oslo-method-software

Copyright (C) 2018 J{\o}rgen Eriksson Midtb{\o}
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

    def __init__(self):
        # self.fname_raw = fname_raw # File name of raw spectrum

        # Allocate matrices to be filled by functions in class later:
        self.raw = Matrix()
        self.unfolded = Matrix()
        self.firstgen = Matrix()
        # self.var_firstgen = Matrix() # variance matrix of first-generation
        # matrix
        self.response = Matrix()  # response matrix

    def unfold(self, fname_resp_mat=None, fname_resp_dat=None, FWHM_factor=10,
               Ex_min=None, Ex_max=None, Eg_min=None, Eg_max=None,
               diag_cut=None,
               verbose=False, plot=False, use_comptonsubtraction=False):
        # = Check that raw matrix is present
        if self.raw.matrix is None:
            raise Exception("Error: No raw matrix is loaded.")

        if fname_resp_mat is None or fname_resp_dat is None:
            if self.response.matrix is None:
                raise Exception(
                    ("fname_resp_mat and/or fname_resp_dat not given, and no"
                     " response matrix is previously loaded.")
                    )

        # Update 2019: Moved unfold function to separate file, so this is just
        # a wrapper.
        self.unfolded = unfold(
            raw=self.raw, fname_resp_mat=fname_resp_mat,
            fname_resp_dat=fname_resp_dat,
            FWHM_factor=FWHM_factor,
            Ex_min=Ex_min, Ex_max=Ex_max, Eg_min=Eg_min,
            diag_cut=diag_cut,
            Eg_max=Eg_max, verbose=verbose, plot=plot,
            use_comptonsubtraction=use_comptonsubtraction
        )

    def first_generation_method(self, Ex_max, dE_gamma,
                                N_iterations=10,
                                multiplicity_estimation="statistical",
                                apply_area_correction=False,
                                verbose=False):
        # = Check that unfolded matrix is present:
        if self.unfolded.matrix is None:
            raise Exception("Error: No unfolded matrix is loaded.")

        # Call first generation method:
        self.firstgen = first_generation_method(matrix_in=self.unfolded,
                                                Ex_max=Ex_max,
                                                dE_gamma=dE_gamma,
                                                N_iterations=N_iterations,
                                                multiplicity_estimation=multiplicity_estimation,
                                                apply_area_correction=apply_area_correction,
                                                verbose=verbose
                                                )
