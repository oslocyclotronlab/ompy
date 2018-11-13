"""
This is the python implementation of the Oslo method.
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
from _matrix_analysis import matrix_analysis
from _library import *
import _error_propagation

class oslo_method():
    def __init__(self, fname_resp_mat=None, fname_resp_dat=None):
        # self.fname_raw = fname_raw # File name of raw spectrum

        # Create an instance of matrix_analysis() which will be the
        # main pipeline for the unfolding and first-generation method
        self.base_analysis = matrix_analysis(fname_resp_mat, fname_resp_dat)

        # Set up shortcut variable names for the raw, unfolded and firstgen
        # matrix objects inside base_analysis.
        self.raw = self.base_analysis.raw
        self.unfolded = self.base_analysis.unfolded
        self.firstgen = self.base_analysis.firstgen

        self.var_firstgen = matrix() # variance matrix of first-generation matrix
        # self.response = matrix() # response matrix -- unsure if this should be a global variable


        # Allocate other variables and settings:
        self.fname_resp_mat = fname_resp_mat
        self.fname_resp_dat = fname_resp_dat
        self.N_Exbins_fg = None
        self.Ex_max_fg = None
        self.dEg_fg = None
        self.error_propagation = None # Placeholder for error_propagation instance

        return None

    def unfold(self, FWHM_factor=10, Ex_min="default", Ex_max="default", Eg_min="default", Eg_max="default", verbose=False, plot=False):
        # Call unfold method on the base_analysis instance:
        self.base_analysis.unfold(FWHM_factor, Ex_min, Ex_max, Eg_min, Eg_max, verbose, plot)
        # Update the top-level copy of unfolded matrix:
        self.unfolded = self.base_analysis.unfolded

        return None

    def first_generation_method(self, N_iterations=10, statistical_or_total=1):
        # Call first_generation_method() on the base_analysis instance:
        self.base_analysis.first_generation_method(N_iterations, statistical_or_total)
        # Update the top-level copy of firstgen matrix:
        self.firstgen = self.base_analysis.firstgen

        return None

    def setup_error_propagation(self, folder="pyma_ensemble_folder", randomness="poisson", seed=None):
        """
        Set up error propagation by creating an instance of class _error_propagation.error_propagation
        """
        self.error_propagation = _error_propagation.error_propagation(base_analysis_instance=self.base_analysis, folder=folder, randomness=randomness, seed=seed)

        return None

    def propagate_errors(self, N_ensemble_members, purge_files=True): 
        if self.error_propagation is None:
            raise Exception("Method setup_error_propagation() must be called before propagate_errors()")

        self.var_firstgen = self.error_propagation.generate_ensemble(N_ensemble_members=N_ensemble_members, purge_files=purge_files)


        return None