"""
Class fit(). It handles the fitting of one-dimensional
functions to the first-generation matrix. 

---

This is pyma, the python implementation of the Oslo method.
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


import numpy as np
import pyma_lib as pml
import pyma_matrix as pmmat


class fit:
    def __init__(self, firstgen, var_firstgen=pmmat.matrix()):
        self.firstgen = firstgen 
        self.var_firstgen = var_firstgen
        return None
        
    def fit(self, Eg_min, Ex_min, Ex_max, estimate_variance_matrix=False):
        """
        Fit transmission coefficient + level density function to
        the first-generation matrix. This code is famously known as
        "rhosigchi" in MAMA. Since it is quite tricky to get the fit
        right, pyma actually runs a compiled version of the original
        rhosicghi Fortran code.
        """

        # = Check that first generation matrix is present 
        if self.firstgen.matrix is None:
            raise Exception("Error: No first generation matrix is loaded.")

        if self.var_firstgen.matrix is None:
            if not estimate_variance_matrix:
                raise Exception("Error: No first generation variance matrix is loaded, but estimate_variance_matrix is set to False.")
            print("The variance will be estimated (with a very uncertain method).", flush=True)

        rho, T = None, None
        if estimate_variance_matrix:
            raise Exception("Rhosigchi with the original variance estimate is not implemented yet.")

        else: # Use variance matrix estimated by Monte Carlo
            import rhosigchi_f2py_importvar as rsc 

            # Check that dimensions meet the requirement. Rhosigchi needs dimension 512x512.
            fg_matrix_orig = self.firstgen.matrix
            Eg_array_orig = self.firstgen.Eg_array
            Ex_array_orig = self.firstgen.Ex_array
            dim_rsc = 512
                
            # Start with Eg because this probably requires the most rebinning: 
            if fg_matrix_orig.shape[1] < dim_rsc:
                # Concatenate array with an array of zeros:
                fg_matrix_dimcorr_Eg = np.concatenate((fg_matrix_orig, np.zeros((fg_matrix_orig.shape[0], dim_rsc-fg_matrix_orig.shape[1]))), axis=1)
                # Make a corresponding Eg array:
                Eg_array_dimcorr = pml.E_array_from_calibration(self.firstgen.calibration["a0x"],self.firstgen.calibration["a1x"],dim_rsc)
            elif fg_matrix_orig.shape[1] > dim_rsc: 
                # Rebin down to correct number of bins:
                fg_matrix_dimcorr_Eg, Eg_array_dimcorr = pml.rebin(fg_matrix_orig, Eg_array_orig, dim_rsc, rebin_axis=1)
            else:
                # Copy original matrix, it's already the right dimension
                fg_matrix_dimcorr_Eg = fg_matrix_orig
                Eg_array_dimcorr = Eg_array_orig

            # Then do the same with Ex:
            if fg_matrix_dimcorr_Eg.shape[0] < dim_rsc:
                # Concatenate array with an array of zeros:
                fg_matrix_dimcorr_Ex = np.concatenate((fg_matrix_dimcorr_Eg, np.zeros((dim_rsc-fg_matrix_dimcorr_Eg.shape[0],fg_matrix_dimcorr_Eg.shape[1]))), axis=0)
                # Make a corresponding Eg array:
                Ex_array_dimcorr = pml.E_array_from_calibration(self.firstgen.calibration["a0y"],self.firstgen.calibration["a1y"],dim_rsc)
            elif fg_matrix_dimcorr_Eg.shape[0] > dim_rsc: 
                # Rebin down to correct number of bins:
                fg_matrix_dimcorr_Ex, Ex_array_dimcorr = pml.rebin(fg_matrix_dimcorr_Eg, Ex_array_orig, dim_rsc, rebin_axis=0)
            else:
                # Copy original matrix, it's already the right dimension
                fg_matrix_dimcorr_Ex = fg_matrix_dimcorr_Eg
                Ex_array_dimcorr = Eg_array_orig

            # Update variable names
            fg_matrix = fg_matrix_dimcorr_Ex
            Ex_array = Ex_array_dimcorr
            Eg_array = Eg_array_dimcorr
                

            print(Ex_array, flush=True)


            var_fg_matrix = np.sqrt(fg_matrix)


            calibration = np.array([Eg_array[0], Eg_array[1]-Eg_array[0], Ex_array[0], Ex_array[1]-Ex_array[0]])
            rho, T = rsc.rhosigchi(fg_matrix,var_fg_matrix,calibration,Eg_min,Ex_min,Ex_max)




        return rho, T, Ex_array, Eg_array