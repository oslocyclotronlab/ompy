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
                                N_iterations=10, statistical_or_total=1,
                                area_correction=True):
        """
        Function implementing the first generation method from Guttormsen et
        al. (NIM 1987).

        The code is heavily influenced by the original implementation by Magne
        in MAMA. Mainly written autumn 2016 at MSU.
        """

        # = Check that unfolded matrix is present:
        if self.unfolded.matrix is None:
            raise Exception("Error: No unfolded matrix is loaded.")

        # Call first generation method:
        firstgen = first_generation_method( ... )

        # 20190204: Remove most (all?) of what happens below!


        # Rename variables for local use:
        unfolded_matrix = self.unfolded.matrix
        Ex_array_mat = self.unfolded.E0_array
        Egamma_array = self.unfolded.E1_array

        # TODO option statistical_or_total=2 (total) does not work

        Ny = len(unfolded_matrix[:, 0])
        Nx = len(unfolded_matrix[0, :])
        calib_in = self.unfolded.calibration()
        bx = calib_in["a10"]
        ax = calib_in["a11"]
        by = calib_in["a00"]
        ay = calib_in["a01"]

        ThresSta = 430.0
        # AreaCorr = 1
        ThresTot = 200.000000
        ThresRatio = 0.3000000
        # ExH = 7520.00000
        ExEntry0s = 300.000000
        ExEntry0t = 0.00000000
        apply_area_correction = area_correction  # TODO rename variable to area_correction throughout

        # Ex_max = 7500 # keV - maximum excitation energy
        # Ex_min = 300 # keV - minimal excitation energy, effectively moving
        # the ground-state energy up because we cannot resolve the low-energy
        # yrast gamma lines. This is weighed up by also using an effective 
        # multiplicity which is lower than the real one, again not considering
        # the low-energy yrast gammas.
        # dE_gamma = 300 # keV - allow gamma energy to exceed excitation energy
        # by this much, to account for experimental resolution
        # Ex_binsize = 40 # keV - bin size that we want on y axis
        # N_Exbins = 120 # Number of excitation energy bins (NB! It will only
        # rebin in whole multiples, so a slight change in N_Exbins might only
        # result in getting some more empty bins on top.)
        # N_Exbins_original = (Ex_max+dE_gamma)/ay # The number of bins between
        # 0 and Ex_max + dE_gamma in the original matrix
        # grouping = int(np.ceil(len(y_array[np.logical_and(0 < y_array,
        # y_array < Ex_max + dE_gamma)])/N_Exbins)) # The integer number of
        # bins that need to be grouped to have approximately N_Exbins bins
        # between Ex_min and Ex_max after compression (rounded up)

        # Make arrays of Ex and Egamma axis values
        # Ex_array = np.linspace(by, Ex_max + dE_gamma, N_Exbins)
        # Egamma_array = np.linspace(0,Nx-1,Nx)*ax + bx # Range of Egamma
        # values # Update: This is passed into the function.

        # matrix_ex_compressed, Ex_array
        #    = rebin(unfolded_matrix[0:int((Ex_max+dE_gamma)/Ex_array_mat.max()*Ny),:], Ex_array_mat[0:int((Ex_max+dE_gamma)/Ex_array_mat.max()*Ny)], N_Exbins, rebin_axis = 0) # This seems crazy. Does it cut away anything at all?
        # HACK: Checking if the compression along Ex is really necessary
        # (shouldn't it be done outside of firstgen method anyway?)
        matrix_ex_compressed = unfolded_matrix
        Ex_array = Ex_array_mat

        # if N_Exbins != Ny:
        # Compress matrix along Ex
        #matrix_ex_compressed = matrix[0:int(N_Exbins*grouping),:].reshape(N_Exbins, grouping, Nx).sum(axis=1)
        # Update 20180828: Trying other rebin functions
        # matrix_ex_compressed = np.zeros((N_Exbins, Nx))
        # for i in range(Nx):
        #   # This is too slow.
        #   # TODO understand if the int((Ex_max + dE_gamma etc...)) stuff is necessary.
        #   # matrix_ex_compressed[:,i] = rebin_by_arrays_1d(matrix[0:int((Ex_max+dE_gamma)/Ex_array_mat.max()*Ny),i], Ex_array_mat[0:int((Ex_max+dE_gamma)/Ex_array_mat.max()*Ny)], Ex_array)
        #   print("i=",i,flush=True)
        #   # print("Ex_array_mat.shape =", Ex_array_mat.shape, flush=True)
        #   # print("matrix.shape =", matrix.shape, flush=True)
        #   matrix_ex_compressed[:,i] = rebin_by_arrays_1d(matrix[:,i], Ex_array_mat, Ex_array)
        # # matrix_ex_compressed = rebin_and_shift_2D_memoryguard(matrix[0:int((Ex_max+dE_gamma)/Ex_array_mat.max()*Ny),:], N_Exbins, rebin_axis = 0) # This seems crazy. Does it cut away anything at all?
        # else:
        # matrix_ex_compressed = matrix

        # print Ny, N_Exbins, N_Exbins_original
        # plt.pcolormesh(Egamma_array, Ex_array, matrix_ex_compressed, norm=LogNorm(vmin=0.001, vmax=matrix_ex_compressed.max()))
        # plt.matshow(matrix_ex_compressed)
        # plt.colorbar()
        # plt.show()

        # Remove counts in matrix for Ex higher than Ex_max:
        matrix_ex_compressed[Ex_array > Ex_max, :] = 0
        # plt.matshow(matrix_ex_compressed)
        # plt.colorbar()
        # plt.show()

        # ==== Calculate multiplicities: ====

        # Setup meshgrids for making boolean indexing arrays
        Egamma_mesh, Ex_mesh = np.meshgrid(Egamma_array, Ex_array)
        Egamma_max = Ex_array + dE_gamma  # Maximal Egamma value for each Ex bin
        Egamma_max_grid = np.meshgrid(np.ones(Nx), Egamma_max)[1]
        if statistical_or_total == 1:
            # Statistical multiplicity calculation (i.e. trying to use
            # statistical/continuum region only)
            # The sliding lower limit for Egamma integral - sliding between
            # ThresTot and ThresSta.
            slide = np.minimum(np.maximum(
                ThresRatio * Ex_mesh, ThresTot), ThresSta)
            # plt.figure(5)
            # plt.plot(slide[:,0])
            # plt.show()
            # sys.exit(0)
            # good_indices = np.where(np.logical_and(slide < Egamma_mesh, Egamma_mesh < Egamma_max_grid) , True, False)
            matrix_ex_compressed_cut = np.where(np.logical_and(
                slide < Egamma_mesh, Egamma_mesh < Egamma_max_grid), matrix_ex_compressed, 0)
        elif statistical_or_total == 2:
            # Total multiplicity calculation
            # good_indices = np.where(Egamma_mesh < Egamma_max_grid, True, False)
            matrix_ex_compressed_cut = np.where(
                Egamma_mesh < Egamma_max_grid, matrix_ex_compressed, 0)
        # for i in range(len(good_indices[:,0])):
        # print len(good_indices[i,good_indices[i,:]]) # OK, it actually works.

        # Cut away counts higher than Egamma = Ex + dE_gamma
        # matrix_ex_compressed_cut = np.where(good_indices, matrix_ex_compressed, 0)
        # plt.figure(1)
        # plt.pcolormesh(Egamma_array, Ex_array, matrix_ex_compressed_cut, norm=LogNorm(vmin=0.01, vmax=matrix_ex_compressed.max()))
        # plt.show()
        # sys.exit(0)

        # Calculate average multiplicity for each Ex channel
        area_matrix_ex_compressed_cut = np.sum(
            matrix_ex_compressed_cut, axis=1)
        Egamma_average = div0(np.sum(
            Egamma_mesh * matrix_ex_compressed_cut, axis=1),
            area_matrix_ex_compressed_cut)
        if statistical_or_total == 1:
            # Statistical multiplicity - use the effective Ex0 value
            multiplicity = div0(
                Ex_array - np.maximum(np.minimum(
                            Ex_array - 200, ExEntry0s), 0), Egamma_average)
        elif statistical_or_total == 2:
            # Total multiplicity - use actual Ex0 = 0
            multiplicity = div0(Ex_array, Egamma_average)

        # plt.figure(2)
        # plt.step(Ex_array, multiplicity) # This rises like a straight line from 0 to about 3-4 - seems very right!
        # plt.show()
        # sys.exit(0)

        # Set up dummy first-generation matrix to start iterations, made of
        # normalized boxes:
        N_Exbins = Ny # TODO clean up and rename energy arrays, Nbins vars, etc
        # N_Exbins = np.argmin(np.abs(Ex_array - (Ex_max+dE_gamma)))  # TODO reimplement the energy arrays in a more consistent way
        H = np.zeros((N_Exbins, Nx))
        for i in range(N_Exbins):
            Ni = len(Egamma_array[Egamma_array < Ex_array[i] + dE_gamma])
            # print("Ni =", Ni, flush=True)
            H[i, Egamma_array < Ex_array[i] + dE_gamma] = 1 / max(Ni, 1)
        # print np.sum(H0, axis=1) # Seems to work!

        # Set up normalization matrix N
        # Get total number of counts in each Ex bin
        area = np.sum(matrix_ex_compressed_cut, axis=1)
        # plt.plot(Ex_array, area)
        # plt.show()

        # Copy the array N_Exbins times down to make a square matrix
        # area_grid = np.tile(area, (N_Exbins, 1))
        # print area_grid.shape
        # multiplicity_grid = np.tile(multiplicity, (N_Exbins, 1))
        # print multiplicity_grid.shape
        # The transpose gives the right result. Haven't twisted my head around
        # exactly why.
        # normalization_matrix = div0((np.transpose(multiplicity_grid)
                                     # * area_grid),
                                    # (multiplicity_grid
                                     # * np.transpose(area_grid))).T
        normalization_matrix_manual = np.zeros((N_Exbins, N_Exbins))
        for i in range(N_Exbins):
            for j in range(N_Exbins):
                normalization_matrix_manual[i, j] = (multiplicity[i]*area[j]
                                                     / (multiplicity[j]
                                                        * area[i]))
        normalization_matrix = normalization_matrix_manual
        normalization_matrix[np.isnan(normalization_matrix)] = 0


        # Set up compression parameters for Egamma axis to be used by H below:
        # Get the maximal allowed gamma energy (need to make H square, thus
        # Egamma <= Ex + dE_gamma, since that's the maximal Ex channel in the
        # compressed matrix)
        # i_Egamma_max = np.where(Egamma_array > Ex_max + dE_gamma)[0][0]
        i_Egamma_max = np.argmin(np.abs(Egamma_array - Ex_max + dE_gamma))
        # print i_Egamma_max, Egamma_array[i_Egamma_max], N_Exbins, int(i_Egamma_max/N_Exbins)
        # i_Egamma_max = i_Egamma_max + N_Exbins - i_Egamma_max%N_Exbins # Make sure the number of indices is a whole multiple of N_Exbins (rounded up)
        # print i_Egamma_max
        grouping_Egamma = int(np.ceil(i_Egamma_max / N_Exbins))
        # Egamma_array_compressed = Egamma_array[0:i_Egamma_max]*grouping_Egamma
        # Egamma_array_compressed = Ex_array



        # Declare variables which will define the limits for the diff spectrum
        # colorbar (for plotting purposes)
        vmin_spec = -200
        vmax_spec = 200
        vmin_diff = -100
        vmax_diff = 100

        # Prepare weight function based on Fermi gas approximation:
        a_f = 16  # 1/MeV
        n_f = 4.2  # Exponent for E_gamma
        # Make the weight array by the formula W(Ex,Eg) = Eg^n_f / (Ex-Eg)^2 *
        # exp(2*sqrt(a*(Ex-Eg)))
        Ex_mesh, Eg_mesh = np.meshgrid(Ex_array, Ex_array, indexing="ij")
        # Make mesh of final energies:
        Ef_mesh = Ex_mesh - Eg_mesh
        # Set everything above Ex=Eg diagonal to zero so the weights also
        # become zero
        Ef_mesh[Ef_mesh < 0] = 0
        # Calculate weights. Remember that energies are in keV, while a is in
        # 1/MeV, so we correct in the exponent:
        W_old = np.where(Eg_mesh > 0, np.power(
            Eg_mesh, n_f) / np.power(Ef_mesh, 2) * np.exp(2 * np.sqrt(a_f * Ef_mesh / 1000)), 0)
        W_old = div0(W_old, W_old.sum(axis=1).reshape(N_Exbins, 1))

        dEg = 1000
        mask_W = make_mask(Ex_array, Ex_array, Ex_array[0], Ex_array[
                           0] + dEg, Ex_array[-1], Ex_array[-1] + dEg)

        # Perform the iterative subtraction:
        for iteration in range(N_iterations):
            # convergence_criterion = 1
            # max_diff = 100
            # while max_diff > convergence_criterion:
            # Store H from previous iteration to compare at the end
            H_old = H
            # Compress the H matrix along gamma axis to make it square and facilitate conversion to excitation energy
            # H_compressed = H[:,0:i_Egamma_max].reshape(N_Exbins, N_Exbins, grouping_Egamma).sum(axis=2)
            # H_compressed, Egamma_array_compressed = rebin(
                # H[:, 0:i_Egamma_max], Egamma_array[0:i_Egamma_max], N_Exbins, rebin_axis=1)
            # Updated 20190130 to rebin_matrix() with energy calibratiON.
            # TODO cut to i_Egamma_max once and for all above instead of here?
            H_compressed = rebin_matrix(
                H[:, 0:i_Egamma_max], Egamma_array[0:i_Egamma_max], Ex_array,
                rebin_axis=1)


            # if iteration == 0:
            # Don't use H as weights for first iteration.
            # W = W_old
            # else:
            if True:
                # Convert first-generation spectra H into weights W
                W = np.zeros((N_Exbins, N_Exbins))
                for i in range(0, N_Exbins):
                    # print H_compressed[i,i:0:-1].shape
                    W[i, 0:i] = H_compressed[i, i:0:-1]
                    # TODO Consider implementing something like Massage(), but try to understand if it is necessary first.

            # Prevent oscillations, following MAMA:
            if iteration > 4:
                W = 0.7 * W + 0.3 * W_old

            # Remove negative weights
            W[W < 0] = 0
            # Apply mask
            # W = W * mask_W
            # Normalize each Ex channel to unity
            # W = np.where(np.invert(np.isnan(W/W.sum(axis=1).astype(float))),  W/W.sum(axis=1).astype(float), 0)
            # Remove Inf and NaN
            W = div0(W, W.sum(axis=1).reshape(N_Exbins, 1))
            # Store for next iteration:
            W_old = np.copy(W)
            # W = np.nan_to_num(W)


            # Calculate product of normalization matrix, weight matrix and raw
            # count matrix
            # Matrix of weighted sum of spectra below
            G = np.dot((normalization_matrix * W), matrix_ex_compressed)

            # Apply area correction
            if apply_area_correction:
                # Setup meshgrids for making boolean indexing arrays
                # Egamma_mesh_compressed, Ex_mesh_compressed = np.meshgrid(Egamma_array_compressed, Ex_array)
                # Egamma_max = Ex_array + dE_gamma # Maximal Egamma value for each Ex bin
                # Egamma_max_grid_compressed = np.meshgrid(np.ones(N_Exbins), Egamma_max)[1]
                # print "Egamma_mesh_compressed, Egamma_max, Egamma_max_grid"
                # print Egamma_mesh_compressed.shape, Egamma_max.shape,
                # Egamma_max_grid.shape
                if statistical_or_total == 1:
                    # Statistical multiplicity calculation (i.e. trying to use statistical/continuum region only)
                    # slide_compressed = np.minimum( np.maximum(ThresRatio*Ex_mesh_compressed, ThresTot), ThresSta ) # The sliding lower limit for Egamma integral - sliding between ThresTot and ThresSta.

                    # good_indices_G = np.where(np.logical_and(slide_compressed < Egamma_mesh_compressed, Egamma_mesh_compressed < Egamma_max_grid) , True, False)
                    G_area = np.where(np.logical_and(
                        slide < Egamma_mesh, Egamma_mesh < Egamma_max_grid), G, 0).sum(axis=1)
                elif statistical_or_total == 2:
                    # Total multiplicity calculation
                    # good_indices_G = np.where(Egamma_mesh_compressed < Egamma_max_grid, True, False)
                    # G_area = np.where(Egamma_mesh_compressed < Egamma_max_grid, G, 0).sum(axis=1)
                    G_area = np.where(
                        Egamma_mesh < Egamma_max_grid, G, 0).sum(axis=1)

                alpha = np.where(G_area > 0, (1 - div0(1, multiplicity))
                                 * div0(area_matrix_ex_compressed_cut, G_area), 1)
                alpha[alpha < 0.85] = 0.85
                alpha[alpha > 1.15] = 1.15

            else:
                alpha = np.ones(N_Exbins)

            # The actual subtraction
            H = matrix_ex_compressed - alpha.reshape((len(alpha), 1)) * G


            # Check convergence
            max_diff = np.max(np.abs(H - H_old))
            print("iteration =", iteration, "max_diff =", max_diff, flush=True)

        # Remove negative counts
        H[H < 0] = 0

        # Update internal variables and return True upon completion
        # return H, H-H_old, Ex_array, Egamma_array

        print("H.shape =", H.shape)
        print("Ex_array.shape =", Ex_array.shape)
        print("Egamma_array.shape =", Egamma_array.shape, flush=True)

        self.firstgen = Matrix(H, Ex_array, Egamma_array)
        return True

    # def fit(self, Eg_min, Ex_min, Ex_max, estimate_variance_matrix=False):
    #     """
    #     Fit transmission coefficient + level density function to
    #     the first-generation matrix. This code is famously known as
    #     "rhosigchi" in MAMA. Since it is quite tricky to get the fit
    #     right, pyma actually runs a compiled version of the original
    #     rhosicghi Fortran code.
    #     """

    #     # = Check that first generation matrix is present
    #     if self.firstgen.matrix is None:
    #         raise Exception("Error: No first generation matrix is loaded.")

    #     if self.var_firstgen.matrix is None:
    #         if not estimate_variance_matrix:
    #             raise Exception(
    #                 "Error: No first generation variance matrix is loaded, but estimate_variance_matrix is set to False.")
    #         print(
    #             "The variance will be estimated (with a very uncertain method).", flush=True)

    #     rho, T = None, None
    #     # Which verison of rhosigchi should be used? The original
    #     # which estimates the variance matrix, or the modified
    #     # one where the variance matrix is imported?
    #     if estimate_variance_matrix:  # Estimate the variance matrix internally in rhosigchi
    #         import rhosigchi_f2py_origvar as rsc

    #     # Use user-supplied variance matrix (typically from error propagated
    #     # ensemble)
    #     else:
    #         import rhosigchi_f2py_importvar as rsc

    #         # Check that dimensions meet the requirement. Rhosigchi needs
    #         # dimension 512x512.
    #         fg_matrix_orig = self.firstgen.matrix
    #         Eg_array_orig = self.firstgen.Eg_array
    #         Ex_array_orig = self.firstgen.Ex_array
    #         dim_rsc = 512

    #         # Start with Eg because this probably requires the most rebinning:
    #         if fg_matrix_orig.shape[1] < dim_rsc:
    #             # Concatenate array with an array of zeros:
    #             fg_matrix_dimcorr_Eg = np.concatenate((fg_matrix_orig, np.zeros(
    #                 (fg_matrix_orig.shape[0], dim_rsc - fg_matrix_orig.shape[1]))), axis=1)
    #             # Make a corresponding Eg array:
    #             Eg_array_dimcorr = E_array_from_calibration(
    #                 self.firstgen.calibration["a0x"], self.firstgen.calibration["a1x"], dim_rsc)
    #         elif fg_matrix_orig.shape[1] > dim_rsc:
    #             # Rebin down to correct number of bins:
    #             fg_matrix_dimcorr_Eg, Eg_array_dimcorr = rebin(
    #                 fg_matrix_orig, Eg_array_orig, dim_rsc, rebin_axis=1)
    #         else:
    #             # Copy original matrix, it's already the right dimension
    #             fg_matrix_dimcorr_Eg = fg_matrix_orig
    #             Eg_array_dimcorr = Eg_array_orig

    #         # Then do the same with Ex:
    #         if fg_matrix_dimcorr_Eg.shape[0] < dim_rsc:
    #             # Concatenate array with an array of zeros:
    #             fg_matrix_dimcorr_Ex = np.concatenate((fg_matrix_dimcorr_Eg, np.zeros(
    #                 (dim_rsc - fg_matrix_dimcorr_Eg.shape[0], fg_matrix_dimcorr_Eg.shape[1]))), axis=0)
    #             # Make a corresponding Eg array:
    #             Ex_array_dimcorr = E_array_from_calibration(
    #                 self.firstgen.calibration["a0y"], self.firstgen.calibration["a1y"], dim_rsc)
    #         elif fg_matrix_dimcorr_Eg.shape[0] > dim_rsc:
    #             # Rebin down to correct number of bins:
    #             fg_matrix_dimcorr_Ex, Ex_array_dimcorr = rebin(
    #                 fg_matrix_dimcorr_Eg, Ex_array_orig, dim_rsc, rebin_axis=0)
    #         else:
    #             # Copy original matrix, it's already the right dimension
    #             fg_matrix_dimcorr_Ex = fg_matrix_dimcorr_Eg
    #             Ex_array_dimcorr = Eg_array_orig

    #         # Update variable names
    #         fg_matrix = fg_matrix_dimcorr_Ex
    #         Ex_array = Ex_array_dimcorr
    #         Eg_array = Eg_array_dimcorr

    #         print(Ex_array, flush=True)

    #         var_fg_matrix = np.sqrt(fg_matrix)

    #         calibration = np.array(
    #             [Eg_array[0], Eg_array[1] - Eg_array[0], Ex_array[0], Ex_array[1] - Ex_array[0]])
    #         rho, T = rsc.rhosigchi(
    #             fg_matrix, var_fg_matrix, calibration, Eg_min, Ex_min, Ex_max)

    #     return rho, T, Ex_array, Eg_array

    # def fit_rhosigpy(self):
    #     """
    #     Perform the rho and T fit using Fabio Zeiser's code rhosig.py from 
    #     https://github.com/oslocyclotronlab/rhosig.py/
    #     """
    #     import sys
    #     from rhosigpy import rhosig as rsp
    #     from rhosigpy import utilities as rsputils

    #     # Copied from Fabio's run_analysis.py script:

    #     # Rebin and cut matrix
    #     pars_fg = {"Egmin": 1.0,
    #                "Exmin": 2.0,
    #                "Emax": 5.0}

    #     oslo_matrix, Nbins, Emid = rsputils.rebin_both_axis(
    #         oslo_matrix, Emid, rebin_fac=4)
    #     oslo_matrix, Emid_Eg, Emid_Ex, Emid_nld = rsputils.fg_cut_Matrix(oslo_matrix,
    #                                                                      Emid, **pars_fg)


# # === Test it ===
# if __name__ == "__main__":
#     import matplotlib.pyplot as plt

#     fname_raw = "data/alfna-Re187.m"

#     # Initialise pyma for current experiment
#     pm = pymama(fname_raw)

#     # Load raw matrix
#     pm.raw.load(fname_raw)

#     # Check that it has loaded a sensible raw matrix:
#     print(pm.raw.matrix.shape)

#     # Plot it
#     # pm.raw.plot(title="raw")

#     # Do unfolding:
#     # pm.unfold(fname_resp_mat, fname_resp_dat, use_comptonsubtraction=False,
#     # verbose=True, plot=True) # Call unfolding routine

#     # Save the unfolded matrix:
#     fname_unfolded = "data/unfolded-Re187.m"
#     # pm.unfolded.save(fname_save_unfolded)

#     # Load the unfolded matrix from file:
#     pm.unfolded.load(fname_unfolded)

#     # # Run first generation method:
#     pm.N_Exbins_fg = pm.unfolded.matrix.shape[0]  # Take all bins
#     # TODO figure out if this is needed and how it relates to max Eg
#     pm.Ex_max_fg = pm.unfolded.Ex_array[-1] - 2000
#     pm.dEg_fg = 1000  # keV
#     pm.first_generation_method()

#     # # Plot first generation matrix
#     pm.firstgen.plot(title="first generation")

#     # # Save it
#     fname_firstgen = "data/firstgen-Re187.m"
#     pm.firstgen.save(fname_firstgen)

#     # Load it
#     # pm.firstgen.load(fname_firstgen)

#     # Fit T and rho
#     Eg_min = 1000
#     Ex_min = 3000
#     Ex_max = 6700  # keV
#     rho, T, Ex_array, Eg_array = pm.fit(
#         Eg_min, Ex_min, Ex_max, estimate_variance_matrix=False)

#     plt.plot(rho, label="rho")
#     plt.plot(T, label="T")
#     plt.legend()
#     plt.show()
