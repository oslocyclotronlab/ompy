# -*- coding: utf-8 -*-
"""
Implementation of the first generation method
(Guttormsen, Ramsøy and Rekstad, Nuclear Instruments and Methods in
Physics Research A 255 (1987).)

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


def first_generation_method(matrix_in,
                            Ex_max, dE_gamma, N_iterations=10,
                            multiplicity_estimation="statistical",
                            apply_area_correction=False,
                            valley_correction_array=None,
                            initial_weight_function="box",
                            verbose=False):
    """
    Function implementing the first generation method from Guttormsen et
    al. (NIM 1987).

    The code is heavily influenced by the original implementation by Magne
    in MAMA. Mainly written autumn 2016 at MSU.

    Args:
        matrix (Matrix): The matrix to apply the first eneration method to
        Ex_max (double): The maximum excitation energy to run the method for
        dE_gamma (double): The amount of leeway to the right of the Ex=Eg
                            diagonal, due to experimental uncertainty
        multiplicity_estimation (str): One of ["statistical", "total"]
        apply_area_correction (bool): Whether to use the area correction method
        valley_correction_array (np.ndarray, optional): Array of weight factors
            for each Ex bin that can be used to manually "turn off"/decrease
            the influence of very large peaks in the method.
        initial_weight_function (str, optional): The initial assumption for the
            weight function to start the first-generation method iterations.
            Possible values: "box", "fermi_gas".
        verbose (bool): Whether to run the method in a verbose, talkative mode

    Todo:
        - Consider removing Ex_max keyword. Can't it just take the whole matrix?
          Compare with MAMA.
    """

    # # DEBUG
    # # Printing keywords
    # print("matrix_in.calibration() =", matrix_in.calibration())
    # print("Ex_max =", Ex_max)
    # print("dE_gamma =", dE_gamma)
    # print("multiplicity_estimation =", multiplicity_estimation)
    # print("apply_area_correction =", apply_area_correction)
    # print("verbose =", verbose)
    # print("", flush=True)
    # # END DEBUG

    # Protect input arrays:
    unfolded_matrix = np.copy(matrix_in.matrix)
    Ex_array_mat = np.copy(matrix_in.E0_array)
    Egamma_array = np.copy(matrix_in.E1_array)


    # Cut the input matrix at or above Ex=0. This is implicitly
    # done in MAMA by the variable IyL.
    i_Ex_low = i_from_E(0, Ex_array_mat)
    if Ex_array_mat[i_Ex_low] < 0:
        i_Ex_low += 1
    unfolded_matrix = unfolded_matrix[i_Ex_low:, :]
    Ex_array_mat = Ex_array_mat[i_Ex_low:]

    # Get some numbers:
    Ny = len(unfolded_matrix[:, 0])
    Nx = len(unfolded_matrix[0, :])
    calib_in = matrix_in.calibration()
    bx = calib_in["a10"]
    ax = calib_in["a11"]
    by = calib_in["a00"]
    ay = calib_in["a01"]

    N_Exbins = Ny  # TODO clean up and rename energy arrays, Nbins vars, etc

    # Check if the valley correction array is present, else
    # fill it with ones
    if valley_correction_array is None:
        valley_correction_array = np.ones(N_Exbins)
    else:
        # If it was sent in, then trim it accordingly with the unfolded matrix
        valley_correction_array = valley_correction_array[i_Ex_low:]

    ThresSta = 430.0
    # AreaCorr = 1
    ThresTot = 200.000000
    ThresRatio = 0.3000000
    # ExH = 7520.00000
    ExEntry0s = 300.000000
    ExEntry0t = 0.00000000

    # Ex_max = 7500 # keV - maximum excitation energy
    # Ex_min = 300 # keV - minimal excitation energy, effectively moving
    # the ground-state energy up because we cannot resolve the low-energy
    # yrast gamma lines. This is weighed up by also using an effective 
    # multiplicity which is lower than the real one, again not considering
    # the low-energy yrast gammas.
    # dE_gamma = 500  # keV - allow gamma energy to exceed excitation energy
    # by this much, to account for experimental resolution

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
    multiplicity = None
    if multiplicity_estimation == "statistical":
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

        # Calculate average multiplicity for each Ex channel
        area_matrix_ex_compressed_cut = np.sum(
            matrix_ex_compressed_cut, axis=1)
        Egamma_average = div0(np.sum(
            Egamma_mesh * matrix_ex_compressed_cut, axis=1),
            area_matrix_ex_compressed_cut)

        # Statistical multiplicity - use the effective Ex0 value
        multiplicity = div0(
            Ex_array - np.maximum(np.minimum(
                        Ex_array - 200, ExEntry0s), 0), Egamma_average)

    elif multiplicity_estimation == "total":
        # TODO fixme
        # Total multiplicity calculation
        # good_indices = np.where(Egamma_mesh < Egamma_max_grid, True, False)
        matrix_ex_compressed_cut = np.where(
            Egamma_mesh < Egamma_max_grid, matrix_ex_compressed, 0)

        # Calculate average multiplicity for each Ex channel
        area_matrix_ex_compressed_cut = np.sum(
            matrix_ex_compressed_cut, axis=1)
        Egamma_average = div0(np.sum(
            Egamma_mesh * matrix_ex_compressed_cut, axis=1),
            area_matrix_ex_compressed_cut)

        # Total multiplicity - use actual Ex0 = 0
        multiplicity = div0(Ex_array, Egamma_average)
        # Set any negative values to zero. Looks like this is what is done
        # in MAMA:
        multiplicity[multiplicity < 0] = 0
        # DEBUG
        # END DEBUG
        assert((multiplicity >= 0).all())

    else:
        raise ValueError("Invalid value for variable"
                         " multiplicity_estimation: ",
                         multiplicity_estimation)
    # for i in range(len(good_indices[:,0])):
    # print len(good_indices[i,good_indices[i,:]]) # OK, it actually works.

    if verbose:
        print("Multiplicities:")
        for i_Ex in range(len(Ex_array)):
            print("Ex = {:f}, multiplicity(Ex) = {:f}".format(Ex_array[i_Ex],
                  multiplicity[i_Ex]))
        print("")


    # Set up dummy first-generation matrix to start iterations, made of
    # normalized boxes:
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
            # normalization_matrix_manual[i, j] = div0(multiplicity[i]*area[j],
            normalization_matrix_manual[i, j] = div0(multiplicity[j]*area[i],
                                                     multiplicity[i]*area[j]
                                                     )
    normalization_matrix = normalization_matrix_manual
    # normalization_matrix[np.isnan(normalization_matrix)] = 0


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

    # Prepare the initial assumption for the weight function:
    W_old = None
    Ex_mesh, Eg_mesh = np.meshgrid(Ex_array, Ex_array, indexing="ij")
    if initial_weight_function == "box":
        W_old = np.ones_like(Eg_mesh)
        W_old = div0(W_old, W_old.sum(axis=1).reshape(N_Exbins, 1))
    elif initial_weight_function == "fermi_gas":
        # Prepare weight function based on Fermi gas approximation:
        a_f = 16  # 1/MeV
        n_f = 4.2  # Exponent for E_gamma
        # Make the weight array by the formula W(Ex,Eg) = Eg^n_f / (Ex-Eg)^2 *
        # exp(2*sqrt(a*(Ex-Eg)))
        # Make mesh of final energies:
        Ef_mesh = Ex_mesh - Eg_mesh
        # Set everything above Ex=Eg diagonal to zero so the weights also
        # become zero
        Ef_mesh[Ef_mesh < 0] = 0
        # Calculate weights. Remember that energies are in keV, while a is in
        # 1/MeV, so we correct in the exponent:
        W_old = np.where(Eg_mesh > 0, div0(np.power(
            Eg_mesh, n_f) , np.power(Ef_mesh, 2) * np.exp(2 * np.sqrt(a_f * Ef_mesh / 1000))), 0)
        W_old = div0(W_old, W_old.sum(axis=1).reshape(N_Exbins, 1))
    else:
        NotImplementedError(
            "unknown value for variable initial_weight_function",
            initial_weight_function)


    mask_W = make_mask(Ex_array, Ex_array, Ex_array[0], Ex_array[
                       0] + dE_gamma, Ex_array[-1], Ex_array[-1] + dE_gamma)

    # Perform the iterative subtraction:
    for iteration in range(N_iterations):
        # convergence_criterion = 1
        # max_diff = 100
        # while max_diff > convergence_criterion:
        # Store H from previous iteration to compare at the end
        H_old = np.copy(H)
        # Compress the H matrix along gamma axis to make it square and facilitate conversion to excitation energy
        # H_compressed = H[:,0:i_Egamma_max].reshape(N_Exbins, N_Exbins, grouping_Egamma).sum(axis=2)
        # H_compressed, Egamma_array_compressed = rebin(
            # H[:, 0:i_Egamma_max], Egamma_array[0:i_Egamma_max], N_Exbins, rebin_axis=1)
        # Updated 20190130 to rebin_matrix() with energy calibratiON.
        H_compressed = rebin_matrix(
            # H[:, 0:i_Egamma_max], Egamma_array[0:i_Egamma_max], Ex_array,
            H, Egamma_array, Ex_array,  # DEBUG 20190206 removed the subset selection. No difference on 164Dy, which is good.
            rebin_axis=1)

        # DEBUG:
        # plt.pcolormesh(Ex_array, Ex_array, H_compressed)
        # plt.show()
        # END DEBUG

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

        # Remove Inf and NaN
        W = np.nan_to_num(W)
        # Remove negative weights
        W[W < 0] = 0
        # Apply mask
        # W = W * mask_W
        # Normalize each Ex channel to unity
        # W = np.where(np.invert(np.isnan(W/W.sum(axis=1).astype(float))),  W/W.sum(axis=1).astype(float), 0)
        W = div0(W, W.sum(axis=1).reshape(N_Exbins, 1))
        # Store for next iteration:
        W_old = np.copy(W)

        # Calculate product of normalization matrix, weight matrix and
        # all-generations count matrix:
        # Matrix of weighted sum of spectra below
        # Todo write better comments on what happens here.
        # What are the dimensions and calibrations?
        G = np.dot((normalization_matrix * W * valley_correction_array),
                   matrix_ex_compressed)

        # Apply area correction
        if apply_area_correction:
        # if False: # DEBUG: Turning off area corr 20190204 to find out why it oversubtracts
            # Setup meshgrids for making boolean indexing arrays
            # Egamma_mesh_compressed, Ex_mesh_compressed = np.meshgrid(Egamma_array_compressed, Ex_array)
            # Egamma_max = Ex_array + dE_gamma # Maximal Egamma value for each Ex bin
            # Egamma_max_grid_compressed = np.meshgrid(np.ones(N_Exbins), Egamma_max)[1]
            # print "Egamma_mesh_compressed, Egamma_max, Egamma_max_grid"
            # print Egamma_mesh_compressed.shape, Egamma_max.shape,
            # Egamma_max_grid.shape
            if multiplicity_estimation == "statistical":
                # Statistical multiplicity calculation (i.e. trying to use statistical/continuum region only)
                # slide_compressed = np.minimum( np.maximum(ThresRatio*Ex_mesh_compressed, ThresTot), ThresSta ) # The sliding lower limit for Egamma integral - sliding between ThresTot and ThresSta.

                # good_indices_G = np.where(np.logical_and(slide_compressed < Egamma_mesh_compressed, Egamma_mesh_compressed < Egamma_max_grid) , True, False)
                G_area = np.where(np.logical_and(
                    slide < Egamma_mesh, Egamma_mesh < Egamma_max_grid), G, 0).sum(axis=1)
            elif multiplicity_estimation == "total":
                # Total multiplicity calculation
                # good_indices_G = np.where(Egamma_mesh_compressed < Egamma_max_grid, True, False)
                # G_area = np.where(Egamma_mesh_compressed < Egamma_max_grid, G, 0).sum(axis=1)
                G_area = np.where(
                    Egamma_mesh < Egamma_max_grid, G, 0).sum(axis=1)

            # DEBUG
            print("multiplicity =", multiplicity)
            print("type(multiplicity) =", type(multiplicity))
            # END DEBUG
            alpha = np.where(G_area > 0, (1 - div0(1, multiplicity))
                             * div0(area_matrix_ex_compressed_cut, G_area), 1)
            alpha[alpha < 0.85] = 0.85
            alpha[alpha > 1.15] = 1.15

        else:
            alpha = np.ones(N_Exbins)

        # The actual subtraction
        # H = matrix_ex_compressed - alpha.reshape((len(alpha), 1)) * G
        H = matrix_ex_compressed - G

        # Check convergence
        max_diff = np.max(np.abs(H - H_old))
        if verbose:
            print("iteration =", iteration, "max_diff =", max_diff, flush=True)

    # Remove negative counts
    # H[H < 0] = 0

    # Update internal variables and return True upon completion
    # return H, H-H_old, Ex_array, Egamma_array

    # DEBUG:
    # print("H.shape =", H.shape)
    # print("Ex_array.shape =", Ex_array.shape)
    # print("Egamma_array.shape =", Egamma_array.shape, flush=True)
    # END DEBUG

    firstgen = Matrix(H,
    # firstgen = Matrix(G,
                      Ex_array,
                      Egamma_array)
    return firstgen
