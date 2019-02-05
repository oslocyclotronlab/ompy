# -*- coding: utf-8 -*-
import numpy as np
from .library import *
from .rebin import *


def first_generation_method_reimplementation(all_generations_matrix,
                                             ):
    """
    Calculates the first-generation matrix by a self-consistent method.

    The first-generation method (Guttormsen et al., NIM A255 (1987):
    Let the Ex and Eg axes be denoted as i and j, respectively.
    Let F_ij be the first-generation matrix that we seek.
    Let A_ij be the all-generations matrix input to the function.
    Let Y_ij be the matrix of younger generations -- everything that is
    not the first generation.
    We then have the relationship

    F_ij = P_ij - Y_ij.

    Furthermore, under certain assumptions (see article),

    Y_ij = sum_{i'<i} n_i' w_i' A_i'j

    Here, n_i' is a normalization coefficient which compensates for
    the differing cross-sections for populating different Ex bins and
    w_i' is the probability to decay from Ex_i to Ex_i'. In fact,
    this is simply given by the (probablity normalized) first-generation
    spectrum F_ii'.

    Args:
        all_generations_matrix (Matrix): input matrix

    Returns.
        first_generation_matrix (Matrix)
    """

    # Use the calibration of the input matrix throughout:
    Ex_array = all_generations_matrix.E0_array
    Eg_array = all_generations_matrix.E1_array

    # Get calibration coefficients:
    aEx, bEx = Ex_array[0], Ex_array[1]-Ex_array[0]
    aEg, bEg = Eg_array[0], Eg_array[1]-Eg_array[0]

    # Get the number of bins corresponding to energy padding dE
    dE = 500  # TODO move out to keyword argument
    i_Ex_dE = int(dE/aEx)

    # Convert to the notation described in the docstring:
    A = all_generations_matrix.matrix

    # === Make normalization matrix ===
    # == Calculate multiplicity vector ==
    # Using multiplicity normalization. We do not implement
    # singles normalization from MAMA.
    # The average multiplicity is simply the ratio between Ex
    # and the average Eg, since that is the average number of gammas
    # that fit between Ex and the ground state.

    # TODO move the assigment below to keyword arg
    # + implement alternative "statistical"
    multiplicity_estimation = "total"
    # Allocate vector of average multiplicities as function of Ex
    M = np.zeros(len(Ex_array))
    if multiplicity_estimation == "total":
        # TODO move the following two lines out to constants.py
        FG_EX_ENTRY_MULTIPLICITY_TOTAL = 0  # Entry point in gound band for total multiplicity estimation
        FG_EG_THRES_MULTIPLICITY_TOTAL = 200  # Lower Eg threshold for total multiplicity estimation

        i_Eg_min = i_from_E(FG_EG_THRES_MULTIPLICITY_TOTAL, Ex_array)
        # Calculate average multiplicity:
        for i_Ex in range(len(Ex_array)):
            Ex = Ex_array[i_Ex]
            i_Eg_max = i_from_E(Ex+dE, Eg_array)
            # Eg_average = sum_j(Eg_j * A_ij) / sum_j(A_ij)
            Eg_average = div0(
                            np.sum(Eg_array[i_Eg_min:i_Eg_max]
                                   * A[i_Ex, i_Eg_min:i_Eg_max]),
                            np.sum(A[i_Ex, i_Eg_min:i_Eg_max])
                             )
            # Set Ex_entry like in MAMA:
            Ex_entry = max(0,
                           min(FG_EX_ENTRY_MULTIPLICITY_TOTAL, Ex - 200)
                           )
            M[i_Ex] = div0((Ex - Ex_entry),
                           Eg_average
                           )

        # Assert that M is positive:
        assert((M >= 0).all())

        # DEBUG:
        print("Multiplicity vector M:")
        for i_Ex in range(len(Ex_array)):
            print("Ex = {:f}, M(Ex) = {:f}".format(Ex_array[i_Ex], M[i_Ex]))
        print("", flush=True)
        # END DEBUG

    else:
        raise ValueError(("Invalid choice for option "
                         "multiplicity_estimation"),
                         multiplicity_estimation
                         )

    # == Calculate S vector ==
    # Begin by making vector S of "cross sections" like in NIM paper.
    # Normalization factor n is given by ratios of this.
    S = np.zeros(len(Ex_array))
    i_Eg_min = i_from_E(FG_EG_THRES_MULTIPLICITY_TOTAL, Ex_array)
    for i_Ex in range(len(Ex_array)):
        Ex = Ex_array[i_Ex]
        i_Eg_max = i_from_E(Ex+dE, Eg_array)
        S[i_Ex] = max(0,
                      div0(
                           np.sum(A[i_Ex, i_Eg_min:i_Eg_max]),
                           M[i_Ex]
                           )
                      )
    # DEBUG:
    print("xsec vector S:")
    for i_Ex in range(len(Ex_array)):
        print("Ex = {:f}, S(Ex) = {:f}".format(Ex_array[i_Ex], S[i_Ex]))
    print("", flush=True)
    # END DEBUG

    # === Start iterating to find F ===
    N_iterations = 3  # TODO move to keyword arg
    # Allocate matrices for F and Y. We store each iteration, because
    # we can:
    F = np.zeros(np.append(N_iterations, A.shape))
    Y = np.zeros(np.append(N_iterations, A.shape))

    # Assume initial trial F to be flat boxes:
    F[0, :, :] = np.ones(A.shape)

    for i_it in range(0, N_iterations-1):
        # In this loop we calculate the (i_it+1)'th iteration
        # of the F and Y spectra.

        # === Step 1: Make weights matrix ===
        # Rebin F along Eg axis to Ex calibration, to use
        # for weights
        F_previous_rebinned = rebin_matrix(F[i_it, :, :],
                                           E_array_in=Eg_array,
                                           E_array_out=Ex_array,
                                           rebin_axis=1)
        # Remove negative values:
        F_previous_rebinned[F_previous_rebinned < 0] = 0
        # Normalize F_previous_rebinned in each Ex bin:
        weights_matrix = div0(F_previous_rebinned,
                              np.sum(F_previous_rebinned, axis=1)
                              )
        # TODO "massage" weights_matrix by removing negatives (and more?
        # see MAMA)

        # === Step 3: Loop over all Ex bins       ===
        # === and calculate Y[i_it] ===
        for i_Ex in range(A.shape[0]):
            # Calculate younger-generations matrix Y.
            # Loop over all Ex' below Ex:
            for i_Exprim in range(0, i_Ex-1):
                # Loop over all Eg bins in each Ex' bin:
                # Caveat: i_Eg is still in Eg_array calibration!
                Exprim = Ex_array[i_Exprim]
                i_Eg_max = i_from_E(Exprim + dE, Eg_array)
                for i_Eg in range(0, i_Eg_max):
                    # But i_Egprim is an index to the weights matrix and shall
                    # hence be in Ex calibration! So:
                    i_Egprim = i_Ex - i_Exprim

                    Y[i_it, i_Ex, i_Eg] += (  # weights w:
                                            weights_matrix[i_Ex, i_Egprim]
                                            # normalization n:
                                            * div0(S[i_Ex], S[i_Exprim])
                                            # all_generations counts:
                                            * A[i_Exprim, i_Eg]
                                            )

                # Apply area correction:
                apply_area_correction = True  # TODO move to keyword arg.
                if apply_area_correction:
                    # alpha = np.where(G_area > 0, (1 - div0(1, multiplicity))
                                     # * div0(area_matrix_ex_compressed_cut, G_area), 1)
                    # Enforce lower and upper limit:
                    alpha = 0.85
                    alpha = max(0.85, min(1.15, alpha))
                    Y[i_it, i_Ex, :] *= alpha



        # === Step 4: Calculate F[i_it+1] ===
        F[i_it+1, :, :] = A - Y[i_it, :, :]

    # Put final result into Matrix instance and return:
    first_generation_matrix = Matrix(matrix=F[-1, :, :],
    # first_generation_matrix = Matrix(matrix=Y[-2, :, :],
                                     E0_array=Ex_array,
                                     E1_array=Eg_array
                                     )
    return first_generation_matrix
