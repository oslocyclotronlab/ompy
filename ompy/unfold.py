# -*- coding: utf-8 -*-
"""
Implementation of the unfolding method
(Guttormsen et al., Nuclear Instruments and Methods in Physics Research A
374 (1996))
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
# External library imports:
import numpy as np
import os

# OMpy imports:
from .library import (Matrix, E_array_from_calibration, i_from_E, make_mask,
                      div0, EffExp)
from .constants import (DE_PARTICLE, DE_GAMMA_1MEV, DE_GAMMA_8MEV,
                        FWHM_1330KEV_OSCAR, FOLDER_PATH_RESPONSE_OSCAR)
from .compton_subtraction_method import shift_matrix
from .response import interpolate_response
from .gauss_smoothing import gauss_smoothing_matrix

global DE_PARTICLE
global DE_GAMMA_1MEV
global DE_GAMMA_8MEV


class Unfolder():
    def __init__(self, folder_path_response=None,
                 calibration=None, matrix=None,
                 fwhm=None):
        """
        Instantiate the class and interpolate the response function
        to match gamma energy calibration of spectrum "raw".

        Args:
            folder_path_response (str): Path to the folder containing response
                                        function information.
            calibration (dict, optional): Calibration to interpolate the
                                          response function to.
            matrix (Matrix, optional): matrix with spectrum. It is just used to
                                       get the
                                       calibration, and has to be input separately
                                       to the unfold() func.
            fwhm (float): Experimental full-width-half-max at E_gamma=1.33 MeV.
        """

        Eg_array = None
        if calibration is None and matrix is None:
            raise Exception("Either calibration or matrix spectrum must be provided")
        elif calibration is not None and matrix is None:
            Eg_array = E_array_from_calibration(a0=calibration["a10"], a1=calibration["a11"],
                                                N=calibration["N1"])
        elif calibration is None and matrix is not None:
            Eg_array = matrix.E1_array
        else:
            raise Exception("Cannot specify both matrix and calibration.")
        if folder_path_response is None:
            # Using default response:
            dirname = os.path.dirname(__file__)
            folder_path_response = os.path.join(dirname, "../",
                                                FOLDER_PATH_RESPONSE_OSCAR)
        if fwhm is None:
            # Using default fwhm value
            fwhm = FWHM_1330KEV_OSCAR


        self.folder_path_response = folder_path_response
        self.fwhm = fwhm

        # Run the interpolation to get a response matrix of correct type
        response_matrix, fwhm_rel, eff_tot, pcmp, pFE, pSE, pDE, p511 =\
            interpolate_response(folder_path_response, Eg_array, fwhm_abs=6.8)

        # Store the resulting arrays inside class:
        self.response_matrix = response_matrix
        self.fwhm_rel = fwhm_rel
        self.eff_tot = eff_tot
        self.pcmp = pcmp
        self.pFE = pFE
        self.pSE = pSE
        self.pDE = pDE
        self.p511 = p511

        # Also make an array of absolute FWHM values, which is the one to use:
        self.fwhm_abs = fwhm*fwhm_rel

    def unfold(self, raw,
               FWHM_factor=10,
               Ex_min=None, Ex_max=None, Eg_min=None,
               diag_cut=None,
               Eg_max=None, verbose=False, plot=False,
               use_comptonsubtraction=False):

        unfolded = unfold(raw, self.response_matrix,
                          FWHM_factor=FWHM_factor,
                          Ex_min=Ex_min, Ex_max=Ex_max, Eg_min=Eg_min,
                          diag_cut=diag_cut,
                          Eg_max=Eg_max, verbose=verbose, plot=plot,
                          use_comptonsubtraction=use_comptonsubtraction,
                          FWHM=self.fwhm_abs, eff=self.eff_tot, pf=self.pFE,
                          pc=self.pcmp, ps=self.pSE, pd=self.pDE, pa=self.p511)
        self.unfolded = unfolded

        return unfolded


def unfold(raw, response,
           FWHM_factor=10,
           Ex_min=None, Ex_max=None, Eg_min=None,
           diag_cut=None,
           Eg_max=None, verbose=False, plot=False,
           use_comptonsubtraction=False,
           FWHM=None, eff=None, pf=None, pc=None, ps=None, pd=None, pa=None):
    """Unfolds the gamma-detector response of a spectrum

    Args:
        raw (Matrix): the raw matrix to unfold, an instance of Matrix()
        response (Matrix): the detector response matrix, an instance of
                           Matrix()
        Ex_min (float): Lower limit for excitation energy
        Ex_max (float): Upper limit for excitation energy
        Eg_min (float): Lower limit for gamma-ray energy
        Eg_max (float): Upper limit for gamma-ray energy
        diag_cut (dict, optional): Points giving upper diagonal boundary on Eg
        verbose (bool): Toggle verbose mode
        plot (bool): Toggle plotting
        use_comptonsubtraction (bool): Toggle whether to use the Compton
                                        subtraction method
        FWHM, eff, ... (np arrays): Arrays of data from resp.dat, interpolated
                                    to correct calibration. Designed to be
                                    input through class wrapper.

    Returns:
        unfolded -- the unfolded matrix as an instance of the Matrix() class

    Todo:
        - Implement the Matrix() and Vector() classes throughout the function.
        - Fix the compton subtraction method implementation.
    """

    # if use_comptonsubtraction:
    #     raise Exception(("The compton subtraction method does not currently"
    #                     " work correctly."))

    # Check that the response matrix has the same energy calibration as the
    # raw matrix's gamma energy axis (axis 1 in matrix notation):
    if (raw.calibration()["a10"] != response.calibration()["a00"]
            and raw.calibration()["a11"] != response.calibration()["a01"]):
        raise Exception(
            "Energy calibration mismatch between raw and response matrices. Numbers:"
            "raw.calibration() =", raw.calibration(),
            "response.calibration() =", response.calibration()
            )
    # if fname_resp_dat is None and use_comptonsubtraction is True:
    #     raise Exception(
    #         "fname_resp_dat not given, and "
    #         "use_comptonsubtraction is True."
    #         )

    # Rename variables for local use:
    # data_raw = raw.matrix
    Ex_array = raw.E0_array
    Eg_array = raw.E1_array


    # = Import data and response matrix =

    # If energy limits are not provided, use extremal array values:
    if Ex_min is None:
        Ex_min = Ex_array[0]
    if Ex_max is None:
        Ex_max = Ex_array[-1]
    if Eg_min is None:
        Eg_min = Eg_array[0]
    if Eg_max is None:
        Eg_max = Eg_array[-1]

    if verbose:
        print("Unfolding in verbose mode. Starting.", flush=True)
        import time
        time_readfiles_start = time.process_time()
    # Import raw mama matrix
    # data_raw, cal, Ex_array, Eg_array = read_mama_2D(fname_data_raw)
    # data_raw, cal, Ex_array, Eg_array = read_mama_2D('/home/jorgenem/gitrepos/pyma/unfolding-testing-20161115/alfna-20160518.m') # Just to verify import works generally
    # cal = {"a0x":Eg_array[0], "a1x":Eg_array[1]-Eg_array[0], "a2x":0, 
         # "a0y":Ex_array[0], "a1y":Eg_array[1]-Eg_array[0], "a2y":0}
    cal_raw = raw.calibration()
    N_Ex, N_Eg = raw.matrix.shape

    if verbose:
        print("Lowest Eg value =", Eg_array[0], flush=True)
        print("Lowest Ex value =", Ex_array[0], flush=True)

    # Import response matrix
    # response = read_mama_2D(fname_resp_mat)
    R = response.matrix
    Eg_array_R = response.E0_array
    cal_R = response.calibration()
    # response = Matrix(R, Eg_array_R, Eg_array_R)  # Both axes are gamma energies here, but it does not matter for Matrix()

    if verbose:
        time_readfiles_end = time.process_time()
        time_rebin_start = time.process_time()

    # Rebin data -- take it in Ex portions to avoid memory problems:
    # N_rebin = int(N_Eg/4)
    N_rebin = N_Eg


    if verbose:
        time_rebin_end = time.process_time()

    if plot:
        # Allocate plots:
        f, ((ax_raw, ax_fold), (ax_unfold, ax_unfold_smooth)) = plt.subplots(2,2)
        ax_raw.set_title("raw")
        ax_fold.set_title("folded")
        ax_unfold.set_title("unfolded")
        if use_comptonsubtraction:
            ax_unfold_smooth.set_title("Compton subtracted")
        # Plot raw matrix:
        from matplotlib.colors import LogNorm
        # cbar_raw = ax_raw.pcolormesh(Eg_array, Ex_array, data_raw, norm=LogNorm(vmin=1))
        cbar_raw = raw.plot(ax=ax_raw)
        f.colorbar(cbar_raw, ax=ax_raw)

    # Check that response matrix matches data, at least in terms of Eg 
    # calibration:
    eps = 1e-3
    if not (np.abs(cal_raw["a10"]-cal_R["a10"]) < eps
            and np.abs(cal_raw["a11"]-cal_R["a11"]) < eps):
        raise Exception("Calibration mismatch between data and response matrices")

    
    
    # = Step 1: Run iterative unfolding =
    if verbose:
        time_unfolding_start = time.process_time()
    
    
    # Set limits for excitation and gamma energy bins to be considered for unfolding
    # Ex_max = 14000 # keV
    # Use index 0 of array as lower limit instead of energy because it can be negative!
    # print("Ex_max =", Ex_max)
    iEx_min, iEx_max = 0, i_from_E(Ex_max, Ex_array)
    Eg_min = 0 # keV - minimum
    # Eg_min = 500 # keV 
    # Eg_max = 14000 # keV
    iEg_min, iEg_max = 0, i_from_E(Eg_max, Eg_array)
    # Max number of iterations:
    Nit = 33 #12 # 8 # 27
    
    # # Make masking array to cut away noise below Eg=Ex+dEg diagonal
    # # Define cut   x1    y1    x2    y2
    # cut_points = [i_from_E(Eg_min + dEg, Eg_array), i_from_E(Ex_min, Ex_array), 
    #               i_from_E(Eg_max+dEg, Eg_array), i_from_E(Ex_max, Ex_array)]
    # # cut_points = [ 72,   5,  1050,  257]
    # # i_array = np.linspace(0,len(Ex_array)-1,len(Ex_array)).astype(int) # Ex axis 
    # # j_array = np.linspace(0,len(Eg_array)-1,len(Eg_array)).astype(int) # Eg axis
    # i_array = np.linspace(0,len(Ex_array)-1,len(Ex_array)).astype(int) # Ex axis 
    # j_array = np.linspace(0,len(Eg_array)-1,len(Eg_array)).astype(int) # Eg axis
    # i_mesh, j_mesh = np.meshgrid(i_array, j_array, indexing='ij')
    # mask = np.where(i_mesh > line(j_mesh, cut_points), 1, 0)
    mask = None
    if diag_cut is not None:
        mask = make_mask(Ex_array, Eg_array,
                         diag_cut["Ex1"], diag_cut["Eg1"],
                         diag_cut["Ex2"], diag_cut["Eg2"])
    else:
        # If diag_cut is not given, use global uncertainty widths from
        # constants.py
        Ex1 = 1000  # MeV
        Eg1 = 1000 + np.sqrt(DE_PARTICLE**2 + DE_GAMMA_1MEV**2)  # MeV
        Ex2 = 8000  # MeV
        Eg2 = 8000 + np.sqrt(DE_PARTICLE**2 + DE_GAMMA_8MEV**2)  # MeV
        mask = make_mask(Ex_array, Eg_array, Ex1, Eg1, Ex2, Eg2)
    # HACK TEST 20181004: Does the mask do any good?:
    # mask = np.ones(mask.shape)

    # rawmat = (raw.matrix*mask)[iEx_min:iEx_max, iEg_min:iEg_max] 
    # 20190131: Removed mask, want to keep unfolding routine as
    # simple as possible.
    rawmat = raw.matrix[iEx_min:iEx_max, iEg_min:iEg_max] 

    mask_cut = mask[iEx_min:iEx_max, iEg_min:iEg_max]

    #Ndof = mask_cut.sum() # This was for a 2D chisquare, which is wrong.
    # We take the chisquare for each Ex bin separately. So rather, 
    # Ndof_vector = mask_cut.sum(axis=1)

    unfoldmat = np.zeros((rawmat.shape[0],rawmat.shape[1]))
    foldmat = np.zeros((rawmat.shape[0],rawmat.shape[1]))
    chisquare_matrix = np.zeros((rawmat.shape[0], Nit))
    fluctuations_matrix = np.zeros((rawmat.shape[0], Nit))
    R = R[iEg_min:iEg_max,iEg_min:iEg_max]
    # Normalize R to conserve probability
    # R = div0(R, R.sum(axis=1))

    Ex_array_cut = Ex_array[iEx_min:iEx_max]
    Eg_array_cut = Eg_array[iEg_min:iEg_max]

    # Calculate fluctuations of the raw spectrum to compare with the unfolded later
    fluctuations_vector_raw = fluctuations(rawmat, Eg_array_cut)
    
    unfoldmat_cube = np.zeros((Nit, rawmat.shape[0], rawmat.shape[1]))
    # Run folding iterations:
    for iteration in range(Nit):
        if iteration == 0:
            unfoldmat = rawmat
            # unfoldmat = mask_cut.copy() # 201810: Trying to debug by checking effect of using a box to start. No difference. 
        else:
            # Option 1, difference:
            unfoldmat = unfoldmat + (rawmat - foldmat) # Difference method 
            # Option 2, ratio:
            # unfoldmat = unfoldmat * div0(rawmat, foldmat) # Ratio method (MAMA seems to use both alternatingly?)
            # Option 3, alternate between them:
            # if iteration % 2 == 0:
            #     unfoldmat = unfoldmat + (rawmat - foldmat) # Difference method 
            # else:
            #     unfoldmat = unfoldmat * div0(rawmat, foldmat) # Ratio method (MAMA seems to use both alternatingly?)
        unfoldmat_cube[iteration,:,:] = unfoldmat

        foldmat = np.dot(R.T, unfoldmat.T).T # Have to do some transposing to get the axis orderings right for matrix product
        # 201810: Trying to debug by checking if it makes a difference to loop over rows as individual vectors (should not make a difference, and does not):
        # for i_row in range(foldmat.shape[0]):
            # foldmat[i_row,:] = np.dot(R.T, unfoldmat[i_row,:].T).T
        foldmat = mask_cut*foldmat # Apply mask for every iteration to suppress stuff below diag
        # 20171110: Tried transposing R. Was it that simple? Immediately looks good.
        #           Or at least much better. There is still something wrong giving negative counts left of peaks.
        #           Seems to come from unfolding and not from compton subtraction
    
        # Calculate reduced chisquare of the "fit" between folded-unfolded matrix and original raw
        # for each Ex bin separately
        # TODO reduced chisquare or normal?
        chisquare_matrix[:,iteration] = div0(np.power(foldmat-rawmat,2),np.where(rawmat>0,rawmat,0)).sum(axis=1) #/ Ndof_vector

        # Also calculate fluctuations in each Ex bin
        fluctuations_matrix[:,iteration] = fluctuations(unfoldmat, Eg_array_cut)

        if verbose:
            # print("Folding iteration = {}, chisquare = {}".format(iteration,chisquare_matrix[:,iteration]), flush=True)
            print("Folding iteration = {}, avg. reduced chisquare = {}".format(
                iteration, np.mean(chisquare_matrix[:, iteration])),
                flush=True)


    # Score the solutions based on chisquare value for each Ex bin
    # and select the best one:
    fluctuations_matrix = fluctuations_matrix/fluctuations_vector_raw[:,None] # TODO check that this broadcasts the vector over the right dimension
    # Get the vector indicating iteration index of best score for each Ex bin:
    weight_fluc = 0.2  # 0.6 # TODO make this an argument
    minimum_iterations = 3 # Minimum iteration number to accept from the scoring
    # Check that it's consistent with chosen max number of iterations:
    if minimum_iterations > Nit:
        minimum_iterations = Nit
    i_score_vector = scoring(chisquare_matrix, fluctuations_matrix,
                             weight_fluc, minimum_iterations)
    unfoldmat = np.zeros(rawmat.shape)
    for i_Ex in range(rawmat.shape[0]):
        unfoldmat[i_Ex, :] = unfoldmat_cube[i_score_vector[i_Ex], i_Ex, :]

    if verbose:
        print("The iteration number with the best score for each Ex bin:")
        for i_Ex in range(rawmat.shape[0]):
            print("i_Ex = {:d}, Ex = {:f}, i_score_vector = {:d}".format(i_Ex,
                  Ex_array[i_Ex], i_score_vector[i_Ex]))

    # Remove negative counts and trim:
    # Update 20190130: Keep unfolding as simple as possible, do these
    # operations manually.
    # unfoldmat[unfoldmat<=0] = 0
    # unfoldmat = mask_cut*unfoldmat

    if plot:
        # Plot:
        cbar_fold = ax_fold.pcolormesh(Eg_array[iEg_min:iEg_max], Ex_array[iEx_min:iEx_max], foldmat, norm=LogNorm(vmin=1))
        f.colorbar(cbar_fold, ax=ax_fold)
    
    if verbose:
        time_unfolding_end = time.process_time()
        time_compton_start = time.process_time()
    
    
    # = Step 2: Compton subtraction =
    if use_comptonsubtraction: # Check if compton subtraction is turned on

        print("DEBUG: Running Compton subtraction method.")
        # We also need the resp.dat file for this.
        # TODO: Consider writing a function that makes the response matrix (R) from this file
        # (or other input), so we don't have to keep track of redundant info.
        # resp = []
        # with open(fname_resp_dat) as file:
            # Read line by line as there is crazyness in the file format
            # lines = file.readlines()
            # for i in range(4,len(lines)):
                # try:
                    # row = np.array(lines[i].split(), dtype="double")
                    # resp.append(row)
                # except:
                    # break
        
        
        # # resp = np.array(resp)
        # # Name the columns for ease of reading
        # E_resp_array = resp[:, 0]
        # FWHM = resp[:, 1]
        # eff = resp[:, 2]
        # pFE = resp[:, 3]
        # pcmp = resp[:, 4]
        # pSE = resp[:, 5]
        # pDE = resp[:, 6]
        # p511 = resp[:, 7]

        # # Correct efficiency by multiplying with EffExp(Eg):
        EffExp_array = EffExp(Eg_array)
        # # eff_corr = np.append(0,eff)*EffExp_array
        # print("From unfold(): eff.shape =", eff.shape, "EffExp_array.shape =", EffExp_array.shape, flush=True)
        eff_corr = eff*EffExp_array # TODO2019 figure out where this fits in

        # 20190708: The resp.dat array is no longer rebinned to the right
        # energy. Instead we interpolate:
        # f_pcmp = interp1d(E_resp_array, pcmp, kind="linear", bounds_error=False, fill_value=0)
        # f_pFE = interp1d(E_resp_array, pFE, kind="linear", bounds_error=False, fill_value=0)
        # f_pSE = interp1d(E_resp_array, pSE, kind="linear", bounds_error=False, fill_value=0)
        # f_pDE = interp1d(E_resp_array, pDE, kind="linear", bounds_error=False, fill_value=0)
        # f_p511 = interp1d(E_resp_array, p511, kind="linear", bounds_error=False, fill_value=0)
        # f_fwhm_rel = interp1d(E_resp_array, fwhm_rel, kind="linear", bounds_error=False, fill_value=0)
        # f_Eff_tot = interp1d(E_resp_array, Eff_tot, kind="linear", bounds_error=False, fill_value=0)
        # # Then get new arrays with interpolated values
        # FWHM = f_fwhm_rel(Eg_array)
        # eff = f_Eff_tot(Eg_array)
        # pf = f_pFE(Eg_array)
        # pc = f_pcmp(Eg_array)
        # ps = f_pSE(Eg_array)
        # pd = f_pDE(Eg_array)
        # pa = f_p511(Eg_array)

        # Debugging: Test normalization of response matrix and response pieces:
        # i_R = 50
        # print("R[{:d},:].sum() =".format(i_R), R[i_R,:].sum())
        # print("(pf+pc+ps+pd+pa)[{:d}] =".format(i_R), pf[i_R]+pc[i_R]+ps[i_R]+pd[i_R]+pa[i_R])

        # We follow the notation of Guttormsen et al (NIM 1996) in what follows.
        # u0 is the unfolded spectrum from above, r is the raw spectrum, 
        # w = us + ud + ua is the folding contributions from everything except Compton,
        # i.e. us = single escape, ua = double escape, ua = annihilation (511).
        # v = pf*u0 + w == uf + w is the estimated "raw minus Compton" spectrum
        # c is the estimated Compton spectrum.
        
    
    
        # Check that there is enough memory:
    
        # # Split this operation into Ex chunks to not exceed memory:
        # # Allocate matrix to fill with result:
        # unfolded = np.zeros(unfoldmat.shape)
    
        # N_Ex_portions = 1 # How many portions to chunk, initially try just 1
        # mem_avail = psutil.virtual_memory()[1]
        # mem_need = 2 * 8 * N_Ex/N_Ex_portions * unfoldmat.shape[1] * unfoldmat.shape[1] # The factor 2 is needed to not crash my system. Possibly numpy copies an array somewhere, doubling required memory?
        # if verbose:
        #     print("Compton subtraction: \nmem_avail =", mem_avail, ", mem_need =", mem_need, ", ratio =", mem_need/mem_avail, flush=True)
        # while mem_need > mem_avail: 
        #     # raise Exception("Not enough memory to construct smoothing arrays. Please try rebinning the data.")
        #     N_Ex_portions += 1 # Double number of portions 
        #     mem_need = 2 * 8 * N_Ex/N_Ex_portions * unfoldmat.shape[1] * unfoldmat.shape[1]
        # if verbose:
        #     print("Adjusted to N_Ex_portions =", N_Ex_portions, "\nmem_avail =", mem_avail, ", mem_need =", mem_need, ", ratio =", mem_need/mem_avail, flush=True)
    
        # N_Ex_per_portion = int(N_Ex/N_Ex_portions)
        # for i in range(N_Ex_portions):
        #     u0 = unfoldmat[i*N_Ex_per_portion:(i+1)*N_Ex_per_portion,:]
        #     r = rawmat[i*N_Ex_per_portion:(i+1)*N_Ex_per_portion,:]
            
        #     # Apply smoothing to the different peak structures. 
        #     # FWHM/FWHM_factor (usually FWHM/10) is used for all except 
        #     # single escape (FWHM*1.1/FWHM_factor)) and annihilation (FWHM*1.0). This is like MAMA.
        #     uf = shift_and_smooth3D(u0, Eg_array, 0.5*FWHM/FWHM_factor, pf, shift=0, smoothing=True)
        #     # print("uf smoothed, integral =", uf.sum())
        #     # uf_unsm = shift_and_smooth3D(u0, Eg_array, 0.5*FWHM/FWHM_factor, pf, shift=0, smoothing=False)
        #     # print("uf unsmoothed, integral =", uf_unsm.sum())
        #     us = shift_and_smooth3D(u0, Eg_array, 0.5*FWHM/FWHM_factor*1.1, ps, shift=511, smoothing=True)
        #     ud = shift_and_smooth3D(u0, Eg_array, 0.5*FWHM/FWHM_factor, pd, shift=1022, smoothing=True)
        #     ua = shift_and_smooth3D(u0, Eg_array, 1.0*FWHM, pa, shift="annihilation", smoothing=True)
        #     w = us + ud + ua
        #     v = uf + w
        #     c = r - v    
        #     # Smoothe the Compton spectrum (using an array of 1's for the probability to only get smoothing):
        #     c_s = shift_and_smooth3D(c, Eg_array, 1.0*FWHM/FWHM_factor, np.ones(len(FWHM)), shift=0, smoothing=True)    
        #     # Subtract smoothed Compton and other structures from raw spectrum and correct for full-energy prob:
        #     u = div0((r - c - w), np.append(0,pf)[iEg_min:iEg_max]) # Channel 0 is missing from resp.dat    
        #     unfolded[i*N_Ex_per_portion:(i+1)*N_Ex_per_portion,:] = div0(u,eff_corr[iEg_min:iEg_max]) # Add Ex channel to array, also correcting for efficiency. Now we're done!



        # 20190709 rewrite:
        # We follow the notation of Guttormsen et al (NIM 1996) in what follows.
        # u0 is the unfolded spectrum from above, r is the raw spectrum, 
        # w = us + ud + ua is the folding contributions from everything except Compton,
        # i.e. us = single escape, ua = double escape, ua = annihilation (511).
        # v = pf*u0 + w == uf + w is the estimated "raw minus Compton" spectrum
        # c is the estimated Compton spectrum.
        #
        # We apply smoothing to the different peak structures. 
        # FWHM/FWHM_factor (usually FWHM/10) is used for all except 
        # single escape (FWHM*1.1/FWHM_factor)) and annihilation (FWHM*1.0). This is like MAMA.

        # TODO: Change the variable Eg_array to be the iEg_min:iEg_max
        # cut version, to avoid having to index it all the time?

        # Rename variables to match notation:
        r = rawmat
        u0 = unfoldmat

        # Full-energy, smoothing but no shift:
        uf = pf[iEg_min:iEg_max] * np.copy(u0)
        uf = gauss_smoothing_matrix(uf, Eg_array[iEg_min:iEg_max],
                                    0.5*FWHM[iEg_min:iEg_max]/FWHM_factor)

        # Single escape, smoothing and shift:
        us = ps[iEg_min:iEg_max] * np.copy(u0)
        us = gauss_smoothing_matrix(us, Eg_array[iEg_min:iEg_max],
                                    0.5*FWHM[iEg_min:iEg_max]/FWHM_factor)
        us = shift_matrix(us, Eg_array[iEg_min:iEg_max], energy_shift=-511)
 
        # Double escape, smoothing and shift:
        ud = pd[iEg_min:iEg_max] * np.copy(u0)
        ud = gauss_smoothing_matrix(ud, Eg_array[iEg_min:iEg_max],
                                    0.5*FWHM[iEg_min:iEg_max]/FWHM_factor)
        ud = shift_matrix(ud, Eg_array[iEg_min:iEg_max], energy_shift=-1024)

        # Single-escape, smoothing and shift:
        ua = np.zeros(u0.shape)
        i511 = i_from_E(511, Eg_array[iEg_min:iEg_max])
        ua[i511, :] = np.sum(pa[iEg_min:iEg_max] * np.copy(u0), axis=1)
        ua = gauss_smoothing_matrix(ua, Eg_array[iEg_min:iEg_max],
                                    1.0*FWHM[iEg_min:iEg_max])
        # ua = shift_matrix(ua, Eg_array[iEg_min:iEg_max], energy_shift=-511)


        # TODO insert FWHMfactor throughout?


        # Put it all together:
        w = us + ud + ua
        v = uf + w
        c = r - v

        # Smoothe the Compton part, which is the main trick:
        c = gauss_smoothing_matrix(c, Eg_array[iEg_min:iEg_max],
                                   1.0*FWHM[iEg_min:iEg_max]/FWHM_factor)

        u = div0((r - c - w), np.append(0,pf)[iEg_min:iEg_max]) # Channel 0 is missing from resp.dat    
        unfolded = div0(u,eff_corr[iEg_min:iEg_max]) # Add Ex channel to array, also correcting for efficiency. Now we're done!



        # end if use_comptonsubtraction
    else:
        unfolded = unfoldmat


    # Diagnostic plotting:
    # f_compt, ax_compt = plt.subplots(1,1)
    # ax_compt.plot(Eg_array[iEg_min:iEg_max], r[0,:], label="r")
    # ax_compt.plot(Eg_array[iEg_min:iEg_max], u0[0,:], label="u0")
    # ax_compt.plot(Eg_array[iEg_min:iEg_max], uf[0,:], label="uf")
    # ax_compt.plot(Eg_array[iEg_min:iEg_max], uf_unsm[0,:], label="uf unsmoothed")
    # ax_compt.plot(Eg_array[iEg_min:iEg_max], us[0,:], label="us")
    # ax_compt.plot(Eg_array[iEg_min:iEg_max], ud[0,:], label="ud")
    # ax_compt.plot(Eg_array[iEg_min:iEg_max], ua[0,:], label="ua")
    # ax_compt.plot(Eg_array[iEg_min:iEg_max], w[0,:], label="w")
    # ax_compt.plot(Eg_array[iEg_min:iEg_max], v[0,:], label="v")
    # ax_compt.plot(Eg_array[iEg_min:iEg_max], c[0,:], label="c")
    # ax_compt.plot(Eg_array[iEg_min:iEg_max], c_s[0,:], label="c_s")
    # ax_compt.plot(Eg_array[iEg_min:iEg_max], u[0,:], label="u")
    # ax_compt.legend()
           

        
    # Trim result:
    unfolded = mask_cut*unfolded

    if verbose:
        time_compton_end = time.process_time()
        # Print timing results:
        print("Time elapsed: \nFile import = {:f} s \nRebinning = {:f} s \nUnfolding = {:f} s".format(time_readfiles_end-time_readfiles_start, time_rebin_end-time_rebin_start, time_unfolding_end-time_unfolding_start), flush=True)
        if use_comptonsubtraction:
            print("Compton subtraction = {:f} s".format(time_compton_end-time_compton_start), flush=True)



    if plot:
        # Plot unfolded and Compton subtracted matrices:
        cbar_unfold = ax_unfold.pcolormesh(Eg_array[iEg_min:iEg_max], Ex_array[iEx_min:iEx_max], unfoldmat, norm=LogNorm(vmin=1))
        f.colorbar(cbar_unfold, ax=ax_unfold)
        if use_comptonsubtraction:
            cbar_unfold_smooth = ax_unfold_smooth.pcolormesh(Eg_array[iEg_min:iEg_max], Ex_array[iEx_min:iEx_max], unfolded, norm=LogNorm(vmin=1))
            f.colorbar(cbar_unfold_smooth, ax=ax_unfold_smooth)
        plt.show()




    # Update global variables:
    unfolded = Matrix(unfolded, Ex_array[iEx_min:iEx_max], Eg_array[iEg_min:iEg_max])

    return unfolded


def scoring(chisquare_matrix, fluctuations_matrix, weight_fluct,
            minimum_iterations):
    """
    Calculates the score of each unfolding iteration for each Ex
    bin based on a weighting of chisquare and fluctuations.

    """
    score_matrix = ((1-weight_fluct) * chisquare_matrix +
                    weight_fluct * fluctuations_matrix)
    # Get index of best (lowest) score for each Ex bin:
    best_iteration = np.argmin(score_matrix, axis=1)
    # Enforce minimum_iterations:
    best_iteration = np.where(minimum_iterations > best_iteration,
             minimum_iterations*np.ones(len(best_iteration), dtype=int),
             best_iteration)
    return best_iteration


def fluctuations(counts_matrix, Eg_array):
    """
    Calculates fluctuations in each Ex bin gamma spectrum by summing
    the absolute diff between the spectrum and a smoothed version of it.

    Returns a column vector of fluctuations in each Ex bin
    """
    from scipy.ndimage import gaussian_filter1d

    a1 = Eg_array[1]-Eg_array[0]
    counts_matrix_smoothed = gaussian_filter1d(counts_matrix, sigma=0.12*a1, axis=1)
    fluctuations = np.sum(np.abs(counts_matrix_smoothed - counts_matrix), axis=1)

    return fluctuations