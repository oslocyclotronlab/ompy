"""
This is pyma, the python implementation of the Oslo method.
It handles two-dimensional matrices of event count spectra, and
implements detector response unfolding, first generation method
and other manipulation of the spectra.

It is heavily inspired by MAMA, written by Magne Guttormsen,
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



class pyma():
    def __init__(self, fname_raw):
        self.fname_raw = fname_raw # File name of raw spectrum

        # Load raw matrix from file:
        matrix_raw, calib_raw, Ex_array_raw, Eg_array_raw = pml.read_mama_2D(fname_raw)
        self.raw = self.matrix(matrix_raw, Ex_array_raw, Eg_array_raw)
        # TODO consider moving raw matrix loading into a separate function. 
        # If so, remember to initialise self.raw = None

        # Allocate storage of unfolded and firstgen matrices to be filled by functions in class later:
        self.unfolded = None
        self.firstgen = None



    class matrix():
        """ 
        The matrix class stores matrices along with calibration and energy axis arrays.

        """
        def __init__(self, matrix, Ex_array, Eg_array):
            self.matrix = matrix
            self.Ex_array = Ex_array
            self.Eg_array = Eg_array

            # Calculate calibration based on energy arrays, assuming linear calibration:
            self.calibration = {"a0x":Eg_array[0], "a1x":Eg_array[1]-Eg_array[0], "a2x":0, 
             "a0y":Ex_array[0], "a1y":Eg_array[1]-Eg_array[0], "a2y":0}

        def plot(self, norm="log"):
            import matplotlib.pyplot as plt
            plot_object = None
            if norm == "log":
                from matplotlib.colors import LogNorm
                plot_object = plt.pcolormesh(self.Eg_array, self.Ex_array, self.matrix, norm=LogNorm())
            return 

        def save(self, fname):
            """
            Save matrix to mama file
            """
            pml.write_mama_2D(self.matrix, fname, self.Ex_array, self.Eg_array, comment="Made by pyma")
            return True

    def load_unfolded(fname_unfolded):
        """
        Load an unfolded matrix from mama file
        
        Inputs:
        fname_unfolded: the file path to the MAMA file
        Outputs: 
        Fills the variable self.unfolded
        Returns True upon completion
        """
        matrix_unfolded, calib_unfolded, Ex_array_unfolded, Eg_array_unfolded = self.read_mama_2D(fname_unfolded)
        self.unfolded = self.matrix(matrix_unfolded, Ex_array_unfolded, Eg_array_unfolded)
        return True

    def load_firstgen(fname_firstgen):
        """
        Load an firstgen matrix from mama file
        
        Inputs:
        fname_firstgen: the file path to the MAMA file
        Outputs: 
        Fills the variable self.firstgen
        Returns True upon completion
        """
        matrix_firstgen, calib_firstgen, Ex_array_firstgen, Eg_array_firstgen = self.read_mama_2D(fname_firstgen)
        self.firstgen = self.matrix(matrix_firstgen, Ex_array_firstgen, Eg_array_firstgen)
        return True    

    



    def unfold(self, fname_resp_mat, fname_resp_dat, FWHM_factor=10, Ex_min="default", Ex_max="default", Eg_min="default", Eg_max="default", verbose=False, plot=False, use_comptonsubtraction=True):
        # = Check that raw matrix is present 
        if "self.raw" is None:
            raise Exception("Error: No raw matrix is loaded.")

        # Rename variables for local use:
        data_raw = self.raw.matrix
        Ex_array = self.raw.Ex_array
        Eg_array = self.raw.Eg_array

        # = Import data and response matrix =

        # If energy limits are not provided, use extremal array values:
        if Ex_min == "default":
            Ex_min = Ex_array[0]
        if Ex_max == "default":
            Ex_max = Ex_array[-1]
        if Eg_min == "default":
            Eg_min = Eg_array[0]
        if Eg_max == "default":
            Eg_max = Eg_array[-1]
        
        if verbose:
            print("Unfolding in verbose mode. Starting.", flush=True)
            import time
            time_readfiles_start = time.process_time()
        # Import raw mama matrix
        # data_raw, cal, Ex_array, Eg_array = read_mama_2D(fname_data_raw)
        # data_raw, cal, Ex_array, Eg_array = read_mama_2D('/home/jorgenem/gitrepos/pyma/unfolding-testing-20161115/alfna-20160518.m') # Just to verify import works generally
        cal = {"a0x":Eg_array[0], "a1x":Eg_array[1]-Eg_array[0], "a2x":0, 
             "a0y":Ex_array[0], "a1y":Eg_array[1]-Eg_array[0], "a2y":0}
        N_Ex, N_Eg = data_raw.shape

        if verbose:
            print("Lowest Eg value =", Eg_array[0], flush=True)
            print("Lowest Ex value =", Ex_array[0], flush=True)
       

        # Import response matrix
        R, cal_R, Eg_array_R, tmp = pml.read_mama_2D(fname_resp_mat)

        if verbose:
            time_readfiles_end = time.process_time()
            time_rebin_start = time.process_time()


        # Rebin data -- take it in Ex portions to avoid memory problems:
        # N_rebin = int(N_Eg/4)
        N_rebin = N_Eg

        if N_rebin != N_Eg:
            # Allocate matrix to store finished result, filled in chunks:
            data_raw_rebinned = np.zeros((N_Ex, N_rebin)) 
        
            N_Ex_portions = 1 # How many portions to chunk, initially try just 1
            mem_avail = psutil.virtual_memory()[1]
            mem_need = 8 * N_Ex/N_Ex_portions * N_Eg * N_rebin
            if verbose:
                print("Rebinning: \nmem_avail =", mem_avail, ", mem_need =", mem_need, ", ratio =", mem_need/mem_avail, flush=True)
            while mem_need > mem_avail: # Empirical limit from my Thinkpad, corresponds to 100 % system memory load (i.e. I'm not sure what the number from psutil really means.)
                # raise Exception("Not enough memory to construct smoothing arrays. Please try rebinning the data.")
                N_Ex_portions += 1 # Double number of portions 
                mem_need = 8 * N_Ex/N_Ex_portions * N_Eg * N_rebin
                if verbose:
                    print("Adjusted to N_Ex_portions =", N_Ex_portions, "\nmem_avail =", mem_avail, ", mem_need =", mem_need, ", ratio =", mem_need/mem_avail, flush=True)
            N_Ex_portions = 10
            N_Ex_per_portion = int(N_Ex/N_Ex_portions)
            for i in range(N_Ex_portions):
                data_raw_rebinned[i*N_Ex_per_portion:(i+1)*N_Ex_per_portion,:], Eg_array_rebinned = rebin_and_shift(data_raw[i*N_Ex_per_portion:(i+1)*N_Ex_per_portion,:], Eg_array, N_rebin, rebin_axis=1)
        
            # Rebin response matrix by interpolating:
            from scipy.interpolate import RectBivariateSpline
            f_R = RectBivariateSpline(Eg_array_R, Eg_array_R, R)
            R_rebinned = f_R(Eg_array_rebinned, Eg_array_rebinned)
        
        
            # Rename everything:    
            data_raw = data_raw_rebinned
            R = R_rebinned    
            Eg_array = Eg_array_rebinned

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
            cbar_raw = ax_raw.pcolormesh(Eg_array, Ex_array, data_raw, norm=LogNorm(vmin=1))
            f.colorbar(cbar_raw, ax=ax_raw)
        


        




        # Check that response matrix matches data, at least in terms of calibration:
        eps = 1e-3
        if not (np.abs(cal["a0x"]-cal_R["a0x"])<eps and np.abs(cal["a1x"]-cal_R["a1x"])<eps and np.abs(cal["a2x"]-cal_R["a2x"])<eps):
            raise Exception("Calibration mismatch between data and response matrices")

        
        
        # = Step 1: Run iterative unfolding =
        if verbose:
            time_unfolding_start = time.process_time()
        
        
        # Set limits for excitation and gamma energy bins to be considered for unfolding
        Ex_low = 0
        Ex_high = 14000 # keV
        # Use index 0 of array as lower limit instead of energy because it can be negative!
        iEx_low, iEx_high = 0, pml.i_from_E(Ex_high, Ex_array)
        Eg_low = 0 # keV - minimum
        # Eg_low = 500 # keV 
        Eg_high = 14000 # keV
        iEg_low, iEg_high = 0, pml.i_from_E(Eg_high, Eg_array)
        Nit = 12#33 # 8 # 27
        
        # # Make masking array to cut away noise below Eg=Ex+dEg diagonal
        dEg = 3000 # keV - padding to allow for energy uncertainty above Ex=Eg diagonal
        # # Define cut   x1    y1    x2    y2
        # cut_points = [i_from_E(Eg_low + dEg, Eg_array), i_from_E(Ex_low, Ex_array), 
        #               i_from_E(Eg_high+dEg, Eg_array), i_from_E(Ex_high, Ex_array)]
        # # cut_points = [ 72,   5,  1050,  257]
        # # i_array = np.linspace(0,len(Ex_array)-1,len(Ex_array)).astype(int) # Ex axis 
        # # j_array = np.linspace(0,len(Eg_array)-1,len(Eg_array)).astype(int) # Eg axis
        # i_array = np.linspace(0,len(Ex_array)-1,len(Ex_array)).astype(int) # Ex axis 
        # j_array = np.linspace(0,len(Eg_array)-1,len(Eg_array)).astype(int) # Eg axis
        # i_mesh, j_mesh = np.meshgrid(i_array, j_array, indexing='ij')
        # mask = np.where(i_mesh > line(j_mesh, cut_points), 1, 0)
        mask = pml.make_mask(Ex_array, Eg_array, Ex_low, Eg_low+dEg, Ex_high, Eg_high+dEg)
        # HACK TEST 20181004: Does the mask do any good?:
        # mask = np.ones(mask.shape)
        
        
        
        rawmat = (data_raw*mask)[iEx_low:iEx_high,iEg_low:iEg_high] 
        
        mask_cut = mask[iEx_low:iEx_high,iEg_low:iEg_high]
        Ndof = mask_cut.sum()
        
        unfoldmat = np.zeros((rawmat.shape[0],rawmat.shape[1]))
        foldmat = np.zeros((rawmat.shape[0],rawmat.shape[1]))
        chisquares = np.zeros(Nit)
        R = R[iEg_low:iEg_high,iEg_low:iEg_high]
        # Normalize R to conserve probability
        # R = div0(R, R.sum(axis=1))
        
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

            foldmat = np.dot(R.T, unfoldmat.T).T # Have to do some transposing to get the axis orderings right for matrix product
            # 201810: Trying to debug by checking if it makes a difference to loop over rows as individual vectors (should not make a difference, and does not):
            # for i_row in range(foldmat.shape[0]):
                # foldmat[i_row,:] = np.dot(R.T, unfoldmat[i_row,:].T).T
            foldmat = mask_cut*foldmat # Apply mask for every iteration to suppress stuff below diag
            # 20171110: Tried transposing R. Was it that simple? Immediately looks good.
            #           Or at least much better. There is still something wrong giving negative counts left of peaks.
            #           Seems to come from unfolding and not from compton subtraction
        
            # Calculate reduced chisquare of the "fit" between folded-unfolded matrix and original raw
            chisquares[iteration] = pml.div0(np.power(foldmat-rawmat,2),np.where(rawmat>0,rawmat,0)).sum() #/ Ndof
            if verbose:
                print("Folding iteration = {}, chisquare = {}".format(iteration,chisquares[iteration]), flush=True)
        
        
        # Remove negative counts and trim:
        unfoldmat[unfoldmat<=0] = 0
        unfoldmat = mask_cut*unfoldmat
        
        if plot:
            # Plot:
            cbar_fold = ax_fold.pcolormesh(Eg_array[iEg_low:iEg_high], Ex_array[iEx_low:iEx_high], foldmat, norm=LogNorm(vmin=1))
            f.colorbar(cbar_fold, ax=ax_fold)
        
        if verbose:
            time_unfolding_end = time.process_time()
            time_compton_start = time.process_time()
        
        
        # = Step 2: Compton subtraction =
        if use_comptonsubtraction: # Check if compton subtraction is turned on
        
            # We also need the resp.dat file for this.
            # TODO: Consider writing a function that makes the response matrix (R) from this file
            # (or other input), so we don't have to keep track of redundant info.
            resp = []
            with open(fname_resp_dat) as file:
                # Read line by line as there is crazyness in the file format
                lines = file.readlines()
                for i in range(4,len(lines)):
                    try:
                        row = np.array(lines[i].split(), dtype="double")
                        resp.append(row)
                    except:
                        break
            
            
            resp = np.array(resp)
            # Name the columns for ease of reading
            FWHM = resp[:,1]
            eff = resp[:,2]
            pf = resp[:,3]
            pc = resp[:,4]
            ps = resp[:,5]
            pd = resp[:,6]
            pa = resp[:,7]
            
            # Correct efficiency by multiplying with EffExp(Eg):
            EffExp_array = EffExp(Eg_array)
            # eff_corr = np.append(0,eff)*EffExp_array
            eff_corr = eff*EffExp_array
        
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
        
            # Split this operation into Ex chunks to not exceed memory:
            # Allocate matrix to fill with result:
            unfolded = np.zeros(unfoldmat.shape)
        
            N_Ex_portions = 1 # How many portions to chunk, initially try just 1
            mem_avail = psutil.virtual_memory()[1]
            mem_need = 2 * 8 * N_Ex/N_Ex_portions * unfoldmat.shape[1] * unfoldmat.shape[1] # The factor 2 is needed to not crash my system. Possibly numpy copies an array somewhere, doubling required memory?
            if verbose:
                print("Compton subtraction: \nmem_avail =", mem_avail, ", mem_need =", mem_need, ", ratio =", mem_need/mem_avail, flush=True)
            while mem_need > mem_avail: 
                # raise Exception("Not enough memory to construct smoothing arrays. Please try rebinning the data.")
                N_Ex_portions += 1 # Double number of portions 
                mem_need = 2 * 8 * N_Ex/N_Ex_portions * unfoldmat.shape[1] * unfoldmat.shape[1]
            if verbose:
                print("Adjusted to N_Ex_portions =", N_Ex_portions, "\nmem_avail =", mem_avail, ", mem_need =", mem_need, ", ratio =", mem_need/mem_avail, flush=True)
        
            N_Ex_per_portion = int(N_Ex/N_Ex_portions)
            for i in range(N_Ex_portions):
                u0 = unfoldmat[i*N_Ex_per_portion:(i+1)*N_Ex_per_portion,:]
                r = rawmat[i*N_Ex_per_portion:(i+1)*N_Ex_per_portion,:]
                
                # Apply smoothing to the different peak structures. 
                # FWHM/FWHM_factor (usually FWHM/10) is used for all except 
                # single escape (FWHM*1.1/FWHM_factor)) and annihilation (FWHM*1.0). This is like MAMA.
                uf = shift_and_smooth3D(u0, Eg_array, 0.5*FWHM/FWHM_factor, pf, shift=0, smoothing=True)
                # print("uf smoothed, integral =", uf.sum())
                # uf_unsm = shift_and_smooth3D(u0, Eg_array, 0.5*FWHM/FWHM_factor, pf, shift=0, smoothing=False)
                # print("uf unsmoothed, integral =", uf_unsm.sum())
                us = shift_and_smooth3D(u0, Eg_array, 0.5*FWHM/FWHM_factor*1.1, ps, shift=511, smoothing=True)
                ud = shift_and_smooth3D(u0, Eg_array, 0.5*FWHM/FWHM_factor, pd, shift=1022, smoothing=True)
                ua = shift_and_smooth3D(u0, Eg_array, 1.0*FWHM, pa, shift="annihilation", smoothing=True)
                w = us + ud + ua
                v = uf + w
                c = r - v    
                # Smooth the Compton spectrum (using an array of 1's for the probability to only get smoothing):
                c_s = shift_and_smooth3D(c, Eg_array, 1.0*FWHM/FWHM_factor, np.ones(len(FWHM)), shift=0, smoothing=True)    
                # Subtract smoothed Compton and other structures from raw spectrum and correct for full-energy prob:
                u = div0((r - c - w), np.append(0,pf)[iEg_low:iEg_high]) # Channel 0 is missing from resp.dat    
                unfolded[i*N_Ex_per_portion:(i+1)*N_Ex_per_portion,:] = div0(u,eff_corr[iEg_low:iEg_high]) # Add Ex channel to array, also correcting for efficiency. Now we're done!

            # end if use_comptonsubtraction
        else:
            unfolded = unfoldmat


        # Diagnostic plotting:
        # f_compt, ax_compt = plt.subplots(1,1)
        # ax_compt.plot(Eg_array[iEg_low:iEg_high], r[0,:], label="r")
        # ax_compt.plot(Eg_array[iEg_low:iEg_high], u0[0,:], label="u0")
        # ax_compt.plot(Eg_array[iEg_low:iEg_high], uf[0,:], label="uf")
        # ax_compt.plot(Eg_array[iEg_low:iEg_high], uf_unsm[0,:], label="uf unsmoothed")
        # ax_compt.plot(Eg_array[iEg_low:iEg_high], us[0,:], label="us")
        # ax_compt.plot(Eg_array[iEg_low:iEg_high], ud[0,:], label="ud")
        # ax_compt.plot(Eg_array[iEg_low:iEg_high], ua[0,:], label="ua")
        # ax_compt.plot(Eg_array[iEg_low:iEg_high], w[0,:], label="w")
        # ax_compt.plot(Eg_array[iEg_low:iEg_high], v[0,:], label="v")
        # ax_compt.plot(Eg_array[iEg_low:iEg_high], c[0,:], label="c")
        # ax_compt.plot(Eg_array[iEg_low:iEg_high], c_s[0,:], label="c_s")
        # ax_compt.plot(Eg_array[iEg_low:iEg_high], u[0,:], label="u")
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
            cbar_unfold = ax_unfold.pcolormesh(Eg_array[iEg_low:iEg_high], Ex_array[iEx_low:iEx_high], unfoldmat, norm=LogNorm(vmin=1))
            f.colorbar(cbar_unfold, ax=ax_unfold)
            if use_comptonsubtraction:
                cbar_unfold_smooth = ax_unfold_smooth.pcolormesh(Eg_array[iEg_low:iEg_high], Ex_array[iEx_low:iEx_high], unfolded, norm=LogNorm(vmin=1))
                f.colorbar(cbar_unfold_smooth, ax=ax_unfold_smooth)
            plt.show()


        # Update global variables:
        self.unfolded = self.matrix(unfolded, Ex_array[iEx_low:iEx_high], Eg_array[iEg_low:iEg_high])

        # return unfolded, Ex_array[iEx_low:iEx_high], Eg_array[iEg_low:iEg_high]
        return True



    def first_generation_spectrum(self, matrix, Ex_array_mat, Egamma_array, N_Exbins, Ex_max, dE_gamma, N_iterations=1, statistical_or_total=1):
        """
        Function implementing the first generation method from Guttormsen et al. (NIM 1987)
        The code is heavily influenced by the original implementation by Magne in MAMA.
        Mainly written autumn 2016 at MSU.
        """

        # TODO option statistical_or_total=2 (total) does not work


        Ny = len(matrix[:,0])
        Nx = len(matrix[0,:])
        # Extract / calculate calibration coefficients
        bx = Egamma_array[0]
        ax = Egamma_array[1] - Egamma_array[0]
        by = Ex_array_mat[0]
        ay = Ex_array_mat[1] - Ex_array_mat[0]

        
        ThresSta = 430.0
        # AreaCorr = 1
        ThresTot =  200.000000
        ThresRatio = 0.3000000
        # ExH = 7520.00000
        ExEntry0s = 300.000000
        ExEntry0t = 0.00000000
        apply_area_correction = True


        # Ex_max = 7500 # keV - maximum excitation energy
        # Ex_min = 300 # keV - minimal excitation energy, effectively moving the ground-state energy up because we cannot resolve the low-energy yrast gamma lines. This is weighed up by also using an effective multiplicity which is lower than the real one, again not considering the low-energy yrast gammas.
        # dE_gamma = 300 # keV - allow gamma energy to exceed excitation energy by this much, to account for experimental resolution
        # Ex_binsize = 40 # keV - bin size that we want on y axis
        # N_Exbins = 120 # Number of excitation energy bins (NB! It will only rebin in whole multiples, so a slight change in N_Exbins might only result in getting some more empty bins on top.)
        # N_Exbins_original = (Ex_max+dE_gamma)/ay # The number of bins between 0 and Ex_max + dE_gamma in the original matrix
        # grouping = int(np.ceil(len(y_array[np.logical_and(0 < y_array, y_array < Ex_max + dE_gamma)])/N_Exbins)) # The integer number of bins that need to be grouped to have approximately N_Exbins bins between Ex_min and Ex_max after compression (rounded up)

        # Make arrays of Ex and Egamma axis values
        Ex_array = np.linspace(by, Ex_max + dE_gamma, N_Exbins)
        # Egamma_array = np.linspace(0,Nx-1,Nx)*ax + bx # Range of Egamma values # Update: This is passed into the function.
        
        matrix_ex_compressed = rebin(matrix[0:int((Ex_max+dE_gamma)/Ex_array_mat.max()*Ny),:], N_Exbins, rebin_axis = 0) # This seems crazy. Does it cut away anything at all?
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
        matrix_ex_compressed[Ex_array>Ex_max, :] = 0
        # plt.matshow(matrix_ex_compressed)
        # plt.colorbar()
        # plt.show()


        # ==== Calculate multiplicities: ====

        # Setup meshgrids for making boolean indexing arrays
        Egamma_mesh, Ex_mesh = np.meshgrid(Egamma_array, Ex_array)
        Egamma_max = Ex_array + dE_gamma # Maximal Egamma value for each Ex bin
        Egamma_max_grid = np.meshgrid(np.ones(Nx), Egamma_max)[1]
        if statistical_or_total == 1:
            # Statistical multiplicity calculation (i.e. trying to use statistical/continuum region only)
            slide = np.minimum( np.maximum(ThresRatio*Ex_mesh, ThresTot), ThresSta ) # The sliding lower limit for Egamma integral - sliding between ThresTot and ThresSta.
            # plt.figure(5)
            # plt.plot(slide[:,0])
            # plt.show()
            # sys.exit(0)
            # good_indices = np.where(np.logical_and(slide < Egamma_mesh, Egamma_mesh < Egamma_max_grid) , True, False)
            matrix_ex_compressed_cut = np.where(np.logical_and(slide < Egamma_mesh, Egamma_mesh < Egamma_max_grid), matrix_ex_compressed, 0)
        elif statistical_or_total == 2:
            # Total multiplicity calculation
            # good_indices = np.where(Egamma_mesh < Egamma_max_grid, True, False)
            matrix_ex_compressed_cut = np.where(Egamma_mesh < Egamma_max_grid, matrix_ex_compressed, 0)
        # for i in range(len(good_indices[:,0])):
        #   print len(good_indices[i,good_indices[i,:]]) # OK, it actually works.
        
        # Cut away counts higher than Egamma = Ex + dE_gamma
        # matrix_ex_compressed_cut = np.where(good_indices, matrix_ex_compressed, 0)
        # plt.figure(1)
        # plt.pcolormesh(Egamma_array, Ex_array, matrix_ex_compressed_cut, norm=LogNorm(vmin=0.01, vmax=matrix_ex_compressed.max()))
        # plt.show()
        # sys.exit(0)

        # Calculate average multiplicity for each Ex channel
        area_matrix_ex_compressed_cut = np.sum(matrix_ex_compressed_cut, axis=1)
        Egamma_average = div0( np.sum(Egamma_mesh * matrix_ex_compressed_cut, axis =1) , area_matrix_ex_compressed_cut )
        if statistical_or_total == 1:
            # Statistical multiplicity - use the effective Ex0 value
            multiplicity = div0( Ex_array - np.maximum( np.minimum(Ex_array - 200, ExEntry0s), 0), Egamma_average)
        elif statistical_or_total == 2:
            # Total multiplicity - use actual Ex0 = 0
            multiplicity = div0( Ex_array, Egamma_average )


        # plt.figure(2)
        # plt.step(Ex_array, multiplicity) # This rises like a straight line from 0 to about 3-4 - seems very right!
        # plt.show()
        # sys.exit(0)

        # Set up dummy first-generation matrix to start iterations, made of normalized boxes:
        H = np.zeros((N_Exbins, Nx))
        for i in range(N_Exbins):
            Ni = len(Egamma_array[Egamma_array<Ex_array[i] + dE_gamma])
            # print("Ni =", Ni, flush=True)
            H[i, Egamma_array < Ex_array[i] + dE_gamma] = 1/max(Ni, 1)
        # print np.sum(H0, axis=1) # Seems to work!

        # Set up normalization matrix N
        area = np.sum(matrix_ex_compressed_cut, axis=1) # Get total number of counts in each Ex bin
        # plt.plot(Ex_array, area)
        # plt.show()

        area_grid = np.tile(area, (N_Exbins, 1)) # Copy the array N_Exbins times down to make a square matrix
        # print area_grid.shape
        multiplicity_grid = np.tile(multiplicity, (N_Exbins, 1)) 
        # print multiplicity_grid.shape
        normalization_matrix = div0(( np.transpose(multiplicity_grid) * area_grid ) , (multiplicity_grid * np.transpose(area_grid) )).T # The transpose gives the right result. Haven't twisted my head around exactly why.
        # normalization_matrix_check = np.zeros((N_Exbins, N_Exbins))
        # for i in range(N_Exbins):
        #   for j in range(N_Exbins):
        #       normalization_matrix_check[i, j] = multiplicity[i]*area[j]/(multiplicity[j]*area[i])
        normalization_matrix[np.isnan(normalization_matrix)] = 0
        # plt.matshow(normalization_matrix, origin='lower', norm=LogNorm(vmin=0.01, vmax=normalization_matrix.max()))
        # plt.show()
        # plt.matshow(normalization_matrix_check, origin='lower') # There is a difference of a transposition, check which is the right one
        # plt.show()

        # Set up compression parameters for Egamma axis to be used by H below:
        i_Egamma_max = np.where(Egamma_array > Ex_max+ dE_gamma)[0][0] # Get the maximal allowed gamma energy (need to make H square, thus Egamma <= Ex + dE_gamma, since that's the maximal Ex channel in the compressed matrix)
        # print i_Egamma_max, Egamma_array[i_Egamma_max], N_Exbins, int(i_Egamma_max/N_Exbins)
        # i_Egamma_max = i_Egamma_max + N_Exbins - i_Egamma_max%N_Exbins # Make sure the number of indices is a whole multiple of N_Exbins (rounded up)
        # print i_Egamma_max
        grouping_Egamma = int(np.ceil(i_Egamma_max/N_Exbins))
        # print grouping_Egamma
        # Egamma_array_compressed = Egamma_array[0:i_Egamma_max]*grouping_Egamma
        Egamma_array_compressed = Ex_array

        # plt.matshow(H[:,0:i_Egamma_max])
        # plt.show()
        # H_extended = np.insert(H[:,0:i_Egamma_max], np.linspace(0,i_Egamma_max, N_Exbins - i_Egamma_max%N_Exbins), H[:,(np.linspace(0,i_Egamma_max, N_Exbins - i_Egamma_max%N_Exbins).astype(int))], axis=1)
        # H_extended[:,grouping_Egamma:-1:grouping_Egamma] /= 2
        # H_extended[:,grouping_Egamma+1:-2:grouping_Egamma] /= 2
        # H_extended = H[:,0:i_Egamma_max].repeat(N_Exbins).reshape(len(H[:,0]),N_Exbins,i_Egamma_max).sum(axis=2)/N_Exbins
        # plt.matshow(H_extended)
        # plt.show()
        # H_compressed = H[:,0:i_Egamma_max+ N_Exbins - i_Egamma_max%N_Exbins].reshape(N_Exbins, N_Exbins, grouping_Egamma).sum(axis=2)
        # plt.matshow(H_compressed)
        # plt.show()
        # H_compressed = rebin(H[:,0:i_Egamma_max], N_Exbins, 1)
        # plt.matshow(H_compressed)
        # plt.show()

        # H_compressed_extended = H_extended.reshape(N_Exbins, N_Exbins, grouping_Egamma).sum(axis=2)
        # plt.matshow(H_compressed_extended)
        # plt.show()

        # sys.exit(0)

        # Declare variables which will define the limits for the diff spectrum colorbar (for plotting purposes)
        vmin_spec = -200
        vmax_spec = 200
        vmin_diff = -100
        vmax_diff = 100

        # Prepare weight function based on Fermi gas approximation:
        a_f = 16 # 1/MeV
        n_f = 4.2 # Exponent for E_gamma
        # Make the weight array by the formula W(Ex,Eg) = Eg^n_f / (Ex-Eg)^2 * exp(2*sqrt(a*(Ex-Eg)))
        Ex_mesh, Eg_mesh = np.meshgrid(Ex_array, Ex_array, indexing="ij")
        # Make mesh of final energies:
        Ef_mesh = Ex_mesh-Eg_mesh
        # Set everything above Ex=Eg diagonal to zero so the weights also become zero
        Ef_mesh[Ef_mesh < 0] = 0
        # Calculate weights. Remember that energies are in keV, while a is in 1/MeV, so we correct in the exponent:
        W_old = np.where(Eg_mesh > 0, np.power(Eg_mesh,n_f) / np.power(Ef_mesh, 2) * np.exp(2*np.sqrt(a_f*Ef_mesh/1000)), 0)
        W_old = div0(W_old, W_old.sum(axis=1).reshape(N_Exbins,1))


        dEg = 1000
        mask_W = make_mask(Ex_array, Ex_array, Ex_array[0], Ex_array[0]+dEg, Ex_array[-1], Ex_array[-1]+dEg)

        # Perform the iterative subtraction:
        for iteration in range(N_iterations):
        # convergence_criterion = 1
        # max_diff = 100
        # while max_diff > convergence_criterion:
            # Store H from previous iteration to compare at the end
            H_old = H
            # Compress the H matrix along gamma axis to facilitate conversion to excitation energy
            # H_compressed = H[:,0:i_Egamma_max].reshape(N_Exbins, N_Exbins, grouping_Egamma).sum(axis=2)
            H_compressed = rebin(H[:,0:i_Egamma_max], N_Exbins, rebin_axis=1)

            # plt.pcolormesh(Egamma_array_compressed, Ex_array, H_compressed)
            # plt.show()

            # if iteration == 0:
                # Don't use H as weights for first iteration.
                # W = W_old
            # else:
            if True:
                # Convert first-generation spectra H into weights W
                W = np.zeros((N_Exbins, N_Exbins))
                for i in range(0,N_Exbins):
                    # print H_compressed[i,i:0:-1].shape
                    W[i,0:i] = H_compressed[i,i:0:-1]
                    # TODO Consider implementing something like Massage(), but try to understand if it is necessary first.
                # plt.figure()
                # plt.pcolormesh(Ex_array, Ex_array, W, norm=LogNorm())
                # plt.show()
                # sys.exit(0)
            # Prevent oscillations, following MAMA:
            if iteration > 4:
                W = 0.7 * W + 0.3 * W_old
            # plt.matshow(W, origin='lower', vmin=W.min(), vmax=W.max())
            # plt.colorbar()
            # plt.title('Before')
            # plt.show()
            # Remove negative weights
            W[W<0] = 0
            # Apply mask 
            # W = W * mask_W
            # Normalize each Ex channel to unity
            # W = np.where(np.invert(np.isnan(W/W.sum(axis=1).astype(float))),  W/W.sum(axis=1).astype(float), 0)
            # Remove Inf and NaN
            W = div0(W, W.sum(axis=1).reshape(N_Exbins,1))
            # Store for next iteration:
            W_old = np.copy(W)
            # W = np.nan_to_num(W) 
            # plt.matshow(W, origin='lower', vmin=W.min(), vmax=W.max())
            # plt.colorbar()
            # plt.title('After')
            # plt.show()

            # sys.exit(0)

            # print "W = "
            # print W
            # print "matrix_ex_compressed = "
            # print matrix_ex_compressed
            # print "product ="
            # plt.matshow(np.dot(W, matrix_ex_compressed), origin='lower', norm=LogNorm())
            # plt.show()

            # Calculate product of normalization matrix, weight matrix and raw count matrix
            G = np.dot( (normalization_matrix * W), matrix_ex_compressed) # Matrix of weighted sum of spectra below
            
            # Apply area correction
            if apply_area_correction:
                # Setup meshgrids for making boolean indexing arrays
                # Egamma_mesh_compressed, Ex_mesh_compressed = np.meshgrid(Egamma_array_compressed, Ex_array)
                # Egamma_max = Ex_array + dE_gamma # Maximal Egamma value for each Ex bin
                # Egamma_max_grid_compressed = np.meshgrid(np.ones(N_Exbins), Egamma_max)[1]
                # print "Egamma_mesh_compressed, Egamma_max, Egamma_max_grid"
                # print Egamma_mesh_compressed.shape, Egamma_max.shape, Egamma_max_grid.shape
                if statistical_or_total == 1:
                    # Statistical multiplicity calculation (i.e. trying to use statistical/continuum region only)
                    # slide_compressed = np.minimum( np.maximum(ThresRatio*Ex_mesh_compressed, ThresTot), ThresSta ) # The sliding lower limit for Egamma integral - sliding between ThresTot and ThresSta.
                    # print "slide_compressed"
                    # print slide_compressed.shape
                    # plt.figure(5)
                    # plt.plot(slide[:,0])
                    # plt.show()
                    # sys.exit(0)
                    # good_indices_G = np.where(np.logical_and(slide_compressed < Egamma_mesh_compressed, Egamma_mesh_compressed < Egamma_max_grid) , True, False)
                    G_area = np.where(np.logical_and(slide < Egamma_mesh, Egamma_mesh < Egamma_max_grid), G, 0).sum(axis=1)
                elif statistical_or_total == 2:
                    # Total multiplicity calculation
                    # good_indices_G = np.where(Egamma_mesh_compressed < Egamma_max_grid, True, False)
                    # G_area = np.where(Egamma_mesh_compressed < Egamma_max_grid, G, 0).sum(axis=1)
                    G_area = np.where(Egamma_mesh < Egamma_max_grid, G, 0).sum(axis=1)
                # G_area = np.where(good_indices_G, G, 0).sum(axis=1)
                # print "print G_area.shape"
                # print G_area.shape
                # print "print G_area"
                # print G_area
                alpha = np.where(G_area > 0, (1 - div0(1,multiplicity)) * div0( area_matrix_ex_compressed_cut, G_area ), 1)
                alpha[alpha < 0.85] = 0.85
                alpha[alpha > 1.15] = 1.15
                # print "alpha.shape"
                # print alpha.shape
                # print "alpha"
                # print alpha
            else:
                alpha = np.ones(N_Exbins)


            # The actual subtraction
            H = matrix_ex_compressed - alpha.reshape((len(alpha), 1))*G
            # print H.shape
            # Plotting:
            # vmin_diff = (H-H_old).min()
            # vmax_diff = (H-H_old).max()
            # vmin_spec = H.min()
            # vmax_spec = H.max()

            # plt.figure(10)
            # plt.subplot(1,2,1)
            # plt.title('First gen spectrum, current')
            # plt.pcolormesh(Egamma_array, Ex_array, H, norm=LogNorm(vmin=0.01, vmax=vmax_spec))
            # plt.colorbar()
            # plt.subplot(1,2,2)
            

            # plt.title('Diff with previous')
            # plt.pcolormesh(Egamma_array, Ex_array, H-H_old, vmin=vmin_diff, vmax=vmax_diff)
            # plt.colorbar()
            # plt.show()



            # Check convergence
            max_diff = np.max(np.abs(H-H_old))
            print("iteration =", iteration, "max_diff =", max_diff, flush=True)

        # Remove negative counts
        H[H<0] = 0
        
        # Return
        return H, H-H_old, Ex_array, Egamma_array




# === Test it ===
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    fname_raw = "data/alfna-Re187.m"

    # Initialise pyma for current experiment
    pm = pyma(fname_raw)
    # Check that it has loaded a sensible raw matrix:
    print(pm.raw.matrix.shape)

    # Do unfolding: 
    fname_resp_mat = "data/response_matrix-Re187-10keV.m"
    fname_resp_dat = "data/resp-Re187-10keV.dat"
    pm.unfold(fname_resp_mat, fname_resp_dat, use_comptonsubtraction=False, verbose=True, plot=True) # Call unfolding routine

    # Plot the unfolded matrix:
    pm.unfolded.plot()

    # Save the unfolded matrix:
    fname_save_unfolded = "data/unfolded-Re187.m"
    pm.unfolded.save(fname_save_unfolded)

    # Load the unfolded matrix from file:
    pm.load_unfolded(fname_save_unfolded)

    # f, (ax1, ax2) = plt.subplots(2,1)
    # ax1 = pm.raw.plot()
    plt.show()