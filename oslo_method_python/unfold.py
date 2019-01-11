from .library import *


def unfold(raw, fname_resp_mat=None, fname_resp_dat=None, FWHM_factor=10,
           Ex_min=None, Ex_max=None, Eg_min=None,
           Eg_max=None, verbose=False, plot=False,
           use_comptonsubtraction=False):
    """Unfolds the gamma-detector response of a spectrum

    Args:
        raw (Matrix): the raw matrix to unfold, an instance of Matrix()
        fname_resp_mat (str): file name of the response matrix, in MAMA format
        fname_resp_dat (str): file name of the resp.dat file made by MAMA
        Ex_min (float): Lower limit for excitation energy
        Ex_max (float): Upper limit for excitation energy
        Eg_min (float): Lower limit for gamma-ray energy
        Eg_max (float): Upper limit for gamma-ray energy
        verbose (bool): Toggle verbose mode
        plot (bool): Toggle plotting
        use_comptonsubtraction (bool): Toggle whether to use the Compton 
                                        subtraction method

    Returns:
        unfolded -- the unfolded matrix as an instance of the Matrix() class

    Todo:
        Implement the Matrix() and Vector() classes throughout the function.
    """

    if fname_resp_mat is None or fname_resp_dat is None:
        raise Exception(
            "fname_resp_mat and/or fname_resp_dat not given, and "
            "no response matrix is previously loaded."
            )

    # Rename variables for local use:
    data_raw = raw.matrix
    Ex_array = raw.Ex_array
    Eg_array = raw.Eg_array

    use_comptonsubtraction = use_comptonsubtraction

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
    cal = {"a0x":Eg_array[0], "a1x":Eg_array[1]-Eg_array[0], "a2x":0, 
         "a0y":Ex_array[0], "a1y":Eg_array[1]-Eg_array[0], "a2y":0}
    N_Ex, N_Eg = data_raw.shape

    if verbose:
        print("Lowest Eg value =", Eg_array[0], flush=True)
        print("Lowest Ex value =", Ex_array[0], flush=True)
   

    # Import response matrix
    R, cal_R, Eg_array_R, tmp = read_mama_2D(fname_resp_mat)
    # Copy it to a global variable
    response = Matrix(R, Eg_array_R, Eg_array_R)  # Both axes are gamma energies here, but it does not matter for Matrix()

    if verbose:
        time_readfiles_end = time.process_time()
        time_rebin_start = time.process_time()


    # Rebin data -- take it in Ex portions to avoid memory problems:
    # N_rebin = int(N_Eg/4)
    N_rebin = N_Eg

    # Update 2019: Do not think rebin should be done inside the unfolding function
    # if N_rebin != N_Eg:
    #     # Allocate matrix to store finished result, filled in chunks:
    #     data_raw_rebinned = np.zeros((N_Ex, N_rebin)) 
    
    #     N_Ex_portions = 1 # How many portions to chunk, initially try just 1
    #     mem_avail = psutil.virtual_memory()[1]
    #     mem_need = 8 * N_Ex/N_Ex_portions * N_Eg * N_rebin
    #     if verbose:
    #         print("Rebinning: \nmem_avail =", mem_avail, ", mem_need =", mem_need, ", ratio =", mem_need/mem_avail, flush=True)
    #     while mem_need > mem_avail: # Empirical limit from my Thinkpad, corresponds to 100 % system memory load (i.e. I'm not sure what the number from psutil really means.)
    #         # raise Exception("Not enough memory to construct smoothing arrays. Please try rebinning the data.")
    #         N_Ex_portions += 1 # Double number of portions 
    #         mem_need = 8 * N_Ex/N_Ex_portions * N_Eg * N_rebin
    #         if verbose:
    #             print("Adjusted to N_Ex_portions =", N_Ex_portions, "\nmem_avail =", mem_avail, ", mem_need =", mem_need, ", ratio =", mem_need/mem_avail, flush=True)
    #     N_Ex_portions = 10
    #     N_Ex_per_portion = int(N_Ex/N_Ex_portions)
    #     for i in range(N_Ex_portions):
    #         data_raw_rebinned[i*N_Ex_per_portion:(i+1)*N_Ex_per_portion,:], Eg_array_rebinned = rebin_and_shift(data_raw[i*N_Ex_per_portion:(i+1)*N_Ex_per_portion,:], Eg_array, N_rebin, rebin_axis=1)
    
    #     # # Rebin response matrix by interpolating:
    #     # NO! This does not work. Need to use more advanced interpolation, see MAMA.
    #     # from scipy.interpolate import RectBivariateSpline
    #     # f_R = RectBivariateSpline(Eg_array_R, Eg_array_R, R)
    #     # R_rebinned = f_R(Eg_array_rebinned, Eg_array_rebinned)
    
    
    #     # Rename everything:    
    #     data_raw = data_raw_rebinned
    #     R = R_rebinned    
    #     Eg_array = Eg_array_rebinned

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
    iEx_low, iEx_high = 0, i_from_E(Ex_high, Ex_array)
    Eg_low = 0 # keV - minimum
    # Eg_low = 500 # keV 
    Eg_high = 14000 # keV
    iEg_low, iEg_high = 0, i_from_E(Eg_high, Eg_array)
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
    mask = make_mask(Ex_array, Eg_array, Ex_low, Eg_low+dEg, Ex_high, Eg_high+dEg)
    # HACK TEST 20181004: Does the mask do any good?:
    mask = np.ones(mask.shape)
    
    
    
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
        chisquares[iteration] = div0(np.power(foldmat-rawmat,2),np.where(rawmat>0,rawmat,0)).sum() #/ Ndof
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
        print("From unfold(): eff.shape =", eff.shape, "EffExp_array.shape =", EffExp_array.shape, flush=True)
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


    # DEBUG: Print Ex array to find out why it 


    # Update global variables:
    unfolded = Matrix(unfolded, Ex_array[iEx_low:iEx_high], Eg_array[iEg_low:iEg_high])

    return unfolded
