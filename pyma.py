# An implementation of MAMA functionality in Python
# Made by Joergen E. Midtboe, University of Oslo
# 2016-2018

import numpy as np 



class pyma():
    def __init__(self, fname_raw):
        self.fname_raw = fname_raw # File name of raw spectrum

        self.matrix_raw, self.calib_raw, self.Ex_array_raw, self.Eg_array_raw = self.read_mama_2D(fname_raw)

        self.matrix_unfolded = None
        self.matrix_firstgen = None


    def read_mama_2D(self,filename):
        # Reads a MAMA matrix file and returns the matrix as a numpy array, 
        # as well as a list containing the four calibration coefficients
        # (ordered as [bx, ax, by, ay] where Ei = ai*channel_i + bi)
        # and 1-D arrays of calibrated x and y values for plotting and similar.
        matrix = np.genfromtxt(filename, skip_header=10, skip_footer=1)
        cal = {}
        with open(filename, 'r') as datafile:
            calibration_line = datafile.readlines()[6].split(",")
            # a = [float(calibration_line[2][:-1]), float(calibration_line[3][:-1]), float(calibration_line[5][:-1]), float(calibration_line[6][:-1])]
            # JEM update 20180723: Changing to dict, including second-order term for generality:
            # print("calibration_line =", calibration_line, flush=True)
            cal = {"a0x":float(calibration_line[1]), "a1x":float(calibration_line[2]), "a2x":float(calibration_line[3]), 
                 "a0y":float(calibration_line[4]), "a1y":float(calibration_line[5]), "a2y":float(calibration_line[6])}
        # TODO: INSERT CORRECTION FROM CENTER-BIN TO LOWER EDGE CALIBRATION HERE.
        # MAKE SURE TO CHECK rebin_and_shift() WHICH MIGHT NOT LIKE NEGATIVE SHIFT COEFF.
        # (alternatively consider using center-bin throughout, but then need to correct when plotting.)
        Ny, Nx = matrix.shape
        y_array = np.linspace(0, Ny-1, Ny)
        y_array = cal["a0y"] + cal["a1y"]*y_array + cal["a2y"]*y_array**2
        x_array = np.linspace(0, Nx-1, Nx)
        x_array = cal["a0x"] + cal["a1x"]*x_array + cal["a2x"]*x_array**2
        # x_array = np.linspace(cal["a0x"], cal["a0x"]+cal["a1x"]*Nx, Nx) # BIG TODO: This is probably center-bin calibration, 
        # x_array = np.linspace(a[2], a[2]+a[3]*(Ny), Ny) # and should be shifted down by half a bin?
                                                        # Update 20171024: Started changing everything to lower bin edge,
                                                        # but started to hesitate. For now I'm inclined to keep it as
                                                        # center-bin everywhere. 
        return matrix, cal, y_array, x_array # Returning y (Ex) first as this is axis 0 in matrix language


    def write_mama_2D(self, matrix, filename, y_array, x_array, comment=""):
        import time
        outfile = open(filename, 'w')

        # Write mandatory header:
        # outfile.write('!FILE=Disk \n')
        # outfile.write('!KIND=Spectrum \n')
        # outfile.write('!LABORATORY=Oslo Cyclotron Laboratory (OCL) \n')
        # outfile.write('!EXPERIMENT=pyma \n')
        # outfile.write('!COMMENT=none|RE:alfna-20FN:RN:UN:FN:RN: \n')
        # outfile.write('!TIME=DATE:'+time.strftime("%d-%b-%y %H:%M:%S", time.localtime())+'   \n')
        # outfile.write('!CALIBRATION EkeV=6, %12.6E, %12.6E, 0.000000E+00, %12.6E, %12.6E, 0.000000E+00 \n' %(Egamma_range[0], (Egamma_range[1]-Egamma_range[0]), Ex_range[0], (Ex_range[1]-Ex_range[0])))
        # outfile.write('!PRECISION=16 \n')
        # outfile.write('!DIMENSION=2,0:%4d,0:%4d \n' %(len(matrix[:,0]), len(matrix[0,:])))
        # outfile.write('!CHANNEL=(0:%4d,0:%4d) \n' %(len(matrix[:,0]), len(matrix[0,:])))
        header_string ='!FILE=Disk \n'
        header_string +='!KIND=Spectrum \n'
        header_string +='!LABORATORY=Oslo Cyclotron Laboratory (OCL) \n'
        header_string +='!EXPERIMENT= pyma \n'
        header_string +='!COMMENT={:s} \n'.format(comment)
        header_string +='!TIME=DATE:'+time.strftime("%d-%b-%y %H:%M:%S", time.localtime())+'   \n'
        header_string +='!CALIBRATION EkeV=6, %12.6E, %12.6E, 0.000000E+00, %12.6E, %12.6E, 0.000000E+00 \n' %(x_array[0], (x_array[1]-x_array[0]), y_array[0], (y_array[1]-y_array[0]))
        header_string +='!PRECISION=16 \n'
        header_string +="!DIMENSION=2,0:{:4d},0:{:4d} \n".format(len(matrix[0,:])-1, len(matrix[:,0])-1)
        header_string +='!CHANNEL=(0:%4d,0:%4d) ' %(len(matrix[0,:])-1, len(matrix[:,0])-1)

        footer_string = "!IDEND=\n"

        # Write matrix:
        # matrix.tofile(filename, sep='       ', format="{:14.8E}")
        # matrix.tofile(filename, sep=' ', format="%-17.8E")
        np.savetxt(filename, matrix, fmt="%-17.8E", delimiter=" ", newline="\n", header=header_string, footer=footer_string, comments="")

        outfile.close()


    def read_response(self, fname_resp_mat, fname_resp_dat):
        # Import response matrix
        R, cal_R, Eg_array_R, tmp = read_mama_2D(fname_resp_mat)
        # We also need info from the resp.dat file:
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
        FWHM = resp[:,1]#*6.8 # Correct with fwhm @ 1.33 MeV?
        eff = resp[:,2]
        pf = resp[:,3]
        pc = resp[:,4]
        ps = resp[:,5]
        pd = resp[:,6]
        pa = resp[:,7]
        
        return R, FWHM, eff, pc, pf, ps, pd, pa, Eg_array_R

    def div0(self, a, b):
        """ division function designed to ignore / 0, i.e. div0( [-1, 0, 1], 0 ) -> [0, 0, 0] """
        with np.errstate(divide='ignore', invalid='ignore'):
            c = np.true_divide( a, b )
            c[ ~ np.isfinite( c )] = 0  # -inf inf NaN
        return c


    def i_from_E(self, E, E_array):
        # Returns index of the E_array value closest to given E
        return np.argmin(np.abs(E_array - E))


    def line(self, x, points):
        """
        Returns a line through coordinates [x1,y1,x2,y2]=points


        """
        a = (points[3]-points[1])/float(points[2]-points[0])
        b = points[1] - a*points[0]
        return a*x + b


    def rebin(self, array, E_range, N_final, rebin_axis=0):
        """
        Rebins an array to have N_final bins

        Inputs:
        array -- the array to rebin
        E_range -- the energy values specifying the calibration of the array 
                   (lower bin edge + assumes linear calibration)
        N_final -- the desired number of bins after rebin
        rebin_axis -- which axis of array to rebin, if multi-dimensional. 
                      In that case, E_range must correspond to chosen axis.


        Returns: 
        array_final -- the rebinned array
        E_range_final -- the array of lower bin edge values in the rebinned array

        
        TODO: Implement a "memory guard" to avoid running out of memory by chunking 
        the .repeat() operations into parts if there is not enough memory to do every-
        thing at once.

        """
        if isinstance(array, tuple): # Check if input array is actually a tuple, which may happen if function is called several times nested for different axes.
            array = array[0]

        N_orig = array.shape[rebin_axis]
        dim = np.insert(array.shape, rebin_axis, N_final)

        # TODO insert a loop here over chunks along another axis than the rebin axis, to try to avoid memory problems
        array_final = (array.repeat(N_final)/N_final).reshape(dim).sum(axis=(rebin_axis+1))

        # Recalculate calibration:
        a0 = E_range[0]
        a1_orig = E_range[1]-E_range[0]
        a1_final = N_orig/N_final*a1_orig
        E_range_final = np.linspace(a0, a0 + a1_final*(N_final-1), N_final)

        return array_final, E_range_final


    def shift_and_smooth3D(self, array, Eg_array, FWHM, p, shift, smoothing=True):
        # Updated 201807: Trying to vectorize so all Ex bins are handled simultaneously.
        # Takes a 2D array of counts, shifts it (downward only!) with energy 'shift'
        # and smooths it with a gaussian of specified 'FWHM'.
        # This version is vectorized to shift, smooth and scale all points
        # of 'array' individually, and then sum together and return.

        # TODO: FIX ME! There is a bug here, it does not do Compton subtraction right.

        # The arrays from resp.dat are missing the first channel.
        p = np.append(0, p) 
        FWHM = np.append(0, FWHM)

        a1_Eg = (Eg_array[1]-Eg_array[0]) # bin width
        N_Ex, N_Eg = array.shape

        # Shift is the same for all energies 
        if shift == "annihilation":
            # For the annihilation peak, all channels should be mapped on E = 511 keV. Of course, gamma channels below 511 keV,
            # and even well above that, cannot produce annihilation counts, but this is taken into account by the fact that p
            # is zero for these channels. Thus, we set i_shift=0 and make a special dimensions_shifted array to map all channels of
            # original array to i(511). 
            i_shift = 0 
        else:
            i_shift = i_from_E(shift, Eg_array) - i_from_E(0, Eg_array) # The number of indices to shift by


        N_Eg_sh = N_Eg - i_shift
        indices_original = np.linspace(i_shift, N_Eg-1, N_Eg-i_shift).astype(int) # Index array for original array, truncated to shifted array length
        if shift == "annihilation": # If this is the annihilation peak then all counts should end up with their centroid at E = 511 keV
            # indices_shifted = (np.ones(N_Eg-i_from_E(511, Eg_array))*i_from_E(511, Eg_array)).astype(int)
            indices_shifted = (np.ones(N_Eg)*i_from_E(511, Eg_array)).astype(int)
        else:
            indices_shifted = np.linspace(0,N_Eg-i_shift-1,N_Eg-i_shift).astype(int) # Index array for shifted array


        if smoothing:
            # Scale each Eg count by the corresponding probability
            # Do this for all Ex bins at once:
            array = array * p[0:N_Eg].reshape(1,N_Eg)
            # Shift array down in energy by i_shift indices,
            # so that index i_shift of array is index 0 of array_shifted.
            # Also flatten array along Ex axis to facilitate multiplication.
            array_shifted_flattened = array[:,indices_original].ravel()
            # Make an array of N_Eg_sh x N_Eg_sh containing gaussian distributions 
            # to multiply each Eg channel by. This array is the same for all Ex bins,
            # so it will be repeated N_Ex times and stacked for multiplication
            # To get correct normalization we multiply by bin width
            pdfarray = a1_Eg* norm.pdf(
                                np.tile(Eg_array[0:N_Eg_sh], N_Eg_sh).reshape((N_Eg_sh, N_Eg_sh)),
                                loc=Eg_array[indices_shifted].reshape(N_Eg_sh,1),
                                scale=FWHM[indices_shifted].reshape(N_Eg_sh,1)/2.355
                            )
                            
            # Remove eventual NaN values:
            pdfarray = np.nan_to_num(pdfarray, copy=False)
            # print("Eg_array[indices_shifted] =", Eg_array[indices_shifted], flush=True)
            # print("pdfarray =", pdfarray, flush=True)
            # Repeat and stack:
            pdfarray_repeated_stacked = np.tile(pdfarray, (N_Ex,1))

            # Multiply array of counts with pdfarray:
            multiplied = pdfarray_repeated_stacked*array_shifted_flattened.reshape(N_Ex*N_Eg_sh,1)

            # Finally, for each Ex bin, we now need to sum the contributions from the smoothing
            # of each Eg bin to get a total Eg spectrum containing the entire smoothed spectrum:
            # Do this by reshaping into 3-dimensional array where each Eg bin (axis 0) contains a 
            # N_Eg_sh x N_Eg_sh matrix, where each row is the smoothed contribution from one 
            # original Eg pixel. We sum the columns of each of these matrices:
            array_out = multiplied.reshape((N_Ex, N_Eg_sh, N_Eg_sh)).sum(axis=1)
            # print("array_out.shape =", array_out.shape)
            # print("array.shape[0],array.shape[1]-N_Eg_sh =", array.shape[0],array.shape[1]-N_Eg_sh)

        else:
            # array_out = np.zeros(N)
            # for i in range(N):
            #     try:
            #         array_out[i-i_shift] = array[i] #* p[i+1]
            #     except IndexError:
            #         pass

            # Instead of above, vectorizing:
            array_out = p[indices_original].reshape(1,N_Eg_sh)*array[:,indices_original]

        # Append zeros to the end of Eg axis so we match the length of the original array:
        if i_shift > 0:
            array_out = np.concatenate((array_out, np.zeros((N_Ex, N_Eg-N_Eg_sh))),axis=1)
        return array_out



    def make_mask(self, Ex_array, Eg_array, Ex1, Eg1, Ex2, Eg2):
        # Make masking array to cut away noise below Eg=Ex+dEg diagonal
        # Define cut   x1    y1    x2    y2
        cut_points = [i_from_E(Eg1, Eg_array), i_from_E(Ex1, Ex_array), 
                      i_from_E(Eg2, Eg_array), i_from_E(Ex2, Ex_array)]
        i_array = np.linspace(0,len(Ex_array)-1,len(Ex_array)).astype(int) # Ex axis 
        j_array = np.linspace(0,len(Eg_array)-1,len(Eg_array)).astype(int) # Eg axis
        i_mesh, j_mesh = np.meshgrid(i_array, j_array, indexing='ij')
        return np.where(i_mesh > line(j_mesh, cut_points), 1, 0)



    def EffExp(self, Eg_array):
        # Function from MAMA which makes an additional efficiency correction based on discriminator thresholds etc.
        # Basically there is a hard-coded set of energies and corresponding efficiencies in MAMA, and it should be 
        # zero below and 1 above this range.
        Egs = np.array([30.,80.,122.,183.,244.,294.,344.,562.,779.,1000])
        Effs = np.array([0.0,0.0,0.0,0.06,0.44,0.60,0.87,0.99,1.00,1.000])
        EffExp_array =  np.zeros(len(Eg_array))
        for iEg in range(len(Eg_array)):
            if Eg_array[iEg] < Egs.min():
                EffExp_array[iEg] = 0
            elif Eg_array[iEg] >= Egs.max():
                EffExp_array[iEg] = 1
            else:
                EffExp_array[iEg] = Effs[np.argmin(np.abs(Egs - Eg_array[iEg]))]

        return EffExp_array



    def unfold(self, data_raw, Ex_array, Eg_array, fname_resp_dat, fname_resp_mat, FWHM_factor=10, Ex_min="default", Ex_max="default", Eg_min="default", Eg_max="default", verbose=False, plot=False, use_comptonsubtraction=True):
        # = Step 0: Import data and response matrix =

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
            ax_unfold_smooth.set_title("Compton subtracted")
            # Plot raw matrix:
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
            print("Time elapsed: \nFile import = {:f} s \nRebinning = {:f} s \nUnfolding = {:f} s \nCompton subtraction = {:f} s".format(time_readfiles_end-time_readfiles_start, time_rebin_end-time_rebin_start, time_unfolding_end-time_unfolding_start, time_compton_end-time_compton_start), flush=True)



        if plot:
            # Plot unfolded and Compton subtracted matrices:
            cbar_unfold = ax_unfold.pcolormesh(Eg_array[iEg_low:iEg_high], Ex_array[iEx_low:iEx_high], unfoldmat, norm=LogNorm(vmin=1))
            f.colorbar(cbar_unfold, ax=ax_unfold)
            cbar_unfold_smooth = ax_unfold_smooth.pcolormesh(Eg_array[iEg_low:iEg_high], Ex_array[iEx_low:iEx_high], unfolded, norm=LogNorm(vmin=1))
            f.colorbar(cbar_unfold_smooth, ax=ax_unfold_smooth)
            plt.show()



        return unfolded, Ex_array[iEx_low:iEx_high], Eg_array[iEg_low:iEg_high]



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
    fname_raw = "data/alfna-Re187.m"
    pm = pyma(fname_raw)
