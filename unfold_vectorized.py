import sys
if sys.version_info[0] < 3:
    raise Exception("Must be using Python 3")
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.stats import norm
import psutil



# === Utility functions. Place in separate library file in the end. ===
def read_mama(filename):
    # Reads a MAMA matrix file and returns the matrix as a numpy array, 
    # as well as a list containing the four calibration coefficients
    # (ordered as [bx, ax, by, ay] where Ei = ai*channel_i + bi)
    # and 1-D arrays of calibrated x and y values for plotting and similar.
    matrix = np.genfromtxt(filename, skip_header=10, skip_footer=1)
    datafile = open(filename, 'r')
    calibration_line = datafile.readlines()[6].split(",")
    # a = [float(calibration_line[2][:-1]), float(calibration_line[3][:-1]), float(calibration_line[5][:-1]), float(calibration_line[6][:-1])]
    # JEM update 20180723: Changing to dict, including second-order term for generality:
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

def write_mama(matrix, filename, Egamma_range, Ex_range, comment=""):
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
    header_string +='!CALIBRATION EkeV=6, %12.6E, %12.6E, 0.000000E+00, %12.6E, %12.6E, 0.000000E+00 \n' %(Egamma_range[0], (Egamma_range[1]-Egamma_range[0]), Ex_range[0], (Ex_range[1]-Ex_range[0]))
    header_string +='!PRECISION=16 \n'
    header_string +='!DIMENSION=2,0:%4d,0:%4d \n' %(len(matrix[0,:])-1, len(matrix[:,0])-1)
    header_string +='!CHANNEL=(0:%4d,0:%4d) ' %(len(matrix[0,:])-1, len(matrix[:,0])-1)

    footer_string = "!IDEND=\n"

    # Write matrix:
    # matrix.tofile(filename, sep='       ', format="{:14.8E}")
    # matrix.tofile(filename, sep=' ', format="%-17.8E")
    np.savetxt(filename, matrix, fmt="%-17.8E", delimiter=" ", newline="\n", header=header_string, footer=footer_string, comments="")

    outfile.close()

def div0( a, b ):
    """ division function designed to ignore / 0, i.e. div0( [-1, 0, 1], 0 ) -> [0, 0, 0] """
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.true_divide( a, b )
        c[ ~ np.isfinite( c )] = 0  # -inf inf NaN
    return c



def i_from_E(E, E_array):
    # Function which returns the index of the E_array value closest to given E
    where_array = np.where(E_array > E)[0]
    # print where_array, len(where_array)
    if len(where_array) > 0:
        i = where_array[0]
        if np.abs(E_array[i]-E) > np.abs(E_array[i-1]-E):
            i -= 1
    else:
        i = len(E_array)-1
    return i



def line(x, points):
    a = (points[3]-points[1])/float(points[2]-points[0])
    b = points[1] - a*points[0]
    # print("a = {}, b = {}".format(a,b))
    return a*x + b


def rebin_and_shift(array, E_range, N_final, rebin_axis=0):
    # Function to rebin an M-dimensional array either to larger or smaller binsize.
    # Written by J{\o}rgen E. Midtb{\o}, University of Oslo, j.e.midtbo@fys.uio.no, github.com/jorgenem
    # Latest change made 20161029.

    # Rebinning is done with simple proportionality. E.g. for down-scaling rebinning (N_final < N_initial): 
    # if a bin in the original spacing ends up between two bins in the reduced spacing, 
    # then the counts of that bin are split proportionally between adjacent bins in the 
    # rebinned array. 
    # Upward binning (N_final > N_initial) is done in the same way, dividing the content of bins
    # equally among adjacent bins.

    # Technically it's done by repeating each element of array N_final times and dividing by N_final to 
    # preserve total number of counts, then reshaping the array from M dimensions to M+1 before summing 
    # along the new dimension of length N_initial, resulting in an array of the desired dimensionality.

    # This version (called rebin_and_shift rather than just rebin) takes in also the energy range array (lower bin edge)
    # corresponding to the counts array, in order to be able to change the calibration. What it does is transform the
    # coordinates such that the starting value of the rebinned axis is zero energy. This is done by shifting all
    # bins, so we are discarding some of the eventual counts in the highest energy bins. However, there is usually a margin.

    if isinstance(array, tuple): # Check if input array is actually a tuple, which may happen if rebin_and_shift() is called several times nested for different axes.
        array = array[0]

    
    N_initial = array.shape[rebin_axis] # Initial number of counts along rebin axis

    # Repeat each bin of array Nfinal times and scale to preserve counts
    array_rebinned = array.repeat(N_final, axis=rebin_axis)/N_final

    if E_range[0] < 0 or E_range[1] < E_range[0]:
        raise Exception("Error in function rebin_and_shift(): Negative zero energy is not supported. (But it should be relatively easy to implement.)")
    else:
        # Calculate number of extra slices in Nf*Ni sized array required to get down to zero energy
        n_extra = int(np.ceil(N_final * (E_range[0]/(E_range[1]-E_range[0]))))
        # Append this matrix of zero counts in front of the array
        indices_append = np.array(array_rebinned.shape)
        indices_append[rebin_axis] = n_extra
        array_rebinned = np.append(np.zeros(indices_append), array_rebinned, axis=rebin_axis)
        array_rebinned = np.split(array_rebinned, [0, N_initial*N_final], axis=rebin_axis)[1]
        indices = np.insert(array.shape, rebin_axis, N_final) # Indices to reshape to
        array_rebinned = array_rebinned.reshape(indices).sum(axis=(rebin_axis+1)) 
        E_range_shifted_and_scaled = np.linspace(0, E_range[-1]-E_range[0], N_final)
    return array_rebinned, E_range_shifted_and_scaled




def shift_and_smooth3D(array, Eg_array, FWHM, p, shift, smoothing=True):
    # Updated 201807: Trying to vectorize so all Ex bins are handled simultaneously.
    # Takes a 2D array of counts, shifts it (downward only!) with energy 'shift'
    # and smooths it with a gaussian of specified 'FWHM'.
    # This version is vectorized to shift, smooth and scale all points
    # of 'array' individually, and then sum together and return.

    # The arrays from resp.dat are missing the first channel.
    p = np.append(0, p) 
    FWHM = np.append(0, FWHM)

    a1_Eg = (Eg_array[1]-Eg_array[0]) # bin width
    N_Ex, N_Eg = array.shape

    # Shift is the same for all energies 
    if shift == "annihilation":
        # For the annihilation peak, all channels should be mapped on E = 511 keV. Of course, gamma channels below 511 keV,
        # and even well above that, cannot produce annihilation counts, but this is taken into account by the fact that p
        # is zero for these channels. Thus, we set i_shift=0 and make a special indices_shifted array to map all channels of
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



# === Unfolding -- convert the following into a function when it's done ===
    
    
if __name__=="__main__":
    fname_data_raw = 'alfna28si.m'
    fname_resp = 'resp.dat'
    fname_resp_mat = 'response-si28-20171112.m'
    
    
    # == Step 0: Import data and response matrix ==
    
    # Import raw mama matrix
    data_raw, cal, Ex_array, Eg_array = read_mama(fname_data_raw)
    # data_raw, cal, Ex_array, Eg_array = read_mama('/home/jorgenem/gitrepos/pyma/unfolding-testing-20161115/alfna-20160518.m') # Just to verify import works generally
    N_Ex, N_Eg = data_raw.shape

    print("Lowest Eg value =", Eg_array[0], flush=True)
    print("Lowest Ex value =", Ex_array[0], flush=True)
   
    # Rebin it:
    N_rebin = int(N_Eg/2)
    # Take it in Ex portions to avoid memory problems
    data_raw_rebinned = np.zeros((N_Ex, N_rebin))
    N_Ex_portions = 10
    N_Ex_per_portion = int(N_Ex/N_Ex_portions)
    for i in range(N_Ex_portions):
        data_raw_rebinned[i*N_Ex_per_portion:(i+1)*N_Ex_per_portion,:], Eg_array_rebinned = rebin_and_shift(data_raw[i*N_Ex_per_portion:(i+1)*N_Ex_per_portion,:], Eg_array, N_rebin, rebin_axis=1)



    # Import response matrix
    R, cal_R, Ex_array_R, Eg_array_R = read_mama(fname_resp_mat)

    # Rebin response matrix by interpolating:
    from scipy.interpolate import RectBivariateSpline
    f_R = RectBivariateSpline(Eg_array_R, Eg_array_R, R)
    R_rebinned = f_R(Eg_array_rebinned, Eg_array_rebinned)


    # Rename everything:    
    data_raw = data_raw_rebinned
    R = R_rebinned    
    Eg_array = Eg_array_rebinned



    # Plot raw matrix:
    f, ((ax_raw, ax_fold), (ax_unfold, ax_unfold_smooth)) = plt.subplots(2,2)
    ax_raw.set_title("raw")
    ax_fold.set_title("folded")
    ax_unfold.set_title("unfolded")
    ax_unfold_smooth.set_title("Compton subtracted")
    cbar_raw = ax_raw.pcolormesh(Eg_array_rebinned, Ex_array, data_raw, norm=LogNorm(vmin=1))
    f.colorbar(cbar_raw, ax=ax_raw)
    


    




    # Check that response matrix matches data, at least in terms of calibration:
    eps = 1e-3
    if not (np.abs(cal["a0x"]-cal_R["a0x"])<eps and np.abs(cal["a1x"]-cal_R["a1x"])<eps and np.abs(cal["a2x"]-cal_R["a2x"])<eps):
        raise Exception("Calibration mismatch between data and response matrices")
    
    
    
    
    # = Step 1: Run iterative unfolding =
    
    
    # Set limits for excitation and gamma energy bins to be considered for unfolding
    Ex_low = 0
    Ex_high = 12000 # keV
    # Use index 0 of array as lower limit instead of energy because it can be negative!
    iEx_low, iEx_high = 0, i_from_E(Ex_high, Ex_array)
    Eg_low = 0
    Eg_high = 12000 # keV
    iEg_low, iEg_high = 0, i_from_E(Eg_high, Eg_array)
    Nit = 10 #27
    
    # Make masking array to cut away noise below Eg=Ex+dEg diagonal
    dEg = 1000 # keV - padding to allow for energy uncertainty above Ex=Eg diagonal
    # Define cut   x1    y1    x2    y2
    cut_points = [i_from_E(Eg_low + dEg, Eg_array), i_from_E(Ex_low, Ex_array), 
                  i_from_E(Eg_high+dEg, Eg_array), i_from_E(Ex_high, Ex_array)]
    # cut_points = [ 72,   5,  1050,  257]
    # i_array = np.linspace(0,len(Ex_array)-1,len(Ex_array)).astype(int) # Ex axis 
    # j_array = np.linspace(0,len(Eg_array)-1,len(Eg_array)).astype(int) # Eg axis
    i_array = np.linspace(0,len(Ex_array)-1,len(Ex_array)).astype(int) # Ex axis 
    j_array = np.linspace(0,len(Eg_array)-1,len(Eg_array)).astype(int) # Eg axis
    i_mesh, j_mesh = np.meshgrid(i_array, j_array, indexing='ij')
    mask = np.where(i_mesh > line(j_mesh, cut_points), 1, 0)
    
    
    
    rawmat = (data_raw*mask)[iEx_low:iEx_high,iEg_low:iEg_high] 
    
    mask_cut = mask[iEx_low:iEx_high,iEg_low:iEg_high]
    Ndof = mask_cut.sum()
    
    unfoldmat = np.zeros((rawmat.shape[0],rawmat.shape[1]))
    foldmat = np.zeros((rawmat.shape[0],rawmat.shape[1]))
    chisquares = np.zeros(Nit)
    R = R[iEg_low:iEg_high,iEg_low:iEg_high]
    
    # Run folding iterations:
    for iteration in range(Nit):
        if iteration == 0:
            unfoldmat = rawmat
        else:
            unfoldmat = unfoldmat + (rawmat - foldmat)
        foldmat = np.dot(R.T, unfoldmat.T).T # Have to do some transposing to get the axis orderings right for matrix product
        foldmat = mask_cut*foldmat # Apply mask for every iteration to suppress stuff below diag
        # 20171110: Tried transposing R. Was it that simple? Immediately looks good.
        #           Or at least much better. There is still something wrong giving negative counts left of peaks.
        #           Seems to come from unfolding and not from compton subtraction
    
        # Calculate reduced chisquare of the "fit" between folded-unfolded matrix and original raw
        chisquares[iteration] = div0(np.power(foldmat-rawmat,2),np.where(rawmat>0,rawmat,0)).sum() / Ndof
        print("Folding iteration = {}, chisquare = {}".format(iteration,chisquares[iteration]), flush=True)
    
    
    # Remove negative counts and trim:
    unfoldmat[unfoldmat<=0] = 0
    unfoldmat = mask_cut*unfoldmat
    
    # Plot:
    cbar_fold = ax_fold.pcolormesh(Eg_array[iEg_low:iEg_high], Ex_array[iEx_low:iEx_high], foldmat, norm=LogNorm(vmin=1))
    f.colorbar(cbar_fold, ax=ax_fold)
    
    
    
    # = Step 2: Compton subtraction =
    
    # We also need the resp.dat file for this.
    # TODO: Consider writing a function that makes the response matrix (R) from this file
    # (or other input), so we don't have to keep track of redundant info.
    resp = []
    with open(fname_resp) as f:
        # Read line by line as there is crazyness in the file format
        lines = f.readlines()
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
    
    # We follow the notation of Guttormsen et al (NIM 1996) in what follows.
    # u0 is the unfolded spectrum from above, r is the raw spectrum, 
    # w = us + ud + ua is the folding contributions from everything except Compton,
    # i.e. us = single escape, ua = double escape, ua = annihilation (511).
    # v = pf*u0 + w == uf + w is the estimated "raw minus Compton" spectrum
    # c is the estimated Compton spectrum.
    
    # Plot compton subtraction spectra for debugging:
    f_compt, ax_compt = plt.subplots(1,1)
    
    # TODO consider parallelizing this loop? Or can it be simply vectorized?
    # Allocate array to store properly unfolded spectrum:
    unfolded = np.zeros(unfoldmat.shape)
    # Select a single channel for testing:
    i_Ex_testing = range(0,iEx_high)


    # Check that there is enough memory:
    mem_avail = psutil.virtual_memory()[1]
    mem_need = 8 * len(i_Ex_testing) * unfoldmat.shape[1] * unfoldmat.shape[1]
    print("mem_avail =", mem_avail, ", mem_need =", mem_need, ", ratio =", mem_need/mem_avail)
    if mem_need > 0.48*mem_avail: # Empirical limit from my Thinkpad, corresponds to 100 % system memory load
        raise Exception("Not enough memory to construct smoothing arrays. Please try rebinning the data.")



    u0 = unfoldmat[i_Ex_testing,:]
    print("u0.shape =", u0.shape, flush=True)
    r = rawmat[i_Ex_testing,:]
    # pf = np.ones(u0.shape[1]) # Test
    uf = shift_and_smooth3D(u0, Eg_array, 0.5*FWHM, pf, shift=0, smoothing=True)
    print("uf smoothed, integral =", uf.sum())
    uf_unsm = shift_and_smooth3D(u0, Eg_array, 0.5*FWHM, pf, shift=0, smoothing=False)
    print("uf unsmoothed, integral =", uf_unsm.sum())
    us = shift_and_smooth3D(u0, Eg_array, 0.5*FWHM, ps, shift=511, smoothing=True)
    ud = shift_and_smooth3D(u0, Eg_array, 0.5*FWHM, pd, shift=1022, smoothing=True)
    ua = shift_and_smooth3D(u0, Eg_array, 1.0*FWHM, pa, shift="annihilation", smoothing=True)
    w = us + ud + ua
    v = uf + w
    c = r - v    
    # Smooth the Compton spectrum (using an array of 1's for the probability to only get smoothing):
    c_s = shift_and_smooth3D(c, Eg_array, 1.0*FWHM, np.ones(len(FWHM)), shift=0, smoothing=True)    
    # Subtract smoothed Compton and other structures from raw spectrum and correct for full-energy prob:
    u = div0((r - c - w),np.append(0,pf)[iEg_low:iEg_high]) # Channel 0 is missing from resp.dat    
    unfolded = div0(u,np.append(0,eff)[iEg_low:iEg_high]) # Add Ex channel to array, also correcting for efficiency. Now we're done!
    # print(unfolded[i_Ex,:], flush=True)    
    # Diagnostic plotting:
    # ax_compt.plot(Eg_array[iEg_low:iEg_high], r[0,:], label="r")
    ax_compt.plot(Eg_array[iEg_low:iEg_high], u0[0,:], label="u0")
    ax_compt.plot(Eg_array[iEg_low:iEg_high], uf[0,:], label="uf")
    ax_compt.plot(Eg_array[iEg_low:iEg_high], uf_unsm[0,:], label="uf unsmoothed")
    ax_compt.plot(Eg_array[iEg_low:iEg_high], us[0,:], label="us")
    ax_compt.plot(Eg_array[iEg_low:iEg_high], ud[0,:], label="ud")
    ax_compt.plot(Eg_array[iEg_low:iEg_high], ua[0,:], label="ua")
    # ax_compt.plot(Eg_array[iEg_low:iEg_high], w[0,:], label="w")
    # ax_compt.plot(Eg_array[iEg_low:iEg_high], v[0,:], label="v")
    # ax_compt.plot(Eg_array[iEg_low:iEg_high], c[0,:], label="c")
    # ax_compt.plot(Eg_array[iEg_low:iEg_high], c_s[0,:], label="c_s")
    # ax_compt.plot(Eg_array[iEg_low:iEg_high], u[0,:], label="u")
    ax_compt.legend()
    
    
    # Trim result:
    unfolded = mask_cut[i_Ex_testing,:]*unfolded
    
    # Plot unfolded and Compton subtracted matrices:
    cbar_unfold = ax_unfold.pcolormesh(Eg_array[iEg_low:iEg_high], Ex_array[iEx_low:iEx_high], unfoldmat, norm=LogNorm(vmin=1))
    # f.colorbar(cbar_unfold, ax=ax_unfold)
    cbar_unfold_smooth = ax_unfold_smooth.pcolormesh(Eg_array[iEg_low:iEg_high], Ex_array[i_Ex_testing], unfolded, norm=LogNorm(vmin=1))
    # f.colorbar(cbar_unfold_smooth, ax=ax_unfold_smooth)
    
    # Save compton subtracted matrix
    write_mama(unfolded, 'unfolded-28Si.m', Eg_array[iEg_low:iEg_high], Ex_array[iEx_low:iEx_high], comment="Unfolded using unfold.py by JEM, during development of pyma, summer 2018")
    
    # f_1d, ax_1d = plt.subplots(1,1)
    # for i in range(len(i_Ex_testing)):
    #     i_Ex = i_Ex_testing[i]
    #     ax_1d.plot(Eg_array[iEg_low:iEg_high], unfolded[i,:], label="i_Ex={:d}".format(i_Ex))
    
    plt.show()