"""
Library of utility functions for pyma.

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

def read_mama_2D(filename):
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


def write_mama_2D(matrix, filename, y_array, x_array, comment=""):
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


def read_response(fname_resp_mat, fname_resp_dat):
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

def div0(a, b):
    """ division function designed to ignore / 0, i.e. div0([-1, 0, 1], 0 ) -> [0, 0, 0] """
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.true_divide(a, b )
        c[ ~ np.isfinite(c )] = 0  # -inf inf NaN
    return c


def i_from_E(E, E_array):
    # Returns index of the E_array value closest to given E
    return np.argmin(np.abs(E_array - E))


def line(x, points):
    """
    Returns a line through coordinates [x1,y1,x2,y2]=points


    """
    a = (points[3]-points[1])/float(points[2]-points[0])
    b = points[1] - a*points[0]
    return a*x + b


def rebin(array, E_range, N_final, rebin_axis=0):
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


def shift_and_smooth3D(array, Eg_array, FWHM, p, shift, smoothing=True):
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



def make_mask(Ex_array, Eg_array, Ex1, Eg1, Ex2, Eg2):
    # Make masking array to cut away noise below Eg=Ex+dEg diagonal
    # Define cut   x1    y1    x2    y2
    cut_points = [i_from_E(Eg1, Eg_array), i_from_E(Ex1, Ex_array), 
                  i_from_E(Eg2, Eg_array), i_from_E(Ex2, Ex_array)]
    i_array = np.linspace(0,len(Ex_array)-1,len(Ex_array)).astype(int) # Ex axis 
    j_array = np.linspace(0,len(Eg_array)-1,len(Eg_array)).astype(int) # Eg axis
    i_mesh, j_mesh = np.meshgrid(i_array, j_array, indexing='ij')
    return np.where(i_mesh > line(j_mesh, cut_points), 1, 0)



def EffExp(Eg_array):
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


def E_array_from_calibration(a0, a1, N):
    """
    Return an array of energy values corresponding to the specified calibration.
    """
    return np.linspace(a0, a0+a1*(N-1), N)