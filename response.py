import sys, os
if sys.version_info[0] < 3:
    raise Exception("Must be using Python 3")
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d

from firstgen import *
from unfold import *


def rebin_by_arrays(counts_in, Ein_array, Eout_array):
    """
    Rebins, i.e. redistributes the numbers from vector counts 
    (which is assumed to have calibration given by Ein_array) 
    into a new vector, which is returned. 
    The total number of counts is preserved.

    In case of upsampling (smaller binsize out than in), 
    the distribution of counts between bins is done by simple 
    proportionality.

    Inputs:
    counts: Numpy array of counts in each bin
    Ein_array: Numpy array of energies, assumed to be linearly spaced, 
               corresponding to the middle-bin energies of counts
    Eout_array: Numpy array of energies corresponding to the desired
                rebinning of the output vector, also middle-bin
    """

    # Get calibration coefficients and number of elements from array:
    Nin = len(Ein_array)
    a0_in, a1_in = Ein_array[0], Ein_array[1]-Ein_array[0]
    Nout = len(Eout_array)
    a0_out, a1_out = Eout_array[0], Eout_array[1]-Eout_array[0]

    # Replace the arrays by bin-edge energy arrays of length N+1 
    # (so that all bins are covered on both sides).
    Ein_array = np.linspace(a0_in - a1_in/2, a0_in - a1_in/2 + a1_in*Nin, Nin+1)
    Eout_array = np.linspace(a0_out - a1_out/2, a0_out - a1_out/2 + a1_out*Nout, Nout+1)




    # Allocate vector to fill with rebinned counts
    counts_out = np.zeros(Nout)
    # Loop over all indices in both arrays. Maybe this can be speeded up?
    for i in range(Nout):
        for j in range(Nin):
            # Calculate proportionality factor based on current overlap:
            overlap = np.minimum(Eout_array[i+1], Ein_array[j+1]) - np.maximum(Eout_array[i], Ein_array[j])
            overlap = overlap if overlap > 0 else 0
            counts_out[i] += counts_in[j] * overlap / a1_in

    return counts_out



def response(folderpath, Eout_array, FWHM):
    """
    Function to make response matrix and related arrays from 
    source files. 
    Assumes the source files are in the folder "folderpath",
    and that they are formatted in a certain standard way.

    The function interpolates the data to give a response matrix with
    the desired energy binning specified by Eout_array.

    Inputs: 
    folderpath: The path to the folder containing Compton spectra and resp.dat
    Eout_array: The desired energies of the output response matrix. 
    FWHM: The experimental relative full-width-half-max at 1.33 MeV.
    """
    # Define helping variables from input
    N_out = len(Eout_array)
    a0_out, a1_out = Eout_array[0], Eout_array[1]-Eout_array[0]

    # Read resp.dat file, which gives information about the energy bins 
    # and discrete peaks
    resp = []
    Nlines = -1
    with open(os.path.join(folderpath, "resp.dat")) as file:
        while True:
            line = file.readline()
            if not line: 
                break
            if line[0:22] == "# Next: Numer of Lines":
                line = file.readline()
                Nlines = int(line)
                print("Nlines =", Nlines)
                break

        line = file.readline()
        # print("line =", line)
        if not line:
            raise Exception("Error reading resp.dat")

        for i in range(Nlines):
            line = file.readline()
            # print("line =", line)
            row = np.array(line.split(), dtype="double")
            resp.append(row)
    
    


    resp = np.array(resp)

    Eg_array, FWHM_rel, Eff_tot, FE, SE, DE, c511 = resp.T # Unpack the resp matrix into its columns

    # Read in Compton spectra for each Eg channel:
    N_Eg = len(Eg_array)
    # Read first Compton spectrum to get number of energy channels in each:
    N_cmp = -1
    a0_cmp, a1_cmp = -1, -1
    with open(os.path.join(folderpath,"cmp"+str(int(Eg_array[0])))) as file:
        lines = file.readlines()
        a0_cmp = float(lines[6].split(",")[1]) # calibration
        a1_cmp = float(lines[6].split(",")[2]) # coefficients [keV]
        N_cmp = int(lines[8][15:]) +1 # 0 is first index
    print("a0_cmp, a1_cmp, N_cmp = ", a0_cmp, a1_cmp, N_cmp)
    compton_matrix = np.zeros((N_Eg, N_cmp))
    # Read the rest:
    for i in range(0,N_Eg):
        fn = "cmp"+str(Eg_array[i])
        compton_matrix[i,:] = np.genfromtxt(os.path.join(folderpath,"cmp"+str(int(Eg_array[i]))), comments="!")

    print("compton_matrix =", compton_matrix)


    # == Make response matrix by interpolating to Eg_array ==
    Ecmp_array = np.linspace(a0_cmp, a1_cmp*(N_cmp-1), N_cmp)
    print("Ecmp_array =", Ecmp_array)

    f, ax = plt.subplots(1,1)
    for i_plt in [0,5,10,20,50]:
        ax.plot(Ecmp_array, compton_matrix[i_plt,:], label="Eg = {:.0f}".format(Eg_array[i_plt]))

    # We need to use the interpolation scheme given in Guttormsen 1996.
    # Brute-force it with loops to make sure I get it right (based on MAMA code)
    # After all, this is done once and for all, so it does not need to be lightning fast

    # Start looping over the rows of the response function,
    # indexed by j to match MAMA code:
    Egmin = 30 # keV -- this is universal (TODO: Is it needed?)
    for j in [0,2,10,99]: # range(N_out):
        # Skip if below lower threshold
        if Eout_array[j] < Egmin:
            continue


        # Find maximal energy for current response function, 6*sigma (TODO: is it needed?)
        Egmax = Eout_array[j] + 6*FWHM*FWHM_rel.max()/2.35 # + 6*FWHM*FWHM_rel[j]/2.35 
        # TODO does the FWHM array need to be interpolated before it is used here? j is not the right index? A quick fix since this is just an upper limit is to safeguard with FWHM.max()
        # TODO check which factors FWHM should be multiplied with. FWHM at 1.33 MeV must be user-supplied? 
        # Also MAMA unfolds with 1/10 of real FWHM for convergence reasons.
        # But let's stick to letting FWHM denote the actual value, and divide by 10 in computations if necessary.
        
        # Find the closest energies among the available response functions, to interpolate between:
        # TODO what to do when E_out[j] is below lowest Eg_array element? Interpolate between two larger?
        i_g_low = 0
        try:
            i_g_low = np.where(Eg_array <= Eout_array[j])[0][-1]
        except IndexError:
            pass
        i_g_high = N_Eg
        try:
            i_g_high = np.where(Eg_array >= Eout_array[j])[0][0]
        except IndexError:
            pass
        if i_g_low == i_g_high:
            if i_g_low > 0:
                i_g_low -= 1
            else:
                i_g_high += 1

        # TODO double check that this works for all j, that it indeed finds E_g above and below
        print("i_g_low =", i_g_low, "i_g_high =", i_g_high, )
        print("Eout_array[{:d}] = {:.1f}".format(j, Eout_array[j]), "Eg_low =", Eg_array[i_g_low], "Eg_high =", Eg_array[i_g_high])

        # Next, select the Compton spectra at index i_g_low and i_g_high. These are called Fs1 and Fs2 in MAMA.
        cmp_low = compton_matrix[i_g_low,:]
        cmp_high = compton_matrix[i_g_high,:]
        # These need to be recalibrated to Eout_array:
        cmp_low = rebin_by_arrays(cmp_low, Ecmp_array, Eout_array)
        cmp_high = rebin_by_arrays(cmp_high, Ecmp_array, Eout_array)

        # Fetch corresponding values for full-energy, etc:
        

        # Normalize total spectrum to 1:
        sum_low = 
        cmp_low = cmp_low 
        cmp_high = cmp_high / cmp_high.sum()
        # TODO: Also normalize other peaks to 

        


        
    sys.exit(0)



    


    # for i_plt in [10,30,60,90]:
    #     ax.plot(Eout_array, R[i_plt,:], label="interpolated, Eout = {:.0f}".format(Eout_array[i_plt]), linestyle="--")



    # ax.legend()
    # plt.show()


    # Normalize and interpolate the other structures of the response:




    # return R, FWHM, eff, pf, pc, ps, pd, pa
    return True



if __name__ == "__main__":
    folderpath = "oscar2017_scale1.0"
    Eg_array = np.linspace(0,2000, 100)
    FWHM = 2.0
    print("response(\"{:s}\") =".format(folderpath), response(folderpath, Eg_array, FWHM))