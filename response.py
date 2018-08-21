import sys, os
if sys.version_info[0] < 3:
    raise Exception("Must be using Python 3")
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d

from firstgen import *
from unfold import *




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
            print("line =", line)
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
    for j in range(N_out):
        # Skip if below lower threshold
        if Eout_array[j] < Egmin:
            continue


        # Find maximal energy for current response function, 6*sigma (TODO: is it needed?)
        Egmax = Eout_array[j] + 6*FWHM*FWHM_rel[j]/2.35 
        # TODO check which factors FWHM should be multiplied with. FWHM at 1.33 MeV must be user-supplied? 
        # Also MAMA unfolds with 1/10 of real FWHM for convergence reasons.
        # But let's stick to letting FWHM denote the actual value, and divide by 10 in computations if necessary.
        
        # Find the closest energies among the available response functions, to interpolate between:
        # TODO what to do when E_out[j] is below lowest Eg_array element? Interpolate between two larger?
        i_g_low = np.where(Eg_array >= Eout_array[j])[0][0]
        i_g_high = np.where(Eg_array >= Eout_array[j])[0][1]
        print("i_g_low =", i_g_low, "i_g_high =", i_g_high, )
        print("Eout_array[{:d}] = {:.1f}".format(j, Eout_array[j]), "Eg_low =", Eg_array[i_g_low], "Eg_high =", Eg_array[i_g_high])

        sys.exit(0)



    


    for i_plt in [10,30,60,90]:
        ax.plot(Eout_array, R[i_plt,:], label="interpolated, Eout = {:.0f}".format(Eout_array[i_plt]), linestyle="--")



    ax.legend()
    plt.show()


    # Normalize and interpolate the other structures of the response:




    # return R, FWHM, eff, pf, pc, ps, pd, pa
    return True



if __name__ == "__main__":
    folderpath = "oscar2017_scale1.0"
    Eg_array = np.linspace(0,2000, 100)
    FWHM = 2.0
    print("response(\"{:s}\") =".format(folderpath), response(folderpath, Eg_array, FWHM))