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


def E_compton(Eg, theta):
    """
    Calculates the energy of an electron that is scattered an angle
    theta by a gamma-ray of energy Eg.
    Adapted from MAMA, file "folding.f", which references 
    Canberra catalog ed.7, p.2.

    Inputs:
    Eg: Energy of gamma-ray in keV
    theta: Angle of scatter in radians

    Returns:
    Energy Ee of scattered electron
    """
    # Return Eg if Eg <= 0.1, else use formula
    print("From E_compton(): Eg =", Eg, ", theta =", theta, ", formula =", Eg*Eg/511*(1-np.cos(theta)) / (1+Eg/511 * (1-np.cos(theta))))
    return np.where(Eg > 0.1, Eg*Eg/511*(1-np.cos(theta)) / (1+Eg/511 * (1-np.cos(theta))), Eg)

def corr(Eg, theta):
    """
    Function to correct number of counts due to delta(theta)
    Adapted from MAMA in the file kelvin.f
    It is dE/dtheta of the E(theta) in Eq. (2) in Guttormsen 1996.
    """
    return (Eg*Eg/511*np.sin(theta))/(1+Eg/511*(1-np.cos(theta)))**2


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
    print("a0_out, a1_out =", a0_out, a1_out)

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

    Eg_sim_array, FWHM_rel, Eff_tot, FE, SE, DE, c511 = resp.T # Unpack the resp matrix into its columns
    a0_sim, a1_sim = Eg_sim_array[0], Eg_sim_array[1]-Eg_sim_array[0]
    print("a0_sim, a1_sim =", a0_sim, a1_sim, flush=True)
    # "Eg_sim" means "gamma, simulated", and refers to the gamma energies where we have simulated Compton spectra.

    # Read in Compton spectra for each Eg channel:
    N_Eg = len(Eg_sim_array)
    # Read first Compton spectrum to get number of energy channels in each:
    N_cmp = -1
    a0_cmp, a1_cmp = -1, -1
    # Get calibration and array length from highest-energy spectrum, because the spectra
    # may have differing length but this is bound to be the longest.
    with open(os.path.join(folderpath,"cmp"+str(int(Eg_sim_array[-1])))) as file:
        lines = file.readlines()
        a0_cmp = float(lines[6].split(",")[1]) # calibration
        a1_cmp = float(lines[6].split(",")[2]) # coefficients [keV]
        N_cmp = int(lines[8][15:]) +1 # 0 is first index
    print("a0_cmp, a1_cmp, N_cmp = ", a0_cmp, a1_cmp, N_cmp)
    compton_matrix = np.zeros((N_Eg, N_cmp))
    # Read the rest:
    for i in range(0,N_Eg):
        fn = "cmp"+str(Eg_sim_array[i])
        cmp_current = np.genfromtxt(os.path.join(folderpath,"cmp"+str(int(Eg_sim_array[i]))), comments="!")
        compton_matrix[i,0:len(cmp_current)] = cmp_current

    print("compton_matrix =", compton_matrix)


    # == Make response matrix by interpolating to Eg_sim_array ==
    Ecmp_array = np.linspace(a0_cmp, a1_cmp*(N_cmp-1), N_cmp)
    print("Ecmp_array =", Ecmp_array)

    # TODO normalize once and for all outside j loop
    f, ax = plt.subplots(1,1)
    # for i_plt in [0,5,10,20,50]:
        # ax.plot(Ecmp_array, compton_matrix[i_plt,:]/compton_matrix[i_plt,:].sum(), label="Eg = {:.0f}".format(Eg_sim_array[i_plt]))

    # We need to use the interpolation scheme given in Guttormsen 1996.
    # Brute-force it with loops to make sure I get it right (based on MAMA code)
    # After all, this is done once and for all, so it does not need to be lightning fast

    # Start looping over the rows of the response function,
    # indexed by j to match MAMA code:
    # Egmin = 30 # keV -- this is universal (TODO: Is it needed?)
    Egmin = max(30, Eout_array[0]) # Don't go below input energy limit
    i_Egmin = int((Egmin - a0_out)/a1_out + 0.5)

    # Allocate response matrix array:
    R = np.zeros((N_out, N_out))
    for j in range(N_out):#[0,2,10,99]: 
        E_j = Eout_array[j]
        # Skip if below lower threshold
        if E_j < Egmin:
            continue


        # Find maximal energy for current response function, 6*sigma (TODO: is it needed?)
        Egmax = E_j + 6*FWHM*FWHM_rel.max()/2.35 # + 6*FWHM*FWHM_rel[j]/2.35 
        i_Egmax = int((Egmax - a0_out)/a1_out + 0.5)
        # TODO does the FWHM array need to be interpolated before it is used here? j is not the right index? A quick fix since this is just an upper limit is to safeguard with FWHM.max()
        # TODO check which factors FWHM should be multiplied with. FWHM at 1.33 MeV must be user-supplied? 
        # Also MAMA unfolds with 1/10 of real FWHM for convergence reasons.
        # But let's stick to letting FWHM denote the actual value, and divide by 10 in computations if necessary.
        
        # Find the closest energies among the available response functions, to interpolate between:
        # TODO what to do when E_out[j] is below lowest Eg_sim_array element? Interpolate between two larger?
        i_g_sim_low = 0
        try:
            i_g_sim_low = np.where(Eg_sim_array <= E_j)[0][-1]
        except IndexError:
            pass
        i_g_sim_high = N_Eg
        try:
            i_g_sim_high = np.where(Eg_sim_array >= E_j)[0][0]
        except IndexError:
            pass
        if i_g_sim_low == i_g_sim_high:
            if i_g_sim_low > 0:
                i_g_sim_low -= 1
            else:
                i_g_sim_high += 1

        Eg_low = Eg_sim_array[i_g_sim_low]
        Eg_high = Eg_sim_array[i_g_sim_high]

        # TODO double check that this works for all j, that it indeed finds E_g above and below

        # Next, select the Compton spectra at index i_g_sim_low and i_g_sim_high. These are called Fs1 and Fs2 in MAMA.
        # NB! The compton_matrix cannot be indexed by Eout indices, but indices corresponding to the Eg steps between GEANT simulations
        # i_g_sim_low = int((Eg_low-a0_sim)/a1_sim + 0.5)
        # i_g_sim_high = int((Eg_high-a0_sim)/a1_sim + 0.5)

        print("Eg_low =", Eg_low, "Eg_high =", Eg_high)
        # print("i_g_sim_low =", i_g_sim_low, "i_g_sim_high =", i_g_sim_high, flush=True)
        print("i_g_sim_low =", i_g_sim_low, "i_g_sim_high =", i_g_sim_high, flush=True)

        cmp_low = compton_matrix[i_g_sim_low,:]
        cmp_high = compton_matrix[i_g_sim_high,:]
        # These need to be recalibrated to Eout_array:
        cmp_low = rebin_by_arrays(cmp_low, Ecmp_array, Eout_array)
        cmp_high = rebin_by_arrays(cmp_high, Ecmp_array, Eout_array)
        print("Eout_array[{:d}] = {:.1f}".format(j, E_j), "Eg_low =", Eg_sim_array[i_g_sim_low], "Eg_high =", Eg_sim_array[i_g_sim_high], flush=True)

        # Fetch corresponding values for full-energy, etc:
        FE_low, SE_low, DE_low, c511_low = FE[i_g_sim_low], SE[i_g_sim_low], DE[i_g_sim_low], c511[i_g_sim_low]
        FE_high, SE_high, DE_high, c511_high = FE[i_g_sim_high], SE[i_g_sim_high], DE[i_g_sim_high], c511[i_g_sim_high]

        # Normalize total spectrum to 1, including FE, SE, etc:
        sum_low = cmp_low.sum() + FE_low + SE_low + DE_low + c511_low
        sum_high = cmp_high.sum() + FE_high + SE_high + DE_high + c511_high

        cmp_low /= sum_low
        FE_low /= sum_low
        SE_low /= sum_low
        DE_low /= sum_low
        c511_low /= sum_low
        cmp_high /= sum_high
        FE_high /= sum_high
        SE_high /= sum_high
        DE_high /= sum_high
        c511_high /= sum_high

        # The interpolation is split into energy regions.
        # Below the back-scattering energy Ebsc we interpolate linearly,
        # then we apply the "fan method" (Guttormsen 1996) in the region
        # from Ebsc up to the Compton edge, then linear extrapolation again the rest of the way.

        # Get maximal energy by taking 6*sigma above full-energy peak
        E_low_max = Eout_array[i_g_sim_low] + 6*FWHM*FWHM_rel[i_g_sim_low]/2.35 # TODO double check that it's the right index on FWHM_rel, plus check calibration of FWHM, FWHM[i_low]
        i_low_max = int((E_low_max - a0_out)/a1_out + 0.5)
        E_high_max = Eout_array[i_g_sim_high] + 6*FWHM*FWHM_rel[i_g_sim_high]/2.35
        i_high_max = int((E_high_max - a0_out)/a1_out + 0.5)

        # Find back-scattering Ebsc and compton-edge Ece energy of the current Eout energy:
        Ece = E_compton(E_j, theta=np.pi)
        print("Ece =", Ece)
        Ebsc = E_j - Ece
        print("Ebsc =", Ebsc)
        # Indices in Eout calibration corresponding to these energies:
        i_ce_out = min(int((Ece - a0_out)/a1_out + 0.5), i_Egmax)
        i_bsc_out = max(int((Ebsc - a0_out)/a1_out + 0.5), i_Egmin)
        print("i_ce_out =", i_ce_out, ", i_bsc_out =", i_bsc_out, ", i_Egmax =", i_Egmax)


        # Interpolate one-to-one up to j_bsc_out:
        for i in range(0,i_bsc_out):
            R[j,i] = cmp_low[i] + (cmp_high[i]-cmp_low[i])*(E_j - Eg_low)/(Eg_high-Eg_low)
            if R[j,i] < 0:
                # print("R[{:d},{:d}] = {:.2f}".format(j,i,R[j,i]), flush=True)
                R[j,i] = 0 # TODO make this faster by indexing at the end

        # Then interpolate with the fan method up to j_ce_out:
        z = 0 # Initialize variable 
        for i in range(i_bsc_out, i_ce_out):
            E_i = Eout_array[i] # Energy of current point in interpolated spectrum
            if E_i > 0.1 and E_i < Ece:
                if np.abs(E_j - E_i) > 0.001:
                    z = E_i/(E_j/511 * (E_j - E_i))
                theta = np.arccos(1-z)
                print("theta = ", theta, flush=True)
                if theta > 0 and theta < np.pi:
                    # Determine interpolation indices in low and high arrays
                    # by Compton formula
                    i_low_interp = max(int((E_compton(Eg_low,theta)-a0_out)/a1_out + 0.5), i_bsc_out)
                    i_high_interp = min(int((E_compton(Eg_high,theta)-a0_out)/a1_out + 0.5), i_high_max)
                    # TODO For some reason the following line gives very negative numbers. Prime suspect is the corr() function
                    # May also need to add check to remove below-zero values.
                    # R[j,i] = cmp_low[i_low_interp]*corr(Eg_low,theta) + (E_j-Eg_low)/(Eg_high-Eg_low) \
                    # * (cmp_high[i_high_interp] * corr(Eg_high,theta) - cmp_low[i_low_interp] * corr(Eg_low,theta))
                    # FA = cmp_high[i_high_interp]*corr(Eg_high, theta) - cmp_low[i_low_interp]*corr(Eg_low,theta)
                    FA = cmp_high[i_high_interp] - cmp_low[i_low_interp]
                    # FB = cmp_low[i_low_interp]*corr(Eg_low,theta) + FA*(E_j - Eg_low)/(Eg_high - Eg_low)
                    # R[j,i] = FA/corr(E_j,theta)
                    print("E_j = {:.2f}, Eg_low = {:.2f}, Eg_high = {:.2f}".format(E_j, Eg_low, Eg_high))
                    R[j,i] = cmp_low[i_low_interp] + (cmp_high[i_high_interp]-cmp_low[i_low_interp])*(E_j-Eg_low)/(Eg_high-Eg_low)
                    if R[j,i] < 0:
                        print("R[{:d},{:d}] = {:.2f}".format(j,i,R[j,i]), flush=True)

                    







        

        
    # sys.exit(0)



    


    for i_plt in [0,5,10,20]:
        ax.plot(Eout_array, R[i_plt,:], label="interpolated, Eout = {:.0f}".format(Eout_array[i_plt]), linestyle="--")



    ax.legend()
    plt.show()


    # Normalize and interpolate the other structures of the response:




    # return R, FWHM, eff, pf, pc, ps, pd, pa
    return True



if __name__ == "__main__":
    # folderpath = "oscar2017_scale1.0"
    folderpath = "oscar2017_scale1.15"
    Eg_sim_array = np.linspace(0,2000, 21)
    FWHM = 2.0
    print("response(\"{:s}\") =".format(folderpath), response(folderpath, Eg_sim_array, FWHM))