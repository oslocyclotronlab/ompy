# import sys, os
# if sys.version_info[0] < 3:
#     raise Exception("Must be using Python 3")
import os
import numpy as np
# import matplotlib.pyplot as plt
from scipy.interpolate import interp1d#, interp2d

# from .firstgen import *
# from .unfold import *
from .rebin import *
from .library import *

DTYPE = np.float64


# TODO 20190527: I just merged master into this old branch to start fixing the
# response interpolation. I moved this file from oslo_method_python/ to ompy/.
# The next step is to cythonize it properly to hopefully make it fast,
# and implement it into the rest of the code structure, then test it.

# j_test = 50



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
    # print("From E_compton(): Eg =", Eg, ", theta =", theta, ", formula =", Eg*Eg/511*(1-np.cos(theta)) / (1+Eg/511 * (1-np.cos(theta))))
    return np.where(Eg > 0.1, Eg*Eg/511*(1-np.cos(theta)) / (1+Eg/511 * (1-np.cos(theta))), Eg)

def corr(Eg, theta):
    """
    Function to correct number of counts due to delta(theta)
    Adapted from MAMA in the file kelvin.f
    It is dE/dtheta of the E(theta) in Eq. (2) in Guttormsen 1996.
    """
    return (Eg*Eg/511*np.sin(theta))/(1+Eg/511*(1-np.cos(theta)))**2


def gaussian(double[:] E_array, double mu, double sigma):
    """
    Returns a normalized Gaussian supported on E_array
    """
    gaussian_array = np.zeros(len(E_array), dtype=DTYPE)
    cdef double[:] gaussian_array_view = gaussian_array
    cdef double prefactor

    prefactor = (1/(sigma*np.sqrt(2*np.pi)))
    cdef int i
    for i in range(len(E_array)):
        gaussian_array_view[i] = (prefactor
                                  * np.exp(
                                    -(E_array[i]-mu)
                                    * (E_array[i]-mu)/(2*sigma*sigma))
                                  )

    return gaussian_array


def gauss_smoothing(double[:] vector_in, double[:] E_array, double fwhm):
    """
    Function which smooths an array of counts by a Gaussian
    of full-width-half-maximum FWHM. Preserves number of counts.
    Args:
        vector_in (array, double): Array of inbound counts to be smoothed
        E_array (array, double): Array with energy calibration of vector_in
        fwhm (double): The full-width-half-maximum value to smooth by

    Returns:
        vector_out: Array of smoothed counts

    """
    if not len(vector_in) == len(E_array):
        raise ValueError("Length mismatch between vector_in and E_array")

    cdef double[:] vector_in_view = vector_in

    vector_out = np.zeros(len(vector_in), dtype=DTYPE)
    # cdef double[:] vector_out_view = vector_out

    cdef int i
    for i in range(len(vector_out)):
        vector_out += (vector_in_view[i]
                       * gaussian(E_array, mu=E_array[i], sigma=fwhm/2.355))

    return vector_out


def response(folderpath, double[:] Eout_array, fwhm_abs):
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
    fwhm_abs: The experimental absolute full-width-half-max at 1.33 MeV.
    """
    # Define helping variables from input
    N_out = len(Eout_array)
    a0_out, a1_out = Eout_array[0], Eout_array[1]-Eout_array[0]
    # print("a0_out, a1_out =", a0_out, a1_out)
    cdef int i

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
                # TODO: The above if test is hardly very robust. Find a better solution.
                line = file.readline()
                Nlines = int(line)
                # print("Nlines =", Nlines)
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

    # Unpack the resp matrix into its columns
    resp = np.array(resp)
    Eg_sim_array, fwhm_rel, Eff_tot, FE, SE, DE, c511 = resp.T
    a0_sim, a1_sim = Eg_sim_array[0], Eg_sim_array[1]-Eg_sim_array[0]
    # print("a0_sim, a1_sim =", a0_sim, a1_sim, flush=True)
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
    # print("a0_cmp, a1_cmp, N_cmp = ", a0_cmp, a1_cmp, N_cmp)
    compton_matrix = np.zeros((N_Eg, N_cmp))
    # Read the rest:
    for i in range(0,N_Eg):
        fn = "cmp"+str(Eg_sim_array[i])
        cmp_current = np.genfromtxt(os.path.join(folderpath,"cmp"+str(int(Eg_sim_array[i]))), comments="!")
        compton_matrix[i,0:len(cmp_current)] = cmp_current

    # print("compton_matrix =", compton_matrix)



    # Normalize total spectrum to 1, including FE, SE, etc:
    # Make a vector containing the sums for each step along the Eg_sim dimension:
    sum_spec = compton_matrix.sum(axis=1) + FE + SE + DE + c511

    # Normalize each part of each spectrum:
    cmp_matrix = div0(compton_matrix,sum_spec.reshape((len(sum_spec),1))) # Still a matrix with Eg_sim along first axis, Ecmp along second
    pcmp = cmp_matrix.sum(axis=1) # Vector of total Compton probability
    # Full energy, single escape, etc:
    pFE = div0(FE,sum_spec)
    pSE = div0(SE,sum_spec)
    pDE = div0(DE,sum_spec)
    p511 = div0(c511,sum_spec)


    # == Interpolate the peak structures except Compton, which is handled separately ==
    # Note that values outside edges of the simulated region are assigned value 0.
    f_pcmp = interp1d(Eg_sim_array, pcmp, kind="linear", bounds_error=False, fill_value=0)
    f_pFE = interp1d(Eg_sim_array, pFE, kind="linear", bounds_error=False, fill_value=0)
    f_pSE = interp1d(Eg_sim_array, pSE, kind="linear", bounds_error=False, fill_value=0)
    f_pDE = interp1d(Eg_sim_array, pDE, kind="linear", bounds_error=False, fill_value=0)
    f_p511 = interp1d(Eg_sim_array, p511, kind="linear", bounds_error=False, fill_value=0)
    f_fwhm_rel = interp1d(Eg_sim_array, fwhm_rel, kind="linear", bounds_error=False, fill_value=0)
    f_Eff_tot = interp1d(Eg_sim_array, Eff_tot, kind="linear", bounds_error=False, fill_value=0)





    # == Make response matrix by interpolating to Eg_sim_array ==
    Ecmp_array = np.linspace(a0_cmp, a1_cmp*(N_cmp-1), N_cmp)
    # print("Ecmp_array =", Ecmp_array)

    # TODO normalize once and for all outside j loop
    # f, ax = plt.subplots(1,1)
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
    # Loop over rows of the response matrix
    for j in range(N_out):#[j_test]: 
        E_j = Eout_array[j]
        # Skip if below lower threshold
        if E_j < Egmin:
            continue


        # Find maximal energy for current response function, 6*sigma (TODO: is it needed?)
        Egmax = E_j + 6*fwhm_abs*f_fwhm_rel(E_j)/2.35 #FWHM_rel.max()/2.35 
        i_Egmax = min(int((Egmax - a0_out)/a1_out + 0.5), N_out)
        # print("i_Egmax =", i_Egmax)
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
        # print("Eg_low =", Eg_low, "Eg_high =", Eg_high)
        # print("i_g_sim_low =", i_g_sim_low, "i_g_sim_high =", i_g_sim_high, flush=True)

        cmp_low = cmp_matrix[i_g_sim_low,:]
        cmp_high = cmp_matrix[i_g_sim_high,:]
        # These need to be recalibrated from Ecmp_array to Eout_array:
        cmp_low = rebin(cmp_low, Ecmp_array, Eout_array)
        cmp_high = rebin(cmp_high, Ecmp_array, Eout_array)
        # print("Eout_array[{:d}] = {:.1f}".format(j, E_j), "Eg_low =", Eg_sim_array[i_g_sim_low], "Eg_high =", Eg_sim_array[i_g_sim_high], flush=True)

        # Fetch corresponding values for full-energy, etc:
        pFE_low, pSE_low, pDE_low, p511_low = pFE[i_g_sim_low], pSE[i_g_sim_low], pDE[i_g_sim_low], p511[i_g_sim_low]
        pFE_high, pSE_high, pDE_high, p511_high = pFE[i_g_sim_high], pSE[i_g_sim_high], pDE[i_g_sim_high], p511[i_g_sim_high]


        # The interpolation is split into energy regions.
        # Below the back-scattering energy Ebsc we interpolate linearly,
        # then we apply the "fan method" (Guttormsen 1996) in the region
        # from Ebsc up to the Compton edge, then linear extrapolation again the rest of the way.

        # Get maximal energy by taking 6*sigma above full-energy peak
        E_low_max = Eg_low + 6*fwhm_abs*f_fwhm_rel(Eg_low)/2.35 #FWHM_rel.max()/2.35 # TODO double check that it's the right index on FWHM_rel, plus check calibration of FWHM, FWHM[i_low]
        i_low_max = min(int((E_low_max - a0_out)/a1_out + 0.5), N_out)
        E_high_max = Eg_high + 6*fwhm_abs*f_fwhm_rel(Eg_high)/2.35 #FWHM_rel.max()/2.35
        i_high_max = min(int((E_high_max - a0_out)/a1_out + 0.5), N_out)
        # print("E_low_max =", E_low_max, "E_high_max =", E_high_max, flush=True)

        # Find back-scattering Ebsc and compton-edge Ece energy of the current Eout energy:
        Ece = E_compton(E_j, theta=np.pi)
        # print("Ece =", Ece)
        Ebsc = E_j - Ece
        # print("Ebsc =", Ebsc)
        # Indices in Eout calibration corresponding to these energies:
        i_ce_out = min(int((Ece - a0_out)/a1_out + 0.5), i_Egmax)
        i_bsc_out = max(int((Ebsc - a0_out)/a1_out + 0.5), i_Egmin)
        # print("i_ce_out =", i_ce_out, ", i_bsc_out =", i_bsc_out, ", i_Egmax =", i_Egmax)


        # ax.axvline(Ebsc)
        # ax.axvline(Ece)


        # Interpolate one-to-one up to j_bsc_out:
        for i in range(0,i_bsc_out):
            R[j,i] = cmp_low[i] + (cmp_high[i]-cmp_low[i])*(E_j - Eg_low)/(Eg_high-Eg_low)
            if R[j,i] < 0:
                # print("R[{:d},{:d}] = {:.2f}".format(j,i,R[j,i]), flush=True)
                R[j,i] = 0 # TODO make this faster by indexing at the end

        # Then interpolate with the fan method up to j_ce_out:
        z = 0 # Initialize variable 
        i_last = i_bsc_out # Keep track of how far up the fan method goes
        i_low_last = i_bsc_out
        i_high_last = i_bsc_out
        for i in range(i_bsc_out, i_ce_out):
            E_i = Eout_array[i] # Energy of current point in interpolated spectrum
            if E_i > 0.1 and E_i < Ece:
                if np.abs(E_j - E_i) > 0.001:
                    z = E_i/(E_j/511 * (E_j - E_i))
                theta = np.arccos(1-z)
                # print("theta = ", theta, flush=True)
                if theta > 0 and theta < np.pi:
                    # Determine interpolation indices in low and high arrays
                    # by Compton formula
                    i_low_interp = max(int((E_compton(Eg_low, theta)
                                            - a0_out)/a1_out + 0.5), i_bsc_out
                                       )
                    i_high_interp = min(int((E_compton(Eg_high, theta)
                                            - a0_out)/a1_out + 0.5), i_high_max
                                        )
                    # TODO For some reason the following line gives very negative numbers. Prime suspect is the corr() function
                    # May also need to add check to remove below-zero values.
                    # R[j,i] = cmp_low[i_low_interp]*corr(Eg_low,theta) + (E_j-Eg_low)/(Eg_high-Eg_low) \
                    # * (cmp_high[i_high_interp] * corr(Eg_high,theta) - cmp_low[i_low_interp] * corr(Eg_low,theta))
                    FA = (cmp_high[i_high_interp]*corr(Eg_high, theta)
                          - cmp_low[i_low_interp]*corr(Eg_low, theta))
                    # FA = cmp_high[i_high_interp] - cmp_low[i_low_interp]
                    FB = cmp_low[i_low_interp]*corr(Eg_low, theta) + FA*(E_j - Eg_low)/(Eg_high - Eg_low)
                    # print("i =", i, flush=True)
                    R[j, i] = FB/corr(E_j, theta)
                    # print("E_j = {:.2f}, Eg_low = {:.2f}, Eg_high = {:.2f}".format(E_j, Eg_low, Eg_high))
                    # R[j,i] = cmp_low[i_low_interp] + (cmp_high[i_high_interp]-cmp_low[i_low_interp])*(E_j-Eg_low)/(Eg_high-Eg_low)
                    # if R[j,i] < 0:
                        # print("R[{:d},{:d}] = {:.2f}".format(j,i,R[j,i]), flush=True)
                    i_last = i
                    i_low_last = i_low_interp
                    i_high_last = i_high_interp

        # Interpolate 1-to-1 the last distance up to E+3*sigma
        # print("i_Egmax =", i_Egmax, "Egmax =", Egmax, ", i_last =", i_last, flush=True)
        # Check if this is needed:
        if i_last >= i_Egmax:
            continue 
        s_low = (i_low_max-i_low_last)/(i_Egmax-i_last)
        s_high = (i_high_max-i_high_last)/(i_Egmax-i_last)
        for i in range(i_last+1, i_Egmax):
            i_low_interp = min(int(i_low_last + s_low*(i-i_last) + 0.5), i_low_max)
            i_high_interp = min(int(i_high_last + s_high*(i-i_last) + 0.5), i_high_max)
            R[j,i] = cmp_low[i_low_interp] + (cmp_high[i_high_interp]-cmp_low[i_low_interp])*(E_j-Eg_low)/(Eg_high-Eg_low)
            # print("Last bit of interpolation: R[{:d},{:d}] = {:.2f}".format(j,i,R[j,i]), flush=True)
            # if R[j,i] < 0:
            #     print("R[j,i] =", R[j,i], flush=True)
            #     R[j,i] = 0

        # DEBUG: Plot cmp_low and cmp_high:
        # ax.plot(Eout_array, cmp_low, label="cmp_low")
        # ax.plot(Eout_array, cmp_high, label="cmp_high")




        # === Add peak structures to the spectrum: ===
        fwhm_current = fwhm_abs * f_fwhm_rel(Eout_array[j])

        # Add full-energy peak, which should be at energy corresponding to
        # index j:
        E_fe = Eout_array[j]
        full_energy = np.zeros(N_out)  # Allocate with zeros everywhere
        full_energy[j] = f_pFE(E_fe)  # Full probability into sharp peak
        # full_energy = gauss_smoothing(full_energy, Eout_array, fwhm_current)  # Smoothe
        R[j, :] += full_energy

        # Add single-escape peak, at index i_se
        E_se = E_fe - 511
        if E_se >= 0:
            i_se = int((E_se - a0_out)/a1_out + 0.5)
            # print("Eout_array[i_se] =", Eout_array[i_se])
            single_escape = np.zeros(N_out)  # Allocate with zeros everywhere
            single_escape[i_se] = f_pSE(E_se)  # Full probability into sharp peak
            # single_escape = gauss_smoothing(single_escape, Eout_array,
            #                                 fwhm_current)  # Smoothe
            R[j, :] += single_escape

        # Add double-escape peak, at index i_de
        E_de = E_fe - 2*511
        if E_de >= 0:
            i_de = int((E_de - a0_out)/a1_out + 0.5)
            # print("Eout_array[i_de] =", Eout_array[i_de])
            double_escape = np.zeros(N_out)  # Allocate with zeros everywhere
            double_escape[i_de] = f_pDE(E_de)  # Full probability into sharp peak
            # double_escape = gauss_smoothing(double_escape, Eout_array,
            #                                 fwhm_current)  # Smoothe
            R[j, :] += double_escape

        # Add 511 annihilation peak, at index i_an
        if E_fe > 511:
            E_511 = 511
            i_511 = int((E_511 - a0_out)/a1_out + 0.5)
            # print("Eout_array[i_511] =", Eout_array[i_511])
            annihilation = np.zeros(N_out)  # Allocate with zeros everywhere
            annihilation[i_511] = f_p511(E_511)  # Full probability into sharp peak
            annihilation = gauss_smoothing(annihilation, Eout_array,
                                           fwhm_current)  # Smoothe
            R[j, :] += annihilation
    
        # === Finally, normalise the row to unity (probability conservation): ===
        R[j, :] = div0(R[j, :], np.sum(R[j, :]))




    # END loop over Eout energies Ej

    # Remove any negative elements from response matrix:
    R[R<0] = 0




                    







        

        



    


    # for i_plt in [j_test]:
        # ax.plot(Eout_array, R[i_plt,:], label="interpolated, Eout = {:.0f}".format(Eout_array[i_plt]), linestyle="--")



    # ax.legend()
    # plt.show()


    
    # Return the response matrix R, as well as the other structures, FWHM and efficiency, interpolated to the Eout_array
    return R, f_fwhm_rel(Eout_array), f_Eff_tot(Eout_array), f_pcmp(Eout_array), f_pFE(Eout_array), f_pSE(Eout_array), f_pDE(Eout_array), f_p511(Eout_array)



if __name__ == "__main__":
    # folderpath = "oscar2017_scale1.0"
    folderpath = "oscar2017_scale1.15"
    Eout_array = np.linspace(200,1400,100)
    FWHM = 300.0
    print("response(\"{:s}\") =".format(folderpath), response(folderpath, Eout_array, FWHM))