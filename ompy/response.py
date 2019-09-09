# import sys, os
# if sys.version_info[0] < 3:
#     raise Exception("Must be using Python 3")
import os
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
from scipy.interpolate import interp1d#, interp2d

# from .firstgen import *
# from .unfold import *
from .rebin import *
from .library import *
from .gauss_smoothing import gauss_smoothing
from .matrix import Matrix

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


def Emid_to_bin(Emid, a0, a1):
    """ Fin the bin of Emin, given te calibration (lower bin edge)

    Assumes linear calibration
    E_lower = a0 + a1 * i
    Emid = E_lower + a1/2
    """
    E_lower = Emid - a1/2
    i = (E_lower-a0)/a1
    return int(i)


def two_channel_split(E_centroid, E_array):
    """
    When E_centroid is between two bins in E_array, this function
    returns the indices of the two nearest bins and the distance to
    the lower bin. The distance to the higher bin is 1-floor_distance

    Args:
        E_centroid (double): The energy of the centroid
        E_array (np.array, double): The energy grid to distribute on
                                    Lower edge calibration.
    """

    a0 = E_array[0]
    a1 = E_array[1]-E_array[0]

    # convert to lower edge
    E_centroid = E_centroid-a1/2

    bin_exact_float = (E_centroid - a0)/a1
    i_floor = int(np.floor((E_centroid - a0)/a1))
    i_ceil = int(np.ceil((E_centroid - a0)/a1))
    floor_distance = (bin_exact_float - i_floor)

    return i_floor, i_ceil, floor_distance


def interpolate_response(folderpath, Eout_array, fwhm_abs, return_table=False):
    """ Interpolate response through "Fan"-mathod (Guttormsen1996)

    Assumes the source files are in the folder "folderpath",
    and that they are formatted in a certain standard way.
    The function interpolates the data to give a response matrix with
    the desired energy binning specified by Eout_array.

    Args:
        folderpath: The path to the folder containing Compton spectra and resp.dat
        Eout_array: The desired energies of the output response matrix.
        fwhm_abs: The experimental absolute full-width-half-max at 1.33 MeV.
                  Note: In the article it is recommended to use 1/10 of the
                  real FWHM for unfolding.
        return_table (optional): Returns "all" output, see below

    Returns:
        response (Matrix): Response matrix with incident energy on the "Ex"
                           axis and the spectral response on the "Eg" axis
        response_table (Dataframe)

    """
    if return_table not in (True, False):
        raise ValueError("return_table must be a bool,now {}".format(return_table))

    assert(1e-1 <= fwhm_abs <= 100), "Check the fwhm_abs, probably it's wrong."\
        "\nNormal Oscar≃30, Now: {}".format(fwhm_abs)

    # Define helping variables from input
    N_out = len(Eout_array)
    a0_out, a1_out = Eout_array[0], Eout_array[1]-Eout_array[0]
    # print("a0_out, a1_out =", a0_out, a1_out)
    # cdef int i

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

    if Eout_array.min() < Eg_sim_array.min():
        print("Note: The response below {:.0f} keV".format(Eg_sim_array.min()),
              "is interpolation only, as there are no simulations available.")

    if Eout_array.max() > Eg_sim_array.max():
        # Actually I don't know why we shouldn't be able to
        # interpolate here, too
        raise ValueError("Maximum energy cannot be larger than largest "
                         "simulated response function, which is "
                         "E_max = {:f}".format(Eg_sim_array.max())
                         )


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
    # compton_matrix = np.tril(compton_matrix)

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
    f_pcmp = interp1d(Eg_sim_array, pcmp, kind="linear", bounds_error=False, fill_value="extrapolate")
    f_pFE = interp1d(Eg_sim_array, pFE, kind="linear", bounds_error=False, fill_value="extrapolate")
    f_pSE = interp1d(Eg_sim_array, pSE, kind="linear", bounds_error=False, fill_value=0)
    f_pDE = interp1d(Eg_sim_array, pDE, kind="linear", bounds_error=False, fill_value=0)
    f_p511 = interp1d(Eg_sim_array, p511, kind="linear", bounds_error=False, fill_value=0)
    f_fwhm_rel_perCent_norm = interp1d(Eg_sim_array, fwhm_rel, kind="linear", bounds_error=False, fill_value="extrapolate")
    f_Eff_tot = interp1d(Eg_sim_array, Eff_tot, kind="linear", bounds_error=False, fill_value=0)

    assert abs(f_fwhm_rel_perCent_norm(1330)-1) < 0.05, \
        "Response function format not as expected. In the Mama-format, the"\
        "'f_fwhm_rel_perCent' column denotes the relative fwhm (= fwhm/E)," \
        "but normalized to 1 at 1.33 MeV."\
        "Now it is: {} at 1.33 MeV.".format(f_fwhm_rel_perCent_norm(1330))
    fwhm_rel_1330 = (fwhm_abs/1330*100)
    f_fwhm_rel_perCent = interp1d(Eg_sim_array, fwhm_rel*fwhm_rel_1330,
                                  kind="linear",
                                  bounds_error=False,
                                  fill_value="extrapolate")

    fwhm_abs_array = Eout_array*f_fwhm_rel_perCent(Eout_array)/100

    # DEBUG:
    # print("p511 vector from resp.dat =", p511)
    # END DEBUG



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
    Egmin = Eout_array[0]
    i_Egmin = Emid_to_bin(Egmin, a0_out, a1_out)

    # Allocate response matrix array:
    R = np.zeros((N_out, N_out))
    # Loop over rows of the response matrix
    # TODO for speedup: Change this to a cython .pyx, declare the j variable.
    #                   + run a Cython profiler, probably use memory views and
    #                   other tweaks to speedup (see rebin.pyx for examples).
    for j in range(N_out):
        E_j = Eout_array[j]
        # Skip if below lower threshold
        if E_j < Egmin:
            continue

        # Find maximal energy for current response function,
        # Changed to 1*sigma, or whatever this means
        # -> Better if the lowest energies of the simulated spectra are above
        # the gamma energy to be extrapolatedu
        Egmax = E_j + 1*fwhm_abs*f_fwhm_rel_perCent_norm(E_j)/2.35 #FWHM_rel.max()/2.35
        i_Egmax = min(Emid_to_bin(Egmax, a0_out, a1_out), N_out)
        # print("i_Egmax =", i_Egmax)

        # MAMA unfolds with 1/10 of real FWHM for convergence reasons.
        # But let's stick to letting FWHM denote the actual value, and divide by 10 in computations if necessary.

        # Find the closest energies among the available response functions, to interpolate between:
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
        # When E_out[j] is below lowest Eg_sim_array element? Interpolate between two larger?
        if i_g_sim_low == i_g_sim_high:
            if i_g_sim_low > 0:
                i_g_sim_low -= 1
            else:
                i_g_sim_high += 1

        Eg_low = Eg_sim_array[i_g_sim_low]
        Eg_high = Eg_sim_array[i_g_sim_high]

        # Next, select the Compton spectra at index i_g_sim_low and i_g_sim_high. These are called Fs1 and Fs2 in MAMA.
        # print("Eg_low =", Eg_low, "Eg_high =", Eg_high)
        # print("i_g_sim_low =", i_g_sim_low, "i_g_sim_high =", i_g_sim_high, flush=True)

        cmp_low = cmp_matrix[i_g_sim_low,:]
        cmp_high = cmp_matrix[i_g_sim_high,:]
        # These need to be recalibrated from Ecmp_array to Eout_array:
        cmp_low = rebin_1D(cmp_low, Ecmp_array, Eout_array)
        cmp_high = rebin_1D(cmp_high, Ecmp_array, Eout_array)
        # print("Eout_array[{:d}] = {:.1f}".format(j, E_j), "Eg_low =", Eg_sim_array[i_g_sim_low], "Eg_high =", Eg_sim_array[i_g_sim_high], flush=True)

        # The interpolation is split into energy regions.
        # Below the back-scattering energy Ebsc we interpolate linearly,
        # then we apply the "fan method" (Guttormsen 1996) in the region
        # from Ebsc up to the Compton edge, then linear extrapolation again the rest of the way.

        # Get maximal energy by taking 6*sigma above full-energy peak
        E_low_max = Eg_low + 6*fwhm_abs_array[i_g_sim_low]/2.35
        i_low_max = min(Emid_to_bin(E_low_max, a0_out, a1_out), N_out-1)
        E_high_max = Eg_high + 6*fwhm_abs_array[i_g_sim_high]/2.35
        i_high_max = min(Emid_to_bin(E_high_max, a0_out, a1_out), N_out-1)
        # print("E_low_max =", E_low_max, "E_high_max =", E_high_max, flush=True)

        # Find back-scattering Ebsc and compton-edge Ece energy of the current Eout energy:
        Ece = E_compton(E_j, theta=np.pi)
        Ebsc = E_j - Ece
        # if E_j==200:
        #     print(E_j)
        #     print("Ece =", Ece)
        #     print("Ebsc =", Ebsc)
        # Indices in Eout calibration corresponding to these energies:
        i_ce_out = min(Emid_to_bin(Ece, a0_out, a1_out), i_Egmax)
        i_bsc_out = max(Emid_to_bin(Ebsc, a0_out, a1_out), i_Egmin)
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
                    Ecmp_ = E_compton(Eg_low, theta)
                    i_low_interp = max(Emid_to_bin(Ecmp_, a0_out, a1_out),
                                       i_bsc_out)
                    Ecmp_ = E_compton(Eg_high, theta)
                    i_high_interp = min(Emid_to_bin(Ecmp_, a0_out, a1_out),                i_high_max)
                    FA = (cmp_high[i_high_interp]*corr(Eg_high, theta)
                          - cmp_low[i_low_interp]*corr(Eg_low, theta))
                    FB = cmp_low[i_low_interp]*corr(Eg_low, theta) + FA*(E_j - Eg_low)/(Eg_high - Eg_low)
                    R[j, i] = FB/corr(E_j, theta)
                    i_last = i
                    i_low_last = i_low_interp
                    i_high_last = i_high_interp


        # Interpolate 1-to-1 the last distance up to E+6*sigma
        # print("i_Egmax =", i_Egmax, "Egmax =", Egmax, ", i_last =", i_last, flush=True)
        # Check if this is needed:
        if i_last >= i_Egmax:
            continue
        s_low = (i_low_max-i_low_last)/(i_Egmax-i_last)
        s_high = (i_high_max-i_high_last)/(i_Egmax-i_last)

        for i in range(i_last, i_Egmax):
            i_low_interp = min(int(i_low_last + s_low*(i-i_last) + 0.5), i_low_max)
            i_high_interp = min(int(i_high_last + s_high*(i-i_last) + 0.5), i_high_max)
            R[j,i] = cmp_low[i_low_interp] + (cmp_high[i_high_interp]-cmp_low[i_low_interp])*(E_j-Eg_low)/(Eg_high-Eg_low)
            # print("Last bit of interpolation: R[{:d},{:d}] = {:.2f}".format(j,i,R[j,i]), flush=True)
            # if R[j,i] < 0:
            #     print("R[j,i] =", R[j,i], flush=True)
            #     R[j,i] = 0

        # coorecton below E_sim[0]
        if E_j < Eg_sim_array[0]:
            R[j,j+1:]=0

        # DEBUG: Plot cmp_low and cmp_high:
        # if 50 < E_j <= 55 or 200 < E_j <= 205 or 500 < E_j <= 505:
        #     if 50 < E_j <= 55: fig, ax = plt.subplots()
        #     # ax.plot(Eout_array, cmp_low, label="cmp_low")
        #     # ax.plot(Eout_array, cmp_high, label="cmp_high")
        #     ax.plot(Eout_array, R[j, :], label="R[j, :]")
        #     ax.plot(Eout_array, R1, "--",label="R[j, :]")
        #     # ax.plot(Eout_array, cmp_high, label="cmp_high")
        #     plt.show()

        # Note: We choose not to smoothe the Compton spectrum, because the
        # simulated Compton spectra stored in file are smoothed already.
        # To apply smoothing to the Compton spectrum, you could do something like
        # R[j, :] = gauss_smoothing(R[j, :], Eout_array,
        #                           fwhm_abs_array)




        # === Add peak structures to the spectrum: ===
        discrete_peaks = np.zeros(N_out)
        E_fe = Eout_array[j] + a1_out/2  # Evaluate energies in middle-bin

        # Add full-energy peak, which should be at energy corresponding to
        # index j:
        # full_energy = np.zeros(N_out)  # Allocate with zeros everywhere
        # full_energy[j] = f_pFE(E_fe)  # Full probability into sharp peak
        discrete_peaks[j] = f_pFE(E_fe)

        # Smoothe it:
        # full_energy = gauss_smoothing(full_energy, Eout_array,
                                      # fwhm_abs_array)
        # R[j, :] += full_energy

        # Add single-escape peak, at index i_se
        E_se = E_fe - 511
        if E_se >= 0 and E_se >= Eout_array[0]:
            i_floor, i_ceil, floor_distance\
                = two_channel_split(E_se, Eout_array)
            # single_escape = np.zeros(N_out)  # Allocate with zeros everywhere
            # Put a portion of the counts into floor bin - the further away,the
            # less counts:
            # single_escape[i_floor] = (1-floor_distance) * f_pSE(E_fe)
            # single_escape[i_ceil] = floor_distance * f_pSE(E_fe)
            discrete_peaks[i_floor] += (1-floor_distance) * f_pSE(E_fe)
            discrete_peaks[i_ceil] += floor_distance * f_pSE(E_fe)
            # single_escape = gauss_smoothing(single_escape, Eout_array,
                                            # fwhm_abs_array)  # Smoothe
            # R[j, :] += single_escape

        # Repeat for double-escape peak, at index i_de
        E_de = E_fe - 2*511
        if E_de >= 0 and E_de >= Eout_array[0]:
            i_floor, i_ceil, floor_distance\
                = two_channel_split(E_de, Eout_array)
            # double_escape = np.zeros(N_out)
            # double_escape[i_floor] = (1-floor_distance) * f_pDE(E_fe)
            # double_escape[i_ceil] = floor_distance * f_pDE(E_fe)
            discrete_peaks[i_floor] += (1-floor_distance) * f_pDE(E_fe)
            discrete_peaks[i_ceil] += floor_distance * f_pDE(E_fe)
            # double_escape = gauss_smoothing(double_escape, Eout_array,
                                            # fwhm_abs_array)  # Smoothe
            # R[j, :] += double_escape

        # Add 511 annihilation peak, at index i_an
        if E_fe > 511 and 511 >= Eout_array[0]:
            E_511 = 511
            i_floor, i_ceil, floor_distance\
                = two_channel_split(E_511, Eout_array)
            # fiveeleven = np.zeros(N_out)
            # fiveeleven[i_floor] = (1-floor_distance) * f_p511(E_fe)
            # fiveeleven[i_ceil] = floor_distance * f_p511(E_fe)
            discrete_peaks[i_floor] += (1-floor_distance) * f_p511(E_fe)
            discrete_peaks[i_ceil] += floor_distance * f_p511(E_fe)
            # fiveeleven = gauss_smoothing(fiveeleven, Eout_array,
                                         # fwhm_abs_array)  # Smoothe
            # R[j, :] += fiveeleven

        # Do common smoothing of the discrete_peaks array:
        discrete_peaks = gauss_smoothing(discrete_peaks, Eout_array,
                                         fwhm_abs_array)  # Smoothe

        R[j, :] += discrete_peaks

        # === Finally, normalise the row to unity (probability conservation): ===
        R[j, :] = div0(R[j, :], np.sum(R[j, :]))

    # END loop over Eout energies Ej

    # Remove any negative elements from response matrix:
    R[R < 0] = 0

    # for i_plt in [j_test]:
        # ax.plot(Eout_array, R[i_plt,:], label="interpolated, Eout = {:.0f}".format(Eout_array[i_plt]), linestyle="--")

    # ax.legend()
    # plt.show()

    response = Matrix(values=R, Eg=Eout_array, Ex=Eout_array)

    if return_table:
        # Return the response matrix, as well as the other structures, FWHM and efficiency, interpolated to the Eout_array
        response_table = {'E': Eout_array,
                          'fwhm_abs': fwhm_abs_array,
                          'fwhm_rel_%': f_fwhm_rel_perCent(Eout_array),
                          'fwhm_rel': f_fwhm_rel_perCent(Eout_array)/100,
                          'eff_tot': f_Eff_tot(Eout_array),
                          'pcmp': f_pcmp(Eout_array),
                          'pFE': f_pFE(Eout_array),
                          'pSE': f_pSE(Eout_array),
                          'pDE': f_pDE(Eout_array),
                          'p511': f_p511(Eout_array)}
        response_table = pd.DataFrame(data=response_table)
        return response, response_table
    else:
        return response



if __name__ == "__main__":
    # folderpath = "oscar2017_scale1.0"
    folderpath = "oscar2017_scale1.15"
    Eout_array = np.linspace(200,1400,100)
    FWHM = 300.0
    print("response(\"{:s}\") =".format(folderpath), response(folderpath, Eout_array, FWHM))
