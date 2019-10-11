# -*- coding: utf-8 -*-
"""
Implementation of a response matrix created from interpolation
of experimental spectra. It takes the path to a folder (old format)
or a zip file as its construction parameters. It has a static method
to mirror that of interpolate_response() in response.py
"""

import os
import zipfile as zf
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union, Optional
from scipy.interpolate import interp1d
import logging

from .rebin import *
from .library import *
from .decomposition import index
from .gauss_smoothing import gauss_smoothing
from .matrix import Matrix

LOG = logging.getLogger(__name__)
logging.captureWarnings(True)

DTYPE = np.float64


class Response():
    """
    The Resonse class that stores the data required to perform
    the interpolation needed to create the response matrix
    """


    def __init__(self,
                 path: Union[str,Path] = None):
        """
        The resonse object is initialized with the path to the source
        files required to perform the interpolation. The path varaiable
        can either be a folder (assuming "old" format) or a zip file
        containing the files otherwise found in the folder in the "old"
        format.
        """

        path = Path(path) if isinstance(path, str) else path

        if path.is_dir():
            self.resp, self.compton_matrix, self.Ecmp_array = self.LoadDir(path) # Better names would be adventagious 
        elif path.is_file():
            self.resp, self.compton_matrix, self.Ecmp_array = self.LoadZip(path)

        self.sum_spec = np.array(self.compton_matrix.sum(axis=1) + self.resp['FE'] + self.resp['SE'] + self.resp['DE'] + self.resp['c511'])
        self.cmp_matrix = div0(self.compton_matrix,self.sum_spec.reshape((len(self.sum_spec),1)))
        self.pcmp = self.cmp_matrix.sum(axis=1) # Vector of total Compton probability
        
        # Full energy, single escape, etc:
        self.pFE = div0(self.resp['FE'],self.sum_spec)
        self.pSE = div0(self.resp['SE'],self.sum_spec)
        self.pDE = div0(self.resp['DE'],self.sum_spec)
        self.p511 = div0(self.resp['c511'],self.sum_spec)

        # == Interpolate the peak structures except Compton, which is handled separately ==
        self.f_pcmp = interp1d(self.resp['Eg'], self.pcmp, kind="linear", bounds_error=False, fill_value="extrapolate")
        self.f_pFE = interp1d(self.resp['Eg'], self.pFE, kind="linear", bounds_error=False, fill_value="extrapolate")
        self.f_pSE = interp1d(self.resp['Eg'], self.pSE, kind="linear", bounds_error=False, fill_value=0)
        self.f_pDE = interp1d(self.resp['Eg'], self.pDE, kind="linear", bounds_error=False, fill_value=0)
        self.f_p511 = interp1d(self.resp['Eg'], self.p511, kind="linear", bounds_error=False, fill_value=0)
        self.f_fwhm_rel_perCent_norm = interp1d(self.resp['Eg'], self.resp['FWHM_rel'], kind="linear", bounds_error=False, fill_value="extrapolate")
        self.f_Eff_tot = interp1d(self.resp['Eg'], self.resp['Eff_tot'], kind="linear", bounds_error=False, fill_value=0)


    def LoadZip(self,
                path: Union[str,Path],
                resp_name: Optional[str] = 'resp.csv',
                spec_prefix: Optional[str] = 'cmp'):
        """
        Method for loading the response file and compton spectra from file.
        Is assumes that path is a zip file that constains at least XX files.
        At least one has to be named 'resp.csv'
        """
        path = Path(path) if isinstance(path, str) else path


        zfile = zf.ZipFile(path, mode='r')

        if not any(resp_name in name for name in zfile.namelist()):
            raise ValueError('Response info not present')

        resp = pd.read_csv(zfile.open(resp_name, 'r'))

        # Verify that resp has all the required columns
        if not set(['Eg', 'FWHM_rel', 'Eff_tot', 'FE', 'SE', 'DE', 'c511']).issubset(resp.columns):
            raise ValueError(f'{resp_name} missing one or more required columns')

        # Verify that zip file contains a spectra for each energy
        files = [spec_prefix+str(int(Eg)) for Eg in sorted(resp['Eg'])]
        if not all(file in zfile.namelist() for file in files):
            raise ValueError(f'One or more compton spectra is missing in {path}')


        # Now we will read in all the Compton spectra
        N_cmp = -1
        a0_cmp, a1_cmp = -1, -1
        # Get calibration and array length from highest-energy spectrum, because the spectra
        # may have differing length but this is bound to be the longest.
        with zfile.open(spec_prefix+str(int(max(resp['Eg']))), 'r') as file:
            lines = file.readlines()
            a0_cmp = float(str(lines[6]).split(",")[1]) # calibration
            a1_cmp = float(str(lines[6]).split(",")[2]) # coefficients [keV]
            N_cmp = int(lines[8][15:]) +1 # 0 is first index
        
        compton_matrix = np.zeros((len(resp['Eg']), N_cmp))
        i = 0
        for file in [zfile.open(file_name) for file_name in files]:
            cmp_current = np.genfromtxt(file, comments="!")
            compton_matrix[i,0:len(cmp_current)] = cmp_current
            i += 1
            
        return resp, compton_matrix, np.linspace(a0_cmp, a1_cmp*(N_cmp - 1), N_cmp)

    def LoadDir(self,
                path: Union[str,Path],
                resp_name: Optional[str] = 'resp.dat',
                spec_prefix: Optional[str] = 'cmp'):
        """
        Method for loading response file and compton spectra from a folder.
        """

        # Read resp.dat file, which gives information about the energy bins
        # and discrete peaks
        resp = []
        Nlines = -1
        with open(os.path.join(path, "resp.dat")) as file:
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
        with open(os.path.join(path,"cmp"+str(int(Eg_sim_array[-1])))) as file:
            lines = file.readlines()
            a0_cmp = float(lines[6].split(",")[1]) # calibration
            a1_cmp = float(lines[6].split(",")[2]) # coefficients [keV]
            N_cmp = int(lines[8][15:]) +1 # 0 is first index
        # print("a0_cmp, a1_cmp, N_cmp = ", a0_cmp, a1_cmp, N_cmp)
        compton_matrix = np.zeros((N_Eg, N_cmp))
        # Read the rest:
        for i in range(0,N_Eg):
            fn = "cmp"+str(Eg_sim_array[i])
            cmp_current = np.genfromtxt(os.path.join(path,"cmp"+str(int(Eg_sim_array[i]))), comments="!")
            compton_matrix[i,0:len(cmp_current)] = cmp_current

        resp = pd.DataFrame(data={
            'Eg': Eg_sim_array,
            'FWHM_rel': fwhm_rel,
            'Eff_tot': Eff_tot,
            'FE': FE,
            'SE': SE,
            'DE': DE,
            'c511': c511})
        return resp, compton_matrix, np.linspace(a0_cmp, a1_cmp*(N_cmp - 1), N_cmp)

    def interpolate(self,
                    Eout: np.ndarray = None,
                    fwhm_abs: float = None):
        """
        Perform the interpolation for the energy range specified in Eout with FWHM at 1332 keV
        given by FWHM_abs (in keV). 
        """

        assert(1e-1 <= fwhm_abs <= 1000), "Check the fwhm_abs, probably it's wrong."\
        "\nNormal Oscar≃30 keV, Now: {} keV".format(fwhm_abs)

        if len(Eout) <= 1:
            raise ValueError(f"Eout should have more elements than 1, now {len(Eout)}")

        N_out = len(Eout)
        a0_out, a1_out = Eout[0], Eout[1]-Eout[0]


        assert abs(self.f_fwhm_rel_perCent_norm(1330)-1) < 0.05, \
            "Response function format not as expected. In the Mama-format, the"\
            "'f_fwhm_rel_perCent' column denotes the relative fwhm (= fwhm/E)," \
            "but normalized to 1 at 1.33 MeV."\
            "Now it is: {} at 1.33 MeV.".format(self.f_fwhm_rel_perCent_norm(1330))
        fwhm_rel_1330 = (fwhm_abs/1330*100)
        f_fwhm_rel_perCent = interp1d(self.resp['Eg'], self.resp['FWHM_rel']*fwhm_rel_1330,
                                      kind="linear",
                                      bounds_error=False,
                                      fill_value="extrapolate")

        fwhm_abs_array = Eout*f_fwhm_rel_perCent(Eout)/100

        Egmin = Eout[0]
        i_Egmin = index(Eout, Egmin)

        R = np.zeros((N_out, N_out))
        # Loop over rows of the response matrix
        # TODO for speedup: Change this to a cython .pyx, declare the j variable.
        #                   + run a Cython profiler, probably use memory views and
        #                   other tweaks to speedup (see rebin.pyx for examples).
        for j in range(N_out):
            E_j = Eout[j]
            # Skip if below lower threshold
            if E_j < Egmin:
                continue

            # Find maximal energy for current response function,
            # Changed to 1*sigma, or whatever this means
            # -> Better if the lowest energies of the simulated spectra are above
            # the gamma energy to be extrapolatedu
            Egmax = E_j + 1*fwhm_abs*self.f_fwhm_rel_perCent_norm(E_j)/2.35 #FWHM_rel.max()/2.35
            i_Egmax = min(index(Eout, Egmax), N_out)
            # print("i_Egmax =", i_Egmax)

            # MAMA unfolds with 1/10 of real FWHM for convergence reasons.
            # But let's stick to letting FWHM denote the actual value, and divide by 10 in computations if necessary.

            # Find the closest energies among the available response functions, to interpolate between:
            i_g_sim_low = 0
            try:
                i_g_sim_low = np.where(self.resp['Eg'] <= E_j)[0][-1]
            except IndexError:
                pass
            i_g_sim_high = len(self.resp['Eg'])
            try:
                i_g_sim_high = np.where(self.resp['Eg'] >= E_j)[0][0]
            except IndexError:
                pass
            # When E_out[j] is below lowest Eg_sim_array element? Interpolate between two larger?
            if i_g_sim_low == i_g_sim_high:
                if i_g_sim_low > 0:
                    i_g_sim_low -= 1
                else:
                    i_g_sim_high += 1

            Eg_low = self.resp['Eg'][i_g_sim_low]
            Eg_high = self.resp['Eg'][i_g_sim_high]

            # Next, select the Compton spectra at index i_g_sim_low and i_g_sim_high. These are called Fs1 and Fs2 in MAMA.
            # print("Eg_low =", Eg_low, "Eg_high =", Eg_high)
            # print("i_g_sim_low =", i_g_sim_low, "i_g_sim_high =", i_g_sim_high, flush=True)

            cmp_low = self.cmp_matrix[i_g_sim_low,:]
            cmp_high = self.cmp_matrix[i_g_sim_high,:]
            # These need to be recalibrated from Ecmp_array to Eout_array:
            cmp_low = rebin_1D(cmp_low, self.Ecmp_array, Eout)
            cmp_high = rebin_1D(cmp_high, self.Ecmp_array, Eout)
            # print("Eout_array[{:d}] = {:.1f}".format(j, E_j), "Eg_low =", Eg_sim_array[i_g_sim_low], "Eg_high =", Eg_sim_array[i_g_sim_high], flush=True)

            # The interpolation is split into energy regions.
            # Below the back-scattering energy Ebsc we interpolate linearly,
            # then we apply the "fan method" (Guttormsen 1996) in the region
            # from Ebsc up to the Compton edge, then linear extrapolation again the rest of the way.

            # Get maximal energy by taking 6*sigma above full-energy peak
            E_low_max = Eg_low + 6*fwhm_abs_array[i_g_sim_low]/2.35
            i_low_max = min(index(Eout, E_low_max), N_out-1)
            E_high_max = Eg_high + 6*fwhm_abs_array[i_g_sim_high]/2.35
            i_high_max =min(index(Eout, E_high_max), N_out-1)
            # print("E_low_max =", E_low_max, "E_high_max =", E_high_max, flush=True)

            # Find back-scattering Ebsc and compton-edge Ece energy of the current Eout energy:
            Ece = self.E_compton(E_j, theta=np.pi)
            Ebsc = E_j - Ece
            # if E_j==200:
            #     print(E_j)
            #     print("Ece =", Ece)
            #     print("Ebsc =", Ebsc)
            # Indices in Eout calibration corresponding to these energies:

            i_ce_out = min(index(Eout, Ece), i_Egmax)
            i_bsc_out = min(index(Eout, Ebsc), i_Egmin)

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
                E_i = Eout[i] # Energy of current point in interpolated spectrum
                if E_i > 0.1 and E_i < Ece:
                    if np.abs(E_j - E_i) > 0.001:
                        z = E_i/(E_j/511 * (E_j - E_i))
                    theta = np.arccos(1-z)
                    # print("theta = ", theta, flush=True)
                    if theta > 0 and theta < np.pi:
                        # Determine interpolation indices in low and high arrays
                        # by Compton formula
                        Ecmp_ = self.E_compton(Eg_low, theta)
                        i_low_interp = min(index(Eout, Ecmp_), i_bsc_out)
                        Ecmp_ = self.E_compton(Eg_high, theta)
                        i_high_interp = min(index(Eout, Ecmp_), i_high_max)
                        FA = (cmp_high[i_high_interp]*self.corr(Eg_high, theta)
                              - cmp_low[i_low_interp]*self.corr(Eg_low, theta))
                        FB = cmp_low[i_low_interp]*self.corr(Eg_low, theta) + FA*(E_j - Eg_low)/(Eg_high - Eg_low)
                        R[j, i] = FB/self.corr(E_j, theta)
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
            if E_j < self.resp['Eg'][0]:
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
            E_fe = Eout[j] + a1_out/2  # Evaluate energies in middle-bin

            # Add full-energy peak, which should be at energy corresponding to
            # index j:
            # full_energy = np.zeros(N_out)  # Allocate with zeros everywhere
            # full_energy[j] = f_pFE(E_fe)  # Full probability into sharp peak
            discrete_peaks[j] = self.f_pFE(E_fe)

            # Smoothe it:
            # full_energy = gauss_smoothing(full_energy, Eout_array,
                                          # fwhm_abs_array)
            # R[j, :] += full_energy

            # Add single-escape peak, at index i_se
            E_se = E_fe - 511
            if E_se >= 0 and E_se >= Eout[0]:
                i_floor, i_ceil, floor_distance\
                    = self.two_channel_split(E_se, Eout)
                # single_escape = np.zeros(N_out)  # Allocate with zeros everywhere
                # Put a portion of the counts into floor bin - the further away,the
                # less counts:
                # single_escape[i_floor] = (1-floor_distance) * f_pSE(E_fe)
                # single_escape[i_ceil] = floor_distance * f_pSE(E_fe)
                discrete_peaks[i_floor] += (1-floor_distance) * self.f_pSE(E_fe)
                discrete_peaks[i_ceil] += floor_distance * self.f_pSE(E_fe)
                # single_escape = gauss_smoothing(single_escape, Eout_array,
                                                # fwhm_abs_array)  # Smoothe
                # R[j, :] += single_escape

            # Repeat for double-escape peak, at index i_de
            E_de = E_fe - 2*511
            if E_de >= 0 and E_de >= Eout[0]:
                i_floor, i_ceil, floor_distance\
                    = self.two_channel_split(E_de, Eout)
                # double_escape = np.zeros(N_out)
                # double_escape[i_floor] = (1-floor_distance) * f_pDE(E_fe)
                # double_escape[i_ceil] = floor_distance * f_pDE(E_fe)
                discrete_peaks[i_floor] += (1-floor_distance) * self.f_pDE(E_fe)
                discrete_peaks[i_ceil] += floor_distance * self.f_pDE(E_fe)
                # double_escape = gauss_smoothing(double_escape, Eout_array,
                                            # fwhm_abs_array)  # Smoothe
                # R[j, :] += double_escape

            # Add 511 annihilation peak, at index i_an
            if E_fe > 511 and 511 >= Eout[0]:
                E_511 = 511
                i_floor, i_ceil, floor_distance\
                    = self.two_channel_split(E_511, Eout)
                # fiveeleven = np.zeros(N_out)
                # fiveeleven[i_floor] = (1-floor_distance) * f_p511(E_fe)
                # fiveeleven[i_ceil] = floor_distance * f_p511(E_fe)
                discrete_peaks[i_floor] += (1-floor_distance) * self.f_p511(E_fe)
                discrete_peaks[i_ceil] += floor_distance * self.f_p511(E_fe)
                # fiveeleven = gauss_smoothing(fiveeleven, Eout_array,
                                             # fwhm_abs_array)  # Smoothe
                # R[j, :] += fiveeleven

            # Do common smoothing of the discrete_peaks array:
            discrete_peaks = gauss_smoothing(discrete_peaks, Eout,
                                             fwhm_abs_array)  # Smoothe

            R[j, :] += discrete_peaks

            # === Finally, normalise the row to unity (probability conservation): ===
            R[j, :] = div0(R[j, :], np.sum(R[j, :]))

        # END loop over Eout energies Ej

        # Remove any negative elements from response matrix:
        R[R < 0] = 0

        response = Matrix(values=R, Eg=Eout, Ex=Eout)

        return response

    @staticmethod
    def interpolate_response(path: Union[str,Path] = None,
                             Eout: np.ndarray = None,
                             fwhm_abs: float = None,
                             response_obj: Optional[bool] = False):
        """
        Static method that performs the same as the old interpolate_response function.
        """
        resp = Response(path)

        if response_obj:
            return resp.interpolate(Eout, fwhm_abs), resp
        else:
            return resp.interpolate(Eout, fwhm_abs)

    @staticmethod
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

    @staticmethod
    def corr(Eg, theta):
        """
        Function to correct number of counts due to delta(theta)
        Adapted from MAMA in the file kelvin.f
        It is dE/dtheta of the E(theta) in Eq. (2) in Guttormsen 1996.
        """
        return (Eg*Eg/511*np.sin(theta))/(1+Eg/511*(1-np.cos(theta)))**2


    @staticmethod
    def two_channel_split(E_centroid, E_array):
        """
        When E_centroid is between two bins in E_array, this function
        returns the indices of the two nearest bins and the distance to
        the lower bin. The distance to the higher bin is 1-floor_distance

        Args:
            E_centroid (double): The energy of the centroid (mid-bin)
            E_array (np.array, double): The energy grid to distribute
        """

        a0 = E_array[0]
        a1 = E_array[1]-E_array[0]

        bin_as_float = (E_centroid - a0)/a1
        i_floor = int(np.floor(bin_as_float))
        i_ceil = int(np.ceil(bin_as_float))
        floor_distance = (bin_as_float - i_floor)

        return i_floor, i_ceil, floor_distance
