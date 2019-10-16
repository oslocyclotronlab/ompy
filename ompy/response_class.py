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
from typing import Union, Optional, Tuple
from scipy.interpolate import interp1d
import logging

from .rebin import rebin_1D
from .library import div0
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
                 path: Union[str, Path]=None):
        """
        The resonse object is initialized with the path to the source
        files required to perform the interpolation. The path varaiable
        can either be a folder (assuming "old" format) or a zip file
        containing the files otherwise found in the folder in the "old"
        format.

        TODO:
            - adapt rutines for the possibility that not all cmp spectra have
              the same binning
        """

        path = Path(path) if isinstance(path, str) else path

        if path.is_dir():
            self.resp, self.compton_matrix, self.Ecmp_array = self.LoadDir(
                path)  # Better names would be adventagious
        elif path.is_file():
            self.resp, self.compton_matrix, self.Ecmp_array = self.LoadZip(
                path)

    def LoadZip(self,
                path: Union[str, Path],
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
        if not set(['Eg', 'FWHM_rel_norm', 'Eff_tot', 'FE', 'SE', 'DE', 'c511']).issubset(resp.columns):
            raise ValueError(f'{resp_name} missing one or more required columns')

        # Verify that zip file contains a spectra for each energy
        files = [spec_prefix + str(int(Eg)) for Eg in sorted(resp['Eg'])]
        if not all(file in zfile.namelist() for file in files):
            raise ValueError(f'One or more compton spectra is missing in {path}')

        # Now we will read in all the Compton spectra
        N_cmp = -1
        a0_cmp, a1_cmp = -1, -1
        # Get calibration and array length from highest-energy spectrum, because the spectra
        # may have differing length but this is bound to be the longest.
        with zfile.open(spec_prefix + str(int(max(resp['Eg']))), 'r') as file:
            lines = file.readlines()
            a0_cmp = float(str(lines[6]).split(",")[1])  # calibration
            a1_cmp = float(str(lines[6]).split(",")[2])  # coefficients [keV]
            N_cmp = int(lines[8][15:]) + 1  # 0 is first index

        compton_matrix = np.zeros((len(resp['Eg']), N_cmp))
        i = 0
        for file in [zfile.open(file_name) for file_name in files]:
            cmp_current = np.genfromtxt(file, comments="!")
            compton_matrix[i, 0:len(cmp_current)] = cmp_current
            i += 1

        return resp, compton_matrix, np.linspace(a0_cmp, a1_cmp * (N_cmp - 1), N_cmp)

    def LoadDir(self,
                path: Union[str, Path],
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
                    # TODO: The above if test is hardly very robust. Find a
                    # better solution.
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
        a0_sim, a1_sim = Eg_sim_array[0], Eg_sim_array[1] - Eg_sim_array[0]
        # print("a0_sim, a1_sim =", a0_sim, a1_sim, flush=True)
        # "Eg_sim" means "gamma, simulated", and refers to the gamma energies where we have simulated Compton spectra.

        # Read in Compton spectra for each Eg channel:
        N_Eg = len(Eg_sim_array)
        # Read first Compton spectrum to get number of energy channels in each:
        N_cmp = -1
        a0_cmp, a1_cmp = -1, -1
        # Get calibration and array length from highest-energy spectrum, because the spectra
        # may have differing length but this is bound to be the longest.
        with open(os.path.join(path, "cmp" + str(int(Eg_sim_array[-1])))) as file:
            lines = file.readlines()
            a0_cmp = float(lines[6].split(",")[1])  # calibration
            a1_cmp = float(lines[6].split(",")[2])  # coefficients [keV]
            N_cmp = int(lines[8][15:]) + 1  # 0 is first index
        # print("a0_cmp, a1_cmp, N_cmp = ", a0_cmp, a1_cmp, N_cmp)
        compton_matrix = np.zeros((N_Eg, N_cmp))
        # Read the rest:
        for i in range(0, N_Eg):
            fn = "cmp" + str(Eg_sim_array[i])
            cmp_current = np.genfromtxt(os.path.join(
                path, "cmp" + str(int(Eg_sim_array[i]))), comments="!")
            compton_matrix[i, 0:len(cmp_current)] = cmp_current

        resp = pd.DataFrame(data={
            'Eg': Eg_sim_array,
            'FWHM_rel_norm': fwhm_rel,
            'Eff_tot': Eff_tot,
            'FE': FE,
            'SE': SE,
            'DE': DE,
            'c511': c511})
        return resp, compton_matrix, np.linspace(a0_cmp, a1_cmp * (N_cmp - 1), N_cmp)

    def get_probabilities(self):
        # total number of counts for each of the loaded responses
        self.sum_spec = self.compton_matrix.sum(axis=1) \
            + self.resp['FE'] + self.resp['SE'] + self.resp['DE'] \
            + self.resp['c511']
        self.sum_spec = np.array(self.sum_spec)

        # normalize "compton" spectra
        self.cmp_matrix = div0(self.compton_matrix,
                               self.sum_spec.reshape((len(self.sum_spec), 1)))
        # Vector of total Compton probability
        self.pcmp = self.cmp_matrix.sum(axis=1)

        # Full energy, single escape, etc:
        self.pFE = div0(self.resp['FE'], self.sum_spec)
        self.pSE = div0(self.resp['SE'], self.sum_spec)
        self.pDE = div0(self.resp['DE'], self.sum_spec)
        self.p511 = div0(self.resp['c511'], self.sum_spec)

        # Interpolate the peak structures except Compton (handled separately)
        def interpolate(y, fill_value="extrapolate"):
            return interp1d(self.resp['Eg'], y,
                            kind="linear", bounds_error=False,
                            fill_value=fill_value)

        self.f_pcmp = interpolate(self.pcmp)
        self.f_pFE = interpolate(self.pFE)
        self.f_pSE = interpolate(self.pSE, fill_value=0)
        self.f_pDE = interpolate(self.pDE, fill_value=0)
        self.f_p511 = interpolate(self.p511, fill_value=0)
        self.f_fwhm_rel_perCent_norm = interpolate(self.resp['FWHM_rel_norm'])
        # TODO: Should this be extrapolated, too?
        self.f_Eff_tot = interpolate(self.resp['Eff_tot'], fill_value=0)

        fwhm_rel_1330 = (self.fwhm_abs / 1330 * 100)
        self.f_fwhm_rel_perCent = interpolate(self.resp['FWHM_rel_norm']
                                              * fwhm_rel_1330)
        def f_fwhm_abs(E):
            return E * self.f_fwhm_rel_perCent(E)/100

        self.f_fwhm_abs = f_fwhm_abs

    def iterpolate_checks(self):
        assert(1e-1 <= self.fwhm_abs <= 1000), \
            "Check the fwhm_abs, probably it's wrong."\
            "\nNormal Oscar≃30 keV, Now: {} keV".format(self.fwhm_abs)

        Eout = self.Eout
        if len(Eout) <= 1:
            raise ValueError("Eout should have more elements than 1" \
                             f"now {len(Eout)}")

        assert abs(self.f_fwhm_rel_perCent_norm(1330) - 1) < 0.05, \
            "Response function format not as expected." \
            "In the Mama-format, the 'f_fwhm_rel_perCent' column denotes"\
            "the relative fwhm (= fwhm/E), but normalized to 1 at 1.33 MeV."\
            f"Now it is: {self.f_fwhm_rel_perCent_norm(1330)} at 1.33 MeV."

    def get_closest_compton(self, E: float) -> Tuple[int, int]:
        """Find and rebin closest energies from available response functions

        Args:
            E (float): Description
        Returns:
            ilow (float), ihigh (float): Indexec of closest energies
        """
        N = len(self.resp['Eg'])
        # ilow = 0
        ihigh = np.searchsorted(self.resp['Eg'], E, side="right")
        if ihigh == N:  # E > self.resp['Eg'].max()
            ihigh -= 1

        if ihigh == 0:  # E < self.resp['Eg'].min()
            ilow = 0
        else:
            ilow = ihigh-1

        Elow = self.resp['Eg'][ilow]
        Ehigh = self.resp['Eg'][ihigh]

        # Next, select the Compton spectra , called Fs1 and Fs2 in MAMA.
        cmp_low = self.cmp_matrix[ilow, :]
        cmp_high = self.cmp_matrix[ihigh, :]
        cmp_low = rebin_1D(cmp_low, self.Ecmp_array, self.Eout)
        cmp_high = rebin_1D(cmp_high, self.Ecmp_array, self.Eout)

        compton = {"ilow": ilow,
                   "ihigh": ihigh,
                   "Elow": Elow,
                   "Ehigh": Ehigh,
                   "counts_low": cmp_low,
                   "counts_high": cmp_high}

        return compton

    def linear_cmp_interpolation(self, E, compton):
        """ Linear interpolation between the compton spectra """
        x = np.array([compton["Elow"], compton["Ehigh"]])
        y = np.vstack([compton["counts_low"], compton["counts_high"]])
        f_cmp = interp1d(x, y, kind="linear", bounds_error=False, fill_value=0,
                         axis=0)
        return f_cmp(E)

    def linear_backscatter(self, E, compton):
        """ Interpolate one-to-one up to the backscatter peak """
        Eedge = self.E_compton(E, theta=np.pi)  # compton-edge energy
        Ebsc = E - Eedge  # back-scattering energy
        i_bsc = index(self.Eout, Ebsc)
        R = np.zeros(self.N_out)

        fcmp = self.linear_cmp_interpolation(E, compton)
        R[:i_bsc+1] = fcmp[:i_bsc+1]
        R[R < 0] = 0
        return R, i_bsc

    def linear_to_end(self, E, compton, i_start, i_stop):
        """ Interpolate one-to-one from the last fan energy to the Emax """
        R = np.zeros(self.N_out)
        fcmp = self.linear_cmp_interpolation(E, compton)
        R[i_start:i_stop+1] = fcmp[i_start:i_stop+1]
        R[R < 0] = 0
        return R

    def fan_method(self, E, compton, i_start, i_stop):
        """ Fan method
        Args:
            i_response (int): loop index in response matrix
            i_bsc (int): index up to where linear interpolation
                to the backscatter peak ran
        Returns:
        """
        R = np.zeros(self.N_out)

        Ece = self.E_compton(E, theta=np.pi)
        i_E_max = min(i_stop, self.N_out)
        i_ce_max = min(index(self.Eout, Ece), i_E_max)

        # Get maximal energy by taking n*sigma above full-energy peak
        # (because compton is not just compton, but anything that is
        #  not discretes)
        Esim_low = compton["Elow"]
        # E_low_max = Esim_low + 6 * self.f_fwhm_abs(Esim_low) / 2.35
        # i_low_max = min(index(self.Eout, E_low_max), self.N_out - 1)

        Esim_high = compton["Ehigh"]
        E_high_max = Esim_high + 6 * self.f_fwhm_abs(Esim_high) / 2.35
        i_high_max = min(index(self.Eout, E_high_max), self.N_out - 1)
        LOG.debug("Maximum energies for fan-method: {E_low_max:.0f}"
                  "{E_high_max:.0f}")

        # Then interpolate with the fan method up to j_ce_out:
        i_last = i_start  # Keep track of how far up the fan method goes

        def lin_interpolation(x, x0, y0, x1, y1):
            return y0 + (y1-y0)*(x-x0)/(x1-x0)

        for i in range(i_start, i_ce_max):
            # In Mama: E -> Egam, Ei -> E [Fabio]
            # Energy of current point in interpolated spectrum
            Ei = self.Eout[i]
            if Ei < 0.1 or Ei > Ece:
                continue
            z = div0(Ei, (E / 511 * (E - Ei)))
            theta = np.arccos(1 - z)
            if theta > 0 and theta < np.pi:
                # Determine interpolation indices in low and high arrays
                # by Compton formula
                Ecmp_ = self.E_compton(Esim_low, theta)
                i_low_interp = min(index(self.Eout, Ecmp_), i_start)
                Ecmp_ = self.E_compton(Esim_high, theta)
                i_high_interp = min(index(self.Eout, Ecmp_), i_high_max)

                c1 = compton["counts_low"][i_low_interp]
                c2 = compton["counts_high"][i_high_interp]

                # apply correction
                c1 *= self.dE_dtheta(Esim_low, theta)
                c2 *= self.dE_dtheta(Esim_high, theta)
                x = [Esim_low, Esim_high]
                y = [c1, c2]

                # essential equation c(E), which is below (2)
                # if Ei < Esim_low or Ei < Esim_high:
                #     print(Ei, Esim_low, Esim_high)
                interpol = lin_interpolation(E, Esim_low, c1, Esim_high, c2)
                R[i] = interpol / self.dE_dtheta(E, theta)
                # interpol = interp1d(x, y,
                #                     fill_value="extrapolate",
                #                     bounds_error=False)
                # R[i] = interpol(E) / self.dE_dtheta(E, theta)
                i_last = i

        # if 1150 < E < 1250:
        #     print(E, index(self.Eout, E), self.Eout[i_start], self.Eout[i_ce_max])
        #     if R<=0:
        #         print(E, R)
            # print(f"i_start {i_start}, E {E}, Ece {Ece}, i_ce_max {i_ce_max}, i_stop{i_stop}")
        if len(R[R < 0]) != 0:
            print("In fan method, some R is negative at: ", E,
                  " with", len(R[R < 0]), "entries")
        R[R < 0] = 0

        return R, i_last

    def discrete_peaks(self, i_response, fwhm_abs_array):
        discrete_peaks = np.zeros(self.N_out)
        Eout = self.Eout
        E_fe = Eout[i_response]

        # Add full-energy peak, which should be at energy corresponding to
        # index i_response:
        # full_energy = np.zeros(N_out)  # Allocate with zeros everywhere
        # full_energy[i_response] = f_pFE(E_fe)  # Full probability into sharp peak
        discrete_peaks[i_response] = self.f_pFE(E_fe)

        # Smoothe it:
        # full_energy = gauss_smoothing(full_energy, Eout_array,
        # fwhm_abs_array)
        # R[i_response, :] += full_energy

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
            discrete_peaks[
                i_floor] += (1 - floor_distance) * self.f_pSE(E_fe)
            discrete_peaks[i_ceil] += floor_distance * self.f_pSE(E_fe)
            # single_escape = gauss_smoothing(single_escape, Eout_array,
            # fwhm_abs_array)  # Smoothe
            # R[i_response, :] += single_escape

        # Repeat for double-escape peak, at index i_de
        E_de = E_fe - 2 * 511
        if E_de >= 0 and E_de >= Eout[0]:
            i_floor, i_ceil, floor_distance\
                = self.two_channel_split(E_de, Eout)
            # double_escape = np.zeros(N_out)
            # double_escape[i_floor] = (1-floor_distance) * f_pDE(E_fe)
            # double_escape[i_ceil] = floor_distance * f_pDE(E_fe)
            discrete_peaks[
                i_floor] += (1 - floor_distance) * self.f_pDE(E_fe)
            discrete_peaks[i_ceil] += floor_distance * self.f_pDE(E_fe)
            # double_escape = gauss_smoothing(double_escape, Eout_array,
            # fwhm_abs_array)  # Smoothe
            # R[i_response, :] += double_escape

        # Add 511 annihilation peak, at index i_an
        if E_fe > 511 and 511 >= Eout[0]:
            E_511 = 511
            i_floor, i_ceil, floor_distance\
                = self.two_channel_split(E_511, Eout)
            # fiveeleven = np.zeros(N_out)
            # fiveeleven[i_floor] = (1-floor_distance) * f_p511(E_fe)
            # fiveeleven[i_ceil] = floor_distance * f_p511(E_fe)
            discrete_peaks[
                i_floor] += (1 - floor_distance) * self.f_p511(E_fe)
            discrete_peaks[i_ceil] += floor_distance * self.f_p511(E_fe)
            # fiveeleven = gauss_smoothing(fiveeleven, Eout_array,
            # fwhm_abs_array)  # Smoothe
            # R[i_response, :] += fiveeleven

        # Do common smoothing of the discrete_peaks array:
        discrete_peaks = gauss_smoothing(discrete_peaks, Eout,
                                         fwhm_abs_array)  # Smoothe
        return discrete_peaks

    def interpolate(self,
                    Eout: np.ndarray = None,
                    fwhm_abs: float = None,
                    return_table: bool = False):
        """ Interpolated the response matrix

        Perform the interpolation for the energy range specified in Eout with
        FWHM at 1332 keV given by FWHM_abs (in keV).

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
        self.Eout = Eout
        self.fwhm_abs = fwhm_abs

        self.get_probabilities()
        self.iterpolate_checks()

        N_out = len(Eout)
        self.N_out = N_out
        fwhm_abs_array = Eout * self.f_fwhm_rel_perCent(Eout) / 100

        R = np.zeros((N_out, N_out))
        # Loop over rows of the response matrix
        # TODO for speedup: Change this to a cython
        for j, E in enumerate(Eout):

            # Find maximal energy for current response (+n*sigma) function,
            # -> Better if the lowest energies of the simulated spectra are
            #    above the gamma energy to be extrapolated
            oneSigma = fwhm_abs * self.f_fwhm_rel_perCent_norm(E) / 2.35
            Egmax = E + 1 * oneSigma
            i_Egmax = min(index(Eout, Egmax), N_out-1)
            # print("i_Egmax =", i_Egmax)
            LOG.debug("Response for E: {E:.0f} calc. up to {Egmax:.0f}")

            compton = self.get_closest_compton(E)

            # The interpolation is split into energy regions.
            # Below the back-scattering energy Ebsc we interpolate linearly,
            # then we apply the "fan method" (Guttormsen 1996) in the region
            # from Ebsc up to the Compton edge, then linear extrapolation again
            # the rest of the way.

            R_linear, i_bsc = self.linear_backscatter(E, compton)
            # R[j, :] += R_linear

            R_fan, i_last = self.fan_method(E, compton,
                                            i_start=i_bsc, i_stop=i_Egmax)
            R[j, :] += R_fan

            R_linear = self.linear_to_end(E, compton,
                                          i_start=i_last, i_stop=i_Egmax)
            # R[j, :] += R_linear

            R[R < 0] = 0
            # coorecton below E_sim[0]
            if E < self.resp['Eg'][0]:
                R[j, j + 1:] = 0

            R[j, :] = gauss_smoothing(R[j, :], self.Eout,
                                      fwhm_abs_array)

            # discrete_peaks = self.discrete_peaks(j, fwhm_abs_array)
            # R[j, :] += discrete_peaks

            # === Finally, normalise the row to unity (probability conservation): ===
            R[j, :] = div0(R[j, :], np.sum(R[j, :]))

        # END loop over Eout energies Ej

        # Remove any negative elements from response matrix:
        R[R < 0] = 0

        response = Matrix(values=R, Eg=Eout, Ex=Eout)

        if return_table:
            # Return the response matrix, as well as the other structures, FWHM and efficiency, interpolated to the Eout_array
            response_table = {'E': Eout,
                              'fwhm_abs': fwhm_abs_array,
                              'fwhm_rel_%': self.f_fwhm_rel_perCent(Eout),
                              'fwhm_rel': self.f_fwhm_rel_perCent(Eout)/100,
                              'eff_tot': self.f_Eff_tot(Eout),
                              'pcmp': self.f_pcmp(Eout),
                              'pFE': self.f_pFE(Eout),
                              'pSE': self.f_pSE(Eout),
                              'pDE': self.f_pDE(Eout),
                              'p511': self.f_p511(Eout)}
            response_table = pd.DataFrame(data=response_table)
            return response, response_table
        else:
            return response

    @staticmethod
    def interpolate_response(path: Union[str, Path]=None,
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
        return np.where(Eg > 0.1, Eg * Eg / 511 * (1 - np.cos(theta)) / (1 + Eg / 511 * (1 - np.cos(theta))), Eg)

    @staticmethod
    def dE_dtheta(Eg, theta):
        """
        Function to correct number of counts due to delta(theta)
        Adapted from MAMA in the file kelvin.f
        It is dE/dtheta of the E(theta) in Eq. (2) in Guttormsen 1996.
        """
        a = (Eg * Eg / 511 * np.sin(theta))
        b = (1 + Eg / 511 * (1 - np.cos(theta)))**2
        return a / b

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
        a1 = E_array[1] - E_array[0]

        bin_as_float = (E_centroid - a0) / a1
        i_floor = int(np.floor(bin_as_float))
        i_ceil = int(np.ceil(bin_as_float))
        floor_distance = (bin_as_float - i_floor)

        return i_floor, i_ceil, floor_distance
