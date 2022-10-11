# -*- coding: utf-8 -*-
"""
Implementation of a response matrix created from interpolation
of experimental spectra. It takes the path to a folder (old format)
or a zip file as its construction parameters.
"""

import os
import zipfile as zf
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union, Optional, Tuple, Dict, Any
from scipy.interpolate import interp1d
import logging
from numba import njit, int32, float32, float64, jit
from collections import OrderedDict
from numba.experimental import jitclass

#from .rebin import rebin_1D
from .library import div0
#from .decomposition import index
#from .gauss_smoothing import gauss_smoothing
from .matrix import Matrix
from .vector import Vector
from .stubs import Pathlike
from . import Toggle, Unitful

LOG = logging.getLogger(__name__)
logging.captureWarnings(True)

DTYPE = np.float64

spec = OrderedDict()
spec['ilow'] = int32
spec['ihigh'] = int32
spec['elow'] = float32
spec['ehigh'] = float32
spec['low'] = float64[::1]
spec['high'] = float64[::1]

@jitclass(spec)
class ComptonNeighbours(object):
    def __init__(self, ilow, ihigh, elow, ehigh, low, high):
        self.ilow = ilow
        self.ihigh = ihigh
        self.elow = elow
        self.ehigh = ehigh
        self.low = low
        self.high = high

    def vstack(self):
        return np.vstack((self.low, self.high))

    def energies(self):
        return np.array([self.elow, self.ehigh])


class Response():
    """ Interpolates response read from file for current setup

    Implementaion of following method
    Guttormsen et al., NIM A 374 (1996) 371–376.
    DOI:10.1016/0168-9002(96)00197-0

    Throughout the class, be aware that "compton" mat refer to all
    non-discrete structures (instead of the real Compton effect only).

    Attributes:
        resp (pd.DataFrame): Information of the `response table`
        compton_matrix (np.ndarray): array with compton counts.
            Shape is (N_incident, N_cmp).
        Ecmp_array (np.ndarray): energy array for the compton counts
        smooth_compton (bool): If True, the compoton array is smoothed
            before further processing. defaults to `False`
        truncate (float): After how many sigma to truncate gaussian smoothing.
            Defaults to 6.

    """
    # if compton was not smoothed before
    smooth_compton = Toggle(False)
    # Whether to smooth the full energy peak
    smooth_fe = Toggle(True)
    fwhm_abs = Unitful('0 keV')

    def __init__(self, path: Pathlike):
        """
        The resonse object is initialized with the path to the source
        files required to perform the interpolation. The path varaiable
        can either be a folder (assuming "old" format) or a zip file
        containing the files otherwise found in the folder in the "old"
        format.

        Args:
            path (str or Path): Path to the required file(s)

        TODO:
            - adapt rutines for the possibility that not all cmp spectra have
              the same binning
        """
        path = Path(path)
        path = path.expanduser()  # if "~" was used for the homedir
        path = path.resolve()  # if relative path was used

        if path.is_dir():
            LOG.debug(f"Loading response from directory: {path}")
            self.resp, self.compton_matrix, self.Ecmp_array = self.LoadDir(
                path)  # Better names would be adventagious
        elif path.is_file():
            LOG.debug(f"Loading response from file: {path}")
            self.resp, self.compton_matrix, self.Ecmp_array = self.LoadZip(
                path)
        elif not path.exists():

            raise ValueError(f"Path {path} does not exist")

        # after how many sigma to truncate gaussian smoothing
        self.truncate: float = 6

    def LoadZip(self,
                path: Pathlike,
                resp_name: str = 'resp.csv',
                spec_prefix: str = 'cmp'):
        """ Method for loading response file and compton spectra from zipfile.

        Is assumes that path is a zip file that constains at least XX files.
        At least one has to be a special summary table to be named `resp_name`.

        Args:
            path (Pathlike): path to folder
            resp_name (str, optional): name of file with
                `response table`
            spec_prefix (str, optional): Prefix for all spectra

        Returns:
            (tuple) containing
                - **resp** (*DataFrame*): Information of the `response table`.
                - **compton_matrix** (*ndarray*): matrix with compton counts.
                    Shape is (N_incident, N_cmp)
                - **last.E** (*ndarray*): energy array

        """
        path = Path(path) if isinstance(path, str) else path

        zfile = zf.ZipFile(path, mode='r')

        if not any(resp_name in name for name in zfile.namelist()):
            raise ValueError('Response info not present')

        resp = pd.read_csv(zfile.open(resp_name, 'r'))

        # Verify that resp has all the required columns
        if not set([
                'Eg', 'FWHM_rel_norm', 'Eff_tot', 'FE', 'SE', 'DE', 'c511'
        ]).issubset(resp.columns):
            raise ValueError(
                f'{resp_name} missing one or more required columns')

        # Verify that zip file contains a spectra for each energy
        files = [spec_prefix + str(int(Eg)) for Eg in sorted(resp['Eg'])]
        if not all(file in zfile.namelist() for file in files):
            raise ValueError(
                f'One or more compton spectra is missing in {path}')

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

        return resp, compton_matrix, np.linspace(a0_cmp, a1_cmp * (N_cmp - 1),
                                                 N_cmp)

    def LoadDir(self,
                path: Union[str, Path],
                resp_name: str | None = 'resp.dat',
                spec_prefix: str | None = 'cmp'):
        """
        Method for loading response file and compton spectra from a folder.

        Args:
            path (Union[str, Path]): path to folder
            resp_name (str | None, optional): name of file with
                `response table`
            spec_prefix (str | None, optional): Prefix for all spectra

        Returns:
            (tuple) containing
                - **resp** (*DataFrame*): Information of the `response table`.
                - **compton_matrix** (*ndarray*): matrix with compton counts.
                    Shape is (N_incident, N_cmp)
                - **last.E** (*ndarray*): energy array
        """

        # Read resp.dat file, which gives information about the energy bins
        # and discrete peaks
        resp = []
        Nlines = -1
        with open(os.path.join(path, resp_name)) as file:
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
        # "Eg_sim" means "gamma, simulated", and refers to the gamma energies
        # where we have simulated Compton spectra.

        # Get calibration and array length from highest-energy spectrum,
        # because the spectra may have differing length,
        # but the last is bound to be the longest.
        N_Eg = len(Eg_sim_array)
        fnames = []
        for i in range(0, N_Eg):
            fnames.append(f"{spec_prefix}{Eg_sim_array[i]:.0f}.m")

        last = Vector(path=os.path.join(path, fnames[-1]))
        N_cmp = last.shape[0]

        compton_matrix = np.zeros((N_Eg, N_cmp))
        # Read the rest:
        for i in range(0, N_Eg):
            cmp_current = Vector(path=os.path.join(path, fnames[i]))
            if cmp_current.calibration() != last.calibration():
                raise NotImplementedError("Currently, all cmp calibrations"
                                          "have to be the same")
            compton_matrix[i, 0:len(cmp_current.values)] = cmp_current.values

        resp = pd.DataFrame(
            data={
                'Eg': Eg_sim_array,
                'FWHM_rel_norm': fwhm_rel,
                'Eff_tot': Eff_tot,
                'FE': FE,
                'SE': SE,
                'DE': DE,
                'c511': c511
            })
        return resp, compton_matrix, last.E

    def get_probabilities(self):
        """ Interpolate full-energy peak probabilities (...) """
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

        Eg = self.resp['Eg'].to_numpy()
        def _interpolate(y, fill_value="extrapolate"):
            return interp1d(Eg,
                            y,
                            kind="linear",
                            bounds_error=False,
                            fill_value=fill_value)

        def interpolate(y, fill_value=0):
            return lambda x: np.interp(x, Eg, y)

        self.f_pcmp = interpolate(self.pcmp)
        self.f_pFE = interpolate(self.pFE)
        self.f_pSE = interpolate(self.pSE, fill_value=0)
        self.f_pDE = interpolate(self.pDE, fill_value=0)
        self.f_p511 = interpolate(self.p511, fill_value=0)
        self.f_fwhm_rel_perCent_norm = interpolate(self.resp['FWHM_rel_norm'])
        # TODO: Should this be extrapolated, too?
        self.f_Eff_tot = interpolate(self.resp['Eff_tot'])

        fwhm_rel_1330 = (self.fwhm_abs.magnitude / 1330 * 100)
        self.f_fwhm_rel_perCent = interpolate(self.resp['FWHM_rel_norm'] *
                                              fwhm_rel_1330)

        def f_fwhm_abs(E):  # noqa
            return E * self.f_fwhm_rel_perCent(E) / 100

        self.f_fwhm_abs = f_fwhm_abs

    #@njit()
    def interpolate(
        self,
        Eout: np.ndarray = None,
        fwhm_abs: float = None,
        return_table: bool = False,
    ) -> Matrix | Tuple[Matrix, pd.DataFrame]:
        """ Interpolated the response matrix

        Perform the interpolation for the energy range specified in Eout with
        FWHM at 1332 keV given by FWHM_abs (in keV).

        The interpolation is split into energy regions. Below the
        back-scattering energy Ebsc we interpolate linearly, then we apply the
        "fan method" (Guttormsen 1996) in the region from Ebsc up to the
        Compton edge, with a Compton scattering angle dependent interpolation.
        From the Compton edge to Egmax we also use a fan, but with a linear
        interpolation.

        Note:
            Below the ~350 keV we only use a linear interpolation, as the
            fan method does not work. This is not described in Guttormsen 1996.

        Args:
            folderpath: The path to the folder containing Compton spectra and
            resp.dat
            Eout_array: The desired energies of the output response matrix.
            fwhm_abs: The experimental absolute full-width-half-max at 1.33
                      MeV. Note: In the article it is recommended to use 1/10
                      of the real FWHM for unfolding.
            return_table (optional): Returns "all" output, see below

        Returns:
            Matrix or (Matrix, pd.DataFrame):
              - response (Matrix): Response matrix with incident energy on the
                "Ex" axis and the spectral response on the "Eg" axis
              - response_table (DataFrame, optional): Table with efficiencies,
                FE, SE (...) probabilities, and so on
        """
        self.Eout = Eout
        self.fwhm_abs = fwhm_abs

        self.get_probabilities()
        self.iterpolate_checks()

        N_out = len(Eout)
        self.N_out = N_out
        fwhm_abs_array = Eout * self.f_fwhm_rel_perCent(Eout) / 100

        R = np.zeros((N_out, N_out))
        Eg = self.resp['Eg'].to_numpy()
        # Loop over rows of the response matrix
        # TODO for speedup: Change this to a cython
        for j, E in enumerate(Eout):
            oneSigma = fwhm_abs_array[j] / 2.35
            Egmax = E + 6 * oneSigma
            i_Egmax = min(index(Eout, Egmax), N_out - 1)
            #LOG.debug(f"Response for E {E:.0f} calc up to {Eout[i_Egmax]:.0f}")

            # TODO This is a 50% bottleneck
            compton = get_closest_compton(E, Eg, self.cmp_matrix, self.Eout ,self.Ecmp_array)

            R_low, i_bsc = linear_backscatter(self.Eout, self.N_out, E, compton)
            R_fan, i_last = \
                fan_method_compton(E, compton, i_start=i_bsc+1, i_stop=i_Egmax,
                                   N_out=self.N_out, Eout=self.Eout)
            if i_last <= i_bsc + 2:  # fan method didn't do anything
                R_high = self.linear_to_end(E,
                                            compton,
                                            i_start=i_bsc + 1,
                                            i_stop=i_Egmax)
                R[j, :] += R_low + R_high
            else:
                R_high = fan_to_end(E,
                                         compton,
                                         i_start=i_last + 1,
                                         i_stop=i_Egmax,
                                         fwhm_abs_array=fwhm_abs_array,
                                    N_out=self.N_out, Eout=self.Eout)
                R[j, :] += R_low + R_fan + R_high

            # coorecton below E_sim[0]
            if E < Eg[0]:
                R[j, j + 1:] = 0

            # TODO This is a 30% bottleneck
            discrete_peaks = self.discrete_peaks(j, fwhm_abs_array)
            R[j, :] += discrete_peaks

            # smooth if compton background was not smoothed before (?)
            # if performed here, no need to smooth twice, see discrete_peaks
            if self.smooth_compton:
                R[j, :] = gauss_smoothing(R[j, :],
                                          self.Eout,
                                          fwhm_abs_array,
                                          truncate=self.truncate)

        # normalize (preserve probability)
        normalize(R)

        # Remove any negative elements from response matrix:
        if len(R[R < 0]) != 0:
            #LOG.debug(f"{len(R[R < 0])} entries in R were set to 0")
            R[R < 0] = 0

        response = Matrix(values=R,
                          Eg=Eout,
                          Ex=Eout,
                          xlabel=r'Measured $\gamma$-energy',
                          ylabel=r'True $\gamma$-energy')

        if return_table:
            # Return the response matrix, as well as the other structures,
            # FWHM and efficiency, interpolated to the Eout_array
            response_table = {
                'E': Eout,
                'fwhm_abs': fwhm_abs_array,
                'fwhm_rel_%': self.f_fwhm_rel_perCent(Eout),
                'fwhm_rel': self.f_fwhm_rel_perCent(Eout) / 100,
                'eff_tot': self.f_Eff_tot(Eout),
                'pcmp': self.f_pcmp(Eout),
                'pFE': self.f_pFE(Eout),
                'pSE': self.f_pSE(Eout),
                'pDE': self.f_pDE(Eout),
                'p511': self.f_p511(Eout)
            }
            response_table = pd.DataFrame(data=response_table)
            return response, response_table
        else:
            return response

    def iterpolate_checks(self):
        """ Check on the inputs to `interpolate` """
        assert(1e-1 <= self.fwhm_abs.magnitude <= 1000), \
            "Check the fwhm_abs, probably it's wrong."\
            "\nNormal Oscar≃30 keV, Now: {} keV".format(self.fwhm_abs.magnitude)

        Eout = self.Eout
        if len(Eout) <= 1:
            raise ValueError("Eout should have more elements than 1" \
                             f"now {len(Eout)}")

        assert abs(self.f_fwhm_rel_perCent_norm(1330) - 1) < 0.05, \
            "Response function format not as expected. " \
            "In the Mama-format, the 'f_fwhm_rel_perCent' column denotes "\
            "the relative fwhm (= fwhm/E), but normalized to 1 at 1.33 MeV. "\
            f"Now it is: {self.f_fwhm_rel_perCent_norm(1330)} at 1.33 MeV."

        LOG.info(f"Note: Spectra outside of {self.resp['Eg'].min()} and "
                 f"{self.resp['Eg'].max()} keV are extrapolation only.")



    def linear_to_end(self, E: float, compton: ComptonNeighbours, i_start: int,
                      i_stop: int) -> np.ndarray:
        """Interpolate one-to-one from the last fan energy to the Emax

        Args:
            E (float): Incident energy
            compton (dict): Dict. with information about the compton spectra
               to interpolate between
            i_start (int): Index where to start (usually end of fan method)
            i_stop (int): Index where to stop (usually E+n*resolution)

        Returns:
            np.ndarray: Response for `E`
        """
        R = np.zeros(self.N_out)
        #fcmp = interpolate_compton(compton, E)
        fcmp = interpolate_compton_2(compton, E)
        R[i_start:i_stop + 1] = fcmp[i_start:i_stop + 1]
        R[R < 0] = 0
        return R

    #@njit

    def discrete_peaks(self, i_response: int,
                       fwhm_abs_array: np.ndarray) -> np.ndarray:
        """Add discrete peaks for a given channel and smooth them

        Args:
            i_response (int): Channel in response matrix
            fwhm_abs_array (np.ndarray): Array with fwhms for each channel

        Returns:
            Array with smoothed discrete peaks
        """
        discrete_peaks = np.zeros(self.N_out)
        Eout = self.Eout
        E_fe = Eout[i_response]
        discrete_peaks[i_response] = self.f_pFE(E_fe)

        # Add single-escape peak
        E_se = E_fe - 511
        if E_se >= 0 and E_se >= Eout[0]:
            i_floor, i_ceil, floor_distance\
                = two_channel_split(E_se, Eout)
            discrete_peaks[i_floor] += (1 - floor_distance) * self.f_pSE(E_fe)
            discrete_peaks[i_ceil] += floor_distance * self.f_pSE(E_fe)

        # Repeat for double-escape peak
        E_de = E_fe - 2 * 511
        if E_de >= 0 and E_de >= Eout[0]:
            i_floor, i_ceil, floor_distance\
                = two_channel_split(E_de, Eout)
            discrete_peaks[i_floor] += (1 - floor_distance) * self.f_pDE(E_fe)
            discrete_peaks[i_ceil] += floor_distance * self.f_pDE(E_fe)

        # Add 511 annihilation peak
        if E_fe > 511 and 511 >= Eout[0]:
            E_511 = 511
            i_floor, i_ceil, floor_distance\
                = two_channel_split(E_511, Eout)
            discrete_peaks[i_floor] += (1 - floor_distance) * self.f_p511(E_fe)
            discrete_peaks[i_ceil] += floor_distance * self.f_p511(E_fe)

            #if not self.smooth_compton:
            # Do common smoothing of the discrete_peaks array:
        discrete_peaks = gauss_smoothing(discrete_peaks,
                                            Eout,
                                            fwhm_abs_array,
                                            truncate=self.truncate)

        return discrete_peaks

@njit
def E_compton(Eg, theta):
    """
    Calculates the energy of an electron that is scattered an angle
    theta by a gamma-ray of energy Eg.

    Note:
        For `Eg <= 0.1` it returns `Eg`. (workaround)

    Args:
        Eg: Energy of incident gamma-ray in keV
        theta: Angle of scatter in radians

    Returns:
        Energy Ee of scattered electron
    """
    Eg_scattered = Eg / (1 + Eg / 511 * (1 - np.cos(theta)))
    electron = Eg - Eg_scattered
    return np.where(Eg > 0.1, electron, Eg)

@njit
def dE_dtheta(Eg, theta):
    """
    Function to correct number of counts due to delta(theta)
    Adapted from MAMA in the file kelvin.f
    It is dE/dtheta of the E(theta) in Eq. (2) in Guttormsen 1996.

    Args:
        Eg: Energy of gamma-ray in keV
        theta: Angle of scatter in radians

    Returns:
        TYPE: dE_dtheta
    """
    a = (Eg * Eg / 511 * np.sin(theta))
    b = (1 + Eg / 511 * (1 - np.cos(theta)))**2
    return a / b

@njit
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

@njit
def index(E, e):
    i = 0
    while i < len(E):
        if E[i] > e:
            return i-1
        i += 1
    return i-1

def interpolate_compton(
        compton: ComptonNeighbours,
        E: float,
        fill_value: str = "extrapolate") -> np.ndarray:
    """Linear interpolation between the compton spectra

    Args:
        E (float): Incident energy
        compton (dict): Dict. with information about the compton spectra
            to interpolate between
        fill_value (str, optional): Fill value beyond boundaries

    Returns:
        f_cmp (nd.array): Interpolated values
    """
    #x = np.array([compton.elow, compton.ehigh])
    #y = np.vstack([compton.low, compton.high])
    x = compton.energies()
    y = compton.vstack()
    f_cmp = interp1d(x,
                        y,
                        kind="linear",
                        bounds_error=False,
                        fill_value=fill_value,
                        axis=0)
    return f_cmp(E)


@njit
def rebin_1D(counts, mids_in, mids_out):
    """Rebin an array of counts from binning mids_in to binning mids_out

    Assumes equidistant binning.

    Args:
        counts: Array of counts to be rebinned
        mids_in: Array of mid-bins energies giving
             the calibration of counts_in
        mids_out: Array of mid-bins energies of the
              counts array after rebin
    Returns:
        counts_out: Array of rebinned counts with calibration
             given by mids_out
    """

    # Get calibration coefficients and number of elements from array:
    Nin = mids_in.shape[0]
    Emin_in, dE_in = mids_in[0], mids_in[1]-mids_in[0]
    Nout = mids_out.shape[0]
    Emin_out, dE_out = mids_out[0], mids_out[1]-mids_out[0]

    # convert to lower-bin edges
    Emin_in -= dE_in/2
    Emin_out -= dE_out/2

    # Allocate rebinned array to fill:
    counts_out = np.zeros(Nout, dtype=DTYPE)
    counts_out_view = counts_out
    for i in range(Nout):
        # Only loop over the relevant subset of j indices where there may be
        # overlap:
        jmin = max(0, int((Emin_out + dE_out*(i-1) - Emin_in)/dE_in))
        jmax = min(Nin-1, int((Emin_out + dE_out*(i+1) - Emin_in)/dE_in))
        # Calculate the bin edge energies manually for speed:
        Eout_i = Emin_out + dE_out*i
        for j in range(jmin, jmax+1):
            # Calculate proportionality factor based on current overlap:
            Ein_j = Emin_in + dE_in*j
            bins_overlap = overlap(Ein_j, Ein_j+dE_in,
                                   Eout_i, Eout_i+dE_out)
            counts_out_view[i] += counts[j] * bins_overlap / dE_in

    return counts_out

@njit
def overlap(edge_in_l, edge_in_u,
           edge_out_l, edge_out_u):
    """ Calculate overlap between energy intervals

       1
    |_____|_____|_____| Binning A
    |___|___|___|___|__ Binning B
      2   3
    Overlap of bins A1 and B2 is 3_
    Overlap of bins A1 and B3 is 1.5_

    Args:
        edge_in_l: Lower edge of input interval
        edge_in_u: Upper edge of input interval
        edge_out_l: Lower edge of output interval
        edge_out_u: Upper edge of output interval
    Returns:
        overlap of the two bins
    """
    overlap = max(0,
                  min(edge_out_u, edge_in_u) -
                  max(edge_out_l, edge_in_l)
                  )
    return overlap


@njit
def fan_method_compton(E: float, compton: ComptonNeighbours, i_start: int,
                       i_stop: int, N_out, Eout) -> Tuple[np.ndarray, int]:
    """Fan method

    Args:
        E (float): Incident energy
        compton (dict): Dict. with information about the compton spectra
            to interpolate between
        i_start (int): Index where to start (usually end of backscatter)
        i_stop (int): Index where to stop (usually E+n*resolution). Note
            that it can be stopped earlier, which will be reported through
            `i_last`

    Note Erlend: This is incorrect. Interpolates only along a single theta-line,
    but as theta-lines fan out for higher Eg, must loop over more bins from
    the higher Eg spectrum and take a weighted sum.

    Returns:
        (np.ndarray, int): `R` is Response for `E`, and `i_last` last
            index of fan-method
    """
    R = np.zeros(N_out)

    Ece = E_compton(E, theta=np.pi)
    i_E_max = min(i_stop, N_out)
    i_ce_max = min(index(Eout, Ece), i_E_max)

    Esim_low = compton.elow
    Esim_high = compton.ehigh

    def lin_interpolation(x, x0, y0, x1, y1):
        return y0 + (y1 - y0) * (x - x0) / (x1 - x0)

    i_last = i_start  # Keep track of how far up the fan method goes
    for i in range(i_start, i_ce_max):
        # In Mama: E -> Egam, Ei -> E [Fabio]
        # Energy of current point in interpolated spectrum
        Ei = Eout[i]
        if Ei < 0.1 or Ei > Ece:
            continue
        z = div0_2(Ei, (E / 511 * (E - Ei)))
        theta = np.arccos(1 - z)
        if theta > 0 and theta < np.pi:
            # Determine interpolation indices in low and high arrays
            # by Compton formula
            Ecmp_ = E_compton(Esim_low, theta)
            i_low_interp = index(Eout, Ecmp_)
            Ecmp_ = E_compton(Esim_high, theta)
            i_high_interp = index(Eout, Ecmp_)

            c1 = compton.low[i_low_interp]
            c2 = compton.high[i_high_interp]

            # apply correction
            c1 *= dE_dtheta(Esim_low, theta)
            c2 *= dE_dtheta(Esim_high, theta)

            # essential equation c(E), which is below (2)
            interpol = lin_interpolation(E, Esim_low, c1, Esim_high, c2)
            R[i] = interpol / dE_dtheta(E, theta)
            i_last = i

    if len(R[R < 0]) != 0:
        #LOG.debug(f"In fan method, {len(R[R < 0])} entries in R"
        #          "are negative and now set to 0")
        R[R < 0] = 0

    return R, i_last


@njit
def fan_to_end(E: float, compton: ComptonNeighbours, i_start: int, i_stop: int,
               fwhm_abs_array: np.ndarray, N_out, Eout) -> np.ndarray:
    """Linear(!) fan interpolation from Compton edge to Emax

    The fan-part is "scaled" by the distance between the Compton edge and
    max(E). To get a reasonable scaling, we have to use ~6 sigma.

    Note:
        We extrapolate up to self.N_out, and not i_stop, as a workaround
        connected to Magne's 1/10th FWHM unfolding [which results
        in a very small i_stop.]

    Args:
        E (float): Incident energy
        compton (dict): Dict. with information about the compton spectra
            to interpolate between
        i_start (int): Index where to start (usually end of fan method)
        i_stop (int): Index where to stop (usually E+n*resolution)
        fwhm_abs_array (np.ndarray): FHWM array, absolute values

    Returns:
        np.ndarray: Response for `E`
    """

    R = np.zeros(N_out)

    Esim_low = compton.elow
    Esim_high = compton.ehigh
    Ecmp1 = E_compton(Esim_low, np.pi)
    Ecmp2 = E_compton(Esim_high, np.pi)
    i_low_edge = index(Eout, Ecmp1)
    i_high_edge = index(Eout, Ecmp2)

    oneSigma = fwhm_abs_array[index(Eout, Esim_low)] / 2.35
    Egmax1 = Esim_low + 6 * oneSigma
    scale1 = (Egmax1 - Ecmp1) / (Eout[i_stop] - Eout[i_start])

    oneSigma = fwhm_abs_array[index(Eout, Esim_high)] / 2.35
    Egmax2 = Esim_high + 6 * oneSigma
    scale2 = (Egmax2 - Ecmp2) / (Eout[i_stop] - Eout[i_start])

    def lin_interpolation(x, x0, y0, x1, y1):
        return y0 + (y1 - y0) * (x - x0) / (x1 - x0)

    i_edge = i_start - 1
    # for i in range(i_edge+1, i_stop):
    for i in range(i_edge + 1, N_out):
        i1 = int(i_low_edge + scale1 * (i - i_edge))
        i2 = int(i_high_edge + scale2 * (i - i_edge))

        if i1 >= len(compton.low):
            i1 = len(compton.low) - 1
        if i2 >= len(compton.high):
            i2 = len(compton.high) - 1

        c1 = compton.low[i1]
        c2 = compton.high[i2]
        y = lin_interpolation(E, Esim_low, c1, Esim_high, c2)
        R[i] = y

    if len(R[R < 0]) != 0:
        #LOG.debug(f"In linear fan method, {len(R[R < 0])} entries in R"
        #          "are negative and now set to 0")
        R[R < 0] = 0

    return R

@njit
def div0_1(a, b):
    """ division function designed to ignore / 0, i.e. div0([-1, 0, 1], 0 ) -> [0, 0, 0] """
    #with np.errstate(divide='ignore', invalid='ignore'):
    #c = np.true_divide(a, b)
    #c[~ np.isfinite(c)] = 0.0  # -inf inf NaN
    c = a/b
    for i in range(len(c)):
        if not np.isfinite(c[i]):
            c[i] = 0.0
    return c

@njit
def div0_2(a, b):
    """ division function designed to ignore / 0, i.e. div0([-1, 0, 1], 0 ) -> [0, 0, 0] """
    #with np.errstate(divide='ignore', invalid='ignore'):
    #c = np.true_divide(a, b)
    #c[~ np.isfinite(c)] = 0.0  # -inf inf NaN
    c = a/b
    if not np.isfinite(c):
        return 0.0
    return c


@njit
def linear_backscatter(Eout, N_out, E: float,
                       compton: ComptonNeighbours) -> Tuple[np.ndarray, int]:
    """Interpolate one-to-one up to the backscatter peak

    Args:
        E (float): Incident energy
        compton (dict): Dict. with information about the compton spectra
            to interpolate between

    Returns:
        (np.ndarray, int): `R` is Response for `E`, and
            `i_bc` is index of backscatter peak

    """
    Eedge = E_compton(E, theta=np.pi)  # compton-edge energy
    Ebsc = E - Eedge  # back-scattering energy
    i_bsc = index(Eout, Ebsc)
    R = np.zeros(N_out)

    fcmp = interpolate_compton_2(compton, E)
    R[:i_bsc + 1] = fcmp[:i_bsc + 1]
    R[R < 0] = 0
    return R, i_bsc


@njit
def get_closest_compton(E: float, Eg, compton, Eout, Ecmp) -> ComptonNeighbours:
    """Find and rebin closest energies from available response functions

    If `E < self.resp['Eg'].min()` the compton matrix will be replaced
    by an array of zeros.

    Args:
        E (float): Description

    Returns:
        Dict with entries `Elow` and `Ehigh`, and `ilow` and `ihigh`, the
        (indices) of closest energies. The arrays `counts_low` and
        `counts_high` are the corresponding arrays of `compton` spectra.
    """
    N = len(Eg)
    # ilow = 0
    # Find the compton spectra corresponding to the closest
    # energy below and above E. Rebin these to the requested energies,
    # but with the final compton energy.
    ihigh = np.searchsorted(Eg, E, side="right")
    if ihigh == N:  # E > self.resp['Eg'].max()
        ihigh -= 1
    ilow = ihigh - 1

    # Select the Compton spectra, called Fs1 and Fs2 in MAMA.
    Ehigh = Eg[ihigh]
    cmp_high = compton[ihigh, :]
    if ilow < 0:  # E < self.resp['Eg'].min()
        Elow = 0
        cmp_low = np.zeros_like(cmp_high)
    else:
        Elow = Eg[ilow]
        cmp_low = compton[ilow, :]

    Enew = np.arange(Eout[0], Ecmp[-1],
                        Eout[1] - Eout[0])
    cmp_low = rebin_1D(cmp_low, Ecmp, Enew)
    cmp_high = rebin_1D(cmp_high, Ecmp, Enew)

    compton = ComptonNeighbours(ilow, ihigh, Elow, Ehigh, cmp_low, cmp_high)

    return compton


@njit
def gauss_smoothing(array_in, E_array,
                    fwhm,
                    truncate=3):
    """
    Function which smooths an array of counts by a Gaussian
    of full-width-half-maximum FWHM. Preserves number of counts.
    Args:
        array_in (array, double): Array of inbound counts to be smoothed
        E_array (array, number): Array with energy calibration of array_in, in
                                 mid-bin calibration
        fwhm (array, double): The full-width-half-maximums. Need to be
                              same size as array_in
        truncate (double, optional): The window width of the Gaussian that is
                                     used to smoothe, in units of sigma.
                                     Defaults to 3.

    Returns:
        array_out: Array of smoothed counts

    """
    if not len(array_in) == len(E_array):
        raise ValueError("Length mismatch between array_in and E_array")
    if not len(array_in) == len(fwhm):
        raise ValueError("Length mismatch between array_in and fwhm")

    array_in_view = array_in

    a0 = E_array[0]
    a1 = E_array[1] - E_array[0]

    array_out = np.zeros(len(array_in), dtype=DTYPE)
    # cdef double[:] array_out_view = array_out

    def find_truncation_indices(E_centroid_current,
                                sigma_current,
                                truncate=truncate):
        E_cut_low = E_centroid_current - truncate * sigma_current
        i_cut_low = int((E_cut_low - a0) / a1)
        i_cut_low = max(0, i_cut_low)
        E_cut_high = E_centroid_current + truncate * sigma_current
        i_cut_high = int((E_cut_high - a0) / a1)
        i_cut_high = max(min(len(array_in), i_cut_high), i_cut_low+1)
        return i_cut_low, i_cut_high

    for i in range(len(array_out)):
        counts = array_in_view[i]
        if counts > 0:
            E_centroid_current = E_array[i]
            sigma_current = fwhm[i]/2.355
            i_cut_low, i_cut_high = find_truncation_indices(E_centroid_current,
                                                            sigma_current)
            pdf = np.zeros(len(array_in), dtype=DTYPE)
            # if using lower bin instead of center bin in both E_mid and mu
            # below-> canceles out
            pdf[i_cut_low:i_cut_high+1] =\
                gaussian(E_array[i_cut_low:i_cut_high+1],
                         mu=E_array[i],
                         sigma=sigma_current
                         )
            pdf = pdf / np.sum(pdf)
            array_out += counts * pdf

    return array_out


@njit
def gaussian(Emids, mu, sigma):
    """
    Returns a normalized Gaussian supported on Emids.

    NB! All arguments (Emids, mu and sigma) must have the
    same units. In OMpy the default unit is keV.

    Args:
        Emids (array, number): Array of energies to evaluate
                               (center bin calibration)
        mu (number): Centroid
        sigma (double): Standard deviation
    Returns:
        gaussian_array (array, double): Array of gaussian
        distribution values matching Emids.
    """
    gaussian_array = np.zeros(len(Emids), dtype=DTYPE)
    gaussian_array_view = gaussian_array

    eps = 1e-6  # Avoid zero division
    sigma += eps

    prefactor = (1/(sigma*np.sqrt(2*np.pi)))
    for i in range(len(Emids)):
        gaussian_array_view[i] = (prefactor
                                  * np.exp(
                                    -(Emids[i]-mu)
                                    * (Emids[i]-mu)/(2*sigma*sigma))
                                  )

    return gaussian_array


@njit
def lerp(x, x0, x1, y0, y1):
    t = (x - x0) / (x1 - x0)
    return (1 - t)*y0 + t*y1


@njit
def interpolate_compton_2(compton: ComptonNeighbours, E: float):
    low = compton.low
    high = compton.high
    intp = np.zeros_like(compton.low)

    t = (E - compton.elow) / (compton.ehigh - compton.elow)
    for i in range(len(intp)):
        intp[i] = (1.0 - t)*low[i] + t*high[i]
    return intp

@njit
def normalize(R):
    for j in range(R.shape[0]):
        R[j, :] = div0_1(R[j, :], np.sum(R[j, :]))
