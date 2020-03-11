# -*- coding: utf-8 -*-
"""
Implementation of the unfolding method
(Guttormsen et al., Nuclear Instruments and Methods in Physics Research A
374 (1996))
---

This file is part of oslo_method_python, a python implementation of the
Oslo method.

Copyright (C) 2018 Jørgen Eriksson Midtbø
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
import pandas
import logging
import warnings
from typing import Optional, Tuple, Callable
import termtables as tt
from scipy.ndimage import gaussian_filter1d
from copy import copy
from .gauss_smoothing import gauss_smoothing_matrix_1D
from .library import div0, i_from_E
from .matrix import Matrix
from .matrixstate import MatrixState
from .rebin import rebin_1D


LOG = logging.getLogger(__name__)
logging.captureWarnings(True)


class Unfolder:
    """Performs Guttormsen unfolding

    The algorithm is roughly as follows:

    Define matrix Rij as the response in channel i when the detector
    is hit by γ-rays with energy corresponding to channel j. Each response
    function is normalized by Σi Rij = 1. The folding is then::

        f = Ru

    where f is the folded spectrum and u the unfolded. If we obtain better and
    better trial spectra u, we can fold and compare them to the observed
    spectrum r.

    As an initial guess, the trial function is u⁰ = r, and the subsequent
    being::
        uⁱ = uⁱ⁻¹ + (r - fⁱ⁻¹)

    until fⁱ≈r.

    Note that no actions are performed on the matrices; they must already
    be cut into the desired size.

    Attributes:
        num_iter (int, optional): The number of iterations to perform. defaults
             to 33. The best iteration is then selected based on the `score`
             method
        R (Matrix): The response matrix
        weight_fluctuation (float, optional): A attempt to penalize
             fluctuations. Defaults to 0.2
        minimum_iterations (int, optional): Minimum number of iterations.
            Defaults to 3.
        window_size (int or np.ndarray): window_size for  (fill and) remove
            negatives on output. Defaults to 20.
        use_compton_subtraction (bool, optional): Set usage of Compton
            subtraction method. Defaults to `True`.
        response_tab (DataFrame, optional): If `use_compton_subtraction=True`
            a table with information ('E', 'eff_tot', ...) must be provided.
        FWHM_tweak_multiplier (Dict, optional): See compton subtraction method
            Necessary keys: `["fe", "se", "de", "511"]`. Magne's suggestion::
                FWHM_tweak_multiplier = {"fe": 1., "se": 1.1,
                                         "de": 1.3, "511": 0.9}
        iscores (np.ndarray, optional): Selected iteration number in the
            unfolding. The unfolding and selection is performed independently
            for each `Ex` row, thus this is an array with `len(raw.Ex)`.
    TODO:
        - Unfolding for a single spectrum (currently needs to be mocked as a
            Matrix).
    """
    def __init__(self, num_iter: int = 33, response: Optional[Matrix] = None):
        """Unfolds the gamma-detector response of a spectrum

        Args:
            num_iter: see above
            reponse: see above
        """
        self.num_iter = num_iter
        self.weight_fluctuation: float = 0.2
        self.minimum_iterations: int = 3
        self.window_size = 20

        self._R: Optional[Matrix] = response
        self.raw: Optional[Matrix] = None
        self.r: Optional[np.ndarray] = None

        self.use_compton_subtraction: bool = True
        self.response_tab: Optional[pandas.DataFrame] = None
        self.FWHM_tweak_multiplier: Optional[dict[str, float]] = None

        self.iscores: Optional[np.ndarray] = None

    def __call__(self, matrix: Matrix) -> Matrix:
        """ Wrapper for `self.apply()` """
        return self.apply(matrix)

    def update_values(self):
        """Verify internal consistency and set default values

        Raises:
            ValueError: If the raw matrix and response matrix have different
                calibrations.
        """
        # Ensure that the given matrix is in fact raw
        if self.raw.state != MatrixState.RAW:
            warnings.warn("Trying to unfold matrix that is not raw")

        assert self.R is not None, "Response R must be set"
        assert self.R.shape[0] == self.R.shape[1],\
            "Response R must be a square matrix"

        LOG.debug("Comparing calibration of raw against response")
        if len(self.raw.Eg) != len(self.R.Eg):
            raise ValueError("Must have equal number of energy bins.")
        if not np.allclose(self.raw.Eg, self.R.Eg):
            raise ValueError("Must have equal energy binning.")

        LOG.debug("Check for negative counts.")
        if np.any(self.raw.values < 0) or np.any(self.R.values < 0):
            raise ValueError("Raw and response cannot have negative counts."
                             "Consider using fill_negatives and "
                             "remove_negatives on the input matixes.")

        self.r = self.raw.values

    def apply(self, raw: Matrix,
              response: Optional[Matrix] = None) -> Matrix:
        """ Run unfolding.

        Selected iteration for each `Ex` row can be obtained via
        `self.iscores`, see above.

        Args:
            raw (Matrix): Raw matrix to unfold
            response (Matrix, optional): Response matrix. Defaults to
                the matrix provided in initialization.

        Returns:
            Matrix: Unfolded matrix

        TODO:
            - Use better criteria for terminating
        """
        if response is not None:
            self.R = response
        self.raw = copy(raw)
        # Set up the arrays
        self.update_values()
        unfolded_cube = np.zeros((self.num_iter, *self.r.shape))
        chisquare = np.zeros((self.num_iter, self.r.shape[0]))
        fluctuations = np.zeros((self.num_iter, self.r.shape[0]))
        folded = np.zeros_like(self.r)

        # Use u⁰ = r as initial guess
        unfolded = self.r
        for i in range(self.num_iter):
            unfolded, folded = self.step(unfolded, folded, i)
            unfolded_cube[i, :, :] = unfolded
            chisquare[i, :] = self.chi_square_red(folded)
            fluctuations[i, :] = self.fluctuations(unfolded)

            if LOG.level >= logging.DEBUG:
                chisq = np.mean(chisquare[i, :])
                LOG.debug(f"Iteration {i}: Avg χ²/ν {chisq}")

        # Score the solutions based on χ² value for each Ex bin
        # and select the best one.
        fluctuations /= self.fluctuations(self.r)
        iscores = self.score(chisquare, fluctuations)
        self.iscores = iscores  # keep if interesting for later
        unfolded = np.zeros_like(self.r)
        for iEx in range(self.r.shape[0]):
            unfolded[iEx, :] = unfolded_cube[iscores[iEx], iEx, :]
        if LOG.level >= logging.DEBUG:
            print_array = np.column_stack((np.arange(len(self.raw.Ex)),
                                           self.raw.Ex.astype(int),
                                           iscores))
            LOG.debug("Selecting following iterations: \n%s",
                      tt.to_string(print_array,
                                   header=('i', 'Ex', 'iteration'))
                      )

        if self.use_compton_subtraction:
            unfolded = self.compton_subtraction(unfolded)

        unfolded = Matrix(unfolded, Eg=self.raw.Eg, Ex=self.raw.Ex)
        unfolded.state = "unfolded"

        self.remove_negative(unfolded)
        return unfolded

    def step(self, unfolded: np.ndarray, folded: np.ndarray,
             step: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Perform a single step of Guttormsen unfolding

        .. parsed-literal::
            Performs the steps
                u = u + (r - f),      for step > 0
                f = uR

        Args:
            unfolded (np.ndarray): unfolded spectrum of iteration `step-1`
            folded (np.ndarray): folded spectrum of of iteration `step-1`
            step (np.ndarray): Iteration number

        Returns:
            Arrays with counts for the unfolded and folded matrix
            (unfolded, folded)

        """
        if step > 0:  # since the initial guess is the raw spectrum
            unfolded = unfolded + (self.r - folded)

        folded = unfolded@self.R.values

        return unfolded, folded

    def chi_square_red(self, folded: np.ndarray) -> np.ndarray:
        """ Compute reduced Χ² of the folded spectrum

        Uses the familiar Χ² = Σᵢ (fᵢ-rᵢ)²/rᵢ
        """
        chi2 = div0(np.power(folded - self.r, 2),
                    np.where(self.r > 0, self.r, 0)).sum(axis=1)
        return chi2/self.r.shape[1]

    def fluctuations(self, counts: np.ndarray,
                     sigma: float = 20) -> np.ndarray:
        """
        Calculates fluctuations in each Ex bin gamma spectrum by summing
        the absolute diff between the spectrum and a smoothed version of it.

        Args:
            counts (np.ndarray): counts of spectrum to act on
                (calibration from `self.raw`).
            sigma (float, optional): width of the gaussian for the smoothing.
                Defaults to 20

        Returns:
            np.ndarray: column vector of fluctuations in each Ex bin
        """

        a1 = self.raw.Eg[1] - self.raw.Eg[0]
        counts_matrix_smoothed = gaussian_filter1d(
            counts, sigma=sigma / a1, axis=1)
        fluctuations = np.sum(
            np.abs(counts_matrix_smoothed - counts), axis=1)

        return fluctuations

    def score(self, chisquare: np.ndarray,
              fluctuations: np.ndarray) -> np.ndarray:
        """
        Calculates the score of each unfolding iteration for each Ex
        bin based on a weighting of chisquare and fluctuations.

        Args:
            chisquare (np.ndarray): chisquare between raw and unfolded
            fluctuations (np.ndarray): measure of fluctuations
                (to be penalized)

        Returns:
            score (the lower the better)
        """
        # Check that it's consistent with chosen max number of iterations:
        if self.minimum_iterations > self.num_iter:
            self.minimum_iterations = self.num_iter

        score_matrix = ((1 - self.weight_fluctuation) * chisquare +
                        self.weight_fluctuation * fluctuations)
        # Get index of best (lowest) score for each Ex bin:
        best_iteration = np.argmin(score_matrix, axis=0)
        # Enforce minimum_iterations:
        best_iteration = np.where(
            self.minimum_iterations > best_iteration,
            self.minimum_iterations * np.ones(len(best_iteration), dtype=int),
            best_iteration)
        return best_iteration

    def fold(self, matrix: Matrix) -> Matrix:
        """ Folds a matrix with the loaded response

        Args:
            matrix (Matrix): Input matrix

        Returns:
            folded (Matrix): Folded matrix
        """
        assert self.R is not None, "Need to load response first"
        folded = matrix.copy()
        folded.values = matrix.values@self.R.values
        return folded

    @property
    def R(self) -> Matrix:
        return self._R

    @R.setter
    def R(self, response: Matrix) -> None:
        # TODO Make setable
        self._R = response

    def compton_subtraction(self, unfolded: np.ndarray) -> np.ndarray:
        """ Compton Subtraction Method in Unfolding of Guttormsen et al (NIM 1996)

        Args:
            unfolded (ndarray): unfolded spectrum

        Returns:
            unfolded spectrum, with compton subtraction applied

        We follow the notation of Guttormsen et al (NIM 1996) in what follows.
        `u0` is the unfolded spectrum from above, `r` is the raw spectrum,
        .. parsed-literal::
            w = us + ud + ua
            is the folding contributions from everything except Compton, i.e.
            us = single escape,
            ua = double escape,
            ua = annihilation (511).

        `v = pf*u0 + w == uf + w` is the estimated "raw minus Compton" spectrum `c` is the estimated Compton spectrum.

        Note:
            - The tweaking of the FWHM ("facFWHM" in Mama) has been delegated
              to the creation of the response matrix. If one wants to unfold
              with, say, 1/10 of the "real" FWHM, this this should be provided
              as input here already.
            - We apply smoothing to the different peak structures as described
              in the article. However, you may also "tweak" the FWHMs per peak
              for something Magne thinks is a better result.

        TODO:
            - When we unfolding with a reduced FWHM, should the compton method
              still work with the actual fhwm?
        """
        LOG.debug("Applying Compton subtraction method")

        if self.response_tab is None:
            raise ValueError("`response_tab` needs to be set for this method")
        tab = self.response_tab

        assert (tab.E == self.R.Eg).all(), \
            "Energies of response table have to match the Eg's"\
            "of the response matrix."

        FWHM = tab.fwhm_abs.values
        eff = tab.eff_tot.values
        pf = tab.pFE.values
        ps = tab.pSE.values
        pd = tab.pDE.values
        pa = tab.p511.values

        keys_needed = ["fe", "se", "de", "511"]
        if self.FWHM_tweak_multiplier is None:
            FWHM_tweak = dict()
            FWHM_tweak["fe"] = 1
            FWHM_tweak["se"] = 1
            FWHM_tweak["de"] = 1
            FWHM_tweak["511"] = 1
        else:
            if all(key in self.FWHM_tweak_multiplier for key in keys_needed):
                FWHM_tweak = self.FWHM_tweak_multiplier
            else:
                raise ValueError("FWHM_tweak_multiplier needs to contain each"
                                 "of this keys: {}".format(keys_needed))
        r = self.raw.values
        u0 = unfolded
        Eg = tab.E.values

        # Full-energy, smoothing but no shift:
        uf = pf * u0
        uf = gauss_smoothing_matrix_1D(uf, Eg, 0.5*FWHM*FWHM_tweak["fe"])

        # Single escape, smoothing and shift:
        us = ps * u0
        us = gauss_smoothing_matrix_1D(us, Eg, 0.5*FWHM*FWHM_tweak["se"])
        us = shift_matrix(us, Eg, energy_shift=-511)

        # Double escape, smoothing and shift:
        ud = pd * u0
        ud = gauss_smoothing_matrix_1D(ud, Eg, 0.5*FWHM*FWHM_tweak["de"])
        ud = shift_matrix(ud, Eg, energy_shift=-1024)

        # 511, smoothing, but no shift:
        ua = np.zeros(u0.shape)
        i511 = i_from_E(511, Eg)
        ua[:, i511] = np.sum(pa * u0, axis=1)
        ua = gauss_smoothing_matrix_1D(ua, Eg, 1.0*FWHM*FWHM_tweak["511"])

        # Put it all together:
        w = us + ud + ua
        v = uf + w
        c = r - v

        # Smoothe the Compton part, which is the main trick:
        c = gauss_smoothing_matrix_1D(c, Eg, 1.0*FWHM)

        # Channel 0 is missing from resp.dat
        # Add Ex channel to array, also correcting for efficiency.
        # u = div0((r - c - w), np.append(0, pf))
        u = div0((r - c - w), pf)
        unfolded = div0(u, eff)

        return unfolded

    def remove_negative(self, matrix: Matrix):
        """ (Fill and) remove negative counts

        Wrapper for Matrix.fill_and_remove_negative()

        Args:
            matrix: Input matrix
        """
        matrix.fill_and_remove_negative(window_size=self.window_size)


def shift(counts_in, E_array_in, energy_shift):
    """
    Shift the counts_in array by amount energy_shift.

    The function is actually a wrapper for the rebin() function that
    "fakes" the input energy calibration to give a shift. It is similar to
    the rebin_and_shift() function defined above, but even simpler.

    Args:
        counts_in (numpy array, float): Array of counts
        E_array_in (numpy array, float): Energies of input counts
        energy_shift (float): Amount to shift the counts by. Negative means
                              shift to lower energies. Default is 0.
    """
    E_array_in_shifted = E_array_in + energy_shift
    counts_out = rebin_1D(counts_in, E_array_in_shifted, E_array_in)
    return counts_out


def shift_matrix(counts_in_matrix, E_array_in, energy_shift):
    """
    Function which takes a matrix of counts and shifts it
    along axis 1.
    """
    counts_out_matrix = np.zeros(counts_in_matrix.shape)
    for i in range(counts_in_matrix.shape[0]):
        counts_out_matrix[i, :] = shift(counts_in_matrix[i, :], E_array_in,
                                        energy_shift=energy_shift)
    return counts_out_matrix


