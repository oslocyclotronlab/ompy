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
import logging
import warnings
from typing import Iterable, Optional
from scipy.ndimage import gaussian_filter1d
from copy import copy
from .library import div0
from .matrix import Matrix
from .matrixstate import MatrixState
from .setable import Setable


LOG = logging.getLogger(__name__)
logging.captureWarnings(True)


class Unfolder:
    """Performs Guttormsen unfolding

    The algorithm is roughly as follows:

    Define matrix Rij as the response in channel i when the detector
    is hit by γ-rays with energy corresponding to channel j. Each response
    function is normalized by Σi Rij = 1. The folding is then
                         f = Ru
    where f is the folded spectrum and u the unfolded. If we obtain better and
    better trial spectra u, we can fold and compare them to the observed
    spectrum r.

    As an initial guess, the trial function is u⁰ = r, and the subsequent being
                      uⁱ = uⁱ⁻¹ + (r - fⁱ⁻¹)
    until fⁱ≈r.

    Note that no actions are performed on the matrices; they must already
    be cut into the desired size.

    Attributes:
        raw (Matrix): The Matrix to unfold
        num_iter (int): The number of iterations to perform
        zeroes (boolean ndarray): Masks everything below the diagonal to false
        r (Matrix): The trapezoidal cut raw Matrix
        R (Matrix): The response matrix
        weight_fluctuation (float):
        minimum_iterations (int):

    TODO: There is the possibility for wrong results if the response
       and the matrix are not rebinned the same. Another question that
       arises is *when* are two binnings equal? What is equal enough?
    """
    def __init__(self, num_iter: int = 33, response: Optional[Matrix] = None):
        """Unfolds the gamma-detector response of a spectrum

        Args:
            num_iter: The number of iterations to perform.j
            reponse: The response Matrix R to use in unfolding.
        TODO:
            - Fix the compton subtraction method implementation.
        """
        self.num_iter: int = 33
        self.weight_fluctuation = 0.2
        self.minimum_iterations = 3
        self.zeroes: Optional[np.ndarray] = None
        self._param = 5
        self._R: Optional[Matrix] = response

    def __call__(self, matrix: Matrix) -> Matrix:
        """ Wrapper for self.apply() """
        return self.apply(matrix)

    def update_values(self):
        """Verify internal consistency and set default values

        Raises:
            AssertionError: If the raw matrix and response matrix
                have different calibration coefficients a10 and a11.
            AttributeError: If no diagonal cut has been made.
        """
        # Ensure that the given matrix is in fact raw
        if self.raw.state != MatrixState.RAW:
            warnings.warn("Trying to unfold matrix that is not raw")

        assert self.R is not None, "Response R must be set"

        LOG.debug("Comparing calibration of raw against response:"
                  f"\n{self.raw.calibration()}"
                  f"\n{self.R.calibration()}")
        calibration_diff = self.raw.calibration_array() -\
            self.R.calibration_array()
        eps = 1e-3
        if not (np.abs(calibration_diff[2:]) < eps).all():
            raise AssertionError(("Calibration mismatch: "
                                  f"{calibration_diff}"
                                  "\nEnsure that the raw matrix and"
                                  " calibration matrix are cut equally."))

        # TODO: Warn if the Matrix is not diagonal
        # raise AttributeError("Call cut_diagonal() before unfolding")

        self.r = self.raw.values
        # TODO Use the arbitrary diagonal mask instead
        self.zeroes = self.raw.diagonal_mask()

    def apply(self, raw: Matrix,
              response: Optional[Matrix] = None) -> Matrix:
        """Run unfolding

        TODO: Use better criteria for terminating
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
            chisquare[i, :] = self.chi_square(folded)
            fluctuations[i, :] = self.fluctuations(unfolded)

            if LOG.level >= logging.DEBUG:
                chisq = np.mean(chisquare[i, :])
                LOG.debug(f"Iteration {i}: Avg χ²/ν {chisq}")

        # Score the solutions based on χ² value for each Ex bin
        # and select the best one.
        fluctuations /= self.fluctuations(self.r)
        iscores = self.score(chisquare, fluctuations)
        unfolded = np.zeros_like(self.r)
        for iEx in range(self.r.shape[0]):
            unfolded[iEx, :] = unfolded_cube[iscores[iEx], iEx, :]

        unfolded = Matrix(unfolded, Eg=self.raw.Eg, Ex=self.raw.Ex)
        unfolded.state = "unfolded"

        # These two lines feel out of place
        # TODO: What they do and where they should be run is very unclear.
        #     Fix later.
        unfolded.fill_negative(window_size=10)
        unfolded.remove_negative()
        return unfolded

    def step(self, unfolded, folded, step):
        """Perform a single step of Guttormsen unfolding

        Performs the steps
            u = u + (r - f)
            f = uR
            set everything below the diagonal of f to 0

        """
        if step > 0:
            unfolded = unfolded + (self.r - folded)
        folded = unfolded@self.R.values

        # Suppress everything below the diagonal
        # *Why* is this necessary? Where does the off-diagonals come from?
        folded[self.zeroes] = 0.0

        return unfolded, folded

    def chi_square(self, folded: np.ndarray) -> np.ndarray:
        """ Compute Χ² of the folded spectrum

        Uses the familiar Χ² = Σᵢ (fᵢ-rᵢ)²/rᵢ
        """
        return div0(np.power(folded - self.r, 2),
                    np.where(self.r > 0, self.r, 0)).sum(axis=1)

    def fluctuations(self, counts: np.ndarray,
                     sigma: float = 0.12) -> np.ndarray:
        """
        Calculates fluctuations in each Ex bin gamma spectrum by summing
        the absolute diff between the spectrum and a smoothed version of it.

        Returns a column vector of fluctuations in each Ex bin
        """

        a1 = self.raw.Eg[1] - self.raw.Eg[0]
        counts_matrix_smoothed = gaussian_filter1d(
            counts, sigma=sigma * a1, axis=1)
        fluctuations = np.sum(
            np.abs(counts_matrix_smoothed - counts), axis=1)

        return fluctuations

    def score(self, chisquare: np.ndarray,
              fluctuations: np.ndarray) -> np.ndarray:
        """
        Calculates the score of each unfolding iteration for each Ex
        bin based on a weighting of chisquare and fluctuations.
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

    @property
    def R(self) -> Matrix:
        return self._R

    @R.setter
    def R(self, response: Matrix) -> None:
        # TODO Make setable
        self._R = response
