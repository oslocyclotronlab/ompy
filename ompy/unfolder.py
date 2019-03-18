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
from typing import Iterable
from .library import div0
from .matrix import Matrix, MatrixState
from scipy.ndimage import gaussian_filter1d


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

    Attributes:
        raw Matrix: The Matrix to unfold
        num_iter int: The number of iterations to perform
        Ex_min float: The lower Ex limit on the energy trapezoidal cut
        Ex_max float: The upper Ex limit on the energy trapezoidal cut
        Eg_min float: The lower Eg limit on the energy trapezoidal cut
        Eg_max float: The upper Eg limit on the energy trapezoidal cut
        mask boolean ndarray: Masks everything below the diagonal to false
        r Matrix: The trapezoidal cut raw Matrix
        R Matrix: The response matrix
        weight_fluctuation float:
        minimum_iterations int:

    """
    def __init__(self, raw: Matrix):
        """Unfolds the gamma-detector response of a spectrum

        Args:
            raw: the raw matrix to unfold, an instance of Matrix()
        Returns:
            unfolded -- the unfolded matrix as an instance of Matrix()

        TODO:
            - Fix the compton subtraction method implementation.
        """
        self.raw: Matrix = raw
        self.num_iter = 33
        self.Ex_min = None
        self.Ex_max = None
        self.Eg_min = None
        self.Eg_max = None

        # Ensure that the given matrix is in fact raw
        if self.raw.state != MatrixState.RAW:
            warnings.warn("Trying to unfold matrix that is not raw")

    def update_values(self):
        """Verify internal consistency and set default values

        """
        eps = 1e-3
        LOG.debug("Comparing calibration of raw against response:"
                  f"\n{self.raw.calibration()}"
                  f"\n{self.R_calibration_array}")
        calibration_diff = self.raw.calibration_array() -\
            self.R_calibration_array
        if not (np.abs(calibration_diff[2:]) < eps).all():
            raise AssertionError(("Calibration mismatch: "
                                  f"{calibration_diff}"
                                  "Ensure that the raw matrix and"
                                  " calibration matrix are cut equally."))

        # If energy limits are not provided, use extremal array values:
        if self.Ex_min is None:
            self.Ex_min = self.raw.Ex[0]
        if self.Ex_max is None:
            self.Ex_max = self.raw.Ex[-1]
        if self.Eg_min is None:
            self.Eg_min = self.raw.Eg[0]
        if self.Eg_max is None:
            self.Eg_max = self.raw.Eg[-1]
        if self.mask is None:
            raise ValueError("Make the mask before unfolding")

        LOG.debug(("Using energy values:"
                   f"\nEg_min: {self.Eg_min}"
                   f"\nEg_max: {self.Eg_max}"
                   f"\nEx_min: {self.Ex_min}"
                   f"\nEx_max: {self.Ex_max}"))

        # Set limits for excitation and gamma energy bins to
        # be considered for unfolding.
        # Use index 0 of array as lower limit instead of energy because
        # it can be negative!
        iEx_min, iEx_max = 0, self.raw.index_Ex(self.Ex_max)
        iEg_min, iEg_max = 0, self.raw.index_Eg(self.Eg_max)

        # Create slices and cut the matrices to appropriate shape
        Egslice = slice(iEg_min, iEg_max)
        Exslice = slice(iEx_min, iEx_max)

        self.r = self.raw.values[Exslice, Egslice]
        self.mask = self.mask[Exslice, Egslice]
        self.R = self.R[Egslice, Egslice]
        self.Eg = self.raw.Eg[Egslice]
        self.Ex = self.raw.Ex[Exslice]

        LOG.debug(f"Exslice: {Exslice}\nEgslice: {Egslice}")
        LOG.debug((f"Cutting matrix from {self.raw.shape}"
                   f" to {self.r.shape}"))

    def load_response(self, filename: str):
        response = Matrix(filename=filename)
        self.R = response.values
        self.R_calibration_array = response.calibration_array()

        self.weight_fluctuation = 0.2
        self.minimum_iterations = 3

    def make_mask(self, E1: Iterable[float], E2: Iterable[float]):
        self.mask = self.raw.line_mask(E1, E2)

    def unfold(self) -> np.ndarray:
        """Run unfolding

        TODO: Use better criteria for terminating
        """

        # Set up the arrays
        self.update_values()
        unfolded_cube = np.zeros((self.num_iter, *self.r.shape))
        chisquare = np.zeros((self.num_iter, self.r.shape[0]))
        fluctuations = np.zeros((self.num_iter, self.r.shape[0]))
        folded = np.zeros_like(self.r)

        # Use u⁰ = r as initial guess
        unfolded = self.r
        for i in range(self.num_iter):
            unfolded, folded = self.unfold_step(unfolded, folded, i)
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

        unfolded[self.mask] = 0
        unfolded = Matrix(unfolded, Eg=self.Eg, Ex=self.Ex)
        unfolded.state = MatrixState.UNFOLDED
        return unfolded

    def unfold_step(self, unfolded, folded, step):
        """Perform a single step of Guttormsen unfolding

        Performs the steps
            u = u + (r - f)
            f = uR
            set everything below the diagonal of f to 0

        """
        if step > 0:
            unfolded = unfolded + (self.r - folded)
        folded = unfolded@self.R  # Should be equal to (R.T@unfolded.T).T

        # Suppress everything below the diagonal
        folded[self.mask] = 0.0

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

        a1 = self.Eg[1] - self.Eg[0]
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
