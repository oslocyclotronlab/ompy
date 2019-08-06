# -*- coding: utf-8 -*-
"""
code for error propagation in oslo_method_python
It handles generation of random ensembles of spectra with
statistical perturbations, and can make first generation variance matrices.

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
import os
import numpy as np
import logging
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Callable, Union, List, Optional
from .matrix import Matrix
from .unfolder import Unfolder

LOG = logging.getLogger(__name__)
logging.captureWarnings(True)


class Ensemble:
    """Generates perturbated matrices to estimate uncertainty

    Attributes:
        raw (Matrix): The raw matrix used as model for the ensemble.
        save_path (str): The path used for saving the generated matrices.
        unfolder (Unfolder): An instance of Unfolder which is used in
            the unfolding of the generated matrices. Only has to support
            obj.unfold(raw: Matrix).
        first_generation_method (FirstGeneration): An instance of
            FirstGeneration for applying the first generation method
            to unfolded matrices. Only has to be callable
            with (unfolded: Matrix)
        regenerate (Bool): Whether to regenerate the matrices.
        std_raw (Matrix): The computed standard deviation of the raw ensemble.
        std_unfolded (Matrix): The computed standard deviation of the unfolded
            ensemble.
        std_firstgen (Matrix): The computed standard deviation of the
            first generation method matrices.
        firstgen_ensemble (np.ndarray): The entire firstgen ensemble.

    TODO: Separate each step - generation, unfolded, firstgening?
    """
    def __init__(self, raw: Optional[Matrix] = None, save_path: Union[str, Path] = "ensemble"):
        self.raw: Optional[Matrix] = raw
        self.save_path: Path = Path(save_path)
        self.unfolder: Optional[Callable[[Matrix], Matrix]] = None
        self.first_generation_method: Optional[Callable[[Matrix], Matrix]] = None
        self.size = 0

        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)

        self.std_raw: Optional[Matrix] = None
        self.std_unfolded: Optional[Matrix] = None
        self.std_firstgen: Optional[Matrix] = None

        self.raw_ensemble: Optional[Matrix] = None
        self.unfolded_ensemble: Optional[Matrix] = None
        self.firstgen_ensemble: Optional[Matrix] = None

    def load(self, path: Optional[Union[str, Path]] = None):
        if path is not None:
            path = Path(path)
        else:
            path = self.save_path

        self.raw = Matrix(path=path / 'raw.npy')
        self.firstgen = Matrix(path=path / 'firstgen.npy')
        raws = list(path.glob("raw_[0-9]*.*"))
        unfolds = list(path.glob("unfolded_[0-9]*.*"))
        firsts = list(path.glob("firstgen_[0-9]*.*"))
        assert len(raws) == len(unfolds) == len(firsts)
        assert len(raws) > 0
        self.size = len(raws)

        self.raw_ensemble = np.empty((self.size, *self.raw.shape))
        self.unfolded_ensemble = np.empty_like(self.raw_ensemble)
        self.firstgen_ensemble = np.empty_like(self.raw_ensemble)

        for i, raw in enumerate(raws):
            self.raw_ensemble[i, :, :] = Matrix(path=raw).values

        for i, unfolded in enumerate(unfolds):
            self.unfolded_ensemble[i, :, :] = Matrix(path=unfolded).values

        for i, firstgen in enumerate(firsts):
            self.firstgen_ensemble[i, :, :] = Matrix(path=firstgen).values

        self.std_raw = Matrix(path=path / 'raw_std.npy')
        self.std_unfolded = Matrix(path=path / 'unfolded_std.npy')
        self.std_firstgen = Matrix(path=path / 'firstgen_std.npy')

    def generate(self, number: int, method: str = 'poisson',
                 regenerate: bool = False) -> None:
        """Generates an ensemble of matrices and estimates standard deviation

        Perturbs the initial raw matrix using either a Gaussian or Poisson
        process, unfolds them and applies the first generation method to them.
        Uses the variation to estimate standard deviation of each step.

        Args:
            number: The number of perturbed matrices to generate.
            method: The stochastic method to use to generate the perturbations
                Can be 'gaussian' or 'poisson'.
            regenerate: Whether to use already generated files (False) or
                generate them all anew (True).
        """
        assert self.raw is not None, "Set the raw matrix"

        self.size = number
        self.regenerate = regenerate
        raw_ensemble = np.zeros((number, *self.raw.shape))
        unfolded_ensemble = np.zeros_like(raw_ensemble)
        firstgen_ensemble = np.zeros_like(raw_ensemble)

        for step in range(number):
            LOG.info(f"Generating {step}")
            raw = self.generate_raw(step, method)
            unfolded = self.unfold(step, raw)
            firstgen = self.first_generation(step, unfolded)

            raw_ensemble[step, :, :] = raw.values
            unfolded_ensemble[step, :, :] = unfolded.values

            # TODO The first generation method might reshape the matrix
            if firstgen_ensemble.shape[1:] != firstgen.shape:
                firstgen_ensemble = np.zeros((number, *firstgen.shape))
            self.firstgen = firstgen
            firstgen_ensemble[step, :, :] = firstgen.values

        self.raw.save(self.save_path / 'raw.npy')
        self.firstgen.save(self.save_path / 'firstgen.npy')
        # Calculate standard deviation
        raw_ensemble_std = np.std(raw_ensemble, axis=0)
        raw_std = Matrix(raw_ensemble_std, self.raw.Eg, self.raw.Ex,
                         state='std')
        raw_std.save(self.save_path / "raw_std.npy")

        unfolded_ensemble_std = np.std(unfolded_ensemble, axis=0)
        unfolded_std = Matrix(unfolded_ensemble_std, self.raw.Eg,
                              self.raw.Ex, state='std')
        unfolded_std.save(self.save_path / "unfolded_std.npy")

        firstgen_ensemble_std = np.std(firstgen_ensemble, axis=0)
        firstgen_std = Matrix(firstgen_ensemble_std, self.firstgen.Eg,
                              self.firstgen.Ex, state='std')
        firstgen_std.save(self.save_path / "firstgen_std.npy")

        self.std_raw = raw_std
        self.std_unfolded = unfolded_std
        self.std_firstgen = firstgen_std

        self.raw_ensemble = raw_ensemble
        self.unfolded_ensemble = unfolded_ensemble
        self.firstgen_ensemble = firstgen_ensemble

    def generate_raw(self, step: int, method: str) -> Matrix:
        """Generate a perturbated matrix

        Looks for an already generated file before generating the matrix.
        Args:
            step: The identifier of the matrix to generate
            method: The name of the method to use. Can be either
                "gaussian" or "poisson"
        Returns:
            The generated matrix
        """
        LOG.debug(f"Generating raw ensemble {step}")
        path = os.path.join(self.save_path, f"raw_{step}.npy")
        raw = self.load_matrix(path)
        if raw is None:
            LOG.debug(f"(Re)generating {path} using {method} process")
            if method == 'gaussian':
                values = self.generate_gaussian()
            elif method == 'poisson':
                values = self.generate_poisson()
            else:
                raise ValueError(f"Method {method} is not supported")
            raw = Matrix(values, Eg=self.raw.Eg, Ex=self.raw.Ex)
            raw.save(path)
        return raw

    def unfold(self, step: int, raw: Matrix) -> Matrix:
        """Unfolds the raw matrix

        Looks for an already generated file before generating the matrix.
        Args:
            step: The identifier of the matrix to unfold.
            raw: The raw matrix to unfold
        Returns:
            The unfolded matrix
        """
        LOG.debug(f"Unfolding raw {step}")
        path = os.path.join(self.save_path, f"unfolded_{step}.npy")
        unfolded = self.load_matrix(path)
        if unfolded is None:
            LOG.debug("Unfolding matrix")
            unfolded = self.unfolder(raw)
            unfolded.save(path)
        return unfolded

    def first_generation(self, step: int, unfolded: Matrix) -> Matrix:
        """Applies first generation method to an unfolded matrix

        Looks for an already generated file before applying first generation.
        Args:
            step: The identifier of the matrix to unfold.
            unfolded: The matrix to apply first generation method to.
        Returns:
            The resulting matrix
        """
        LOG.debug(f"Performing first generation on unfolded {step}")
        path = os.path.join(self.save_path, f"firstgen_{step}.npy")
        firstgen = self.load_matrix(path)
        if firstgen is None:
            LOG.debug("Calculating first generation matrix")
            firstgen = self.first_generation_method(unfolded)
            firstgen.save(path)
        return firstgen

    def generate_gaussian(self) -> np.ndarray:
        """Generates an array with Gaussian perturbations of self.raw

        Returns:
            The resulting array
        """
        std = np.sqrt(np.where(self.raw.value > 0, self.raw.values, 0))
        perturbed = np.random.normal(size=self.raw.shape, loc=self.raw.values,
                                     scale=std)
        perturbed[perturbed < 0] = 0
        return perturbed

    def generate_poisson(self) -> np.ndarray:
        """Generates an array with Poisson perturbations of self.raw

        Returns:
            The resulting array
        """
        std = np.where(self.raw.values > 0, self.raw.values, 0)
        perturbed = np.random.poisson(std)
        return perturbed

    def load_matrix(self, path: str) -> Union[Matrix, None]:
        """Check if file exists and should not be regenerated

        Args:
            path: Path to file to load
        Returns:
            Matrix if file exists and can be loaded, None otherwise.
        """
        if os.path.isfile(path) and not self.regenerate:
            LOG.debug(f"Loading {path}")
            return Matrix(path=path)

    def get_raw(self, index: Union[int, List[int]]) -> Union[Matrix,
                                                             List[Matrix]]:
        """Get the raw matrix(ces) created in ensemble generation.

        Args:
            index: The index of the ensemble. If an iterable,
                a list of matrices will be returned.
        Returns:
            The matrix(ces) corresponding to index(ces)
        """
        try:
            matrices = []
            for i in index:
                matrices.append(Matrix(self.raw_ensemble[i], self.raw.Eg,
                                       self.raw.Ex))
            return matrices
        except TypeError:
            pass
        return Matrix(self.raw_ensemble[index], self.raw.Eg,
                      self.raw.Ex)

    def get_unfolded(self,
                     index: Union[int, List[int]]) -> Union[Matrix,
                                                            List[Matrix]]:
        """Get the unfolded matrix(ces) created in ensemble generation.

        Args:
            index: The index of the ensemble. If an iterable,
                a list of matrices will be returned.

        Returns:
            The matrix(ces) corresponding to index(ces)
        """
        try:
            matrices = []
            for i in index:
                matrices.append(Matrix(self.unfolded_ensemble[i], self.raw.Eg,
                                       self.raw.Ex))
            return matrices
        except TypeError:
            pass
        return Matrix(self.unfolded_ensemble[index], self.raw.Eg,
                      self.raw.Ex, state='unfolded')

    def get_firstgen(self,
                     index: Union[int, List[int]]) -> Union[Matrix,
                                                            List[Matrix]]:
        """Get the first generation matrix(ces) created in ensemble generation.

        Args:
            index: The index of the ensemble. If an iterable,
                a list of matrices will be returned.
        Returns:
            The matrix(ces) corresponding to index(ces)
        """
        try:
            matrices = []
            for i in index:
                matrices.append(Matrix(self.firstgen_ensemble[i],
                                       self.firstgen.Eg,
                                       self.firstgen.Ex))
            return matrices
        except TypeError:
            pass
        return Matrix(self.firstgen_ensemble[index],
                      self.firstgen.Eg,
                      self.firstgen.Ex, state='firstgen')

    def plot(self):
        fig, ax = plt.subplots(ncols=3, sharey=True, constrained_layout=True)
        self.std_raw.plot(ax=ax[0], title='Raw')
        self.std_unfolded.plot(ax=ax[1], title='Unfolded')
        self.std_firstgen.plot(ax=ax[2], title='First Generation')
        ax[1].set_ylabel(None)
        ax[2].set_ylabel(None)
        fig.suptitle("Standard Deviation")
        return fig, ax
