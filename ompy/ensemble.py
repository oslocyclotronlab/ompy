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
from .matrix import Matrix
import os
import numpy as np
import logging
from typing import Callable
from .unfolder import Unfolder

LOG = logging.getLogger(__name__)
logging.captureWarnings(True)


class Ensemble:
    """Generates perturbated matrices to estimate uncertainty

    TODO: Separate each step - generation, unfolded, firstgening?
    """
    def __init__(self, raw: Matrix, save_path: str):
        self.raw = raw
        self.save_path = save_path
        self.unfolder: Unfolder = None
        self.first_generation_method: Callable = None

    def save(self, matrix: Matrix, name: str):
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)

        path = os.path.join(self.save_path, name)
        matrix.save(path)

    def generate(self, number: int, method: str = 'poisson',
                 regenerate: bool = False) -> None:
        self.regenerate = regenerate
        raw_ensemble = np.zeros((number, *self.raw.shape))
        unfolded_ensemble = np.zeros_like(raw_ensemble)
        firstgen_ensemble = np.zeros_like(raw_ensemble)

        for step in range(number):
            raw = self.generate_raw(step, method)
            unfolded = self.unfold(step, raw)
            firstgen = self.first_generation(step, unfolded)

            raw_ensemble[step, :, :] = raw.values
            unfolded_ensemble[step, :, :] = unfolded.values

            # TODO The first generation method might reshape the matrix
            if firstgen_ensemble.shape != firstgen.shape:
                firstgen_ensemble = np.zeros((number, *firstgen.shape))
            self.firstgen = firstgen
            firstgen_ensemble[step, :, :] = firstgen.values

        # Calculate standard deviation
        raw_ensemble_std = np.std(raw_ensemble, axis=0)
        raw_std = Matrix(raw_ensemble_std, self.raw.Eg, self.raw.Ex,
                         state='std')
        self.save(raw_std, "raw.m")

        unfolded_ensemble_std = np.std(unfolded_ensemble, axis=0)
        unfolded_std = Matrix(unfolded_ensemble_std, self.raw.Eg,
                              self.raw.Ex, state='std')
        self.save(unfolded_std, "unfolded_std.m")

        firstgen_ensemble_std = np.std(firstgen_ensemble, axis=0)
        firstgen_std = Matrix(firstgen_ensemble_std, self.firstgen.Eg,
                              self.firstgen.Ex, state='std')
        self.save(firstgen_std, "first_std.m")

        self.std_raw = raw_std
        self.std_unfolded = unfolded_std
        self.std_firstgen = firstgen_std

        self.firstgen_ensemble = firstgen_ensemble

    def generate_raw(self, step: int, method: str) -> Matrix:
        LOG.info(f"Generating raw ensemble {step}")
        path = os.path.join(self.save_path, f"raw_{step}.m")
        if os.path.isfile(path) and not self.regenerate:
            LOG.debug(f"Loading {path}")
            current = Matrix(filename=path)
        else:
            LOG.debug(f"(Re)generating {path} using {method} process")
            if method == 'gaussian':
                values = self.generate_gaussian()
            elif method == 'poisson':
                values = self.generate_poisson()
            else:
                raise ValueError(f"Method {method} is not supported")
            current = Matrix(values, Eg=self.raw.Eg, Ex=self.raw.Ex)
            current.save(path)
        return current

    def unfold(self, step: int, raw: Matrix) -> Matrix:
        LOG.info(f"Unfolding raw {step}")
        path = os.path.join(self.save_path, f"unfolded_{step}.m")
        if os.path.isfile(path) and not self.regenerate:
            LOG.debug("Loading {path}")
            unfolded = Matrix(filename=path)
        else:
            LOG.debug("Unfolding matrix")
            unfolded = self.unfolder.unfold(raw)
            unfolded.save(path)
        return unfolded

    def first_generation(self, step: int, unfolded: Matrix) -> Matrix:
        LOG.info(f"Performing first generation on unfolded {step}")
        path = os.path.join(self.save_path, f"firstgen_{step}.m")
        if os.path.isfile(path) and not self.regenerate:
            LOG.debug("Loading {path}")
            firstgen = Matrix(filename=path)
        else:
            LOG.debug("Calculating first generation matrix")
            firstgen = self.first_generation_method(unfolded)
            firstgen.save(path)
        return firstgen

    def generate_gaussian(self) -> np.ndarray:
        std = np.sqrt(np.where(self.raw.value > 0, self.raw.values, 0))
        perturbed = np.random.normal(size=self.raw.shape, loc=self.raw.values,
                                     scale=std)
        perturbed[perturbed < 0] = 0
        return perturbed

    def generate_poisson(self) -> np.ndarray:
        std = np.sqrt(np.where(self.raw.values > 0, self.raw.values, 0))
        perturbed = np.random.poisson(std)
        return perturbed
