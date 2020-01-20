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
import numpy as np
import logging
import matplotlib.pyplot as plt
import os
from pathos.multiprocessing import ProcessPool
from pathos.helpers import cpu_count
from itertools import repeat

from typing import Callable, Union, List, Optional, Any, Tuple
from pathlib import Path
from numpy import ndarray
from .matrix import Matrix
from .rebin import rebin_2D
from .action import Action

if 'JPY_PARENT_PID' in os.environ:
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm

LOG = logging.getLogger(__name__)
logging.captureWarnings(True)


class Ensemble:
    """Generates perturbated matrices to estimate uncertainty

    Note:
        When adding any functionality that runs withing the parallelized loop
        make sure to use the random generator exposed via the arguments. If
        one uses np.random instead, this will be the same an exact copy for
        each process.

    Attributes:
        raw (Matrix): The raw matrix used as model for the ensemble. If a
            background is provided, this will initially be the the
            "prompt+bg" matrix.
        bg: Background matrix to subtract.
        bg_ratio: Prompt is obtainied by `raw - bg_ratio * bg`. Defaults to 1.
        path (str): The path used for saving the generated matrices.
        unfolder (Unfolder): An instance of Unfolder which is used in
            the unfolding of the generated matrices. Only has to support
            obj.unfold(raw: Matrix).
        first_generation_method (FirstGeneration): An instance of
            FirstGeneration for applying the first generation method
            to unfolded matrices. Only has to be callable
            with (unfolded: Matrix)
        regenerate (bool): If true, regenerate the matrices instead of
            loading from `path`.
        std_raw (Matrix): The computed standard deviation of the raw ensemble.
        std_unfolded (Matrix): The computed standard deviation of the unfolded
            ensemble.
        std_firstgen (Matrix): The computed standard deviation of the
            first generation method matrices.
        firstgen_ensemble (np.ndarray): The entire firstgen ensemble.
        action_raw (Action[Matrix]): An arbitrary action to apply to each
            generated raw matrix. Defaults to NOP.
        action_prompt_w_bg (Action[Matrix]): An arbitrary action to apply to
            each generated prompt_w_bg matrix. Defaults to NOP.
        action_bg (Action[Matrix]): An arbitrary action to apply to
            each generated bg matrix. Defaults to NOP.
        action_unfolded (Action[Matrix]): An arbitrary action to apply to each
            generated unfolded matrix. Defaults to NOP.
        action_firstgen (Action[Matrix]): An arbitrary action to apply to each
            generated firstgen matrix. Defaults to NOP.
        nprocesses (int): Number of processes for multiprocessing.
            Defaults to number of available cpus-1 (with mimimum 1).
        seed (int): Random seed for reproducibility of results


    TODO:
        - Separate each step - generation, unfolded, firstgening?
        - (Re)generation with book keeping is a recurring pattern.
          Try to abstract it away.
    """
    def __init__(self, raw: Optional[Matrix] = None,
                 bg: Optional[Matrix] = None,
                 bg_ratio: float = 1,
                 path: Optional[Union[str, Path]] = None):
        """ Sets up attributes and loads a saved ensemble if provided.

        Args:
            raw: The model matrix to peturbate. If a background is provided,
                this is the "prompt+bg" matrix.
            bg: Background matrix to subtract.
            bg_ratio: Prompt is obtainied by `raw - bg_ratio * bg`. Defaults to
                1. This is the case for equal time gate length of `prompt+bg`
                and `bg`.
            path: The path where to save the ensemble. If set,
                the ensemble will try to load from the path, but will
                fail *silently* if it is unable to. It is recommended to call
                load([path]) explicitly.
        """
        self.raw: Optional[Matrix] = raw
        self.bg: Optional[Matrix] = bg
        self.bg_ratio: Optional[float] = bg_ratio
        self.prompt_w_bg: Optional[Matrix] = raw

        self.unfolder: Optional[Callable[[Matrix], Matrix]] = None
        self.first_generation_method: Optional[Callable[[Matrix], Matrix]] = None
        self.size = 0
        self.regenerate = False
        self.action_prompt_w_bg = Action('matrix')
        self.action_bg = Action('matrix')
        self.action_raw = Action('matrix')
        self.action_unfolded = Action('matrix')
        self.action_firstgen = Action('matrix')

        self.std_raw: Optional[Matrix] = None
        self.std_unfolded: Optional[Matrix] = None
        self.std_firstgen: Optional[Matrix] = None

        self.raw_ensemble: Optional[Matrix] = None
        self.unfolded_ensemble: Optional[Matrix] = None
        self.firstgen_ensemble: Optional[Matrix] = None

        self.seed: int = 987654
        self.nprocesses: int = cpu_count()-1 if cpu_count() > 1 else 1

        if path is not None:
            self.path = Path(path)
            self.path.mkdir(exist_ok=True)
            self.load()
        else:
            # Should the ensemble always try to load?
            self.path = Path('ensemble')
            self.path.mkdir(exist_ok=True)

        self.raw.state = "raw"

    def load(self, path: Optional[Union[str, Path]] = None):
        """ Loads a saved ensamble

        Currently only supports '.npy' format.

        Args:
            path: The path to the ensemble directory. If the
                value is None, 'self.path' will be used.
        """
        path = Path(path) if path is not None else Path(self.path)

        self.raw = Matrix(path=path / 'raw.npy')
        self.firstgen = Matrix(path=path / 'firstgen.npy')
        raws = list(path.glob("raw_[0-9]*.*"))
        unfolds = list(path.glob("unfolded_[0-9]*.*"))
        firsts = list(path.glob("firstgen_[0-9]*.*"))
        assert len(raws) == len(unfolds) == len(firsts), "Corrupt ensemble"
        assert len(raws) > 0, "Found no saved ensemble members"
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
        assert self.unfolder is not None, "Set unfolder"
        assert self.first_generation_method is not None, \
            "Set first generation method"

        self.size = number
        self.regenerate = regenerate

        LOG.info(f"Start normalization with {self.nprocesses} cpus")
        pool = ProcessPool(nodes=self.nprocesses)
        ss = np.random.SeedSequence(self.seed)
        iterator = pool.imap(self.step, range(number), ss.spawn(number),
                             repeat(method))
        ensembles = np.array(list(tqdm(iterator, total=number)))
        pool.close()
        pool.join()
        pool.clear()

        raw_ensemble = ensembles[:, 0, :, :]
        unfolded_ensemble = ensembles[:, 1, :, :]
        firstgen_ensemble = ensembles[:, 2, :, :]

        # TODO Move this to a save step
        self.raw.save(self.path / 'raw.npy')
        # saving for firstgen is in step due to pickling
        self.firstgen = Matrix(path=self.path / 'firstgen.npy')

        # Calculate standard deviation
        raw_ensemble_std = np.std(raw_ensemble, axis=0)
        raw_std = Matrix(raw_ensemble_std, self.raw.Eg, self.raw.Ex,
                         state='std')
        raw_std.save(self.path / "raw_std.npy")

        unfolded_ensemble_std = np.std(unfolded_ensemble, axis=0)
        unfolded_std = Matrix(unfolded_ensemble_std, self.raw.Eg,
                              self.raw.Ex, state='std')
        unfolded_std.save(self.path / "unfolded_std.npy")

        firstgen_ensemble_std = np.std(firstgen_ensemble, axis=0)
        firstgen_std = Matrix(firstgen_ensemble_std, self.firstgen.Eg,
                              self.firstgen.Ex, state='std')
        firstgen_std.save(self.path / "firstgen_std.npy")

        self.std_raw = raw_std
        self.std_unfolded = unfolded_std
        self.std_firstgen = firstgen_std

        self.raw_ensemble = raw_ensemble
        self.unfolded_ensemble = unfolded_ensemble
        self.firstgen_ensemble = firstgen_ensemble

    def step(self, step: int, rsequence: np.random.SeedSequence,
             method: str):
        """Single step in `self.generate`

        Args:
            step (int): step (loop) number
            rsequence (np.random.SeedSequence): SeedSequence for random seed
            method (str): see `self.generate`

        Returns:
            raw, unfolded, firstgen
        """
        LOG.info(f"Generating {step}")
        rstate = np.random.default_rng(rsequence)
        if self.bg is not None:
            prompt_w_bg = self.generate_perturbed(step, method,
                                                  state="prompt+bg",
                                                  rstate=rstate)
            bg = self.generate_perturbed(step, method, state="bg",
                                         rstate=rstate)
            raw = self.subtract_bg(step, prompt_w_bg, bg)
        else:
            raw = self.generate_perturbed(step, method, state="raw",
                                          rstate=rstate)
        unfolded = self.unfold(step, raw)
        firstgen = self.first_generation(step, unfolded)

        if step == 0:  # workaround
            firstgen.save(self.path / 'firstgen.npy')

        assert(raw.shape == unfolded.shape and raw.shape == firstgen.shape), \
            ("For now, all matrices have to have the same shape. Currently, "
             f"shapes: raw: {raw.shape}, unfolded: {unfolded.shape} and "
             f"firstgen: {firstgen.shape}")
        return raw.values, unfolded.values, firstgen.values

    def generate_perturbed(self, step: int, method: str, state: str,
                           rstate: np.random.Generator) -> Matrix:
        """Generate a perturbated matrix

        Looks for an already generated file before generating the matrix.

        Args:
            step: The identifier of the matrix to generate
            method: The name of the method to use. Can be either
                "gaussian" or "poisson"
            state: Either "raw", "prompt+bg" or "bg".
            rstate: numpy random generator (random state)

        Returns:
            The generated matrix
        """
        allowed = ["raw", "prompt+bg", "bg"]
        if state not in allowed:
            raise NotImplementedError(f"Matrix must be a state in {allowed}")
        LOG.debug(f"Generating {state} ensemble {step}")
        path = self.path / f"{state}_{step}.npy"
        mat = self.load_matrix(path)
        if mat is None:
            # if np.any(self.mat.values < 0):
            #     raise ValueError("input matrix has to have positive"
            #                      "entries only. Consider using fill and or"
            #                      "remove negatives")
            LOG.debug(f"(Re)generating {path} using {method} process")
            if method == 'gaussian':
                values = self.generate_gaussian(state, rstate)
            elif method == 'poisson':
                values = self.generate_poisson(state, rstate)
            else:
                raise ValueError(f"Method {method} is not supported")
            base_mat = self.matrix_from_state(state)
            mat = Matrix(values, Eg=base_mat.Eg, Ex=base_mat.Ex, state=state)
            mat.save(path)
        self.action_from_state(state).act_on(mat)
        return mat

    def subtract_bg(self, step: int, matrix: Matrix, bg: Matrix) -> Matrix:
        """ Subtract bg from "raw" (prompt+bg) matrix

        Looks for an already generated file before generating the matrix.
        Args:
            step: The identifier of the matrix to act on.
            matrix: The matrix to subtract from (usually, prompt+bg)
            bg: The bg matrix.
        Returns:
            raw: The prompts only matrix ("raw" matrix).
        """
        LOG.debug(f"Subtracting bg from prompt+bg {step}")
        path = self.path / f"raw_{step}.npy"
        raw = self.load_matrix(path)
        if raw is None:
            LOG.debug("Raw matrix")
            raw = matrix - self.bg_ratio * bg
            raw.remove_negative()
            raw.save(path)
        self.action_raw.act_on(raw)
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
        path = self.path / f"unfolded_{step}.npy"
        unfolded = self.load_matrix(path)
        if unfolded is None:
            LOG.debug("Unfolding matrix")
            unfolded = self.unfolder(raw)
            unfolded.save(path)
        self.action_unfolded.act_on(unfolded)
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
        path = self.path / f"firstgen_{step}.npy"
        firstgen = self.load_matrix(path)
        if firstgen is None:
            LOG.debug("Calculating first generation matrix")
            firstgen = self.first_generation_method(unfolded)
            firstgen.save(path)
        self.action_firstgen.act_on(firstgen)
        return firstgen

    def rebin(self, out_array: np.ndarray, member: str) -> None:
        """ Rebins the first generations matrixes and recals std

        Args:
           out_array: mid-bin energy calibration of output
                      matrix along rebin axis
           member: which member to rebin, currently only FG available
        """
        if member != 'firstgen':
            raise NotImplementedError("Not implemented for raw and unfolding "
                                      "yet. if done, need to redo unfolding "
                                      "and/or first gen method")

        ensemble = self.firstgen_ensemble
        matrix = self.firstgen

        do_Ex = not np.array_equal(out_array, matrix.Ex)
        do_Eg = not np.array_equal(out_array, matrix.Eg)
        if (not do_Ex) and (not do_Eg):
            return

        rebinned = np.zeros((self.size, out_array.size, out_array.size))
        for i in tqdm(range(self.size)):
            values = ensemble[i]
            if do_Ex:
                values = rebin_2D(values, mids_in=matrix.Ex,
                                  mids_out=out_array, axis=0)
            if do_Eg:
                values = rebin_2D(values, mids_in=matrix.Eg,
                                  mids_out=out_array, axis=1)
            rebinned[i] = values

        # correct fg matrix (different attribute) and axis
        values = matrix.values
        if do_Ex:
            values = rebin_2D(values, mids_in=matrix.Ex,
                              mids_out=out_array, axis=0)
        if do_Eg:
            values = rebin_2D(values, mids_in=matrix.Eg,
                              mids_out=out_array, axis=1)
        matrix = Matrix(values, out_array, out_array)

        # recalculate std
        firstgen_ensemble_std = np.std(rebinned, axis=0)
        firstgen_std = Matrix(firstgen_ensemble_std, matrix.Eg,
                              matrix.Ex, state='std')
        firstgen_std.save(self.path / "firstgen_std.npy")

        self.firstgen = matrix
        self.firstgen_ensemble = rebinned
        self.std_firstgen = firstgen_std

    def generate_gaussian(self, state: str,
                          rstate: Optional[np.random.Generator] = np.random.default_rng) -> np.ndarray: # noqa
        """Generates an array with Gaussian perturbations of a matrix.
        Note that entries are truncated at 0 (only positive).

        Args:
            state: State of the matrx/which matrix should be taken as a base
            rstate (optional): numpy random generator (random state)
        Returns:
            The resulting array
        """
        mat = self.matrix_from_state(state)
        std = np.sqrt(np.where(mat.values > 0, mat.values, 0))
        perturbed = rstate.normal(size=mat.shape, loc=mat.values,
                                  scale=std)
        perturbed[perturbed < 0] = 0
        return perturbed

    def generate_poisson(self, state: str,
                         rstate: Optional[np.random.Generator] = np.random.default_rng) -> np.ndarray:  # noqa
        """Generates an array with Poisson perturbations of a matrix

        Args:
            state: State of the matrx/which matrix should be taken as a base
            rstate (optional): numpy random generator (random state)
        Returns:
            The resulting array
        """
        mat = self.matrix_from_state(state)
        std = np.where(mat.values > 0, mat.values, 0)
        perturbed = rstate.poisson(std)
        return perturbed

    def load_matrix(self, path: Union[Path, str]) -> Union[Matrix, None]:
        """Check if file exists and should not be regenerated

        Args:
            path: Path to file to load
        Returns:
            Matrix if file exists and can be loaded, None otherwise.
        """
        path = Path(path)
        if path.exists() and not self.regenerate:
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

    def action_from_state(self, state: str) -> Action:
        """ Return the action corresponding to a given state
        Args:
            state: The state
        Returns:
            action: The corresponding action
        """
        if state == "raw":
            action = self.action_raw
        elif state == "prompt+bg":
            action = self.action_prompt_w_bg
        elif state == "bg":
            action = self.action_bg
        elif state == "unfolded":
            action = self.action_unfolded
        elif state == "firstgen":
            action = self.action_firstgen
        else:
            raise NotImplementedError(f"State {state} is not a known state")
        return action

    def matrix_from_state(self, state: str) -> Action:
        """ Return the matrix corresponding to a given state
        Args:
            state: The state
        Returns:
            Matrix: The corresponding matrix
        """
        if state == "raw":
            matrix = self.raw
        elif state == "prompt+bg":
            matrix = self.prompt_w_bg
        elif state == "bg":
            matrix = self.bg
        # elif state == "unfolded":
        #     matrix = self.unfolded
        # elif state == "firstgen":
        #     matrix = self.firstgen
        else:
            raise NotImplementedError(f"State {state} is not a known state")
        return matrix

    def plot(self, *, ax: Any = None,
             vmin: Optional[float] = None,
             vmax: Optional[float] = None,
             add_cbar: bool = True,
             scale_by: str = 'all',
             **kwargs) -> Tuple[Any, ndarray]:
        """ Plot the computed standard deviations

        Args:
            ax (optional): A matplotlib axis to plot onto.
            vmin (optional, float): The lower cutoff for colors.
            vmax (optional, float): The upper cutoff for colors.
            add_cbar (optional, bool): Whether to add a colorbar.
                Defaults to True.
            scale_by (optional, str): Which std matrix to set color
                limits by. Can be "raw", "unfolded", "firstgen" or "all".
                Defaults to "all".

        Returns:
            The matplotlib figure and axis
        """
        if ax is not None:
            if len(ax) < 3:
                raise ValueError("Three axes must be provided")
            fig = ax.figure
        else:
            fig, ax = plt.subplots(ncols=3, sharey=True, constrained_layout=True)

        extrema = lambda x: (np.min(x), np.max(x)) # noqa
        choices = {"raw": extrema(self.std_raw.values),
                   "unfolded": extrema(self.std_unfolded.values),
                   "firstgen": extrema(self.std_firstgen.values)}
        choices["all"] = extrema([v for v in choices.values()])
        if scale_by not in choices:
            raise ValueError(f"`scale_by` can not be {scale_by}")

        vminset = True
        if vmin is None:
            vminset = False
            vmin = choices[scale_by][0]
            vmin = 1 if vmin <= 0 else vmin
        vmaxset = True
        if vmax is None:
            vmaxset = False
            vmax = choices[scale_by][1]

        # Actual plotting
        self.std_raw.plot(ax=ax[0], title='Raw', add_cbar=False,
                          vmin=vmin, vmax=vmax, **kwargs)
        self.std_unfolded.plot(ax=ax[1], title='Unfolded', add_cbar=False,
                               vmin=vmin, vmax=vmax, **kwargs)
        im, _, _ = self.std_firstgen.plot(ax=ax[2], title='First Generation',
                               vmin=vmin, vmax=vmax, add_cbar=False, **kwargs)

        # Y labels only clutter
        ax[1].set_ylabel(None)
        ax[2].set_ylabel(None)

        # Handle the colorbar
        if add_cbar:
            if vminset and vmaxset:
                fig.colorbar(im, extend='both')
            elif vminset:
                fig.colorbar(im, extend='min')
            elif vmaxset:
                fig.colorbar(im, extend='max')
            else:
                fig.colorbar(im)
        fig.suptitle("Standard Deviation")
        return fig, ax
