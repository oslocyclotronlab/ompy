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
import os
from .library import *
from .matrix_analysis import *
from .matrix import Matrix
import copy


class ErrorPropagation:
    def __init__(self, matrix_analysis_instance,
                 folder="error_propagation_ensemble",
                 randomness="poisson",
                 random_seed=None):

        self.matrix_analysis = matrix_analysis_instance
        self.folder = folder
        self.randomness = randomness
        # Allocate variables to be filled by class methods later:
        self.std_raw = None
        self.std_unfolded = None
        self.std_firstgen = None
        # Create folder
        if not os.path.exists(folder):
            os.mkdir(folder)

        # Set random seed
        if random_seed is not None:
            np.random.seed(random_seed)

        # Check if the passed matrix_analysis_instance contains
        # raw, unfolded and firstgen matrices.
        # If so, copy them to ensemble directory.
        # If not, run through and make them (except raw,
        # which must be there).

        # Raw:
        if self.matrix_analysis.raw.values is None:
            raise Exception("Error: No raw matrix passed to ErrorPropagation.")
        else:
            self.matrix_analysis.raw.save(
                os.path.join(folder, "raw-orig.m")
                )
        # Unfolded:
        if self.matrix_analysis.unfolded.values is None:
            self.matrix_analysis.unfold()
        else:
            self.matrix_analysis.unfolded.save(
                os.path.join(folder, "unfolded-orig.m")
                )
        # First generation
        if self.matrix_analysis.firstgen.values is None:
            self.matrix_analysis.first_generation_method()
        else:
            self.matrix_analysis.firstgen.save(
                os.path.join(folder, "firstgen-orig.m")
                )

    def generate_ensemble(self,
                          N_ensemble_members,
                          randomness="poisson",
                          verbose=True,
                          purge_files=False):
        """
        Function which generates an ensemble of raw spectra, unfolds and
        first-generation-methods them

        Args:
            N_ensemble_members (int): How many ensemble members to simulate
            randomness (str): How to approximate the statistical distribution
                              of counts in the raw matrix.
                              Must be one of ("gaussian", "poisson")
            verbose (bool): Whether to print information as the calculation
                            proceeds.
            purge_files (bool): If the ensemble directory is not empty, should
                                existing files be overwritten?
        Returns:
            std_firstgen (Matrix): The variance in the ensemble of first-
                                   generation matrices, taken independently
                                   in each pixel.
        """

        folder = self.folder
        matrix_analysis = self.matrix_analysis

        # TODO copy things from generate_ensemble.py.
        # Set up the folder and file structure,
        # run the loop and perturb each member.
        # Consider how best to do the member handling -- instantiate the pyma class inside itself for each member?


        # fname_resp_mat = '../data/response_matrix-Re187-10keV.m'
        # fname_resp_dat = '../data/resp-Re187-10keV.dat'
        # R, FWHM, eff, pc, pf, ps, pd, pa, Eg_array_resp = pml.read_response(fname_resp_mat, fname_resp_dat)

        # purge_files = False#True # Set to True if we want to regenerate the saved matrices

        # === Unfold and get firstgen without perturbations: ===
        # === Unfold it ===:
        # fname_unfolded_orig = os.path.join(folder, name+"-unfolded-orig.m")
        # if os.path.isfile(fname_unfolded_orig) and not purge_files:
        #     unfolded_orig, tmp, Ex_array_unf, Eg_array_unf = read_mama_2D(fname_unfolded_orig)
        #     if verbose:
        #         print("Read unfolded matrix from file", flush=True)
        # else:
        #     if verbose:
        #         print("Unfolding matrix", flush=True)
        #     # Unfold:
        #     unfolded_orig, Ex_array_unf, Eg_array_unf = unfold(data_raw, Ex_array, Eg_array, fname_resp_dat, fname_resp_mat, verbose=False, use_comptonsubtraction=False)
        #     write_mama_2D(unfolded_orig, fname_unfolded_orig, Ex_array_unf, Eg_array_unf, comment="unfolded matrix, original (not perturbed).")

        #     if verbose:
        #         print("unfolded_orig.shape =", unfolded_orig.shape, flush=True)

        # # === Extract first generation spectrum ===:
        # Ex_max = 12000 # keV - maximum excitation energy
        # dE_gamma = 1000 # keV - allow gamma energy to exceed excitation energy by this much, to account for experimental resolution
        # # N_Exbins = 300
        # N_Exbins = len(Ex_array_unf)
        # fname_firstgen_orig = os.path.join(folder, name+"-firstgen-orig.m")
        # if os.path.isfile(fname_firstgen_orig) and not purge_files:
        #     firstgen_orig, tmp, Ex_array_fg, Eg_array_fg = read_mama_2D(fname_firstgen_orig)
        #     print("Read first generation matrix from file", flush=True)
        # else:
        #     print("Calculating first generation matrix", flush=True)
        #     print("unfolded_orig.shape =", unfolded_orig.shape, flush=True)
        #     # Find first generation spectrum:
        #     firstgen_orig, diff, Ex_array_fg, Eg_array_fg = first_generation_spectrum(unfolded_orig, Ex_array_unf, Eg_array_unf, N_Exbins, Ex_max, dE_gamma, N_iterations=10)
        #     write_mama_2D(firstgen_orig, fname_firstgen_orig, Ex_array_fg, Eg_array_fg, comment="first generation matrix, original (not perturbed).")




        # === Proceed with statistical treatment, making a perturbed ensemble ===




        # N_stat = 10 # How many perturbed copies do we want in our ensemble?

        # Allocate arrays to store ensemble members: TODO maybe it's not needed? Just do everything in loop one by one? It's fast enough if we read matrices from file.
        # data_raw_ensemble = np.empty(np.append(data_raw.shape,N_stat))
        # unfolded_ensemble = np.empty(np.append(data_raw.shape,N_stat))
        # firstgen_ensemble = np.empty(np.append(data_raw.shape,N_stat))

        # TESTING: Plot matrices to see effect of perturbations:
        # Nx, Ny = 2, 2
        # from matplotlib.colors import LogNorm
        # def map_iterator_to_grid(counter, Nx, Ny):
        #   # Returns i, j coordinate pairs to map a single iterator onto a 2D grid for subplots.
        #   # Counts along each row from left to right, then increments row number
        #   i = counter // Nx
        #   j = counter % Nx
        #   return i, j
        # f_raw, axs_raw = plt.subplots(Nx,Ny)
        # axs_raw[0,0].set_title("raw")
        # f_unf, axs_unf = plt.subplots(Nx,Ny)
        # axs_unf[0,0].set_title("unf")
        # f_fg, axs_fg = plt.subplots(Nx,Ny)
        # axs_fg[0,0].set_title("fg")
        # END TESTING

        # Allocate a cube array to store all ensemble members. We need them to make the std matrix.
        raw_ensemble = np.zeros((N_ensemble_members,
                                matrix_analysis.raw.values.shape[1],
                                matrix_analysis.raw.values.shape[0])
                                )
        unfolded_ensemble = np.zeros((N_ensemble_members,
                                      matrix_analysis.unfolded.values.shape[1],
                                      matrix_analysis.unfolded.values.shape[0])
                                     )
        firstgen_ensemble = np.zeros((N_ensemble_members,
                                      matrix_analysis.firstgen.values.shape[0],
                                      matrix_analysis.firstgen.values.shape[1])
                                     )

        # Loop over and generate the random perturbations, then unfold and
        # first-generation-method them:
        for i in range(N_ensemble_members):
            if verbose:
                print("=== Begin ensemble member ", i, " ===", flush=True)
            # === Perturb initial raw matrix ===:
            fname_raw_current = os.path.join(folder, "raw-"+str(i)+".m")

            # Allocate matrix_analysis instance for current ensemble member:
            ma_curr = copy.deepcopy(matrix_analysis)
            # Check if the current ensemble member exists on file, create it if
            # not:
            if os.path.isfile(fname_raw_current) and not purge_files:
                ma_curr.raw.load(fname_raw_current, suppress_warning=True)
                if verbose:
                    print("Read raw matrix from file", flush=True)
            else:
                if verbose:
                    print("Generating raw matrix", flush=True)
                # matrix_ensemble_current = np.maximum(matrix + np.random.normal(size=matrix_shape)*np.sqrt(matrix), np.zeros(matrix_shape)) # Each bin of the matrix is perturbed with a gaussian centered on the bin count, with standard deviation sqrt(bin count). Also, no negative counts are accepted.
                if randomness=="gaussian":
                    # Assuming sigma \approx sqrt(n) where n is number of
                    # counts in bin.
                    matrix_perturbed = np.random.normal(
                            size=matrix_analysis.raw.values.shape,
                            loc=matrix_analysis.raw.values,
                            scale=np.sqrt(np.where(
                                matrix_analysis.raw.values > 0,
                                matrix_analysis.raw.values, 0)
                                                   )
                                                        )
                    matrix_perturbed[matrix_perturbed < 0] = 0
                elif randomness == "poisson":
                    # Use the original number of counts as the estimator for lambda (expectation value) in the Poisson
                    # distribution at each bin, and draw a completely new matrix:
                    matrix_perturbed = np.random.poisson(
                                        np.where(
                                            matrix_analysis.raw.values > 0,
                                            matrix_analysis.raw.values, 0
                                                )
                                                        )
                else:
                    raise ValueError(("Unknown value for randomness variable:",
                                      str(randomness))
                                     )

                # Update the "raw" member of ma_curr:
                ma_curr.raw = Matrix(matrix_perturbed,
                                     matrix_analysis.raw.Eg,
                                     matrix_analysis.raw.Ex)
                # Save ensemble member to disk:
                ma_curr.raw.save(fname_raw_current)

            # Store raw ensemble member in memory:
            raw_ensemble[i, :, :] = ma_curr.raw.values

                # if verbose:
                    # print("data_raw_ensemblemember.shape =", data_raw_ensemblemember.shape, flush=True)

            # === Unfold it ===:
            fname_unfolded_current = os.path.join(folder, "unfolded-"+str(i)+".m")
            if os.path.isfile(fname_unfolded_current) and not purge_files:
                ma_curr.unfolded.load(fname_unfolded_current,
                                      suppress_warning=True)
                # unfolded_ensemblemember, tmp, Ex_array_unf, Eg_array_unf = read_mama_2D(fname_unfolded_current)
                if verbose:
                    print("Read unfolded matrix from file", flush=True)
            else:
                if verbose:
                    print("Unfolding matrix", flush=True)
                # Unfold:
                # unfolded_ensemblemember, Ex_array_unf, Eg_array_unf = unfold(data_raw_ensemblemember, Ex_array, Eg_array, fname_resp_dat, fname_resp_mat, verbose=False, use_comptonsubtraction=False)
                ma_curr.unfold()
                ma_curr.unfolded.save(fname_unfolded_current)

                # if verbose:
                    # print("unfolded_ensemblemember.shape =", unfolded_ensemblemember.shape, flush=True)

            # Store unfolded ensemble member in memory:
            unfolded_ensemble[i, :, :] = ma_curr.unfolded.values

            # === Extract first generation spectrum ===:
            # Ex_max = 12000 # keV - maximum excitation energy
            # dE_gamma = 1000 # keV - allow gamma energy to exceed excitation energy by this much, to account for experimental resolution
            # N_Exbins = 300
            # N_Exbins = len(Ex_array_unf)
            fname_firstgen_current = os.path.join(folder, "firstgen-"+str(i)+".m")
            if os.path.isfile(fname_firstgen_current) and not purge_files:
                ma_curr.firstgen.load(fname_firstgen_current,
                                      suppress_warning=True)
                # firstgen_ensemblemember, tmp, Ex_array_fg, Eg_array_fg = read_mama_2D(fname_firstgen_current)
                if verbose:
                    print("Read first generation matrix from file", flush=True)
            else:
                if verbose:
                    print("Calculating first generation matrix", flush=True)
                # print("unfolded_ensemblemember.shape =", unfolded_ensemblemember.shape, flush=True)
                # Find first generation spectrum:
                # firstgen_ensemblemember, diff, Ex_array_fg, Eg_array_fg = first_generation_spectrum(unfolded_ensemblemember, Ex_array_unf, Eg_array_unf, N_Exbins, Ex_max, dE_gamma, N_iterations=10)
                # write_mama_2D(firstgen_ensemblemember, fname_firstgen_current, Ex_array_fg, Eg_array_fg, comment="first generation matrix, ensemble member no. "+str(i))
                ma_curr.first_generation_method()
                ma_curr.firstgen.save(fname_firstgen_current)

            firstgen_ensemble[i, :, :] = ma_curr.firstgen.values


            # TESTING: Plot ensemble of first-gen matrices:
            # i_plt, j_plt = map_iterator_to_grid(i, Nx, Ny)
            # ma_curr.raw.plot(ax=axs_raw[i_plt, j_plt])
            # ma_curr.unfolded.plot(ax=axs_unf[i_plt, j_plt])
            # ma_curr.firstgen.plot(ax=axs_fg[i_plt, j_plt])
            # END TESTING



            # End loop over perturbed ensemble

        # TODO consider if firstgen_ensemble should be kept as a class object
        # -- would it be useful for anything? Or does it just take up memory?

        # === Calculate standard deviation ===:
        # raw:
        raw_ensemble_std = np.std(raw_ensemble, axis=0)
        std_raw = Matrix(raw_ensemble_std,
                         matrix_analysis.raw.Eg,
                         matrix_analysis.raw.Ex,
                         )
        fname_raw_std = os.path.join(folder, "raw_std.m")
        std_raw.save(fname_raw_std)
        # unfolded:
        unfolded_ensemble_std = np.std(unfolded_ensemble, axis=0)
        std_unfolded = Matrix(unfolded_ensemble_std,
                         matrix_analysis.unfolded.Eg,
                         matrix_analysis.unfolded.Ex,
                         )
        fname_unfolded_std = os.path.join(folder, "unfolded_std.m")
        std_unfolded.save(fname_unfolded_std)
        # firstgen:
        firstgen_ensemble_std = np.std(firstgen_ensemble, axis=0)
        std_firstgen = Matrix(firstgen_ensemble_std,
                              matrix_analysis.firstgen.Eg,
                              matrix_analysis.firstgen.Ex,
                              )
        fname_firstgen_std = os.path.join(folder, "firstgen_std.m")
        std_firstgen.save(fname_firstgen_std)

        # TESTING:
        # plt.show()
        # END TESTING

        self.std_firstgen = std_firstgen
        self.std_unfolded = std_unfolded
        self.std_raw = std_raw
        # TODO add self.std_unfolded

        # Also store a list containing all ensemble members of firstgen:
        self.firstgen_ensemble = firstgen_ensemble
        # self.firstgen_ensemble = []
        # for i_ens in range(N_ensemble_members):


        # return std_firstgen
