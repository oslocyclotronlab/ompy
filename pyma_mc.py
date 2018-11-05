"""
Class mc(), the Monte Carlo module in pyma. 
It handles generation of random ensembles of spectra with 
statistical perturbations, and can make first generation variance matrices.

---

This is pyma, the python implementation of the Oslo method.
It handles two-dimensional matrices of event count spectra, and
implements detector response unfolding, first generation method
and other manipulation of the spectra.

It is heavily inspired by MAMA, written by Magne Guttormsen and others,
available at https://github.com/oslocyclotronlab/oslo-method-software

Copyright (C) 2018 J{\o}rgen Eriksson Midtb{\o}
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

from pymama import pymama
import pyma_lib as pml
import pyma_matrix as pmmat

class pyma_mc:
    def __init__(self, pymama_instance, folder="pyma_ensemble_folder", seed=None):
        self.pm_orig = pymama_instance
        self.folder = folder
        # Create folder
        if not os.path.exists(folder):
            os.mkdir(folder)

        # Set random seed
        if seed is not None:
            np.random.seed(seed)

        # Check if the passed pymama_instance contains
        # raw, unfolded and firstgen matrices. 
        # If so, copy them to ensemble directory.
        # If not, run through and make them (except raw,
        # which must be there).

        # Raw:
        if self.pm_orig.raw.matrix is None:
            raise Exception("Error: No raw matrix passed to pyma_mc. ")
        else:
            self.pm_orig.raw.save(os.path.join(folder, "raw_orig.m"))
        # Unfolded:
        if self.pm_orig.unfolded.matrix is None:
            self.pm_orig.unfold()
        else:
            self.pm_orig.unfolded.save(os.path.join(folder, "raw_orig.m"))
        # First generation
        if self.pm_orig.firstgen.matrix is None:
            self.pm_orig.first_generation_method()
        else:
            self.pm_orig.firstgen.save(os.path.join(folder, "raw_orig.m"))



    def generate_ensemble(self, N_members, verbose=False):
        """
        Function which generates an ensemble of raw spectra, unfolds and first-generation-methods them

        """


        folder = self.folder

        # TODO copy things from generate_ensemble.py.
        # Set up the folder and file structure,
        # run the loop and perturb each member.
        # Consider how best to do the member handling -- instantiate the pyma class inside itself for each member?


        # fname_resp_mat = '../data/response_matrix-Re187-10keV.m'
        # fname_resp_dat = '../data/resp-Re187-10keV.dat'
        # R, FWHM, eff, pc, pf, ps, pd, pa, Eg_array_resp = pml.read_response(fname_resp_mat, fname_resp_dat)

        purge_files = False#True # Set to True if we want to regenerate the saved matrices

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
        Nx, Ny = 2, 2
        from matplotlib.colors import LogNorm
        def map_iterator_to_grid(counter, Nx, Ny):
          # Returns i, j coordinate pairs to map a single iterator onto a 2D grid for subplots.
          # Counts along each row from left to right, then increments row number
          i = counter // Nx
          j = counter % Nx
          return i, j
        # f_raw, axs_raw = plt.subplots(Nx,Ny)
        # axs_raw[0,0].set_title("raw")
        # f_unf, axs_unf = plt.subplots(Nx,Ny)
        # axs_unf[0,0].set_title("unf")
        # f_fg, axs_fg = plt.subplots(Nx,Ny)
        # axs_fg[0,0].set_title("fg")






        # Allocate a cube array to store all firstgen ensemble members. We need them to make the firstgen variance matrix.
        firstgen_ensemble = np.zeros((N_stat,firstgen_orig.shape[0],firstgen_orig.shape[1]))
        for i in range(N_members):
            print("Begin ensemble member ",i, flush=True)
            # === Perturb initial raw matrix ===:
            fname_raw_current = os.path.join(folder, name+"-raw-"+str(i)+".m")
            if os.path.isfile(fname_raw_current) and not purge_files:
                data_raw_ensemblemember, tmp, tmp, tmp = read_mama_2D(fname_raw_current)
                print("Read raw matrix from file", flush=True)
            else:
                print("Generating raw matrix", flush=True)
                # matrix_ensemble_current = np.maximum(matrix + np.random.normal(size=matrix_shape)*np.sqrt(matrix), np.zeros(matrix_shape)) # Each bin of the matrix is perturbed with a gaussian centered on the bin count, with standard deviation sqrt(bin count). Also, no negative counts are accepted.
                data_raw_ensemblemember = data_raw + np.random.normal(size=data_raw.shape, scale=np.sqrt(np.where(data_raw > 0, data_raw, 0))) # Assuming sigma \approx n^2 / N where n is current bin count and N is total count, according to sigma^2 = np(1-p) for normal approx. to binomial distribution.
                data_raw_ensemblemember[data_raw_ensemblemember < 0] = 0
                write_mama_2D(data_raw_ensemblemember, fname_raw_current, Ex_array, Eg_array, comment="raw matrix, ensemble member no. "+str(i))
                # data_raw_ensemble[:,:,i] = data_raw_ensemblemember

                print("data_raw_ensemblemember.shape =", data_raw_ensemblemember.shape, flush=True)

            # === Unfold it ===:
            fname_unfolded_current = os.path.join(folder, name+"-unfolded-"+str(i)+".m")
            if os.path.isfile(fname_unfolded_current) and not purge_files:
                unfolded_ensemblemember, tmp, Ex_array_unf, Eg_array_unf = read_mama_2D(fname_unfolded_current)
                print("Read unfolded matrix from file", flush=True)
            else:
                print("Unfolding matrix", flush=True)
                # Unfold:
                unfolded_ensemblemember, Ex_array_unf, Eg_array_unf = unfold(data_raw_ensemblemember, Ex_array, Eg_array, fname_resp_dat, fname_resp_mat, verbose=False, use_comptonsubtraction=False)
                write_mama_2D(unfolded_ensemblemember, fname_unfolded_current, Ex_array_unf, Eg_array_unf, comment="unfolded matrix, ensemble member no. "+str(i))

                print("unfolded_ensemblemember.shape =", unfolded_ensemblemember.shape, flush=True)



            # === Extract first generation spectrum ===: 
            Ex_max = 12000 # keV - maximum excitation energy
            dE_gamma = 1000 # keV - allow gamma energy to exceed excitation energy by this much, to account for experimental resolution
            # N_Exbins = 300
            N_Exbins = len(Ex_array_unf)
            fname_firstgen_current = os.path.join(folder, name+"-firstgen-"+str(i)+".m")
            if os.path.isfile(fname_firstgen_current) and not purge_files:
                firstgen_ensemblemember, tmp, Ex_array_fg, Eg_array_fg = read_mama_2D(fname_firstgen_current)
                print("Read first generation matrix from file", flush=True)
            else:
                print("Calculating first generation matrix", flush=True)
                print("unfolded_ensemblemember.shape =", unfolded_ensemblemember.shape, flush=True)
                # Find first generation spectrum:
                firstgen_ensemblemember, diff, Ex_array_fg, Eg_array_fg = first_generation_spectrum(unfolded_ensemblemember, Ex_array_unf, Eg_array_unf, N_Exbins, Ex_max, dE_gamma, N_iterations=10)
                write_mama_2D(firstgen_ensemblemember, fname_firstgen_current, Ex_array_fg, Eg_array_fg, comment="first generation matrix, ensemble member no. "+str(i))


            firstgen_ensemble[i,0:firstgen_ensemblemember.shape[0],0:firstgen_ensemblemember.shape[1]] = firstgen_ensemblemember


            # TESTING: Plot ensemble of first-gen matrices:
            # i_plt, j_plt = map_iterator_to_grid(i,Nx,Ny)
            # axs_raw[i_plt, j_plt].pcolormesh(Eg_array, Ex_array, data_raw_ensemblemember, norm=LogNorm(vmin=1,vmax=1e3))
            # axs_unf[i_plt, j_plt].pcolormesh(Eg_array_unf, Ex_array_unf, unfolded_ensemblemember, norm=LogNorm(vmin=1,vmax=1e3))
            # axs_fg[i_plt, j_plt].pcolormesh(Eg_array_fg, Ex_array_fg, firstgen_ensemblemember, norm=LogNorm(vmin=1,vmax=1e3))



            # End loop over perturbed ensemble



        # === Calculate variance ===:
        firstgen_ensemble_variance = np.var(firstgen_ensemble, axis=0)
        fname_firstgen_variance = os.path.join(folder, name+"-firstgen_variance.m")
        write_mama_2D(firstgen_ensemble_variance, fname_firstgen_variance, Ex_array_fg, Eg_array_fg, comment="variance of first generation matrix ensemble")



        return True



