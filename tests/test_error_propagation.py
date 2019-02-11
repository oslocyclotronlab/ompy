# -*- coding: utf-8 -*-
from context import oslo_method_python as om
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

# Import raw matrix
fname_raw = "Dy164_raw.m"
# Set up a MatrixAnalysis instance
ma = om.MatrixAnalysis(fname_raw=fname_raw)
print("ma.raw.calibration() =", ma.raw.calibration())

# Do unfolding on it:
fname_resp_mat = "Dy164_response_matrix.m"
fname_resp_dat = "Dy164_response_parameters.dat"
unfold_fill_and_remove_negative = True
unfold_verbose = False
ma.unfold(fname_resp_mat=fname_resp_mat, fname_resp_dat=fname_resp_dat,
          fill_and_remove_negative=unfold_fill_and_remove_negative)
# BEGIN unfolding asserts
assert(fname_resp_mat == ma.unfold_fname_resp_mat)
assert(fname_resp_dat == ma.unfold_fname_resp_dat)
assert(unfold_fill_and_remove_negative == ma.unfold_fill_and_remove_negative)
assert(unfold_verbose == ma.unfold_verbose)
# END unfolding asserts

# Do first generation method on it:
Ex_max = 12000
dE_gamma = 500
N_iterations = 5
fill_and_remove_negative = True
fg_fill_and_remove_negative = True
multiplicity_estimation = "statistical"
apply_area_correction = False
fg_verbose = False
ma.first_generation_method(Ex_max=Ex_max,
                           dE_gamma=dE_gamma,
                           N_iterations=N_iterations,
                           fill_and_remove_negative=fg_fill_and_remove_negative,
                           multiplicity_estimation=multiplicity_estimation,
                           apply_area_correction=apply_area_correction,
                           verbose=fg_verbose
                           )

# BEGIN first generation method asserts
assert(Ex_max == ma.fg_Ex_max)
assert(dE_gamma == ma.fg_dE_gamma)
assert(N_iterations == ma.fg_N_iterations)
assert(multiplicity_estimation == ma.fg_multiplicity_estimation)
assert(apply_area_correction == ma.fg_apply_area_correction)
assert(fg_verbose == ma.fg_verbose)
# END first generation method asserts


# === Run error propagation ===
# Instantiate the propagation class with MatrixAnalysis instance ma:
ep = om.ErrorPropagation(ma,
                         folder="error_propagation_ensemble",
                         randomness="poisson",
                         seed=73)

N_ensemble_members = 2
ep.generate_ensemble(N_ensemble_members = N_ensemble_members,
                     verbose=True,
                     purge_files=True)

import sys
sys.exit(0)

# === Plotting ===
f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
# Raw:
cbar1 = ma.raw.plot(ax=ax1, title="raw")
f.colorbar(cbar1, ax=ax1)
# Unfolded:
cbar2 = ma.unfolded.plot(ax=ax2, title="unfolded")
f.colorbar(cbar2, ax=ax2)
# Firstgen:
cbar3 = ma.firstgen.plot(ax=ax3, title="first generation")

plt.show()