import numpy as np 
import matplotlib.pyplot as plt

# Import the pyma modules:
from pymama import pymama
import pyma_matrix as pmmat
import pyma_lib as pml
from pyma_fit import pyma_fit
from pyma_mc import pyma_mc


# Initialise pyma for current experiment
fname_raw = "data/alfna-Re187.m"
pm = pymama()

# Load raw matrix
pm.raw.load(fname_raw)

# Check that it has loaded a sensible raw matrix:
print(pm.raw.matrix.shape)



# Plot it
# pm.raw.plot(title="raw")

# Do unfolding: 
# fname_resp_mat = "data/response_matrix-Re187-10keV.m"
# fname_resp_dat = "data/resp-Re187-10keV.dat"
# pm.unfold(fname_resp_mat, fname_resp_dat, use_comptonsubtraction=False, verbose=True, plot=True) # Call unfolding routine


# Save the unfolded matrix:
fname_unfolded = "data/unfolded-Re187.m"
# pm.unfolded.save(fname_save_unfolded)

# Load the unfolded matrix from file:
pm.unfolded.load(fname_unfolded)


# # Run first generation method:
pm.N_Exbins_fg = pm.unfolded.matrix.shape[0] # Take all bins
pm.Ex_max_fg = pm.unfolded.Ex_array[-1] - 2000 # TODO figure out if this is needed and how it relates to max Eg
pm.dEg_fg = 1000 # keV
# pm.first_generation(N_Exbins=N_Exbins_fg, Ex_max=Ex_max_fg, dE_gamma=dEg_fg)
pm.first_generation_method()

# # Plot first generation matrix
# pm.firstgen.plot(title="first generation")

# # Save it
fname_firstgen = "data/firstgen-Re187.m"
# pm.firstgen.save(fname_firstgen)

# Load it
pm.firstgen.load(fname_firstgen)


# Run the error propagation module to get the variance matrix
folder = "ensemble_Re187"
N_members = 1
pmc = pyma_mc(pm, folder=folder, randomness="gaussian", seed=42)
pmc.generate_ensemble(N_members=N_members, verbose=True, purge_files=True)



# Fit T and rho
Eg_min = 1000
Ex_min = 3000
Ex_max = 6700 # keV
rho, T, Ex_array, Eg_array = pm.fit(Eg_min, Ex_min, Ex_max, estimate_variance_matrix=False)

plt.plot(rho, label="rho")
plt.plot(T, label="T")
plt.legend()
plt.show()