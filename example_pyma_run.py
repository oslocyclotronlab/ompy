import numpy as np 
import matplotlib.pyplot as plt

# # Import the pyma modules:
# from pymama import pymama
# import pyma_matrix as pmmat
# import pyma_lib as pml
# from pyma_fit import pyma_fit
# from pyma_mc import pyma_mc

import sys
sys.path.insert(0,"/home/jorgenem/gitrepos") # Needed to make Python look for this module in the directory above
import pyma




# Initialise pymama for current experiment
fname_raw = "data/alfna-Re187.m"
fname_resp_mat = "data/response_matrix-Re187-10keV.m"
fname_resp_dat = "data/resp-Re187-10keV.dat"
pma = pyma.pymama(fname_resp_mat=fname_resp_mat, fname_resp_dat=fname_resp_dat)

# Load raw matrix
pma.raw.load(fname_raw)

# Check that it has loaded a sensible raw matrix:
print(pma.raw.matrix.shape)



# Plot it
# pma.raw.plot(title="raw")

# Do unfolding: 
# pma.unfold(fname_resp_mat, fname_resp_dat, use_comptonsubtraction=False, verbose=True, plot=False) # Call unfolding routine


# Save the unfolded matrix:
fname_unfolded = "data/unfolded-Re187.m"
# pma.unfolded.save(fname_save_unfolded)

# Load the unfolded matrix from file:
pma.unfolded.load(fname_unfolded)


# # Run first generation method:
pma.N_Exbins_fg = pma.unfolded.matrix.shape[0] # Take all bins
pma.Ex_max_fg = pma.unfolded.Ex_array[-1] - 2000 # TODO figure out if this is needed and how it relates to max Eg
pma.dEg_fg = 1000 # keV
# pma.first_generation(N_Exbins=N_Exbins_fg, Ex_max=Ex_max_fg, dE_gamma=dEg_fg)
pma.first_generation_method()

# # Plot first generation matrix
# pma.firstgen.plot(title="first generation")

# # Save it
fname_firstgen = "data/firstgen-Re187.m"
# pma.firstgen.save(fname_firstgen)

# Load it
pma.firstgen.load(fname_firstgen)


# Run the error propagation module to get the variance matrix
folder = "ensemble_Re187"
N_members = 200
pmc = pyma.mc(pma, folder=folder, randomness="gaussian", seed=42)
var_firstgen = pmc.generate_ensemble(N_members=N_members, verbose=True, purge_files=False, use_comptonsubtraction=False)



# Fit T and rho
Eg_min = 1000
Ex_min = 3000
Ex_max = 6700 # keV
pmf = pyma.fit(pma.firstgen, var_firstgen)

print("firstgen variance matrix: var_firstgen.matrix.max() =", var_firstgen.matrix.max(), flush=True)

rho, T, Ex_array, Eg_array = pmf.fit(Eg_min, Ex_min, Ex_max, estimate_variance_matrix=False)

plt.plot(rho, label="rho")
plt.plot(T, label="T")
plt.legend()
plt.show()