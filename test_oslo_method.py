import numpy as np
from oslo_method import oslo_method

# Give location of response matrix and other necessary response data:
fname_resp_mat = "data/response_matrix-Re187-10keV.m"
fname_resp_dat = "data/resp-Re187-10keV.dat"

# Create instance of main class:
om = oslo_method(fname_resp_mat, fname_resp_dat)

# Load raw matrix:
fname_raw = "data/alfna-Re187.m"
om.raw.load(fname_raw)

# TODO find a better system for setting the parameters for om.base_analysis.

# Run unfolding and save result
om.base_analysis.use_comptonsubtraction = False
om.unfold()
fname_unfolded = "data/unfolded-Re187.m"
om.unfolded.save(fname_unfolded)

# Run first generation method and save result:
om.base_analysis.N_Exbins_fg = om.base_analysis.unfolded.matrix.shape[0] # Take all bins
om.base_analysis.Ex_max_fg = om.base_analysis.unfolded.Ex_array[-1] - 2000 # TODO figure out if this is needed and how it relates to max Eg
om.base_analysis.dEg_fg = 1000 # keV
om.first_generation_method()
fname_firstgen = "data/firstgen-Re187.m"
om.firstgen.save(fname_firstgen)


fname_ensemble_folder = "ensemble_Re187_poisson"
om.setup_error_propagation()
N_ensemble_members = 2
om.propagate_errors(N_ensemble_members=N_ensemble_members, purge_files=True)

# TODO figure out why the first_generation_method() prints the exact same max_diff for all the perturbation members...


om.var_firstgen.plot()
plt.show()

