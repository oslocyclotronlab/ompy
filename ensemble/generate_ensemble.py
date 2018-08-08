import numpy as np 
import matplotlib.pyplot as plt 
import sys
sys.path.insert(0, "../")
from firstgen import *
from unfold import *




# Read raw matrix
fname_data_raw = '../alfna28si.m'
data_raw, cal, Ex_array, Eg_array = read_mama(fname_data_raw)

# TODO implement rebinning before calling unfold()
# The optimal solution is to construct the response
# matrix here.
fname_resp_mat = '../response-si28-20171112.m'
fname_resp = '../resp.dat'



N_stat = 1 # How many perturbed copies do we want in our ensemble?
data_raw_ensemble = np.empty(np.append(data_raw.shape,N_stat))
firstgen_ensemble = np.empty(np.append(firstgen_matrix.shape,N_stat))

np.random.seed(2)
for i in range(N_stat):
	# matrix_ensemble_current = np.maximum(matrix + np.random.normal(size=matrix_shape)*np.sqrt(matrix), np.zeros(matrix_shape)) # Each bin of the matrix is perturbed with a gaussian centered on the bin count, with standard deviation sqrt(bin count). Also, no negative counts are accepted.
	data_raw_ensemble_current = data_raw + np.random.normal(size=data_raw.shape)*np.sqrt(np.where(data_raw > 0, data_raw, 0)) # Assuming sigma \approx n^2 / N where n is current bin count and N is total count, according to sigma^2 = np(1-p) for normal approx. to binomial distribution.
	data_raw_ensemble_current[data_raw_ensemble_current < 0] = 0
	data_raw_ensemble[:,:,i] = data_raw_ensemble_current

	# Unfold:
	unfolded, Ex_array_unf, Eg_array_unf = unfold(data_raw, Ex_array, Eg_array, fname_resp, fname_resp_mat)
	Ex_max = 12000 # keV - maximum excitation energy
	dE_gamma = 300 # keV - allow gamma energy to exceed excitation energy by this much, to account for experimental resolution
	N_Exbins = 300

	# First generation spectrum:
	firstgen, diff, Ex_array_fg, Eg_array_fg = first_generation_spectrum(unfolded, Eg_array_unf, Ex_array_unf, N_Exbins, Ex_max, dE_gamma, N_iterations=20)

	print("iteration i =", i)
