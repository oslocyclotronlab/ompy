from unfold import *
import matplotlib.pyplot as plt 
import sys


f, (ax_mat, ax_proj) = plt.subplots(2,1)

# fname = "data/alfna-Si28-14keV.m"
# fname = "mama-testing/unfolded-si28.m"
fname = "mama-testing/unfolded-si28-14keV.m"
# fname = "mama-testing/firstgen-si28.m"
# fname = "data/alfna-Re187.m"
# fname = "data/response_matrix-Si28-7keV.m"
# fname = "python_unfolded-28Si.m"
# fname = "python_unfolded-28Si-14keV-roounfold.m"
# fname = "python_unfolded-28Si-14keV-1p0FWHM-roounfold.m"
fname = "python_unfolded-28Si-14keV.m"

matrix, cal, Ex_array, Eg_array = read_mama_2D(fname)

from matplotlib.colors import LogNorm
ax_mat.pcolormesh(Eg_array, Ex_array, matrix, norm=LogNorm(vmin=1e0, vmax=1e4), cmap="jet")

ax_mat.set_title("fname = "+fname)

# Project down on Eg axis for a chosen Ex range
i_proj_low = 255
i_proj_high = 256


ax_proj.plot(Eg_array, matrix[i_proj_low:i_proj_high,:].sum(axis=0), label="proj {:.2f} to {:.2f}".format(Ex_array[i_proj_low],Ex_array[i_proj_high]))
ax_proj.legend()

ax_proj.set_title("Total number of counts in proj. = {:.2f}".format(matrix[i_proj_low:i_proj_high,:].sum(), flush=True))
f.subplots_adjust(hspace=0.3)

plt.show()