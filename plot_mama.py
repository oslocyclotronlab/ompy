from unfold import *
import matplotlib.pyplot as plt 
import sys


f, ax = plt.subplots(1,1)

fname = "mama-testing/unfolded-si28.m"
# fname = "mama-testing/firstgen-si28.m"
# fname = "data/alfna-Re187.m"
# fname = "python_unfolded-28Si.m"
matrix, cal, Ex_array, Eg_array = read_mama_2D(fname)

from matplotlib.colors import LogNorm
cbar = ax.pcolormesh(Eg_array, Ex_array, matrix, norm=LogNorm(vmin=1e0))#, vmax=1e4))
f.colorbar(cbar, ax=ax)


plt.show()