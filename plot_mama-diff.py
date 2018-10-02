from unfold import *
import matplotlib.pyplot as plt 
import sys


f, ax = plt.subplots(1,1)

fname1 = "mama-testing/unfolded-si28.m"
# fname = "data/alfna-Re187.m"
# fname1 = "mama-testing/firstgen-si28.m"
matrix1, cal1, Ex_array1, Eg_array1 = read_mama_2D(fname1)
fname2 = "python_unfolded-28Si.m"
matrix2, cal2, Ex_array2, Eg_array2 = read_mama_2D(fname2)

from matplotlib.colors import LogNorm
cbar = ax.pcolormesh(Eg_array2, Ex_array2, np.abs(matrix1[matrix2.shape] - matrix2), norm=LogNorm(vmin=1e-1))
f.colorbar(cbar, ax=ax)


plt.show()