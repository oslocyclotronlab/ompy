import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.colors import LogNorm
import sys
sys.path.insert(0, "../")
sys.path.insert(0, "../../rhosig.py/")
from firstgen import *
from unfold import *
import rhosig as rsg # Fabio's rhosig implementation





fn_firstgen = "../firstgen-28Si.m"
firstgen, cal, Ex_array, Eg_array = read_mama(fn_firstgen)
plt.pcolormesh(Eg_array, Ex_array, firstgen, norm=LogNorm(vmin=1))


# Rebin to square grid
Nbins = 200
# firstgen_rebinned, cal_rebinned, Ex_rebinned, Eg_rebinned = rebin_and_shift(firstgen, Ex_array, Nbins, rebin_axis=0)

# plt.pcolormesh(Ex_rebinned, Eg_rebinned, firstgen_rebinned)

plt.show()

sys.exit(0)

rsg.decompose_matrix(P_in=firstgen, Emid=Emid, fill_value=1e-1)