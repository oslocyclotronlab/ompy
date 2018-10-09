from unfold import *
import numpy as np 
import matplotlib.pyplot as plt 

import rhosigchi_f2py_importvar as rsc 

# fname_fg = "firstgen-Re187.m"
fname_fg = "ensemble/ensemble_si28_14keV/si28-firstgen-orig.m"
fname_varfg = "ensemble/ensemble_si28_14keV/si28-firstgen-variance.m"

fg_in, cal_fg, Ex_array_fg, Eg_array_fg = read_mama_2D(fname_fg)
varfg_in, cal_varfg, Ex_array_varfg, Eg_array_varfg = read_mama_2D(fname_fg)

fg_in, E_array_rebinned = rebin_new(rebin_new(fg_in, Ex_array_fg, 512, rebin_axis=0), Eg_array_fg, 512, rebin_axis=1)
varfg_in, E_array_rebinned = rebin_new(rebin_new(varfg_in, Ex_array_fg, 512, rebin_axis=0), Eg_array_fg, 512, rebin_axis=1)




calib = np.array([cal_fg["a0x"], cal_fg["a1x"], cal_fg["a0y"], cal_fg["a1y"]])
eg_min = 1000 # keV
ex_min = 5000
ex_max = 10000
np.random.seed(2)
# varfg_in = np.random.uniform(low=0.1, high=0.2, size=(fg_in.shape)) # Variance matrix, make from ensemble run


rho, T = rsc.rhosigchi(fg_in,varfg_in,calib,eg_min,ex_min,ex_max)


f, (ax_rho, ax_T) = plt.subplots(2,1)
ax_rho.plot(rho)
ax_rho.set_title("rho")
ax_rho.set_yscale("log")
ax_T.plot(T)
ax_T.set_title("T")
ax_T.set_yscale("log")


plt.show()


