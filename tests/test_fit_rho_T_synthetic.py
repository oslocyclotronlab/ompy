# -*- coding: utf-8 -*-
from context import oslo_method_python as om
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm


Ex_min = 500
Ex_max = 2000
Eg_min = 0
bin_width_out = 120

calib_out = {"a0": -500, "a1": bin_width_out}

# Set up the energy array common to rho and T
E_array = om.E_array_from_calibration(a0=calib_out["a0"],
                                      a1=calib_out["a1"],
                                      E_max=Ex_max)
Nbins = len(E_array)

# Define true rho and T to construct the P_true
# T_array = np.random.uniform(low=1.8, high=2, size=Nbins)
# rho_array = np.random.normal(loc=E_array, size=Nbins, scale=0.1)
# rho_true = np.exp(1e-6*E_array)
rho_true = 100 + (0.01*E_array)**3
T_true = 0.001*E_array**2 + 0.01*E_array + 10

P_true = om.construct_P(rho_true, T_true, E_array)
firstgen = om.Matrix(matrix=P_true, E0_array=E_array, E1_array=E_array)
# Use ones as error matrix, which should in practice turn off error weighting
firstgen.std = np.ones(firstgen.matrix.shape)

f2D, ((ax_P_true, ax_P_fit), (ax_P_diff, ax2D_4)) = plt.subplots(2, 2)
cbar = ax_P_true.pcolormesh(E_array, E_array, P_true, norm=LogNorm())
ax_P_true.set_title("P_true")
f2D.colorbar(cbar, ax=ax_P_true)


rho, T = om.fit_rho_T(firstgen, bin_width_out,
                      Ex_min, Ex_max, Eg_min)


f1D, (ax_rho, ax_T) = plt.subplots(1, 2)
rho.plot(ax=ax_rho, label="fit")
ax_rho.plot(E_array+calib_out["a1"]/2, rho_true, label="truth")
ax_rho.legend()
ax_rho.set_yscale("log")

T.plot(ax=ax_T, label="fit")
ax_T.plot(E_array+calib_out["a1"]/2, T_true, label="truth")
ax_T.legend()
ax_T.set_yscale("log")


P_fit = om.construct_P(rho_true, T_true, E_array)
cbar = ax_P_fit.pcolormesh(E_array, E_array, P_fit, norm=LogNorm())
ax_P_fit.set_title("P_fit")
f2D.colorbar(cbar, ax=ax_P_fit)

cbar = ax_P_diff.pcolormesh(E_array, E_array, P_fit-P_true)
f2D.colorbar(cbar, ax=ax_P_diff)


plt.show()
