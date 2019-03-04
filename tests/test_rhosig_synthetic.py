# -*- coding: utf-8 -*-
from context import oslo_method_python as om
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
import copy


Ex_min = 0
Ex_max = 2000
Eg_min = 0
bin_width_out = 120

calib_out = {"a0": -500, "a1": bin_width_out}

# Set up the energy array common to rho and T
E_array = om.E_array_from_calibration(a0=calib_out["a0"],
                                      a1=calib_out["a1"],
                                      E_max=Ex_max)
Nbins = len(E_array)
print("Nbins = ", Nbins, flush=True)

# Define true rho and T to construct the P_true
# T_array = np.random.uniform(low=1.8, high=2, size=Nbins)
rho_true = np.exp(1e-4*E_array) + 10
# rho_true = 100 + (0.01*E_array)**3
T_true = 0.001*E_array**2 + 0.01*E_array + 100

# Add some noise?:
# rho_true = np.random.normal(loc=rho_true, size=Nbins, scale=0.01)
# T_true = np.random.normal(loc=T_true, size=Nbins, scale=0.01)

P_true = om.construct_P(rho_true, T_true, E_array)
firstgen = om.Matrix(matrix=P_true, E0_array=E_array, E1_array=E_array)

# Error matrix:
# Use ones as error matrix, which should in practice turn off error weighting
firstgen_std = copy.deepcopy(firstgen)
firstgen_std.matrix = np.ones(firstgen.matrix.shape)
# OR: Use the square root of the number of counts:
# firstgen_std = copy.deepcopy(firstgen)
# firstgen_std.matrix = np.sqrt(firstgen.matrix)

f2D, ((ax_P_true, ax_P_fit), (ax_P_diff, ax2D_4)) = plt.subplots(2, 2)
cbar = ax_P_true.pcolormesh(E_array, E_array, P_true, norm=LogNorm())
ax_P_true.set_title("P_true")
f2D.colorbar(cbar, ax=ax_P_true)

E_array_midbin = E_array + calib_out["a1"]/2
rho_fit, T_fit = om.decompose_matrix(firstgen.matrix, firstgen_std.matrix,
                    E_array_midbin, E_array_midbin, E_array_midbin,
                    method="Powell")


rho = om.Vector(rho_fit, E_array)
T = om.Vector(T_fit, E_array)


f1D, (ax_rho, ax_T) = plt.subplots(1, 2)
rho.plot(ax=ax_rho, label="fit")
ax_rho.plot(E_array+calib_out["a1"]/2, rho_true, label="truth")
ax_rho.legend()
ax_rho.set_yscale("log")

T.plot(ax=ax_T, label="fit")
ax_T.plot(E_array+calib_out["a1"]/2, T.transform(const=2e2, alpha=0.002),
          label="fit, transformed")
ax_T.plot(E_array+calib_out["a1"]/2, T_true, label="truth")
ax_T.legend()
ax_T.set_yscale("log")


P_fit = om.PfromRhoT(rho.vector, T.vector, len(E_array),
                     E_array_midbin, E_array_midbin, E_array_midbin)
cbar = ax_P_fit.pcolormesh(E_array, E_array, P_fit, norm=LogNorm())
ax_P_fit.set_title("P_fit")
f2D.colorbar(cbar, ax=ax_P_fit)

P_diff = P_fit - P_true
cbar = ax_P_diff.pcolormesh(E_array, E_array, P_diff)
f2D.colorbar(cbar, ax=ax_P_diff)

# Print some quantities
print("P_diff.max() =", P_diff.max())
print("rho (fitted) =", rho.vector)
print("T (fitted) =", T.vector)


plt.show()
