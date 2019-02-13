# -*- coding: utf-8 -*-

from context import oslo_method_python as om
import matplotlib.pyplot as plt
import numpy as np
import copy
from matplotlib.colors import LogNorm

firstgen = om.Matrix()
# TODO: Make a test matrix (could be random numbers)
# to ship with the package, to use for unittesting
# on unfolding, firstgen method and here.
# Should probably save all steps of the analysis (raw,
# unfolded, firstgen) as separate matrices.
# firstgen.load("../firstgen-28Si.m")
# Testing with a proper firstgen for a heavy nucleus (development):
# firstgen.load("/home/jorgenem/MEGA/doktorgrad/oslometode_usikkerhetspropagering/Dy164/data/fg")

firstgen.load("error_propagation_ensemble/firstgen-orig.m")

# Set up firstgen_std:
firstgen_std = copy.deepcopy(firstgen)
firstgen_std.load("error_propagation_ensemble/firstgen_std.m")
# firstgen_std.matrix = np.ones(firstgen.matrix.shape)
# firstgen_std.matrix = np.sqrt(firstgen.matrix)  # In lieu of something better
# print("firstgen.std =", firstgen.std)

bin_width_out = 200
Ex_min = 2000
Ex_max = 8000
Eg_min = 1000

E_array_out = om.E_array_from_calibration(a0=-500,
                                          a1=bin_width_out,
                                          E_max=Ex_max)

rho, T = om.fit_rho_T(firstgen, firstgen_std, bin_width_out,
                      Ex_min, Ex_max, Eg_min,
                      method="Powell",
                      negatives_penalty=1e10)


# Plot
f, ((axrho, axT), (axgsf, axdiw)) = plt.subplots(2, 2)
rho.plot(ax=axrho, yscale="log", title="rho")
axT.set_title("T")
T.plot(ax=axT, yscale="log", title="T")
# Also plot gsf, i.e. T/Eg^3
gsf = om.div0(T.vector, T.E_array**3)
axgsf.plot(T.E_array, gsf)
axgsf.set_title("gsf")
axgsf.set_yscale("log")

# Plot a projection onto the gamma-energy axis of the first generation
# matrix and the fitted firstgen matrix (similar to "does it work"):


firstgen.plot()

E_limits = [5000, 6000]

# Make rebinned firstgen matrix
firstgen_rebinned_matrix = om.rebin_matrix(firstgen.matrix, firstgen.E0_array,
                                        E_array_out, rebin_axis=0)
firstgen_rebinned_matrix = om.rebin_matrix(firstgen_rebinned_matrix,
                                     firstgen.E1_array,
                                     E_array_out, rebin_axis=1)
firstgen_rebinned = om.Matrix(firstgen_rebinned_matrix, E_array_out, E_array_out)
firstgen_rebinned.plot_projection(E_limits=E_limits, axis=1,
                         ax=axdiw, label="exp", normalize=True)

# # DEBUG
# # i_E_low = om.i_from_E(E_limits[0], firstgen.E1_array)
# # i_E_high = om.i_from_E(E_limits[1], firstgen.E1_array)
# i_E_low = np.argmin(np.abs(E_limits[0] - firstgen.E0_array))
# i_E_high = np.argmin(np.abs(E_limits[1] - firstgen.E0_array))

# print(i_E_low, i_E_high)
# axdiw.plot(firstgen.E1_array,
#            firstgen.matrix[i_E_low:i_E_high, :].sum(axis=0),
#            label="exp manual"
#            )
# print(firstgen.matrix[i_E_low:i_E_high, :])
# END DEBUG

P_fit = om.construct_P(rho.vector, T.vector, rho.E_array)
P_fit = P_fit / P_fit.sum(axis=1)  # Normalize to unity
P_fit = om.Matrix(P_fit, rho.E_array, rho.E_array)
P_fit.plot_projection(E_limits=E_limits, axis=1,
                      ax=axdiw, label="fit", normalize=True)



# === And plot the P matrices of constructed and fitted ===
f_P, ((axPexp, axPfit), (axPdiff, axP4)) = plt.subplots(2,2)

P_exp = om.div0(firstgen_rebinned.matrix,
                np.sum(firstgen_rebinned.matrix, axis=1)[:, None])
axPexp.pcolormesh(E_array_out, E_array_out, P_exp, norm=LogNorm())

axPfit.pcolormesh(E_array_out, E_array_out, P_fit.matrix, norm=LogNorm())



plt.show()
