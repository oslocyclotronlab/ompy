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



# === Begin plotting ===

f2D, ((ax_P_true, ax_P_fit), (ax_P_diff, ax2D_4)) = plt.subplots(2, 2)
firstgen_cut = firstgen.cut_rect(
                              axis="both",
                              E_limits=[Ex_min, Ex_max, Eg_min, Ex_max],
                              inplace=False
                              )
cbar = ax_P_true.pcolormesh(firstgen_cut.E1_array,
                            firstgen_cut.E0_array,
                            firstgen_cut.matrix,
                            norm=LogNorm())
ax_P_true.set_title("P_true")
f2D.colorbar(cbar, ax=ax_P_true)



f1D, (ax_rho, ax_T) = plt.subplots(1, 2)
rho.plot(ax=ax_rho, label="fit")
ax_rho.legend()
ax_rho.set_yscale("log")

T.plot(ax=ax_T, label="fit")
# ax_T.plot(Emid_Eg, T.transform(const=2e2, alpha=0.002),
          # label="fit, transformed")
ax_T.legend()
ax_T.set_yscale("log")

# Run the cut again just to get energy arrays for plotting:
pars_fg = {"Egmin" : Eg_min,
           "Exmin" : Ex_min,
           "Emax" : Ex_max}
E_array_midbin = E_array_out + bin_width_out/2
tmp, Emid_Eg, Emid_Ex, Emid_nld = om.fg_cut_matrix(firstgen.matrix,
                                                        E_array_midbin, **pars_fg)

P_fit = om.PfromRhoT(rho.vector, T.vector, len(Emid_Ex),
                     Emid_Eg, Emid_nld, Emid_Ex)
cbar = ax_P_fit.pcolormesh(Emid_Eg, Emid_Ex, P_fit, norm=LogNorm())
ax_P_fit.set_title("P_fit")
f2D.colorbar(cbar, ax=ax_P_fit)

# P_diff = P_fit - P_true
# cbar = ax_P_diff.pcolormesh(E_array, E_array, P_diff)
# f2D.colorbar(cbar, ax=ax_P_diff)

# Print some quantities
# print("P_diff.max() =", P_diff.max())
print("rho (fitted) =", rho.vector)
print("T (fitted) =", T.vector)


plt.show()