# -*- coding: utf-8 -*-

from context import oslo_method_python as om
import matplotlib.pyplot as plt
import numpy as np

firstgen = om.Matrix()
# TODO: Make a test matrix (could be random numbers)
# to ship with the package, to use for unittesting
# on unfolding, firstgen method and here.
# Should probably save all steps of the analysis (raw,
# unfolded, firstgen) as separate matrices.
# firstgen.load("../firstgen-28Si.m")
# Testing with a proper firstgen for a heavy nucleus (development):
firstgen.load("/home/jorgenem/MEGA/doktorgrad/oslometode_usikkerhetspropagering/Dy164/data/fg")
firstgen.std = np.sqrt(firstgen.matrix)  # In lieu of something better
# print("firstgen.std =", firstgen.std)

bin_width_out = 120
Ex_min = 4000
Ex_max = 8000
Eg_min = 1000
print("Ex_min =", Ex_min)
print("Ex_max =", Ex_max)
print("Eg_min =", Eg_min)

rho, T = om.fit_rho_T(firstgen, bin_width_out,
                      Ex_min, Ex_max, Eg_min)


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


firstgen.plot_projection(E_limits=E_limits, axis=1,
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
# # END DEBUG

P_fit = om.construct_P(rho.vector, T.vector, rho.E_array)
P_fit = P_fit / P_fit.sum(axis=1)  # Normalize to unity
P_fit = om.Matrix(P_fit, rho.E_array, rho.E_array)
P_fit.plot_projection(E_limits=E_limits, axis=1,
                      ax=axdiw, label="fit", normalize=True)

plt.show()
