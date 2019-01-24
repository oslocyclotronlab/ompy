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

calib_out = {"a0": 0, "a1": 120}
Ex_min = firstgen.E0_array[0]
Ex_max = 8000
Eg_min = Ex_min
print("Ex_min =", Ex_min)
print("Ex_max =", Ex_max)
print("Eg_min =", Eg_min)

rho, T = om.fit_rho_T(firstgen, calib_out,
                      Ex_min, Ex_max, Eg_min)


# Plot
f, (axrho, axT, axgsf) = plt.subplots(1, 3)
axrho.set_title("rho")
rho.plot(ax=axrho, yscale="log")
axT.set_title("T")
T.plot(ax=axT, yscale="log")
# Also plot gsf, i.e. T/Eg^3
gsf = om.div0(T.vector, T.E_array**3)
axgsf.plot(T.E_array, gsf)
axgsf.set_title("gsf")
axgsf.set_yscale("log")
plt.show()
