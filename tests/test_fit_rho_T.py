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
firstgen.load("../firstgen-28Si.m")
firstgen.std = np.sqrt(firstgen.matrix)  # In lieu of something better

calib_out = {"a0": -20, "a1": 30}
Ex_min = firstgen.E0_array[0]
Ex_max = firstgen.E0_array[-1]
Eg_min = Ex_min

rho, T = om.fit_rho_T(firstgen, calib_out,
                      Ex_min, Ex_max, Eg_min)


# Plot
f, (axrho, axT) = plt.subplots(1, 2)
rho.plot(ax=axrho)
T.plot(ax=axT)
plt.show()
