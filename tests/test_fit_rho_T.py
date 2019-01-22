# -*- coding: utf-8 -*-

from context import oslo_method_python as om
import matplotlib.pyplot as plt

rho, T = om.fit_rho_T(firstgen, firstgen_stdev, calib_out,
                      Ex_min, Ex_max, Eg_min)

# Plot
f, (axrho, axT) = plt.subplots(1, 2)
rho.plot(ax=axrho)
T.plot(ax=axT)
