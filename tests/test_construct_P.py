# -*- coding: utf-8 -*-

from context import oslo_method_python as om
import matplotlib.pyplot as plt
import numpy as np



Ex_max = 2000
calib_out = {"a0": -500, "a1": 120}

# Set up the energy array common to rho and T
E_array = om.E_array_from_calibration(a0=calib_out["a0"],
                                   a1=calib_out["a1"],
                                   E_max=Ex_max)
Nbins = len(E_array)

# T_array = np.random.uniform(low=1.8, high=2, size=Nbins)
# rho_array = np.random.normal(loc=E_array, size=Nbins, scale=0.1)
rho_array = np.exp(0.01*E_array)
# rho_array = np.zeros(Nbins)
# rho_array[0:3] = 1
T_array = 0.001*E_array**2

P_theo = om.construct_P(rho_array, T_array, E_array)

f, ax = plt.subplots(1,1)
from matplotlib.colors import LogNorm
cbar = ax.pcolormesh(E_array, E_array, P_theo, norm=LogNorm())
f.colorbar(cbar, ax=ax)
plt.show()