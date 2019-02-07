# -*- coding: utf-8 -*-
from context import oslo_method_python as om
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

# Import raw matrix
fname_raw = "Dy164_raw.m"
# Set up a MatrixAnalysis instance
ma = om.MatrixAnalysis(fname_raw=fname_raw)
print("ma.raw.calibration() =", ma.raw.calibration())

# Do unfolding and firstgen on it:
# TODO put all parameters into MatrixAnalysis() as I go along
fname_resp_mat = "Dy164_response_matrix.m"
fname_resp_dat = "Dy164_response_parameters.dat"
fill_and_remove_negative = True
ma.unfold(fname_resp_mat=fname_resp_mat, fname_resp_dat=fname_resp_dat,
          fill_and_remove_negative=fill_and_remove_negative)
assert(fname_resp_mat == ma.unfold_fname_resp_mat)
assert(fname_resp_dat == ma.unfold_fname_resp_dat)
assert(fill_and_remove_negative == ma.unfold_fill_and_remove_negative)
# Todo implement fill_negative and remove_negative as matrix_analysis
# functions, and add self. parameters saying whether to call them after
# unfold and/or after firstgen


# TODO do asserts between ma self parameters and inputs, to check
# that all settings are correctly picked up and used in unfold() and firstgen()




# === Plotting ===
f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
# Raw:
cbar1 = ma.raw.plot(ax=ax1, title="raw")
f.colorbar(cbar1, ax=ax1)
# Unfolded:
cbar2 = ma.unfolded.plot(ax=ax2, title="unfolded")
f.colorbar(cbar2, ax=ax2)

plt.show()