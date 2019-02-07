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


# TODO do asserts between ma self parameters and inputs, to check
# that all settings are correctly picked up and used in unfold() and firstgen()




# === Plotting ===
f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
ma.raw.plot(ax=ax1, title="raw")

plt.show()