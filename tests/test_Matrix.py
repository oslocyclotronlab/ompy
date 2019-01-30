# -*- coding: utf-8 -*-
# Test that the Matrix class is behaving correctly, 
# including plotting
# TODO convert asserts to unittest system

from context import oslo_method_python as om
import matplotlib.pyplot as plt
import numpy as np

decimal = 5  # Precision for np.testing.array_almost_equal

# Set up test array
E0_array = np.linspace(-200, 200, 201)
E1_array = np.linspace(-100, 300, 101)
counts = np.random.normal(loc=0.01*np.meshgrid(E0_array, E1_array,
                                               indexing="ij")[0],
                          size=(len(E0_array), len(E1_array)))
mat = om.Matrix(counts, E0_array, E1_array)

# == mat.calibration() ==
cal = mat.calibration()
assert(cal["a00"] == mat.E0_array[0])
assert(cal["a01"] == (mat.E0_array[1]-mat.E0_array[0]))
assert(cal["a10"] == mat.E1_array[0])
assert(cal["a11"] == (mat.E1_array[1]-mat.E1_array[0]))

# Allocate several subplots for different tests:
f, (ax1, ax2) = plt.subplots(1, 2)

# == mat.plot() ==
cbar = mat.plot(ax=ax1, zscale="linear", title="plot() works")
f.colorbar(cbar, ax=ax1)


# == mat.cut_rect() ==
cut_axis = 0
E0_limits = [E0_array[5], E0_array[-5]]
if E0_limits[1] <= E0_limits[0]:
    E0_limits[1] = E0_limits + cal["a01"]
mat.cut_rect(axis=cut_axis, E_limits=E0_limits, inplace=True)
mat.plot(ax=ax2, zscale="linear", title="cut_rect() works")
# TODO write a test to verify that it worked instead of plotting.

# == show plots ==
plt.show()
