# -*- coding: utf-8 -*-
# Test that the Matrix class is behaving correctly, 
# including plotting
# TODO convert asserts to unittest system

from context import oslo_method_python as om
import matplotlib.pyplot as plt
import numpy as np

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

# == mat.plot() ==
f, ax = plt.subplots(1,1)
cbar = mat.plot(ax=ax, zscale="linear", title="plot title")
f.colorbar(cbar, ax=ax)
plt.show()
