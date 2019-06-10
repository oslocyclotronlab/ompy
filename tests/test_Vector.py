# -*- coding: utf-8 -*-
# Test that the Vector class is behaving correctly, 
# including plotting
# TODO convert asserts to unittest system

from context import oslo_method_python as om
import matplotlib.pyplot as plt
import numpy as np

# Set up test array
E_array = np.linspace(-200, 200, 201)
counts = np.random.normal(loc=0.01*E_array, size=len(E_array))
vec = om.Vector(counts, E_array)

# == vec.calibration() ==
cal = vec.calibration()
assert(cal["a0"] == vec.E_array[0])
assert(cal["a1"] == (vec.E_array[1]-vec.E_array[0]))

# == vec.plot() ==
vec.plot()
plt.show()
