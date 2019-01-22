# -*- coding: utf-8 -*-
# Test that the Vector class is behaving correctly, 
# including plotting
# TODO convert asserts to unittest system

from context import oslo_method_python as om
import matplotlib.pyplot as plt
import numpy as np

# Set up test array
E_array = np.linspace(-501, 3020, 101)
counts = np.random.normal(loc=0.01*E_array, size=len(E_array))
vec = om.Vector(counts, E_array)

# Test rebinning on a simple case where neighbouring
# bins are joined
E_array_out = E_array[::2]

print(E_array)
print(E_array_out)

counts_rebinned = om.rebin(counts, E_array, E_array_out)
print(counts)
print(counts_rebinned)