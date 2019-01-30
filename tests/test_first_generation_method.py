# -*- coding: utf-8 -*-

from context import oslo_method_python as om
import matplotlib.pyplot as plt

# Test the first_generation_method() and helper functions

fname_unfolded = "/home/jorgenem/MEGA/doktorgrad/oslometode_usikkerhetspropagering/Dy164/data/alfnaun"
# Call it through the MatrixAnalysis class for now,
# but firstgen should be made callable as a standalone function.
ma = om.MatrixAnalysis()

ma.unfolded.load(fname_unfolded)

f2D, (ax2D1, ax2D2) = plt.subplots(1, 2)
ax2D1.plot()
