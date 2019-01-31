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
cbar = ma.unfolded.plot(ax=ax2D1, title="Dy164 unfolded",
                        zscale="log", zmin=1e-3)
f2D.colorbar(cbar, ax=ax2D1)

# Run first generation method
Ex_max = 8500
dE_gamma = 500
ma.first_generation_method(Ex_max=Ex_max, dE_gamma=dE_gamma)
ma.firstgen.plot(ax=ax2D2, title="Dy164 first-generation",
                        zscale="log")
f2D.colorbar(cbar, ax=ax2D2)


plt.show()
