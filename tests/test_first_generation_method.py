# -*- coding: utf-8 -*-

from context import oslo_method_python as om
import matplotlib.pyplot as plt

# Test the first_generation_method() and helper functions

fname_unfolded = "/home/jorgenem/MEGA/doktorgrad/oslometode_usikkerhetspropagering/Dy164/data/alfnaun"
matrix_in = om.Matrix()
matrix_in.load(fname_unfolded)

f2D, (ax2D1, ax2D2) = plt.subplots(1, 2)
cbar = matrix_in.plot(ax=ax2D1, title="Dy164 unfolded",
                        zscale="log", zmin=1e-3)
f2D.colorbar(cbar, ax=ax2D1)

# Run first generation method
Ex_max = 8500
dE_gamma = 500
matrix_fg = om.first_generation_method(matrix_in, Ex_max=Ex_max, dE_gamma=dE_gamma)
matrix_fg.plot(ax=ax2D2, title="Dy164 first-generation",
                        zscale="log")
f2D.colorbar(cbar, ax=ax2D2)


plt.show()
