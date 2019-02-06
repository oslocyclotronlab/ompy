# -*- coding: utf-8 -*-

from context import oslo_method_python as om
import matplotlib.pyplot as plt

# Test the first_generation_method() and helper functions

fname_unfolded = "/home/jorgenem/MEGA/doktorgrad/oslometode_usikkerhetspropagering/Dy164/data/alfnaun"
matrix_in = om.Matrix()
matrix_in.load(fname_unfolded)

f2D, ((ax2D1, ax2D2), (ax2D3, ax2D4)) = plt.subplots(2, 2)
cbar = matrix_in.plot(ax=ax2D1, title="Dy164 unfolded",
                      zscale="log", zmin=1e-3, zmax=5e3)
f2D.colorbar(cbar, ax=ax2D1)

# Run first generation method
Ex_max = 14000
dE_gamma = 500
N_iterations = 10
matrix_fg = om.first_generation_method(matrix_in, Ex_max=Ex_max,
                                       dE_gamma=dE_gamma,
                                       N_iterations=N_iterations,
                                       )
cbar = matrix_fg.plot(ax=ax2D2, title="firstgen om python", zscale="log",
                      zmin=1e-3, zmax=5e3)
f2D.colorbar(cbar, ax=ax2D2)


# Load MAMA-firstgen to compare
fname_fg_mama = "/home/jorgenem/MEGA/doktorgrad/oslometode_usikkerhetspropagering/Dy164/data/fg"
matrix_fg_mama = om.Matrix()
matrix_fg_mama.load(fname_fg_mama)
cbar = matrix_fg_mama.plot(ax=ax2D3, title="firstgen mama", zmin=1e-3, zmax=5e3)
f2D.colorbar(cbar, ax=ax2D3)


plt.show()
