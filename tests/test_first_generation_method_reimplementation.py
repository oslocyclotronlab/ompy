from context import oslo_method_python as om
import matplotlib.pyplot as plt

# Load all-generations spectrum
fname_unfolded = "/home/jorgenem/MEGA/doktorgrad/oslometode_usikkerhetspropagering/Dy164/data/alfnaun"
allgen = om.Matrix()
allgen.load(fname_unfolded)

# Calculate firstgen using om:
firstgen = om.first_generation_method_reimplementation(allgen)

# Load firstgen calculated with MAMA to compare:
fname_fg_mama = "/home/jorgenem/MEGA/doktorgrad/oslometode_usikkerhetspropagering/Dy164/data/fg"
firstgen_mama = om.Matrix()
firstgen_mama.load(fname_fg_mama)


# === Plot them ===
f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
# All generations
cbar1 = allgen.plot(ax=ax1, title="all generations matrix", zmin=1e-2)
f.colorbar(cbar1, ax=ax1)
# First generation
cbar2 = firstgen.plot(ax=ax2, title="firstgen reimplementation OM Python",
                      zmin=1e-2, zmax=5e3)
f.colorbar(cbar2, ax=ax2)
# First generation from MAMA
cbar3 = firstgen_mama.plot(ax=ax3, title="firstgen MAMA",
                      zmin=1e-2)
f.colorbar(cbar3, ax=ax3)


plt.show()
