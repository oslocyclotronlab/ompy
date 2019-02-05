from context import oslo_method_python as om
import matplotlib.pyplot as plt

# Load all-generations spectrum
fname_unfolded = "/home/jorgenem/MEGA/doktorgrad/oslometode_usikkerhetspropagering/Dy164/data/alfnaun"
allgen = om.Matrix()
allgen.load(fname_unfolded)

firstgen = om.first_generation_method_reimplementation(allgen)


# === Plot them ===
f, (ax1, ax2) = plt.subplots(1, 2)
# All generations
cbar1 = allgen.plot(ax=ax1, title="all generations matrix", zmin=1e-2)
f.colorbar(cbar1, ax=ax1)
# First generation
cbar2 = firstgen.plot(ax=ax2, title="firstgen reimplementation OM Python",
                      zmin=1e-2)
f.colorbar(cbar2, ax=ax2)

plt.show()
