# -*- coding: utf-8 -*-
from context import oslo_method_python as om
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
import copy


Ex_min = 4000
Ex_max = 6000
Eg_min = 1000
bin_width_out = 100

calib_out = {"a0": -500, "a1": bin_width_out}

# Set up the energy array common to rho and T
E_array = om.E_array_from_calibration(a0=calib_out["a0"],
                                      a1=calib_out["a1"],
                                      E_max=Ex_max)
Nbins = len(E_array)
print("Nbins = ", Nbins, flush=True)

# Load firstgen
firstgen_in = om.Matrix()
firstgen_in.load("/home/jorgenem/MEGA/doktorgrad/oslometode_usikkerhetspropagering/RAINIER/synthetic_data/Jint_EB06_mama_4res/1Gen.m")
# Error matrix:
# Use ones as error matrix, which should in practice turn off error weighting
firstgen_std_in = copy.deepcopy(firstgen_in)
firstgen_std_in.matrix = np.sqrt(firstgen_in.matrix)

# Rebin and cut:
firstgen = om.Matrix()
# axis 0:
firstgen.matrix = om.rebin_matrix(firstgen_in.matrix, firstgen_in.E0_array,
                               E_array, rebin_axis=0)
# axis 1:
firstgen.matrix = om.rebin_matrix(firstgen.matrix, firstgen_in.E1_array,
                               E_array, rebin_axis=1)

firstgen_std = om.Matrix()
firstgen_std.matrix = om.interpolate_matrix_2D(firstgen_std_in.matrix,
                                     firstgen_std_in.E0_array,
                                     firstgen_std_in.E1_array,
                                     E_array,
                                     E_array
                                     )
# Set energy axes accordingly
firstgen.E0_array = E_array
firstgen.E1_array = E_array
firstgen_std.E0_array = E_array
firstgen_std.E1_array = E_array
# Verify that it got rebinned and assigned correctly:
calib_firstgen = firstgen.calibration()
assert (
        calib_firstgen["a00"] == calib_out["a0"] and
        calib_firstgen["a01"] == calib_out["a1"] and
        calib_firstgen["a10"] == calib_out["a0"] and
        calib_firstgen["a11"] == calib_out["a1"]
       ), "firstgen does not have correct calibration."
calib_firstgen_std = firstgen_std.calibration()
assert (
        calib_firstgen_std["a00"] == calib_out["a0"] and
        calib_firstgen_std["a01"] == calib_out["a1"] and
        calib_firstgen_std["a10"] == calib_out["a0"] and
        calib_firstgen_std["a11"] == calib_out["a1"]
       ), "firstgen_std does not have correct calibration."


# Make cuts to the matrices using Fabio's utility:
pars_fg = {"Egmin" : Eg_min,
           "Exmin" : Ex_min,
           "Emax" : Ex_max}
E_array_midbin = E_array + calib_out["a1"]/2
firstgen_matrix, Emid_Eg, Emid_Ex, Emid_nld = om.fg_cut_matrix(firstgen.matrix,
                                                        E_array_midbin, **pars_fg)
firstgen_std_matrix, Emid_Eg, Emid_Ex, Emid_nld = om.fg_cut_matrix(firstgen_std.matrix,
                                                        E_array_midbin, **pars_fg)


f2D, ((ax_P_true, ax_P_fit), (ax_P_diff, ax2D_4)) = plt.subplots(2, 2)
firstgen_cut = firstgen.cut_rect(
                              axis="both",
                              E_limits=[Ex_min, Ex_max, Eg_min, Ex_max],
                              inplace=False
                              )
cbar = ax_P_true.pcolormesh(firstgen_cut.E1_array,
                            firstgen_cut.E0_array,
                            firstgen_cut.matrix,
                            norm=LogNorm())
ax_P_true.set_title("P_true")
f2D.colorbar(cbar, ax=ax_P_true)

rho_fit, T_fit = om.decompose_matrix(firstgen_matrix, firstgen_std_matrix,
                                     Emid_Eg=Emid_Eg,
                                     Emid_nld=Emid_nld,
                                     Emid_Ex=Emid_Ex,
                                     method="Powell")

print("len(rho_fit) =", len(rho_fit))
print("len(T_fit) =", len(T_fit))

rho = om.Vector(rho_fit, Emid_nld-calib_out["a1"]/2)
T = om.Vector(T_fit, Emid_Eg-calib_out["a1"]/2)


f1D, (ax_rho, ax_T) = plt.subplots(1, 2)
rho.plot(ax=ax_rho, label="fit")
ax_rho.legend()
ax_rho.set_yscale("log")

T.plot(ax=ax_T, label="fit")
# ax_T.plot(Emid_Eg, T.transform(const=2e2, alpha=0.002),
          # label="fit, transformed")
ax_T.legend()
ax_T.set_yscale("log")


P_fit = om.PfromRhoT(rho.vector, T.vector, len(Emid_Ex),
                     Emid_Eg, Emid_nld, Emid_Ex)
cbar = ax_P_fit.pcolormesh(Emid_Eg, Emid_Ex, P_fit, norm=LogNorm())
ax_P_fit.set_title("P_fit")
f2D.colorbar(cbar, ax=ax_P_fit)

# P_diff = P_fit - P_true
# cbar = ax_P_diff.pcolormesh(E_array, E_array, P_diff)
# f2D.colorbar(cbar, ax=ax_P_diff)

# Print some quantities
# print("P_diff.max() =", P_diff.max())
print("rho (fitted) =", rho.vector)
print("T (fitted) =", T.vector)


plt.show()
