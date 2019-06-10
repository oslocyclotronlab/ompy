# -*- coding: utf-8 -*-

from context import oslo_method_python as om
import matplotlib.pyplot as plt

# Test the unfold() function:

fname_raw = "../tests/Dy164_raw.m"
matrix_raw = om.Matrix()
matrix_raw.load(fname_raw)

fname_resp_mat = "../tests/Dy164_response_matrix.m"
fname_resp_dat = "../tests/Dy164_response_parameters.dat"
diag_cut = {"Ex1": 0, "Eg1": 800, "Ex2": 7300, "Eg2": 7500}
# Call the unfolding algorithm:
matrix_unfolded = om.unfold(matrix_raw,
                            fname_resp_dat=fname_resp_dat,
                            fname_resp_mat=fname_resp_mat, diag_cut=diag_cut,
                            verbose=True)


# === Plot raw and unfolded ===
# Raw:
f2D, (ax2D1, ax2D2) = plt.subplots(1, 2)
cbar = matrix_raw.plot(ax=ax2D1, title="Dy164 raw",
                      zscale="log", zmin=1e-3, zmax=5e3)
f2D.colorbar(cbar, ax=ax2D1)
# Unfolded:
cbar = matrix_unfolded.plot(ax=ax2D2, title="Dy164 unfolded",
                      zscale="log", zmin=1e-3, zmax=5e3)
f2D.colorbar(cbar, ax=ax2D2)
plt.show()
