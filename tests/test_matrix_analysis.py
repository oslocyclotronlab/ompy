# -*- coding: utf-8 -*-

from context import oslo_method_python as om
import matplotlib.pyplot as plt


# Check that all the machinery is well oiled by running through the analysis pipeline.

# A highly compressed Si28 Ex-Eg matrix is provided in this directory as a test case.
# There is also a response matrix and response parameter file made by MAMA with the
# corresponding energy calibration.
# In the future (as of Jan. 2019), this package should contain functionality to generate the response
# functions internally.

fname_resp_mat = "../tests/Dy164_response_matrix.m"
fname_resp_dat = "../tests/Dy164_response_parameters.dat"

# Set up an instance of the matrix analysis class
ma = om.MatrixAnalysis()

# Load the Dy164 raw matrix
ma.raw.load("../tests/Dy164_raw.m")
# Drop Ex lower than 0 and larger than Sn, about 8300 keV
ma.raw.cut_rect(axis=0, E_limits=[0, 8400])

# == Unfolding==
diag_cut = {"Ex1": 0, "Eg1": 800, "Ex2": 7300, "Eg2": 7500}
ma.unfold(fname_resp_dat=fname_resp_dat, fname_resp_mat=fname_resp_mat,
          diag_cut=diag_cut)
# Remove negatives remaining
ma.unfolded.remove_negative()
# == Firstgen ==
Ex_max = 8500
dE_gamma = 500
ma.first_generation_method(Ex_max, dE_gamma)

# Plot them
f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
ma.raw.plot(ax=ax1, title="raw", zmin=1e-3, zmax=5e3)
ma.unfolded.plot(ax=ax2, title="unfolded", zmin=1e-3, zmax=5e3)
ma.firstgen.plot(ax=ax3, title="firstgen", zmin=1e-3, zmax=5e3)

plt.show()
# == Should ideally have some unit tests: ==
# import unittest
# class BasicTestSuite(unittest.TestCase):
#     """Basic test cases."""

#     def test_absolute_truth_and_meaning(self):
#         assert True


# if __name__ == '__main__':
#     unittest.main()