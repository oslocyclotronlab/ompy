# -*- coding: utf-8 -*-

from context import oslo_method_python as om


# Check that all the machinery is well oiled by running through the analysis pipeline.

# A highly compressed Si28 Ex-Eg matrix is provided in this directory as a test case.
# There is also a response matrix and response parameter file made by MAMA with the
# corresponding energy calibration.
# In the future (as of Jan. 2019), this package should contain functionality to generate the response
# functions internally.



fname_resp_mat = "response_matrix.m"
fname_resp_dat = "response_parameters.dat"

# Set up an instance of the matrix analysis class
ma = om.MatrixAnalysis()

# Load the Si28 raw matrix
ma.raw.load("Si28_raw_matrix_compressed.m")
ma.raw.plot()

# == Unfolding==
ma.unfold(fname_resp_dat=fname_resp_dat, fname_resp_mat=fname_resp_mat)
ma.unfolded.plot()




# == Firstgen ==
ma.first_generation_method()

ma.firstgen.plot()


# == Should ideally have some unit tests: ==
# import unittest
# class BasicTestSuite(unittest.TestCase):
#     """Basic test cases."""

#     def test_absolute_truth_and_meaning(self):
#         assert True


# if __name__ == '__main__':
#     unittest.main()