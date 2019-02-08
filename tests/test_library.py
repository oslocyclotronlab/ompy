# -*- coding: utf-8 -*-
# Unittests of (TODO: all) functions in library
import numpy as np
from context import oslo_method_python as om
import unittest


# === Matrix ===
class TestMatrix(unittest.TestCase):
    def test_init(self):
        with self.assertRaises(ValueError):
            matrix = np.zeros((3, 3))
            E0_array = np.zeros(2)  # Shape mismatch to matrix
            E1_array = np.zeros(3)
            mat = om.Matrix(matrix=matrix, E0_array=E0_array,
                            E1_array=E1_array)
        with self.assertRaises(ValueError):
            matrix = np.zeros((5, 3))
            E0_array = np.zeros(5)
            E1_array = np.zeros(5)  # Shape mismatch to matrix
            mat = om.Matrix(matrix=matrix, E0_array=E0_array,
                            E1_array=E1_array)
        with self.assertRaises(ValueError):
            matrix = np.zeros((3, 3))
            std = np.zeros((3, 2))  # Shape mismatch to matrix
            E0_array = np.zeros(3)
            E1_array = np.zeros(3)
            mat = om.Matrix(matrix=matrix, E0_array=E0_array,
                            E1_array=E1_array, std=std)


if __name__ == '__main__':
    unittest.main()

    
    
    # === E_array_from_calibration ===
    decimal = 5  # Precision of element-wise unittest comparison
    a0, a1 = -10.0, 5.0  # Calibration
    E_array_true = np.array([-10.,  -5.,   0.,   5.,  10.,
                             15.,  20.,  25.,  30.,  35.])
    
    # Test giving number of bins N
    N = 10
    E_array_test = om.E_array_from_calibration(a0, a1, N=N)
    np.testing.assert_array_almost_equal(E_array_test,
                                         E_array_true,
                                         decimal=decimal
                                         )
    
    # Test giving max energy E_max
    E_max = 36.0
    E_array_test = om.E_array_from_calibration(a0, a1, E_max=E_max)
    np.testing.assert_array_almost_equal(E_array_test,
                                         E_array_true,
                                         decimal=decimal
                                         )
    