# -*- coding: utf-8 -*-
# Unittests of (TODO: all) functions in library
import numpy as np
from context import oslo_method_python as om


# == E_array_from_calibration ==
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
