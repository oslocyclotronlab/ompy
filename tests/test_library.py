import pytest
import numpy as np
import ompy as om


def test_E_array_from_calibration():
    expected = np.asarray([0.5, 1.5, 2.5, 3.5])
    E = om.E_array_from_calibration(0.5, 1, N=4)
    np.testing.assert_allclose(expected, E)
    E = om.E_array_from_calibration(0.5, 1, E_max=3.5)
    np.testing.assert_allclose(expected, E)
    E = om.E_array_from_calibration(0.5, 1, E_max=3.1)
    np.testing.assert_allclose(expected, E)
    E = om.E_array_from_calibration(0.5, 1, E_max=2.6)
    np.testing.assert_allclose(expected[:-1], E)
    E = om.E_array_from_calibration(0.5, 1, E_max=3.9)
    np.testing.assert_allclose(expected, E)

