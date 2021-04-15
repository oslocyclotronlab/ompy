import pytest
import ompy as om
import warnings
from numpy.testing import assert_equal, assert_allclose
import numpy as np


def compare_unitful(x, y):
    try:
        x.units
        x_unitful = True
    except AttributeError:
        x_unitful = False

    try:
        y.units
        y_unitful = True
    except AttributeError:
        y_unitful = False

    if x_unitful and y_unitful:
        assert_equal(x.units, y.units)
        assert_allclose(x.magnitude, y.magnitude)
    elif x_unitful and not y_unitful:
        assert_allclose(x.magnitude, y)
    elif not x_unitful and y_unitful:
        assert_allclose(x, y.magnitude)
    else:
        assert_allclose(x, y)


class TestZerosLike:
    def test_vector(self):
        values = np.linspace(0, 10)
        E = np.linspace(10, 100)
        vec = om.Vector(values=values, E=E)
        zeros = om.zeros_like(vec)
        compare_unitful(zeros.E, vec.E)
        zeros.verify_integrity()
        assert_allclose(zeros.values, np.zeros(len(zeros)))
        assert_equal(len(vec), len(zeros))

    def test_matrix(self):
        values = np.linspace(0, 10, 100).reshape((10, 10))
        Eg = np.linspace(0, 1, 10)
        Ex = np.linspace(-1, 2, 10)
        mat = om.Matrix(Eg=Eg, Ex=Ex, values=values)

        zeros = om.zeros_like(mat)
        compare_unitful(zeros.Eg, mat.Eg)
        compare_unitful(zeros.Ex, mat.Ex)
        assert_allclose(zeros.values, np.zeros(mat.values.shape))
        zeros.verify_integrity()

    # def test_shape(self):
    #     zeros = om.zeros(10)
    #     assert_allclose(zeros.values, np.zeros(10))

    #     zeros = om.zeros((11, ))
    #     assert_allclose(zeros.values, np.zeros(11))

    #     zeros = om.zeros((11, 21))
    #     assert_allclose(zeros.values, np.zeros((11, 21)))

    #     with pytest.raises(ValueError):
    #         om.zeros((11, 12, 12))

    # def test_array(self):
    #     arr = np.zeros(10)
    #     zeros = om.zeros(arr)
    #     assert_allclose(arr, zeros.values)
    #     arr = np.zeros((24, 12))
    #     zeros = om.zeros(arr)
    #     assert_allclose(arr, zeros.values)

    #     arr = np.zeros((10, 10, 10))
    #     with pytest.raises(ValueError):
    #         om.zeros(arr)
