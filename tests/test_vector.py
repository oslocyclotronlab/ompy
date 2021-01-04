import pytest
import ompy as om
import warnings
from numpy.testing import assert_equal, assert_allclose
import numpy as np


def test_init():
    E = np.linspace(0, 1, 100)
    vals = np.linspace(2, 3.4, 100)
    vec = om.Vector(values=vals, E=E)
    assert_equal(E, vec.E)
    assert_equal(vals, vec.values)

    # Provide values only
    with pytest.raises(AssertionError):
        vec = om.Vector(vals)

    # No values defaults to zeros
    vec = om.Vector(E=E)
    assert_equal(np.zeros_like(E), vec.values)

    with pytest.raises(ValueError):
        om.Vector(vals, [1, 2, 3, 4, 5])


def test_len():
    N = 100
    E = np.linspace(0, 1, N)
    vals = np.linspace(2, 3.4, N)
    vec = om.Vector(values=vals, E=E)
    assert_equal(len(vec), N)


def test_save_load_no_std():
    E = np.linspace(0, 1, 100)
    vals = np.linspace(2, 3.4, 100)
    vec = om.Vector(values=vals, E=E)

    formats = ['.npy', '.txt', '.tar', '.m', '.csv']
    for form in formats:
        print(form)
        vec.save('/tmp/no_std'+form)
        vec_from_file = om.Vector(path='/tmp/no_std'+form)
        assert_allclose(vec_from_file.E, E)
        assert_allclose(vec_from_file.values, vals)


def test_save_load():
    E = np.linspace(0, 1, 100)
    vals = np.linspace(2, 3.4, 100)
    std = np.random.randn(100)*0.1

    vec = om.Vector(values=vals, E=E, std=std)

    formats = ['.npy', '.txt', '.tar', '.csv']
    for form in formats:
        vec.save('/tmp/std'+form)
        vec_from_file = om.Vector(path='/tmp/std'+form)
        assert_allclose(vec_from_file.E, vec.E)
        assert_allclose(vec_from_file.values, vec.values)
        assert_allclose(vec_from_file.std, vec.std)


def test_save_load_tar():
    E = np.linspace(0, 1, 100)
    vals = np.random.random((100, 100))

    mat = om.Matrix(values=vals, Ex=E, Eg=E)
    mat.save('/tmp/mat.tar')

    with pytest.raises(ValueError):
        vec = om.Vector(path='/tmp/mat.tar')


def test_save_std_warning():
    E = np.linspace(0, 1, 100)
    vals = np.linspace(2, 3.4, 100)
    std = np.random.randn(100)*0.1

    vec = om.Vector(values=vals, E=E, std=std)

    with pytest.warns(UserWarning):
        vec.save('/tmp/error.m')


def test_closest():
    E = np.array([0., 1., 2., 3., 4.])
    values = np.array([10., 9., 8., 7., 6.])
    std = 0.1*values

    E_new = np.array([0.5, 1.5, 3.])
    values_new = np.array([10., 9., 7.])
    std_new = 0.1*values_new

    vector = om.Vector(values=values, E=E)
    vector_res = vector.closest(E_new)

    assert_equal(vector_res.E, E_new)
    assert_equal(vector_res.values, values_new)

    vector = om.Vector(values=values, E=E, std=std)
    vector_res = vector.closest(E_new)
    assert_equal(vector_res.E, E_new)
    assert_equal(vector_res.values, values_new)
    assert_equal(vector_res.std, std_new)

    # Make sure the change is inplace.
    assert vector.closest(E_new, inplace=True) is None
    assert_equal(vector.E, E_new)
    assert_equal(vector.values, values_new)
    assert_equal(vector.std, std_new)

    # Make sure that x-values outside the
    # range gives zero
    E_new = [-1.5, 1.5, 3.5]
    vector_res = vector.closest(E_new)

    assert vector_res.values[0] == 0
    assert vector_res.std[0] == 0

    E_new = np.array([0.5, 1.5, 3., 6.])
    vector_res = vector.closest(E_new)

    assert vector_res.values[-1] == 0
    assert vector_res.std[-1] == 0

    # Make sure that RuntimeError is raised
    E = np.array([0., 1., 2., 4., 3.])
    values = np.array([10., 9., 8., 7., 6.])
    std = 0.1*values

    vector = om.Vector(values=values, E=E, std=std)
    with pytest.raises(RuntimeError):
        vector_res = vector.closest(E_new)


def test_cumsum():
    E = np.array([0., 1.5, 3., 4.5, 6., 7.5])
    values = np.array([0., 0., 1., 2., 3., 4.])
    std = 0.1*values

    expect = np.cumsum(values)
    expect_std = np.sqrt(np.cumsum(std**2))

    vec = om.Vector(values=values, E=E, std=std)

    vec_cum = vec.cumulative(factor=None, inplace=False)
    assert_equal(vec_cum.E, E)
    assert_equal(vec_cum.values, expect)
    assert_equal(vec_cum.std, expect_std)

    vec_cum = vec.cumulative(factor=2., inplace=False)
    assert_equal(vec_cum.E, E)
    assert_equal(vec_cum.values, 2.*expect)
    assert_equal(vec_cum.std, 2.*expect_std)

    vec_cum = vec.cumulative(factor='de', inplace=False)
    assert_equal(vec_cum.E, E)
    assert_equal(vec_cum.values, (E[1]-E[0])*expect)
    assert_equal(vec_cum.std, (E[1]-E[0])*expect_std)

    assert vec.cumulative(factor=None, inplace=True) is None
    assert_equal(vec.E, E)
    assert_equal(vec.values, expect)
    assert_equal(vec.std, expect_std)

    with pytest.raises(AssertionError):
        assert_equal(vec.values, values)

    with pytest.raises(ValueError):
        vec.cumulative(factor='dx', inplace=True)

    E = np.array([0., 1.5, 3., 4.5, 6., 7.5, 8.5])
    values = np.array([0., 0., 1., 2., 3., 4., 5.])

    vec = om.Vector(values=values, E=E)

    with pytest.raises(RuntimeError):
        vec.cumulative(factor='de', inplace=True)


def test_cut():
    E = np.arange(-1, 10, 1)
    values = np.linspace(33, 43, 11)
    vector = om.Vector(values=values, E=E)

    vector.cut(Emin=0, Emax=8)

    Ecut = np.arange(0, 9, 1)
    valcut = np.arange(34, 43)
    assert_equal(vector.E, Ecut)
    assert_equal(vector.values, valcut)


@pytest.mark.filterwarnings('ignore:divide by zero encountered in true_divide:RuntimeWarning')  # noqa
def test_numericals():
    E = np.array([0, 1, 2])
    values1 = np.array([0, 1, -2.])
    vector1 = om.Vector(values=values1, E=E)

    values2 = values1+1
    vector2 = om.Vector(values=values2, E=E)

    factor = 5.

    for op in ("/", "*", "+", "-"):
        eval(f"assert_equal((vector1{op}vector2).values, values1{op}values2)")
        eval(f"assert_equal((vector2{op}vector1).values, values2{op}values1)")
        eval(f"assert_equal((vector1{op}factor).values, values1{op}factor)")
        eval(f"assert_equal((factor{op}vector1).values, factor{op}values1)")

    assert_equal((vector2@vector1).values, values2@values1)
    assert_equal((vector1@vector2).values, values1@values2)


# This does not work as of now...
# def test_mutable():
#     E = np.array([0, 1, 2])
#     E_org = E.copy()

#     values = np.array([0, 1, -2.])
#     vector = om.Vector(values=values, E=E)

#     # chaning the original array shouldn't change the vector array
#     # (due to the setter)!
#     E += 1
#     assert_equal(vector.E, E_org)
