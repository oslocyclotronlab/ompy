import pytest
from contextlib import contextmanager
import numpy as np
from numpy.testing import assert_equal, assert_allclose
import ompy as om
from test_matrix import compare_unitful, assert_matrix, assert_matrices


vals_rebinned = [10 - 2 * 2.5, 2.5 + 5, 20 - 2 * 5, 0.25 * 20 + 0.25 * 30]


def test_rebin_vector():
    before = om.Vector(values=[10, 20, 30], E=[10., 20, 30])
    after = om.Vector(values=vals_rebinned, E=[10., 15, 20, 25])
    rebinned = before.rebin(mids=after.E, inplace=False)
    assert_allclose(rebinned.E.magnitude, after.E.magnitude)
    assert_allclose(rebinned.values, after.values)


def test_rebin_both():
    shape = (1001, 1001)
    Elim = (0, 10000)
    before = om.Matrix(values=np.ones(shape),
                       Eg=np.linspace(Elim[0], Elim[1], shape[0]),
                       Ex=np.linspace(Elim[0], Elim[1], shape[1]))

    new_shape = (101, 101)
    new_Elim = Elim
    after = before.rebin(axis='both',
                         mids=np.linspace(new_Elim[0], new_Elim[1],
                                          new_shape[0]),
                         inplace=False)

    assert len(after.Eg) != len(before.Eg)
    assert len(after.Ex) != len(before.Ex)
    compare_unitful(after.Eg, after.Ex)

    before.rebin(axis='both',
                 mids=np.linspace(new_Elim[0], new_Elim[1], new_shape[0]),
                 inplace=True)
    assert len(after.Eg) == len(before.Eg)
    assert len(after.Ex) == len(before.Ex)
    compare_unitful(before.Ex, before.Ex)
    compare_unitful(before.Eg, after.Eg)


@pytest.mark.xfail
def test_rebin_vector_non_equidistant(before, after):
    before = om.Vector(values=[10, 20, 30], E=[10., 20, 30])
    after = om.Vector(values=[10 - 2 * 2.5, 2.5 + 5, 20 - 2 * 5, 30],
                      E=[10., 15, 20, 30])
    rebinned = before.rebin(mids=after.E, inplace=False)
    compare_unitful(rebinned.E, after.E)
    assert_allclose(rebinned.values, after.values)


def test_rebin_matrix_Eg():
    values = np.array([[10, 20, 30], [10, 20, 30], [10, 20, 30]])
    before = om.Matrix(values=values, Ex=[10., 20, 30], Eg=[10., 20, 30])

    values = np.array([vals_rebinned, vals_rebinned, vals_rebinned])
    after = om.Matrix(values=values, Ex=[10, 20, 30], Eg=[10., 15, 20, 25])
    rebinned = before.rebin(axis="Eg", mids=after.Eg, inplace=False)
    assert_matrices(rebinned, after)


def test_rebin_matrix_Ex():
    values = np.array([[10, 20, 30], [10, 20, 30], [10, 20, 30]]).T
    before = om.Matrix(values=values, Eg=[10., 20, 30], Ex=[10., 20, 30])

    values = np.array([vals_rebinned, vals_rebinned, vals_rebinned]).T
    after = om.Matrix(values=values, Eg=[10, 20, 30], Ex=[10., 15, 20, 25])
    rebinned = before.rebin(axis="Ex", mids=after.Ex, inplace=False)
    assert_matrices(rebinned, after)


np.random.seed(678456456)
vec1 = om.Vector(values=[1., 1, 1, 1], E=[2., 3, 4, 5])
vec2 = om.Vector(values=[0.2, 1, 4, 1], E=[2., 2.5, 4, 5])
vec3 = om.Vector(values=np.random.uniform(size=50),
                 E=np.linspace(-10, 10, num=50))


@pytest.mark.parametrize(
    "vec, factor",
    [
        (vec1, 1),
        (vec2, 1.5),
        (vec2, 2.),
        (vec3, 5),
        # (vec2, 3),
        pytest.param(
            vec1, 1 / 4, marks=pytest.mark.xfail(reason="see issue #122"))
    ],
)
def test_rebin_factor_preserve_counts(vec, factor):
    sum_before = vec.values.sum()
    vec = vec.rebin(factor=factor, inplace=False)
    sum_after = vec.values.sum()
    assert_allclose(sum_before, sum_after)


@contextmanager
def does_not_raise():
    yield


@pytest.mark.parametrize(
    "vec, mids, expectation",
    [(vec3, np.linspace(vec3.E[0], vec3.E[-1], num=7), does_not_raise()),
     (vec3, np.linspace(vec3.E[0] - 2, vec3.E[-1], num=9), does_not_raise()),
     (vec3, np.linspace(vec3.E[0] - 2, vec3.E[-1] + 2,
                        num=3), does_not_raise()),
     (vec3, np.linspace(vec3.E[1], vec3.E[-1],
                        num=31), pytest.raises(AssertionError)),
     pytest.param(vec3,
                  np.linspace(vec3.E[0], vec3.E[-1], num=51),
                  does_not_raise(),
                  marks=pytest.mark.xfail(reason="see issue #122"))],
)
def test_rebin_mids_preserve_counts(vec, mids, expectation):
    with expectation:
        sum_before = vec.values.sum()
        vec = vec.rebin(mids=mids, inplace=False)
        sum_after = vec.values.sum()
        assert_allclose(sum_before, sum_after)
