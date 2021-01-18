import pytest
import ompy as om
import numpy as np
from numpy.testing import assert_equal, assert_almost_equal
from typing import Tuple


def ones(shape: Tuple[int, int]) -> om.Matrix:
    """ Creates a mock matrix with ones in the upper diagonal

    A 5×5 looks like this:
        ░░░░░░░░░░
        ░░░░░░░░
        ░░░░░░
        ░░░░
        ░░
    Args:
        shape: The shape of the matrix
    Returns:
        The matrix
    """
    mat = np.ones(shape)
    mat = np.tril(mat)
    return om.Matrix(values=mat)


@pytest.fixture()
def Si28():
    return om.example_raw('Si28')


@pytest.mark.parametrize(
        "axis,Emin,Emax,shape",
        [('Ex', None, None, (10, 10)),
         ('Ex', None, 8,    (8, 10)),
         ('Eg', None, 8,    (10, 8)),
         ('Eg', 1,    None, (10, 9)),
         ('Ex', 5.5,  5.5,  (1, 10))],)
def test_cut_shape(axis, Emin, Emax, shape):
    mat = ones((10, 10)).cut(axis, Emin, Emax, inplace=False)
    assert mat.shape == shape
    assert len(mat.Eg) == shape[1]
    assert len(mat.Ex) == shape[0]


@pytest.mark.parametrize(
        "Emin,Emax,vector",
        [(-2, 1, np.array([-1, 0])),
         (-1, None, np.array([0, 1])),
         (None, 1.6, np.array([-1, 0, 1])),
         (-1.2, 1.6, np.array([-1, 0, 1]))])
def test_cut_limit(Emin, Emax, vector):
    values = np.zeros((5, 3))
    values[:, 0] = -1
    values[:, 2] = 1
    Eg = [-1.2, 0.2, 1.6]
    mat = om.Matrix(values=values, Eg=Eg)
    mat.cut('Eg', Emin, Emax)
    assert mat[0, :].shape == vector.shape
    assert np.all(mat[0, :] == vector)


def test_cut_Si(Si28):
    Si28.cut('Ex', Emin=0.0)
    assert Si28.Ex[0] > 0


@pytest.mark.parametrize(
        "E,index",
        [(-10.5, 0),
         (-11, 0),
         (-10.1, 0),
         (-9.4, 1),
         (10, 20),
         (9.5, 20),
         (8.6, 19)])
def test_index(E, index):
    mat = ones((10, 10))
    mat.Ex = np.arange(-10.5, 10.5)
    assert mat.index_Ex(E) == index


@pytest.mark.filterwarnings('ignore:divide by zero encountered in true_divide:RuntimeWarning')  # noqa
def test_numericals():
    E = np.array([0, 1, 2])
    values1 = np.array([[0, 1, 2.], [-2, 1, 2.],  [2, 3, -10.]])
    matrix1 = om.Matrix(values=values1, Ex=E, Eg=E)

    values2 = values1+1
    matrix2 = om.Matrix(values=values2, Ex=E, Eg=E)

    factor = 5.

    for op in ("/", "*", "+", "-"):
        eval(f"assert_equal((matrix1{op}matrix2).values, values1{op}values2)")
        eval(f"assert_equal((matrix2{op}matrix1).values, values2{op}values1)")
        eval(f"assert_equal((matrix1{op}factor).values, values1{op}factor)")
        eval(f"assert_equal((factor{op}matrix1).values, factor{op}values1)")

    assert_equal((matrix2@matrix1).values, values2@values1)
    assert_equal((matrix1@matrix2).values, values1@values2)


@pytest.mark.parametrize(
        "Ex,Eg",
        [(np.linspace(0, 10., num=10), np.linspace(10, 20., num=15)),
         ([0, 1, 2, 3, 7, 10.], [0, 1, 2, 3, 80, 90.])
         ])
def test_bin_shift(Ex, Eg):
    values = np.ones((len(Ex), len(Eg)), dtype="float")
    mat = om.Matrix(values=values, Ex=Ex, Eg=Eg)

    assert_almost_equal(Ex, mat.Ex)
    assert_almost_equal(Eg, mat.Eg)

    mat.to_lower_bin()
    mat.to_mid_bin()
    assert_almost_equal(Ex, mat.Ex)
    assert_almost_equal(Eg, mat.Eg)

    mat.to_mid_bin()
    mat.to_lower_bin()
    assert_almost_equal(Ex, mat.Ex)
    assert_almost_equal(Eg, mat.Eg)


@pytest.mark.parametrize(
        "Ex,Eg",
        [(np.linspace(0, 10., num=10), np.linspace(10, 20., num=15)),
         ([0, 1, 2, 3, 7, 10.], [0, 1, 2, 3, 80, 90.])
         ])
def test_save_warning(Ex, Eg):
    values = np.ones((len(Ex), len(Eg)), dtype="float")
    mat = om.Matrix(values=values, Ex=Ex, Eg=Eg, std=0.5*values)
    with pytest.warns(UserWarning):
        mat.save("/tmp/mat.npy")


@pytest.mark.parametrize(
        "Ex,Eg",
        [(np.linspace(0, 10., num=10), np.linspace(10, 20., num=15)),
         ([0, 1, 2, 3, 7, 10.], [0, 1, 2, 3, 80, 90.])
         ])
def test_save_std_exception(Ex, Eg):
    values = np.ones((len(Ex), len(Eg)), dtype="float")
    mat = om.Matrix(values=values, Ex=Ex, Eg=Eg)
    with pytest.raises(RuntimeError):
        mat.save("/tmp/mat.npy", which='std')


@pytest.mark.parametrize(
        "Ex,Eg",
        [(np.linspace(0, 10., num=10), np.linspace(10, 20., num=15)),
         ([0, 1, 2, 3, 7, 10.], [0, 1, 2, 3, 80, 90.])
         ])
def test_save_which_error(Ex, Eg):
    values = np.ones((len(Ex), len(Eg)), dtype="float")
    mat = om.Matrix(values=values, Ex=Ex, Eg=Eg, std=0.5*values)
    with pytest.raises(NotImplementedError):
        mat.save("/tmp/mat.npy", which='Im not real')


@pytest.mark.parametrize(
        "Ex,Eg",
        [(np.linspace(0, 10., num=10), np.linspace(10, 20., num=15)),
         ([0, 1, 2, 3, 7, 10.], [0, 1, 2, 3, 80, 90.])
         ])
def test_shape_ZerosMatrix(Ex, Eg):
    values = np.zeros((len(Ex), len(Eg)), dtype="float")
    mat = om.ZerosMatrix(Ex=Ex, Eg=Eg)
    assert_equal(mat.values, values)


def test_ZerosMatrix_fail_without_enough_info():
    energies = np.array([1, 2, 3])
    with pytest.raises(AssertionError):
        om.ZerosMatrix(Ex=energies)
    with pytest.raises(AssertionError):
        om.ZerosMatrix(Eg=energies)


def test_fill_matrix():
    """ TODO: add more cases such as making sure we are close to the correct
    number, etc.
    """
    Ex = np.array([1., 2., 3.])
    Eg = np.array([1., 2., 3., 4., 5., 6.])
    values = np.zeros((len(Ex), len(Eg)))
    mat = om.ZerosMatrix(Ex=Ex, Eg=Eg)

    mat.fill(1.9, 2.4)
    values[1][1] += 1
    assert_equal(mat.values, values)

    mat.fill(0., 6.)
    values[-1][0] += 1
    assert_equal(mat.values, values)


# This does not work as of now...
# def test_mutable():
#     E = np.array([0, 1, 2])
#     E_org = E.copy()

#     values = np.array([[0, 1, 2.], [-2, 1, 2.],  [2, 3, -10.]])
#     matrix = om.Matrix(values=values, Ex=E, Eg=E)

#     # chaning the Ex array shouldn't change the Eg array (due to the setter)!
#     matrix.Ex[0] += 1
#     assert_equal(matrix.Eg, E_org)

