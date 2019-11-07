import pytest
import ompy as om
import numpy as np
from numpy.testing import assert_equal, assert_almost_equal
import os

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
    mat = om.ones((10, 10)).cut(axis, Emin, Emax, inplace=False)
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
    mat = om.ones((10, 10))
    mat.Ex = np.arange(-10.5, 10.5)
    assert mat.index_Ex(E) == index


@pytest.mark.filterwarnings('ignore:divide by zero encountered in true_divide:RuntimeWarning')
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

def test_FitPeak():
    peak_fit = {'const': 1182930.5206675457, 'mean': 4435.811321829209, 'std': 99.67972415903449, 'slope': 1.999450091327474, 'intercept': -7826.141269345372}
    path = os.path.dirname(os.path.realpath(__file__))
    mat = om.Matrix(path=path+"/../example_data/test_PeakFit.m")
    result = mat.FitPeak((3790, 4250), (4650, 5200), (6382, 8850)) 

    assert_almost_equal(peak_fit['const'], result['const'], decimal=1)
    assert_almost_equal(peak_fit['mean'], result['mean'], decimal=1)
    assert_almost_equal(peak_fit['std'], result['std'], decimal=1)
    assert_almost_equal(peak_fit['slope'], result['slope'], decimal=1)
    assert_almost_equal(peak_fit['intercept'], result['intercept'], decimal=1)

# This does not work as of now...
# def test_mutable():
#     E = np.array([0, 1, 2])
#     E_org = E.copy()

#     values = np.array([[0, 1, 2.], [-2, 1, 2.],  [2, 3, -10.]])
#     matrix = om.Matrix(values=values, Ex=E, Eg=E)

#     # chaning the Ex array shouldn't change the Eg array (due to the setter)!
#     matrix.Ex[0] += 1
#     assert_equal(matrix.Eg, E_org)

