import pytest
import ompy as om
import numpy as np


@pytest.fixture()
def Si28():
    return om.load_example_raw('Si28')


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

