import warnings
from typing import Tuple

import numpy as np
import ompy as om
import pytest
from numpy.testing import assert_allclose, assert_equal
from ompy import Matrix


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


def assert_matrix(mat, values, Ex, Eg, std=None):
    compare_unitful(mat.Ex, Ex)
    compare_unitful(mat.Eg, Eg)
    assert_allclose(mat.values, values)
    if std is None:
        assert_equal(mat.std, None)
    else:
        assert_allclose(mat.std, std)


def assert_matrices(mat, expect):
    assert_matrix(mat, expect.values, expect.Ex,
                  expect.Eg, expect.std)


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


class TestInit:
    def test_wrong_arguments(self):
        with pytest.raises(ValueError):
            _ = om.Matrix()
        with pytest.raises(ValueError):
            _ = om.Matrix(Eg=[1, 2], Ex=[2, 3])
        with pytest.raises(ValueError):
            _ = om.Matrix(Eg=[1, 2], values=[12, 3])
        with pytest.raises(ValueError):
            _ = om.Matrix(Ex=[1, 2], values=[12, 3])

    def test_values(self):
        Ex = np.linspace(-1, 1, 100)
        Eg = np.linspace(0, 1, 50)
        values = np.linspace(0, 100, 100*50)

        with pytest.raises(ValueError):
            _ = om.Matrix(values=values, Eg=Eg, Ex=Ex)

        values = values.reshape((100, 50))

        with pytest.raises(ValueError):
            _ = Matrix(Eg=Eg, Ex=Eg, values=values)
        with pytest.raises(ValueError):
            _ = Matrix(Eg=Ex, Ex=Ex, values=values)

        mat = Matrix(Eg=Eg, Ex=Ex, values=values)
        mat.verify_integrity()
        assert_matrix(mat, values, Ex, Eg)

        with pytest.raises(ValueError):
            Matrix(Eg=Eg, Ex=Ex, values=values, std=Ex)

        with pytest.raises(ValueError):
            Matrix(Eg=Eg, Ex=Ex, values=values, std=values.T)

    def test_copy(self):
        Ex = np.linspace(-1, 1, 100)
        Eg = np.linspace(0, 1, 50)
        values = np.linspace(0, 100, 100*50).reshape((100, 50))
        std = 0.1*values

        Ex2 = np.linspace(-1, 1, 100)
        Eg2 = np.linspace(0, 1, 50)
        values2 = np.linspace(0, 100, 100*50).reshape((100, 50))
        std2 = 0.1*values2

        mat = Matrix(Eg=Eg, Ex=Ex, values=values, std=std,
                     copy=True)
        assert_matrix(mat, values, Ex, Eg, std)

        values[0] += 1
        Ex[0] += 1
        Eg[0] += 1
        std[0] += 1
        assert_matrix(mat, values2, Ex2, Eg2, std2)

    def test_units(self):
        Ex = np.linspace(-1, 1, 100)
        Eg = np.linspace(0, 1, 50)
        values = np.linspace(0, 100, 100*50).reshape((100, 50))
        mat = Matrix(Ex=Ex, Eg=Eg, values=values)
        assert(mat.Ex_units == om.u.keV)
        assert(mat.Eg_units == om.u.keV)

        mat = Matrix(Ex=Ex, Eg=Eg, values=values, units='MeV')
        assert(mat.Ex_units == om.u.MeV)
        assert(mat.Eg_units == om.u.MeV)
        assert_matrix(mat, values, Ex, Eg)

        Ex = np.linspace(0, 1, 10) * om.ureg.MeV
        Eg = np.linspace(0, 1, 10) * om.ureg.GeV
        values = np.linspace(0, 100, 100).reshape((10, 10))
        mat = Matrix(Ex=Ex, Eg=Eg, values=values)
        assert(mat.Ex_units == om.u.MeV)
        assert(mat.Eg_units == om.u.GeV)
        assert_matrix(mat, values, Ex, Eg)

    def test_path(self):
        warnings.warn("Test not implemented")


class TestArithmetic:
    def test(self):
        Ex = np.linspace(0, 1, 100)
        Eg = np.linspace(0, 1, 50)
        values = np.linspace(-1, 1, 50*100).reshape((100, 50))
        mat = Matrix(Ex=Ex, Eg=Eg, values=values)
        mat2 = Matrix(Ex=Ex, Eg=Eg, values=values, units='MeV')
        mat3 = Matrix(Ex=Eg, Eg=Ex, values=values.T)
        for op in ['+', '-', '*', '/']:
            fact = eval(f"values{op}values")
            trial = eval(f"mat{op}mat")
            assert_allclose(trial.values, fact)

            fact = eval(f"values{op}5.0")
            trial = eval(f"mat{op}5.0")
            assert_allclose(trial.values, fact)

            fact = eval(f"2.1{op}values")
            trial = eval(f"2.1{op}mat")
            assert_allclose(trial.values, fact)

            with pytest.raises(ValueError):
                eval(f"mat{op}mat2")
            with pytest.raises(ValueError):
                eval(f"mat{op}mat3")


def test_to():
    Ex = np.linspace(0, 1, 10) * om.ureg.MeV
    Eg = np.linspace(0, 1, 10) * om.ureg.GeV
    values = np.linspace(0, 100, 100).reshape((10, 10))
    mat = Matrix(values=values, Ex=Ex, Eg=Eg)
    mat2 = mat.to('keV')

    assert(mat.Ex_units == om.u.MeV)
    assert(mat.Eg_units == om.u.GeV)
    assert(mat2.Ex_units == om.u.keV)
    assert(mat2.Eg_units == om.u.keV)
    assert_allclose(mat2.values, values)
    assert_allclose(mat2.Eg, Eg.magnitude * 1e6)
    assert_allclose(mat2.Ex, Ex.magnitude * 1e3)


def test_to_E():
    Ex = np.linspace(0, 1, 10) * om.ureg.MeV
    Eg = np.linspace(0, 1, 10) * om.ureg.GeV
    values = np.linspace(0, 100, 100).reshape((10, 10))
    mat = Matrix(values=values, Ex=Ex, Eg=Eg)
    assert_equal(type(mat.to_same_Ex(0.1)), float)
    assert_equal(type(mat.to_same_Ex(0.1*om.ureg.keV)), float)
    assert_equal(type(mat.to_same_Ex('800 keV')), float)
    assert_equal(type(mat.to_same_Eg(0.1)), float)
    assert_equal(type(mat.to_same_Eg(0.1*om.ureg.keV)), float)
    assert_equal(type(mat.to_same_Eg('800 keV')), float)

    assert_allclose(mat.to_same_Ex(0.1), 0.1)
    assert_allclose(mat.to_same_Ex(0.1*om.ureg.keV), 0.1e-3)
    assert_allclose(mat.to_same_Ex('800 keV'), 0.8)
    assert_allclose(mat.to_same_Eg(0.1), 0.1)
    assert_allclose(mat.to_same_Eg(0.1*om.ureg.keV), 0.1e-6)
    assert_allclose(mat.to_same_Eg('800 keV'), 0.8e-3)


@pytest.fixture()
def Si28():
    return om.example_raw('Si28')


def test_clone():
    Ex = np.linspace(-1, 1)
    Eg = np.linspace(-2, 4)
    values = np.random.random((50, 50))
    std = 0.5*values

    mat = om.Matrix(Ex=Ex, Eg=Eg, values=values, std=std)
    mat2 = mat.clone()
    assert_matrices(mat, mat2)
    mat3 = mat.clone(Ex=0.2*Eg)
    assert_allclose(mat3.values, values)
    assert_allclose(mat3.std, std)
    assert_allclose(mat3.Eg, Eg)
    assert_allclose(mat3.Ex, 0.2*Eg)

    mat3 = mat.clone(Eg=0.2*Ex)
    assert_allclose(mat3.values, values)
    assert_allclose(mat3.std, std)
    assert_allclose(mat3.Eg, 0.2*Ex)
    assert_allclose(mat3.Ex, Ex)

    mat3 = mat.clone(values=values.T)
    assert_allclose(mat3.values, values.T)
    assert_allclose(mat3.std, std)
    assert_allclose(mat3.Eg, Eg)
    assert_allclose(mat3.Ex, Ex)

    mat3 = mat.clone(std=0.5*std)
    assert_allclose(mat3.values, values)
    assert_allclose(mat3.std, 0.5*std)
    assert_allclose(mat3.Eg, Eg)
    assert_allclose(mat3.Ex, Ex)

    mat3 = mat.clone(Eg=Ex, Ex=Eg, values=values**2, std=0.5*std)
    assert_allclose(mat3.values, values**2)
    assert_allclose(mat3.std, 0.5*std)
    assert_allclose(mat3.Eg, Ex)
    assert_allclose(mat3.Ex, Eg)

    with pytest.raises(RuntimeError):
        mat.clone(duck=5)


class TestCut:
    def test_unitless(self):
        Ex = np.linspace(0, 120, 120)
        Eg = np.linspace(0, 180, 180)*om.ureg.MeV
        values = np.random.random((120, 180))
        mat = Matrix(values=values, Ex=Ex, Eg=Eg)

        cut = mat.cut('Ex', None, None)
        assert_equal(cut.shape, mat.shape)
        assert_equal(cut.shape, values.shape)
        assert_matrix(cut, values, Ex, Eg)

        cut = mat.cut('Ex', Emin=60)
        assert_allclose(cut.Ex.max(), 120)
        assert_allclose(cut.Ex.min(), Ex[Ex>60][0])
        assert_allclose(cut.Eg.max(), Eg.magnitude.max())
        assert_allclose(cut.Eg.min(), Eg.magnitude.min())

        cut = mat.cut('Ex', Emax=60)
        assert_allclose(cut.Ex.max(), Ex[Ex<60][-1])
        assert_allclose(cut.Ex.min(), 0)
        assert_allclose(cut.Eg.max(), Eg.magnitude.max())
        assert_allclose(cut.Eg.min(), Eg.magnitude.min())

        cut = mat.cut('Ex', Emin=50, Emax=100)
        assert_allclose(cut.Ex.max(), Ex[Ex<100][-1])
        assert_allclose(cut.Ex.min(), Ex[Ex>50][0])
        assert_allclose(cut.Eg.max(), Eg.magnitude.max())
        assert_allclose(cut.Eg.min(), Eg.magnitude.min())

        Eg = Eg.magnitude
        cut = mat.cut('Eg', Emin=60)
        assert_allclose(cut.Eg.max(), 180)
        assert_allclose(cut.Eg.min(), Eg[Eg>60][0])
        assert_allclose(cut.Ex.max(), Ex.max())
        assert_allclose(cut.Ex.min(), Ex.min())

        cut = mat.cut('Eg', Emax=60)
        assert_allclose(cut.Eg.max(), Eg[Eg<60][-1])
        assert_allclose(cut.Eg.min(), 0)
        assert_allclose(cut.Ex.max(), Ex.max())
        assert_allclose(cut.Ex.min(), Ex.min())

        cut = mat.cut('Eg', Emin=50, Emax=100)
        assert_allclose(cut.Eg.max(), Eg[Eg<100][-1])
        assert_allclose(cut.Eg.min(), Eg[Eg>50][0])
        assert_allclose(cut.Ex.max(), Ex.max())
        assert_allclose(cut.Ex.min(), Ex.min())

    def test_units(self):
        Ex = np.linspace(0, 120, 120)
        Eg = np.linspace(0, 180, 180)*om.ureg.MeV
        values = np.random.random((120, 180))
        mat = Matrix(values=values, Ex=Ex, Eg=Eg)

        cut = mat.cut('Ex', Emin='50 keV')
        assert_allclose(cut.Ex.min(), Ex[Ex>50][0])
        cut = mat.cut('Ex', Emin=om.Q_(50, 'keV'))
        assert_allclose(cut.Ex.min(), Ex[Ex>50][0])
        cut = mat.cut('Ex', Emin=50*om.ureg.keV)
        assert_allclose(cut.Ex.min(), Ex[Ex>50][0])

        Eg2 = Eg.magnitude
        cut = mat.cut('Eg', Emin='50 MeV')
        assert_allclose(cut.Eg.min(), Eg2[Eg2>50][0])
        cut = mat.cut('Eg', Emin=om.Q_(50, 'MeV'))
        assert_allclose(cut.Eg.min(), Eg2[Eg2>50][0])
        cut = mat.cut('Eg', Emin=50*om.ureg.MeV)
        assert_allclose(cut.Eg.min(), Eg2[Eg2>50][0])

        cut = mat.cut('Eg', Emin='6000 keV', Emax='0.1 GeV')
        assert_allclose(cut.Eg.min(), Eg2[Eg2>6][0])
        assert_allclose(cut.Eg.max(), Eg2[Eg2<100][-1])


def test_cut_Si(Si28):
    Si28.cut('Ex', Emin=0.0, inplace=True)
    assert Si28.Ex[0] > 0


class TestIndex:
    def test_index(self):
        Ex = np.linspace(0, 1, 10)
        Eg = np.linspace(-1, 5, 11)
        values = np.random.random((10, 11))
        mat = Matrix(Ex=Ex, Eg=Eg, values=values)
        for i, E in enumerate(Ex):
            assert_equal(i, mat.index_Ex(E))
        for i, E in enumerate(Eg):
            assert_equal(i, mat.index_Eg(E))
        assert_equal(0, mat.index_Ex(-10))
        assert_equal(0, mat.index_Eg(-10))
        assert_equal(9, mat.index_Ex(10))
        assert_equal(10, mat.index_Eg(10))

    def test_units(self):
        Ex = np.linspace(0, 1, 10)*om.ureg('MeV')
        Eg = np.linspace(-1, 5, 11)
        values = np.random.random((10, 11))
        mat = Matrix(Ex=Ex, Eg=Eg, values=values)
        for i, E in enumerate(Ex):
            assert_equal(i, mat.index_Ex(E))
        for i, E in enumerate(Ex.to('keV')):
            assert_equal(i, mat.index_Ex(E))
        for i, E in enumerate(Eg):
            assert_equal(i, mat.index_Eg(E))
        assert_equal(0, mat.index_Ex(-10))
        assert_equal(0, mat.index_Eg(-10))
        assert_equal(9, mat.index_Ex(10))
        assert_equal(10, mat.index_Eg(10))
        assert_equal(2, mat.index_Eg(10*om.ureg('eV')))



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

    assert_allclose(Ex, mat.Ex)
    assert_allclose(Eg, mat.Eg)

    mat.to_lower_bin()
    mat.to_mid_bin()
    assert_allclose(Ex, mat.Ex)
    assert_allclose(Eg, mat.Eg)

    mat.to_mid_bin()
    mat.to_lower_bin()
    assert_allclose(Ex, mat.Ex)
    assert_allclose(Eg, mat.Eg)


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


def test_loc():
    Ex = np.linspace(0, 10)
    Eg = np.linspace(-2.1, 103, 104)
    values = np.random.random((len(Ex), len(Eg)))
    std = 0.1*values
    mat = om.Matrix(values=values, Ex=Ex, Eg=Eg, std=std)
    mat2 = mat.loc[:, :]
    assert_matrices(mat2, mat)

    mat2 = mat.loc[0:, :]
    assert_matrices(mat2, mat)

    mat2 = mat.loc[:10, :]
    assert_allclose(mat2.Ex, np.linspace(0, 10)[:-1])

    mat2 = mat.loc[:'<10 keV', :]
    assert_allclose(mat2.Ex, np.linspace(0, 10)[:-1])

    mat2 = mat.loc[:'>10 keV', :]
    assert_matrices(mat2, mat)

    #mat2 = mat.loc[:, '1 keV':'>98 keV']

# This does not work as of now...
# def test_mutable():
#     E = np.array([0, 1, 2])
#     E_org = E.copy()

#     values = np.array([[0, 1, 2.], [-2, 1, 2.],  [2, 3, -10.]])
#     matrix = om.Matrix(values=values, Ex=E, Eg=E)

#     # chaning the Ex array shouldn't change the Eg array (due to the setter)!
#     matrix.Ex[0] += 1
#     assert_equal(matrix.Eg, E_org)

