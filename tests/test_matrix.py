from numpy import ndarray
import ompy as om
import numpy as np
from numpy.testing import assert_allclose
from ompy import Matrix, Vector
import pytest

"""
TODO:
-[x] Math
-[ ] Rebin
-[x] Index
-[x] < > <= >=
-[x] == !=
-[x] Locators
-[ ] Plotting
-[x] Neg
-[x] Abs
-[x] Sum
-[x] IO
"""

def assert_content_compatible(a, b):
    assert_allclose(a.values, b.values)
    assert a.X_index.is_compatible_with(b.X_index)
    assert a.Y_index.is_compatible_with(b.Y_index)

def assert_content_equal(a, b):
    assert_allclose(a.X_index.bins, b.X_index.bins)
    assert_allclose(a.Y_index.bins, b.Y_index.bins)
    assert_allclose(a.X_index.boundary, b.X_index.boundary)
    assert_allclose(a.Y_index.boundary, b.Y_index.boundary)
    assert_allclose(a.values, b.values)

def assert_metadata_equal(a, b):
    assert a.metadata == b.metadata
    assert a.X_index.meta == b.X_index.meta
    assert a.Y_index.meta == b.Y_index.meta

def test_simple_constructor():
    x = np.linspace(-4.5, 324, 1001)
    y = np.linspace(56, 10002.0, 56)
    values = np.random.random((len(x), len(y)))
    u = om.Matrix(X=x, Y=y, values=values)
    assert_allclose(u.X_index, x)
    assert_allclose(u.Y_index, y)
    assert_allclose(u.values, values)

def test_math():
    x = np.linspace(-4.5, 324, 1001)
    y = np.linspace(56, 10002.0, 56)
    values = np.random.random((len(x), len(y)))
    u = om.Matrix(X=x, Y=y, values=values)
    ops = [lambda x: x+1, lambda x: x-1, lambda x: x*2, lambda x: x/2, lambda x: -x, lambda x: abs(x)]
    for op in ops:
        v = op(u)
        assert_allclose(v.values, op(values))
        assert v.X_index == u.X_index
        assert v.Y_index == u.Y_index
        assert v.metadata == u.metadata
    ops = [lambda x: 1+x, lambda x: 1-x, lambda x: 2*x, lambda x: 2/x]
    for op in ops:
        v = op(u)
        assert_allclose(v.values, op(values))
        assert v.X_index == u.X_index
        assert v.Y_index == u.Y_index
        assert v.metadata == u.metadata
    ops = [lambda x, y: x+y, lambda x, y: x-y, lambda x, y: x*y, lambda x, y: x/y]
    v = 2.454*(u-3.6)
    v = om.Matrix(Eg=x, Ex=y, values=v.values, xlabel="Eg", ylabel="Ex", name="Test")
    for op in ops:
        w = op(u, v)
        assert_allclose(w.values, op(values, v.values))
        assert w.X_index == u.X_index
        assert w.Y_index == u.Y_index
        assert w.metadata == u.metadata
        w = op(v, u)
        assert_allclose(w.values, op(v.values, values))
        assert w.X_index == v.X_index
        assert w.Y_index == v.Y_index
        assert w.metadata == v.metadata

def test_math_dot():
    x = np.linspace(-4.5, 324, 1001)
    y = np.linspace(56, 10002.0, 56)
    values = np.random.random((len(x), len(y)))
    u = om.Matrix(X=x, Y=y, values=values)
    v = om.Matrix(X=y, Y=x, values=values.T, name="Test", xlabel="Eg", ylabel="Ex")
    w: Matrix = u@v
    assert_allclose(w.values, values@values.T)
    assert w.X_index == u.X_index
    assert w.Y_index == v.Y_index

    walues = np.random.random((len(y)))
    v = om.Vector(X=y, values=walues)
    w: Vector = u@v
    assert_allclose(w.values, values@walues)
    assert w._index == u.X_index

    w: np.ndarray = u@walues
    assert_allclose(w, values@walues)


@pytest.fixture
def matrix():
    x = np.linspace(-4.5, 324, 1001)
    y = np.linspace(56, 10002.0, 56)
    values = np.random.random((len(x), len(y)))
    return om.Matrix(X=x, Y=y, values=values, xlabel="Eg", ylabel="Ex", name="Test", unit='eV')


def test_io_npz(matrix, tmp_path):
    path = tmp_path / "test.npz"
    matrix.save(path)
    loaded = om.Matrix.from_path(path)
    assert_content_equal(matrix, loaded)
    assert_metadata_equal(matrix, loaded)


def test_io_npy(matrix, tmp_path):
    path = tmp_path / "test.npy"
    matrix.save(path)
    loaded = om.Matrix.from_path(path)
    assert_content_compatible(matrix, loaded)


def test_io_txt(matrix, tmp_path):
    path = tmp_path / "test.txt"
    matrix.save(path)
    loaded = om.Matrix.from_path(path)
    assert_content_compatible(matrix, loaded)


def test_io_tar(matrix, tmp_path):
    path = tmp_path / "test.tar"
    matrix.save(path)
    loaded = om.Matrix.from_path(path)
    assert_content_compatible(matrix, loaded)


def test_io_mama(matrix, tmp_path):
    path = tmp_path / "test.m"
    matrix.save(path)
    loaded = om.Matrix.from_path(path)
    assert_content_compatible(matrix, loaded)


def test_sum(matrix):
    x = np.linspace(-4.5, 324, 1001)
    y = np.linspace(56, 10002.0, 56)
    values = np.random.random((len(x), len(y)))
    v = om.Matrix(X=x, Y=y, values=values, xlabel="Eg", ylabel="Ex", name="Test", unit='eV')
    assert_allclose(v.sum(), values.sum())
    u = v.sum(axis=0)
    assert_allclose(u.values, values.sum(axis=0))
    assert u._index == v.Y_index
    #assert u.name == v.name
    assert u.vlabel == v.vlabel
    u = v.sum(axis=1)
    assert_allclose(u.values, values.sum(axis=1))
    assert u._index == v.X_index
    #assert u.name == v.name
    assert u.vlabel == v.vlabel


@pytest.fixture
def matrix_a():
    x = np.linspace(-4.5, 324, 1001)
    y = np.linspace(56, 10002.0, 56)
    values = np.random.random((len(x), len(y)))
    return om.Matrix(X=x, Y=y, values=values, xlabel="Eg", ylabel="Ex", name="A", unit='eV')

@pytest.fixture
def matrix_b():
    x = np.linspace(-4.5, 324, 1001)
    y = np.linspace(56, 10002.0, 56)
    values = np.random.random((len(x), len(y)))
    return om.Matrix(X=x, Y=y, values=values, xlabel="Eg", ylabel="Ex", name="B", unit='eV')

def test_eq(matrix_a, matrix_b):
    assert ((matrix_a == matrix_b) == (matrix_a.values == matrix_b.values)).all()

def test_ne(matrix_a, matrix_b):
    assert ((matrix_a != matrix_b) == (matrix_a.values != matrix_b.values)).all()

def test_lt(matrix_a, matrix_b):
    assert ((matrix_a < matrix_b) == (matrix_a.values < matrix_b.values)).all()

def test_gt(matrix_a, matrix_b):
    assert ((matrix_a > matrix_b) == (matrix_a.values > matrix_b.values)).all()

def test_le(matrix_a, matrix_b):
    assert ((matrix_a <= matrix_b) == (matrix_a.values <= matrix_b.values)).all()

def test_ge(matrix_a, matrix_b):
    assert ((matrix_a >= matrix_b) == (matrix_a.values >= matrix_b.values)).all()
