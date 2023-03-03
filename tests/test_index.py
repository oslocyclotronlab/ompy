import pytest
import ompy as om
import numpy as np
from numpy.testing import assert_allclose
from ompy.array.index_fn import is_monotone, is_uniform
from ompy.array.index import LeftUniformIndex, MidUniformIndex, Calibration, Index
from ompy.array.index import LeftNonUniformIndex, MidNonUniformIndex
from ompy.array.index import widths_from_mid_left, widths_from_mid_right
from ompy import Quantity, DimensionalityError, Unit


def test_LeftUniformIndex():
    x = np.array([1, 2, 3, 4, 5]) + 5
    X = LeftUniformIndex(x[:-1], boundary=x[-1])
    assert X.boundary == x[-1]
    assert_allclose(X.bins, x[:-1])


def test_len():
    x = np.array([1, 2, 3, 4, 5]) + 5
    X = LeftUniformIndex(x[:-1], boundary=x[-1])
    assert len(X) == len(x) - 1
    X = MidUniformIndex(x[1:], boundary=x[0])
    assert len(X) == len(x) - 1


def test_from_array():
    x = np.array([1, 2, 3, 4, 5]) + 5
    X = LeftUniformIndex.from_array(x, extrapolate_boundary=False)
    assert X.boundary == x[-1]
    assert_allclose(X.bins, x[:-1])

    X = LeftUniformIndex.from_array(x, extrapolate_boundary=True)
    assert_allclose(X.bins, x)
    assert X.boundary == x[-1] + (x[-1] - x[-2])
    assert len(X) == len(x)


def test_from_array_mid():
    x = np.array([1, 2, 3, 4, 5]) + 5
    X = MidUniformIndex.from_array(x, extrapolate_boundary=False)
    assert X.boundary == x[0] + (x[1] - x[0]) / 2
    assert_allclose(X.bins, x[1:])

    X = MidUniformIndex.from_array(x, extrapolate_boundary=True)
    assert_allclose(X.bins, x)
    assert X.boundary == x[0] - (x[1] - x[0]) / 2
    assert len(X) == len(x)


def test_hash():
    x = np.array([1, 2, 3, 4, 5]) + 5
    X = LeftUniformIndex.from_array(x, extrapolate_boundary=False)
    assert hash(X) == hash(X)
    Y = LeftUniformIndex.from_array(x, extrapolate_boundary=False)
    assert hash(X) == hash(Y)
    Z = LeftUniformIndex.from_array(1.0001 * x, extrapolate_boundary=False)
    assert hash(X) != hash(Z)
    M = MidUniformIndex.from_array(x, extrapolate_boundary=False)
    assert hash(M) == hash(M)
    assert hash(X) != hash(M)


def test_eq():
    x = np.array([1, 2, 3, 4, 5]) + 5
    X = LeftUniformIndex.from_array(x, extrapolate_boundary=False)
    assert X == X
    Y = LeftUniformIndex.from_array(x, extrapolate_boundary=False)
    assert X == Y
    Z = LeftUniformIndex.from_array(1.0001 * x, extrapolate_boundary=False)
    assert X != Z
    M = MidUniformIndex.from_array(x, extrapolate_boundary=False)
    assert M == M
    assert X != M


def test_constructor():
    for cls in (LeftUniformIndex, MidUniformIndex):
        x = np.array([1, 2, 3, 4, 5, 5])
        with pytest.raises(ValueError):
            cls.from_array(x)
        x = np.array([])
        with pytest.raises(ValueError):
            cls(x, boundary=1)
        x = np.array([3, 2, 1])
        with pytest.raises(ValueError):
            cls.from_array(x)
        x = np.array([1, 2, 3, 4, 5, 5.999, 7, 8])
        with pytest.raises(ValueError):
            cls.from_array(x)


def test_monotone():
    x = np.array([1, 2, 3, 4, 5])
    assert is_monotone(x)
    x = np.array([1, 2, 3, 4, 5, 5])
    assert not is_monotone(x)
    x = np.array([1, 2, 3, 4, 5, 4])
    assert not is_monotone(x)
    x = np.array([1, 1, 2, 3, 4])
    assert not is_monotone(x)


def test_uniform():
    x = np.array([1, 2, 3, 4, 5])
    assert is_uniform(x)
    x = np.array([1, 2, 3, 4, 5, 5])
    assert not is_uniform(x)
    x = np.array([1, 2, 3, 4, 5, 4])
    assert not is_uniform(x)
    x = np.array([1, 1, 2, 3, 4])
    assert not is_uniform(x)
    x = np.linspace(-4.3, 1034.2, 10023)
    assert is_uniform(x)
    x[45] = 1.01 * x[45]
    assert not is_uniform(x)


def test_extrapolate():
    x = np.array([3, 4, 5, 6])
    X = LeftUniformIndex.extrapolate(x)
    assert len(X) == len(x) + 1
    assert_allclose(X[:-1], x)
    assert X[-1] == x[-1] + (x[-1] - x[-2])
    X = LeftUniformIndex.extrapolate(x, n=10)
    assert len(X) == len(x) + 10
    assert_allclose(X[:len(x)], x)
    assert X[-1] == x[-1] + (x[-1] - x[-2]) * 10
    X = LeftUniformIndex.extrapolate(x, n=1, direction='left')
    assert len(X) == len(x) + 1
    assert_allclose(X[1:], x)
    assert X[0] == x[0] - (x[1] - x[0])
    X = LeftUniformIndex.extrapolate(x, n=0)
    assert_allclose(X, x)
    X = LeftUniformIndex.extrapolate(x, n=-2, direction='right')
    assert_allclose(X, x[:-2])
    X = LeftUniformIndex.extrapolate(x, n=-2, direction='left')
    assert_allclose(X, x[2:])
    with pytest.raises(ValueError):
        LeftUniformIndex.extrapolate(x, direction='both')
    with pytest.raises(ValueError):
        LeftUniformIndex.extrapolate(x, n=-5)

    # Similar tests, but for Mid
    x = np.array([3, 4, 5, 6])
    X = MidUniformIndex.extrapolate(x)
    assert len(X) == len(x) + 1
    assert_allclose(X[1:], x)
    assert X[0] == x[0] - (x[1] - x[0])
    X = MidUniformIndex.extrapolate(x, n=10)
    assert len(X) == len(x) + 10
    assert_allclose(X[10:], x)
    assert X[0] == x[0] - (x[1] - x[0]) * 10
    X = MidUniformIndex.extrapolate(x, n=1, direction='right')
    assert len(X) == len(x) + 1
    assert_allclose(X[:-1], x)
    assert X[-1] == x[-1] + (x[-1] - x[-2])
    X = MidUniformIndex.extrapolate(x, n=0)
    assert_allclose(X, x)
    X = MidUniformIndex.extrapolate(x, n=-2, direction='right')
    assert_allclose(X, x[:-2])
    X = MidUniformIndex.extrapolate(x, n=-2, direction='left')
    assert_allclose(X, x[2:])
    with pytest.raises(ValueError):
        MidUniformIndex.extrapolate(x, direction='both')
    with pytest.raises(ValueError):
        MidUniformIndex.extrapolate(x, n=-5)


def test_to_left_mid():
    x = np.array([3, 4, 5, 6])
    X = LeftUniformIndex.from_array(x)
    assert X == X.to_left()
    assert X != X.to_mid()
    Y = MidUniformIndex.from_array(x)
    assert Y == Y.to_mid()
    assert Y != Y.to_left()
    assert X.to_mid().to_left() == X
    assert Y.to_left().to_mid() == Y


def test_index_left():
    x = np.linspace(-4.3, 1034.2, 103)
    X = LeftUniformIndex.from_array(x)
    for i in range(len(x)):
        assert X.index(x[i]) == i
    dx = x[1] - x[0]
    eps = np.random.random(len(x)) * dx
    for i in range(len(x)):
        assert X.index(x[i] + eps[i]) == i
    with pytest.raises(IndexError):
        X.index(x[0] - 1e-6)
    with pytest.raises(IndexError):
        X.index(x[-1] + (x[-1] - x[-2]) + 1e-6)


def test_index_mid():
    x = np.linspace(-4.3, 1034.2, 103)
    X = MidUniformIndex.from_array(x)
    for i in range(len(x)):
        assert X.index(x[i]) == i
    dx = x[1] - x[0]
    eps = (np.random.random(len(x)) - 0.5) * dx
    for i in range(len(x)):
        assert X.index(x[i] + eps[i]) == i
    with pytest.raises(IndexError):
        X.index(x[0] - (x[1] - x[0]) / 2 - 1e-6)
    with pytest.raises(IndexError):
        X.index(x[-1] + (x[-1] - x[-2]) / 2 + 1e-6)


def test_index_units():
    x = np.linspace(-4.3, 1034.2, 103)
    X = LeftUniformIndex.from_array(x)
    keV = Quantity(1, 'keV')
    MeV = Quantity(1, 'MeV')
    for i in range(len(x)):
        assert X.index(x[i] * keV) == i
    for i in range(len(x)):
        v = x[i] * MeV / 1e3 + 1 * keV
        assert X.index(v) == i
    with pytest.raises(DimensionalityError):
        X.index(Quantity(1, 'kg'))


def test_unit_conversion():
    x = np.linspace(-4.3, 1034.2, 103)
    X = LeftUniformIndex.from_array(x)
    keV = Unit('keV')
    MeV = Unit('MeV')

    Y = X.to_unit(keV)
    assert Y == X
    Y = X.to_unit(MeV)
    assert Y.unit == Unit('MeV')
    assert_allclose(Y.bins, X.bins / 1e3)
    assert_allclose(Y.boundary, X.boundary / 1e3)
    Z = X.to_unit(MeV).to_unit(keV)
    assert_allclose(Z.bins, X.bins)
    assert_allclose(Z.boundary, X.boundary)


def test_parse():
    """ Test the parsing of slices from ints, floats, Quantity and strings to slices of ints """
    x = np.linspace(-4.3, 1034.2, 103)
    X = LeftUniformIndex.from_array(x)
    s = slice(0, 10)
    assert X.index_slice(s, strict=False) == s
    i = X.index(0)
    j = X.index(10)
    assert X.index_slice(s, strict=True) == slice(i, j)
    s = slice(None, 10)
    assert X.index_slice(s, strict=True) == slice(None, j)
    s = slice(0, None)
    assert X.index_slice(s, strict=True) == slice(i, None)
    s = slice(20.65, Quantity(300.2, 'keV'))
    i = X.index(20.65)
    j = X.index(300.2)
    assert X.index_slice(s, strict=True) == slice(i, j)
    x = X[70]
    y = X[81]
    s = slice(f"< {x} keV", f"> {y} keV")
    S = X.index_slice(s, strict=True)
    assert S == slice(69, 82)

    assert X.index_expression(5, strict=False) == 5
    assert X.index_expression(5, strict=True) == X.index(5)
    assert X.index_expression(Quantity(5, 'keV'), strict=True) == X.index(5)
    assert X.index_expression(f"< {x} keV", strict=True) == 69
    assert X.index_expression(f"{x} keV", strict=True) == 70
    assert X.index_expression(f"> {x} keV", strict=True) == 71

def test_calibration():
    c = Calibration([1, 2])
    assert len(c) == 2
    assert c.order == 1
    assert c[0] == 1
    assert c[1] == 2
    assert c.a0 == 1
    assert c.a1 == 2
    assert c.start == 1
    assert c.step == 2
    assert_allclose(c.linspace(101), np.linspace(1, 1 + 2*100, 101))
    cs = {'a0': 1, 'a1': 2, 'a2': 0.01}
    c = Calibration.from_dict(cs)
    assert c.order == 2
    assert c.a0 == 1
    assert c.a1 == 2
    assert c.a2 == 0.01
    with pytest.raises(ValueError):
        c = Calibration([1])
    with pytest.raises(ValueError):
        c = Calibration([])
    x = np.linspace(-4.3, 1034.2, 103)
    X = LeftUniformIndex.from_array(x)
    c = X.to_calibration()
    print(c)
    print(c.linspace(103))
    Y = LeftUniformIndex.from_calibration(c, len(X))
    assert X == Y

def test_to_dict_uniform():
    x = np.linspace(-4.3, 1034.2, 103)
    X = LeftUniformIndex.from_array(x, label='Fishes')
    d = X.to_dict()
    Y = Index.from_dict(d)
    assert X == Y
    X = MidUniformIndex.from_array(x, label='Fishes')
    d = X.to_dict()
    Y = Index.from_dict(d)
    assert X == Y

def test_to_same():
    x = np.linspace(-4.3, 1034.2, 103)
    X = LeftUniformIndex.from_array(x, label='Fishes')
    Y = MidUniformIndex.from_array(0.2*x, unit='MeV')
    Z = X.to_same(Y)
    assert Z.unit == Y.unit
    assert Z.is_mid()
    assert Z.alias == X.alias
    assert Z.label == X.label
    Z = Y.to_same(X)
    assert Z.unit == X.unit
    assert Z.is_left()
    assert Z.alias == Y.alias
    assert Z.label == Y.label
    Z = X.to_same(Y).to_same(X)
    assert Z.meta == X.meta
    assert_allclose(Z.bins, X.bins)
    assert_allclose(Z.boundary, X.boundary)
    Z = X.to_same(X)
    assert X == Z

def test_is_compatible_uniform():
    x = np.linspace(-4.3, 1034.2, 103)
    X = LeftUniformIndex.from_array(x, label='Fishes')
    Y = X.to_mid()
    assert X.is_compatible_with(Y)
    Y = X.to_unit('MeV')
    assert X.is_compatible_with(Y)
    Y = X.to_mid().to_unit('MeV')
    assert X.is_compatible_with(Y)
    assert Y.is_compatible_with(X)
    Y = LeftUniformIndex.from_array(1e-3+x)
    assert not X.is_compatible_with(Y)
    Y = MidUniformIndex.from_array(x)
    assert not X.is_compatible_with(Y)
    assert not Y.is_compatible_with(X)
    Y = X[1:]
    assert not X.is_compatible_with(Y)
    Y = LeftUniformIndex.from_array(x, label='Fishes', unit='MeV')
    assert not X.is_compatible_with(Y)
    Y = LeftUniformIndex.from_array(x, label='Fishes', unit='kg')
    assert not X.is_compatible_with(Y)

def test_widths_from_midt_left():
    dx = np.asarray([1, 1, 2, 3, 3, 2, 5, 5], dtype=float)
    x = np.empty(len(dx) + 1)
    x[0] = 0
    x[1:] = x[0] + np.cumsum(dx)
    dx_ = np.append(dx, 7)
    m = x + 1/2*dx_
    widths = widths_from_mid_left(m, x[0])
    assert_allclose(widths, dx_)

    dx = np.random.random(500)
    x = np.empty(len(dx) + 1)
    x[0] = -5*np.random.rand()
    x[1:] = x[0] + np.cumsum(dx)
    dx_ = np.append(dx, 7.3)
    m = x + 1/2*dx_
    widths = widths_from_mid_left(m, x[0])
    assert_allclose(widths, dx_)

def test_widths_from_mid_right():
    dx = np.asarray([1, 1, 2, 3, 3, 2, 5, 5], dtype=float)
    x = np.empty(len(dx) + 1)
    x[0] = 0
    x[1:] = x[0] + np.cumsum(dx)
    dx_ = np.append(dx, 7)
    m = x + 1/2*dx_
    widths = widths_from_mid_right(m, x[-1] + dx_[-1])
    assert_allclose(widths, dx_)

    dx = np.random.random(500)
    x = np.empty(len(dx) + 1)
    x[0] = -5*np.random.rand()
    x[1:] = x[0] + np.cumsum(dx)
    dx_ = np.append(dx, 7.3)
    m = x + 1/2*dx_
    widths = widths_from_mid_right(m, x[-1] + dx_[-1])
    assert_allclose(widths, dx_)

def test_nonuniform():
    x = np.asarray([0, 1, 2, 4, 6, 10, 20, 50], dtype=float)
    i = LeftNonUniformIndex.from_array(x)
    assert_allclose(i.bins, x)
    assert_allclose([i.boundary], [x[-1] + (x[-1] - x[-2])])

    # Test idempotency
    k = i.to_left()
    assert k == i

    # Test conversion
    k = i.to_mid().to_left()
    assert k.is_content_close(i)
    assert k.is_metadata_equal(i)

    # Mid is more complicated as it lacks a degree of freedom that is
    # impossible to recover
    with pytest.raises(ValueError):
        MidNonUniformIndex.from_array(x)
    with pytest.raises(ValueError):
        MidNonUniformIndex.from_array(x, boundary=5, width=5)
    with pytest.raises(ValueError):
        MidNonUniformIndex.from_array(x, boundary=5, direction='m')
    with pytest.raises(ValueError):
        MidNonUniformIndex.from_array(x, width=5, direction='m')
    with pytest.raises(ValueError):
        MidNonUniformIndex.from_array(x, boundary=-1)

    N = 500
    dx = np.random.random(N)
    x = np.empty(len(dx) + 1)
    x[0] = -5*np.random.rand()
    x[1:] = x[0] + np.cumsum(dx)
    dx_ = np.append(dx, 7.3)
    m = x + 1/2*dx_
    j = MidNonUniformIndex.from_array(m, boundary=x[0])
    assert_allclose(j.boundary, x[0])
    assert_allclose(j.bins, m)
    assert_allclose(j.steps(), dx_)
    # idempotency
    assert j.to_mid() == j
    # Conversion
    k = j.to_left().to_mid()
    assert k.is_content_close(j)
    assert k.is_metadata_equal(j)

    width = dx[0]
    k = MidNonUniformIndex.from_array(m, width=width)
    assert_allclose(k.boundary, x[0])
    assert_allclose(k.bins, m)
    assert_allclose(k.steps(), dx_)
    assert k.is_content_close(j)
    assert k.is_metadata_equal(j)
    k = MidNonUniformIndex.from_array(m, boundary=x[-1] + dx_[-1], direction='right')
    assert k.is_content_close(j)
    assert k.is_metadata_equal(j)
    k = MidNonUniformIndex.from_array(m, width=dx_[-1], direction='right')
    assert k.is_content_close(j)
    assert k.is_metadata_equal(j)

    with pytest.raises(ValueError):
        MidNonUniformIndex.from_array(m, extrapolate_boundary=True, width=5)
    with pytest.raises(ValueError):
        MidNonUniformIndex.from_array(m, extrapolate_boundary=True, boundary=5)

    dx = np.asarray([3, 3, 3, 4, 5, 6, 7, 7])
    x = np.empty(len(dx) + 1)
    x[0] = -5*np.random.rand()
    x[1:] = x[0] + np.cumsum(dx)
    dx_ = np.append(dx, 7)
    m = x + 1/2*dx_
    j = MidNonUniformIndex.from_array(m, extrapolate_boundary=True)
    assert_allclose(j.boundary, x[0])
    assert_allclose(j.bins, m)
    assert_allclose(j.steps(), dx_)
    j = MidNonUniformIndex.from_array(m, extrapolate_boundary=True, direction='right')
    assert_allclose(j.boundary, x[0])
    assert_allclose(j.bins, m)
    assert_allclose(j.steps(), dx_)


def test_nonuniform_index():
    N = 500
    dx = np.random.random(N)
    x = np.empty(len(dx) + 1)
    x[0] = 5 * np.random.rand()
    x[1:] = x[0] + np.cumsum(dx)
    dx_ = np.append(dx, 7.3)
    m = x + 1 / 2 * dx_
    M = MidNonUniformIndex.from_array(m, boundary=x[0])
    assert_allclose(M.bins, m)
    assert_allclose(M.steps(), dx_)
    L = LeftNonUniformIndex.from_array(x)
    samples = x + np.random.random(len(x)) * dx_
    for i, s in enumerate(samples[:-1]):
        assert_allclose(L.index(s), i)
    samples = m + 1.9*(np.random.random(len(m)) - 0.5) * dx_/2
    for i, s in enumerate(samples):
        j = M.index(s)
        assert_allclose(M.index(s), i)
    last = x[-1] + dx_[-1]
    with pytest.raises(IndexError):
        M.index(x[0] - 1e-6)
    with pytest.raises(IndexError):
        M.index(last + 1e-6)
    with pytest.raises(IndexError):
        L.index(x[0] - 1e-6)
    with pytest.raises(IndexError):
        L.index(last + 1e-6)
    L = LeftNonUniformIndex.from_array(np.append(x, last),
                                       extrapolate_boundary=False)
    assert M.index(last - 1e-6) == N
    assert L.index(last - 1e-6) == N
    assert M.index(x[0] + 1e-6) == 0
    assert L.index(x[0] + 1e-6) == 0
