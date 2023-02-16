import pytest
from ompy.array import Vector, Index
from ompy.array.index import LeftUniformIndex, MidUniformIndex
import numpy as np
from numpy.testing import assert_allclose
from ompy import Unit
import ompy as om

"""
TODO:
If meta are equal, clone the meta. 
"""


def test_simple_constructor():
    x = np.linspace(0.0, 1232, 122)
    v = 2*x**2
    V = Vector(X=x, Y=v)
    with pytest.raises(ValueError):
        Vector(X=x, Y=v[:-1])
    with pytest.raises(ValueError):
        Vector(X=x[:-1], Y=v)
    assert_allclose(V.X, x)
    assert_allclose(V.values, v)

def test_index_constructor():
    x = np.linspace(0.0, 1232, 122)
    v = 2*x**2
    i = LeftUniformIndex.from_array(x)
    V = Vector(X=i, Y=v)
    assert_allclose(V.X, x)
    assert_allclose(V.values, v)
    assert V._index == i

    j = i.update_metadata(alias='Fish')
    U = Vector(X=j, Y=v)
    assert U._index == j
    assert U._index.alias == 'Fish'

def test_metadata_constructor():
    x = np.linspace(0.0, 1232, 122)
    v = 2*x**2
    V = Vector(X=x, values=v)
    assert V.unit == Unit('keV')
    assert V.xlabel == 'Energy'
    assert V.ylabel == 'Counts'
    assert V.name == ''
    assert V.alias == ''
    assert V.valias == ''
    V = Vector(X=x, values=v, unit='MeV', xlabel='E', vlabel='C', name='test', xalias='t', valias='c')
    assert V.unit == Unit('MeV')
    assert V.xlabel == 'E'
    assert V.ylabel == 'C'
    assert V.name == 'test'
    assert V.alias == 't'
    assert V.valias == 'c'
    V = Vector(E=x, values=v)
    assert V.alias == 'E'
    with pytest.raises(ValueError):
        Vector(X=x, Y=v, E=x)
    with pytest.raises(ValueError):
        Vector(X=x, Y=v, values=v)
    with pytest.raises(ValueError):
        Vector(E=x, values=v, xalias='I')
    with pytest.raises(ValueError):
        Vector(E=x, Y=v, valias='I')
    V = Vector(X=x, values=v, xalias='E', valias='C')
    assert V.alias == 'E'
    assert V.valias == 'C'
    V = Vector(E=x, C=V)
    assert V.alias == 'E'
    assert V.valias == 'C'

    x = LeftUniformIndex.from_array(x).to_unit('MeV')
    V = Vector(X=x, values=v)
    assert V.unit == Unit('MeV')

def test_index():
    x = np.linspace(-6.7, 1232, 122)
    v = 2*x**2
    w = np.copy(v)
    V = Vector(X=x, values=v)
    for i, (x, v) in enumerate(zip(x, v)):
        assert V.index(x) == i
        assert V[i] == v
    with pytest.raises(IndexError):
        V.index(-7)
    with pytest.raises(IndexError):
        V.index(1243)
    V[0] = 0
    assert V[0] == 0
    V[-1] = 0.242
    assert V[-1] == 0.242
    t = np.random.rand(15)
    V[5:20] = t
    assert_allclose(V[5:20], t)

def test_iloc():
    x = np.linspace(-6.7, 1232, 122)
    v = 2*x**2
    w = np.copy(v)
    V = Vector(X=x, values=v, xlabel="Fish", vlabel="Counts")
    for i, (xx, vv) in enumerate(zip(x, v)):
        assert V.iloc[i] == vv
    W = V.iloc[5:20]
    assert_allclose(W.X, x[5:20])
    assert_allclose(W.values, v[5:20])
    assert W.metadata == V.metadata
    V.iloc[6:30] += 10
    assert_allclose(V.values[6:30], w[6:30] + 10)

def test_vloc():
    x = np.linspace(-6.7, 1232, 122)
    v = 2*x**2
    w = np.copy(v)
    V = Vector(X=x, values=v, xlabel="Fish", vlabel="Counts")
    for i, (xx, vv) in enumerate(zip(x, v)):
        assert V.vloc[xx] == vv

    start = x[5]
    stop = x[20]
    W = V.vloc[start:stop]
    assert_allclose(W.X, x[5:20])
    assert_allclose(W.values, v[5:20])
    assert W.metadata == V.metadata
    V.vloc[start:stop] += 10
    assert_allclose(V.values[5:20], w[5:20] + 10)

def test_loc():
    x = np.linspace(-6.7, 1232, 122)
    v = 2*x**2
    w = np.copy(v)
    V = Vector(X=x, values=v, xlabel="Fish", vlabel="Counts")
    for i, (xx, vv) in enumerate(zip(x, v)):
        assert V.loc[xx] == vv
        assert V.loc[i] == vv

    s0 = x[5]
    s1 = x[20]
    for start, stop in zip((5, s0, 5, s0), (20, s1, s1, 20)):
        W = V.loc[start:stop]
        assert_allclose(W.X, x[5:20])
        assert_allclose(W.values, v[5:20])
        assert W.metadata == V.metadata

    V.loc[5:s1] += 10
    assert_allclose(V.values[5:20], w[5:20] + 10)

def test_copy():
    x = np.linspace(-6.7, 1232, 122)
    v = 2*x**2
    w = np.copy(v)
    V = Vector(X=x, values=v, copy=False)
    V.iloc[5:20] += 10
    assert_allclose(V[5:20], v[5:20])
    W = Vector(X=x, values=w, copy=True)
    W.iloc[5:20] += 10
    assert_allclose(W[5:20], w[5:20] + 10)

def test_math_add():
    x = np.linspace(-6.7, 1232, 122)
    v = 2*x**2
    w = np.random.rand(len(x))
    V = Vector(X=x, values=v, xlabel='Test', vlabel='Count')
    W = Vector(X=x, values=w)
    Z = 2 + V
    assert_allclose(Z.values, 2 + v)
    assert Z.metadata == V.metadata
    assert Z._index == V._index
    Z = V + 2
    assert_allclose(Z.values, v + 2)
    assert Z.metadata == V.metadata
    assert Z._index == V._index

    Z = V + W
    assert_allclose(Z.values, v + w)
    assert Z.metadata == V.metadata
    assert Z._index == V._index

    Z = W + V
    assert_allclose(Z.values, w + v)
    assert Z.metadata == W.metadata
    assert Z._index == W._index

    W = Vector(X=x+1, values=w)
    with pytest.raises(ValueError):
        V + W
    with pytest.raises(ValueError):
        W + V
    Z = V + w
    assert_allclose(Z.values, v+w)
    Z = w + V
    #assert_allclose(Z.values, w+v)
    with pytest.raises(ValueError):
        V + np.random.rand(10)
    with pytest.raises(ValueError):
        np.random.rand(10) + v

def test_math_sub():
    x = np.linspace(-6.7, 1232, 122)
    v = 2*x**2
    w = np.random.rand(len(x))
    V = Vector(X=x, values=v, xlabel='Test', vlabel='Count')
    W = Vector(X=x, values=w)
    Z = 2 - V
    assert_allclose(Z.values, 2 - v)
    assert Z.metadata == V.metadata
    assert Z._index == V._index
    Z = V - 2
    assert_allclose(Z.values, v - 2)
    assert Z.metadata == V.metadata
    assert Z._index == V._index

    Z = V - W
    assert_allclose(Z.values, v - w)
    assert Z.metadata == V.metadata
    assert Z._index == V._index

    Z = W - V
    assert_allclose(Z.values, w - v)
    assert Z.metadata == W.metadata
    assert Z._index == W._index

    W = Vector(X=x-1, values=w)
    with pytest.raises(ValueError):
        V - W
    with pytest.raises(ValueError):
        W - V
    Z = V - w
    assert_allclose(Z.values, v - w)
    Z = w - V
    #assert_allclose(Z.values, w - v)
    with pytest.raises(ValueError):
        V - np.random.rand(10)
    with pytest.raises(ValueError):
        np.random.rand(10) - V

def test_math_mul():
    x = np.linspace(-6.7, 1232, 122)
    v = 2*x**2
    w = np.random.rand(len(x))
    V = Vector(fish=x, values=v, xlabel='Test', vlabel='Count')
    W = Vector(duck=x, values=w)
    Z = 2 * V
    assert_allclose(Z.values, 2 * v)
    assert Z.metadata == V.metadata
    assert Z._index == V._index
    assert Z._index.meta == V._index.meta
    Z = V * 2
    assert_allclose(Z.values, v * 2)
    assert Z.metadata == V.metadata
    assert Z._index == V._index
    assert Z._index.meta == V._index.meta

    Z = V * W
    assert_allclose(Z.values, v * w)
    assert Z.metadata == V.metadata
    assert Z._index == V._index

    Z = W * V
    assert_allclose(Z.values, w * v)
    assert Z.metadata == W.metadata
    assert Z._index == W._index

    W = Vector(X=x-1, values=w)
    with pytest.raises(ValueError):
        V * W
    with pytest.raises(ValueError):
        W * V
    Z = V * w
    assert_allclose(Z.values, v*w)
    Z = w * V
    #assert_allclose(Z.values, v*w)
    with pytest.raises(ValueError):
        V * np.random.rand(10)
    with pytest.raises(ValueError):
        np.random.rand(10) * V

def test_math_div():
    x = np.linspace(-6.7, 1232, 122)
    v = 2*x**2
    w = np.random.rand(len(x))
    V = Vector(X=x, values=v, xlabel='Test', vlabel='Count')
    W = Vector(X=x, values=w)
    Z = 2 / V
    assert_allclose(Z.values, 2 / v)
    assert Z.metadata == V.metadata
    assert Z._index == V._index
    Z = V / 2
    assert_allclose(Z.values, v / 2)
    assert Z.metadata == V.metadata
    assert Z._index == V._index

    Z = V / W
    assert_allclose(Z.values, v / w)
    assert Z.metadata == V.metadata
    assert Z._index == V._index

    Z = W / V
    assert_allclose(Z.values, w / v)
    assert Z.metadata == W.metadata
    assert Z._index == W._index

    W = Vector(X=x-1, values=w)
    with pytest.raises(ValueError):
        V / W
    with pytest.raises(ValueError):
        W / V
    Z = V / w
    assert_allclose(Z.values, v/w)
    Z = v / W
    #assert_allclose(Z.values, w/v)
    with pytest.raises(ValueError):
        V / np.random.rand(10)
    with pytest.raises(ValueError):
        np.random.rand(10) / V

import tempfile
def is_equal(v, w, values: bool = True, index: bool = True,
             meta: bool = True, unit: bool = True) -> bool:
    if index:
        assert v._index == w._index
    if meta:
        assert v.metadata == w.metadata
    if values:
        assert_allclose(v.values, w.values)

def is_compatible(v, w) -> bool:
    assert_allclose(v.values, w.values)
    assert v.is_compatible_with(w)

def test_io_npz():
    x = np.linspace(-6.7, 1232, 122)
    v = 2*x**2
    V = Vector(X=x, values=v, xlabel='Test', vlabel='Count')
    with tempfile.NamedTemporaryFile() as f:
        V.save(f.name)
        W = Vector.from_path(f.name)
        is_equal(V, W)
    with tempfile.NamedTemporaryFile() as f:
        V.save(f.name, filetype='npz')
        W = Vector.from_path(f.name, filetype='npz')
        is_equal(V, W)
    with tempfile.NamedTemporaryFile() as f:
        V.save(f.name)
        W = Vector.from_path(f.name, filetype='npz')
        is_equal(V, W)
    with tempfile.NamedTemporaryFile() as f:
        V.save(f.name, filetype='npz')
        W = Vector.from_path(f.name)
        is_equal(V, W)
    V = V.to_mid().to_unit('MeV')
    with tempfile.NamedTemporaryFile() as f:
        V.save(f.name)
        W = Vector.from_path(f.name)
        is_equal(V, W)

def test_io_mama():
    x = np.linspace(-6.7, 1232, 122)
    v = 2*x**2
    V = Vector(X=x, values=v, xlabel='Test', vlabel='Count')
    with tempfile.NamedTemporaryFile() as f:
        V.save(f.name, filetype='mama')
        W = Vector.from_path(f.name, filetype='mama')
        is_compatible(V, W)

def test_io_npy():
    x = np.linspace(-6.7, 1232, 122)
    v = 2 * x ** 2
    V = Vector(X=x, values=v, xlabel='Test', vlabel='Count')
    with tempfile.NamedTemporaryFile() as f:
        V.save(f.name, filetype='npy')
        W = Vector.from_path(f.name, filetype='npy')
        is_compatible(V, W)
def test_io_txt():
    x = np.linspace(-6.7, 1232, 122)
    v = 2 * x ** 2
    V = Vector(X=x, values=v, xlabel='Test', vlabel='Count')
    with tempfile.NamedTemporaryFile() as f:
        V.save(f.name, filetype='txt')
        W = Vector.from_path(f.name, filetype='txt')
        is_compatible(V, W)

def test_io_csv():
    x = np.linspace(-6.7, 1232, 122)
    v = 2 * x ** 2
    V = Vector(X=x, values=v, xlabel='Test', vlabel='Count')
    with tempfile.NamedTemporaryFile() as f:
        V.save(f.name, filetype='csv')
        W = Vector.from_path(f.name, filetype='csv')
        is_compatible(V, W)

def test_mathmul():
    x = np.linspace(-4, 4, 100)
    y = np.random.random(len(x))
    u = Vector(X=x, values=y)
    t = np.random.random((len(x)))
    v = Vector(X=x, values=t, xlabel='Test', vlabel='Count')
    assert_allclose(u@v, y@t)

    y_ = np.linspace(0, 1, 251)
    u_ = np.random.random((len(x), len(y_)))
    u = om.Matrix(X=x, Y=y_, values=u_)
    w = v@u
    assert_allclose(w.values, t@u_)
    assert w._index == u.Y_index

    assert_allclose(v@u_, t@u_)
    assert_allclose(v@t, t@t)