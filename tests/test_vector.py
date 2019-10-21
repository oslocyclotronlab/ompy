import pytest
import ompy as om
from numpy.testing import assert_equal
import numpy as np


def test_init():
    E = np.linspace(0, 1, 100)
    vals = np.linspace(2, 3.4, 100)
    vec = om.Vector(values=vals, E=E)
    assert_equal(E, vec.E)
    assert_equal(vals, vec.values)

    # No energy defaults to midbin
    vec = om.Vector(vals)
    assert_equal(np.arange(0.5, 100.5, 1), vec.E)
    assert_equal(vals, vec.values)

    # No values defaults to zeros
    vec = om.Vector(E=E)
    assert_equal(np.zeros_like(E), vec.values)

    with pytest.raises(ValueError):
        om.Vector(vals, [1, 2, 3, 4, 5])


def test_cut():
    E = np.arange(-1, 10, 1)
    values = np.linspace(33, 43, 11)
    vector = om.Vector(values=values, E=E)

    vector.cut(Emin=0, Emax=8)

    Ecut = np.arange(0, 9, 1)
    valcut = np.arange(34, 43)
    assert_equal(vector.E, Ecut)
    assert_equal(vector.values, valcut)

@pytest.mark.filterwarnings('ignore:divide by zero encountered in true_divide:RuntimeWarning')
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
