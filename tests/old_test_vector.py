import pytest
import ompy as om
import warnings
from numpy.testing import assert_equal, assert_allclose
import numpy as np


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


def assert_vector(vec, fact):
    compare_unitful(vec.E_true, fact.E_true)
    assert_allclose(vec.values, fact.values)
    if fact.std is None:
        assert_equal(vec.std, None)
    else:
        assert_allclose(vec.std, fact.std)


class TestInit:
    def test_zero_size(self):
        with pytest.raises(ValueError):
            _ = om.Vector()

    def test_energy(self):
        vals = np.linspace(-1, 2, 100)
        E = np.linspace(0, 1, 100)
        v = om.Vector(vals, E=E)
        assert_allclose(v.E, E)
        assert_equal(v.E.shape, v.values.shape)

    def test_values(self):
        with pytest.raises(ValueError):
            _ = om.Vector(values=[1, 2, 3, 4])
        return
        values = np.linspace(-4, 5, 100)
        v = om.Vector(values)
        v2 = om.Vector(values=values)
        assert_allclose(v.values, v2.values)
        assert_allclose(v.values, values)

    def test_both(self):
        E = np.linspace(0, 1, 100)
        vals = np.linspace(2, 3.4, 100)
        vec = om.Vector(values=vals, E=E)
        vec2 = om.Vector(vals, E)
        compare_unitful(E, vec.E)
        assert_allclose(vals, vec.values)
        assert_allclose(vec2.values, vec.values)
        compare_unitful(vec2.E, vec.E)

        # Wrong size
        with pytest.raises(ValueError):
            om.Vector(vals, [1, 2, 3, 4, 5])

        with pytest.raises(ValueError):
            om.Vector(vals, E, std=[1, 2, 3])

        with pytest.raises(ValueError):
            vec = om.Vector(vals, E=np.random.random(100))
            vec.verify_integrity(check_equidistant=True)

        vec = om.Vector([1.0], [2.0])
        assert vec.is_equidistant()


    def test_unit(self):
        val = np.linspace(0, 1, 100)
        E = np.linspace(0, 1, 100)
        vec = om.Vector(val, E=E)
        assert(vec.units == om.ureg('keV'))
        val = np.linspace(0, 1, 10)
        E = np.linspace(0, 100, 10)*om.ureg('MeV')
        vec2 = om.Vector(val, E=E)
        assert(vec2.units == om.ureg('MeV'))
        E = np.linspace(0, 100, 10)
        vec3 = om.Vector(val, E=E, units='MeV')
        assert(vec3.units == om.ureg('MeV'))

    def test_reference(self):
        E = np.linspace(0, 1)
        val = 2*E
        std = 0.1*E
        # Numpy seems to copy sometimes and use reference other times.
        vec = om.Vector(E=E, values=val, copy=False, std=std)
        assert_allclose(vec.E, E)
        assert_allclose(vec.values, val)
        assert_allclose(vec.std, std)

    def test_load(self):
        E = np.linspace(0, 1)
        val = 2*E
        std = 0.1*E
        vec = om.Vector(E=1e-3*E, values=val, std=std, units='MeV')
        vec.save("/tmp/test.npy")
        vec = om.Vector(path="test.npy")
        assert_allclose(vec.E, E)
        assert_allclose(vec.values, val)
        assert_allclose(vec.std, std)

        with pytest.warns(UserWarning):
            om.Vector(path='test.npy', values=val)


class TestArithmetic:
    def test_mul(self):
        values = np.array([1, 2, 3, 4, 5])
        E = np.linspace(0, 100, 5)
        vec = om.Vector(values=values, E=E)
        assert_allclose(values*values, (vec*values).values)
        assert_allclose(5*values, (vec*5).values)
        assert_allclose(values*values, (vec*vec).values)

        vec2 = om.Vector(values=2*values, E=E)
        assert_allclose(2*values*values, (vec*vec2).values)

        vec3 = om.Vector(values=1+values, E=1e-3*E, units='MeV')
        assert_allclose((1+values)*values, (vec*vec3).values)

        vec4 = om.Vector(values=2*values, E=2*E)
        with pytest.raises(ValueError):
            vec*vec4

        vec5 = om.Vector(values=2*values, E=E, units='MeV')
        with pytest.raises(ValueError):
            vec*vec5

        with pytest.raises(ValueError):
            vec*np.linspace(0, 1)

    def test_add(self):
        values = np.array([1, 2, 3, 4, 5])
        E = np.linspace(0, 100, 5)
        vec = om.Vector(values=values, E=E)
        assert_allclose(values+values, (vec+values).values)
        assert_allclose(5+values, (vec+5).values)
        assert_allclose(values+values, (vec+vec).values)

        vec2 = om.Vector(values=2*values, E=E)
        assert_allclose(2*values+values, (vec+vec2).values)

        vec3 = om.Vector(values=1+values, E=1e-3*E, units='MeV')
        assert_allclose((1+values)+values, (vec+vec3).values)

        vec4 = om.Vector(values=2*values, E=2*E)
        with pytest.raises(ValueError):
            vec+vec4

        vec5 = om.Vector(values=2*values, E=E, units='MeV')
        with pytest.raises(ValueError):
            vec+vec5

        with pytest.raises(ValueError):
            vec+np.linspace(0, 1)

    def test_sub(self):
        values = np.array([1, 2, 3, 4, 5])
        E = np.linspace(0, 100, 5)
        vec = om.Vector(values=values, E=E)
        assert_allclose(values-values, (vec-values).values)
        assert_allclose(values-5, (vec-5).values)
        assert_allclose(5-values, (5-vec).values)
        assert_allclose(values-values, (vec-vec).values)

        vec2 = om.Vector(values=2*values, E=E)
        assert_allclose(2*values-values, (vec2 - vec).values)

        vec3 = om.Vector(values=1+values, E=1e-3*E, units='MeV')
        assert_allclose((1+values)-values, (vec3-vec).values)

        vec4 = om.Vector(values=2*values, E=2*E)
        with pytest.raises(ValueError):
            vec-vec4

        vec5 = om.Vector(values=2*values, E=E, units='MeV')
        with pytest.raises(ValueError):
            vec-vec5

        with pytest.raises(ValueError):
            vec-np.linspace(0, 1)

    def test_div(self):
        values = np.array([1, 2, 3, 4, 5])
        E = np.linspace(0, 100, 5)
        vec = om.Vector(values=values, E=E)
        assert_allclose(values/values, (vec/values).values)
        assert_allclose(values/5, (vec/5).values)
        assert_allclose(5/values, (5/vec).values)
        assert_allclose(values/values, (vec/vec).values)

        vec2 = om.Vector(values=2*values, E=E)
        assert_allclose(2*values/values, (vec2 / vec).values)

        vec3 = om.Vector(values=1+values, E=1e-3*E, units='MeV')
        assert_allclose((1+values)/values, (vec3/vec).values)

        vec4 = om.Vector(values=2*values, E=2*E)
        with pytest.raises(ValueError):
            vec/vec4

        vec5 = om.Vector(values=2*values, E=E, units='MeV')
        with pytest.raises(ValueError):
            vec/vec5

        with pytest.raises(ValueError):
            vec/np.linspace(0, 1)


class TestIndex:
    def test_dimensionless(self):
        E = np.linspace(-2.1, 9.9, 100)
        val = np.random.random(size=100)
        vec = om.Vector(values=val, E=E)
        for i, e in enumerate(E):
            assert_equal(i, vec.index(e))
        assert_equal(0, vec.index(-10))
        assert_equal(99, vec.index(100))

    def test_with_units(self):
        E = np.linspace(-2.1, 9.9, 100)
        val = np.random.random(size=100)
        vec = om.Vector(values=val, E=E.copy())
        E *= om.ureg.keV
        E = E.to('MeV')
        for i, e in enumerate(E):
            assert_equal(i, vec.index(e))
        assert_equal(0, vec.index(-10*om.u.keV))
        assert_equal(99, vec.index(0.1*om.u.MeV))


class TestRebin:
    def test_mids(self):
        E = np.linspace(0, 1, 10)
        values = 3*E
        v = om.Vector(values, E)
        newE = np.linspace(0, 1, 5)
        v.rebin(mids=newE, inplace=True)
        assert_allclose(v.E, newE)
        assert_equal(v.values.shape, newE.shape)

        with pytest.raises(ValueError):
            v.rebin(mids=E, factor=0.5)

        with pytest.raises(ValueError):
            v.rebin(factor=-0.1)

        with pytest.raises(ValueError):
            v.rebin(factor=0.0)

    def test_mids_with_unit(self):
        E = np.linspace(0, 1, 10)
        values = 3*E
        v = om.Vector(values, E)

        newE = np.linspace(0, 1, 6) * om.ureg.keV
        newE = newE.to('MeV')

        v.rebin(mids=newE, inplace=True)
        assert_allclose(v.E, newE.to_unit('keV').magnitude)
        assert_equal(v.values.shape, newE.shape)

    def test_binwidth(self):
        E = np.linspace(0, 10, 100)
        values = 2*E
        v = om.Vector(values=values, E=E)
        v2 = v.rebin(binwidth=0.5)
        assert_allclose(v2.de, 0.5*np.ones_like(v2.E))

        with pytest.raises(ValueError):
            v.rebin(binwidth=-0.5)

        with pytest.raises(ValueError):
            v.rebin(binwidth=0)

    def test_non_equidistant(self):
        E = np.linspace(0, 1, 10)
        v = 2*E
        vec = om.Vector(v, E)
        vec2 = vec.clone()
        vec2.E[4] = 2
        with pytest.raises(RuntimeError):
            vec.rebin(mids=vec2.E)
        with pytest.raises(RuntimeError):
            vec2.rebin(mids=vec.E)

    def test_upsampling(self):
        E = np.linspace(0, 1, 50)
        v = 2*E
        vec = om.Vector(values=v, E=E)
        with pytest.warns(UserWarning):
            vec.rebin(factor=0.5)
        with pytest.warns(UserWarning):
            E = np.linspace(0, 2, 30)
            vec.rebin(mids=E)


def test_to():
    E = np.linspace(0, 1)
    v = 2*E
    vec = om.Vector(E=E, values=v)
    vec2 = vec.to('MeV')
    assert_allclose(vec2.E_true, 1e-3 * E)
    assert vec2.units == om.ureg('MeV')
    vec.to('MeV', inplace=True)
    assert_allclose(vec.E, 1e-3*E)
    assert vec.units == om.ureg('MeV')


def test_rebin_like():
    E = np.linspace(0, 1, 10)
    values = 3*E
    v = om.Vector(values, E)

    E = np.linspace(0, 1, 5)
    values = E
    v2 = om.Vector(values, E.copy())

    v.rebin_like(v2, inplace=True)
    compare_unitful(v.E, E)

    v = om.Vector(values, E, units='MeV')
    v.rebin_like(v2, inplace=True)
    assert(v.units == om.ureg.MeV)


def test_clone():
    E = np.linspace(-1, 1)
    values = 2*E
    std = 0.5*values
    vec = om.Vector(E=E, values=values)
    vec2 = vec.clone()
    assert_vector(vec, vec2)
    vec3 = vec.clone(E=values)
    assert_allclose(vec3.values, vec.values)
    assert_allclose(vec3.E, values)
    vec4 = vec.clone(std=std)
    assert_allclose(vec4.values, vec.values)
    assert_allclose(vec4.E, vec.E)
    assert_allclose(vec4.std, std)
    vec5 = vec4.clone()
    assert_vector(vec4, vec5)
    vec6 = vec4.clone(E=values, values=E, std=0.2*E)
    assert_allclose(vec6.E, values)
    assert_allclose(vec6.values, E)
    assert_allclose(vec6.std, 0.2*E)

    with pytest.raises(RuntimeError):
        vec5.clone(duck=7)


def test_len():
    for N in [1, 2, 5, 100, 1501]:
        N = 100
        E = np.linspace(0, 1, N)
        vals = np.linspace(2, 3.4, N)
        vec = om.Vector(values=vals, E=E)
        assert_equal(len(vec), N)


def test_calibration():
    E = [0.3, 0.3+0.9, 0.3+2*0.9]
    vec = om.Vector(values=E, E=E)
    calib = vec.calibration()
    assert_allclose(calib['a0'], 0.3)
    assert_allclose(calib['a1'], 0.9)


def test_save_load_no_std():
    E = np.linspace(0, 1, 100)
    vals = np.linspace(2, 3.4, 100)
    vec = om.Vector(values=vals, E=E)

    formats = ['.npy', '.txt', '.tar', '.m', '.csv']
    for form in formats:
        vec.save('/tmp/no_std'+form)
        vec_from_file = om.Vector(path='/tmp/no_std'+form)
        assert_equal(vec_from_file.units, vec.units)
        assert_allclose(vec_from_file.E, vec.E)
        assert_allclose(vec_from_file.values, vals)

    with pytest.raises(ValueError):
        vec.save("/tmp/test.npy", filetype='.chicken')


def test_save_load():
    E = np.linspace(0, 1, 100)
    vals = np.linspace(2, 3.4, 100)
    std = np.random.randn(100)*0.1

    vec = om.Vector(values=vals, E=E, std=std)

    formats = ['.npy', '.txt', '.tar', '.csv']
    for fmt in formats:
        vec.save('/tmp/std'+fmt)
        vec_from_file = om.Vector(path='/tmp/std'+fmt)
        assert_equal(vec_from_file.units, vec.units)
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

    compare_unitful(vector_res.E, E_new)
    assert_equal(vector_res.values, values_new)

    vector = om.Vector(values=values, E=E, std=std)
    vector_res = vector.closest(E_new)
    compare_unitful(vector_res.E, E_new)
    assert_equal(vector_res.values, values_new)
    assert_equal(vector_res.std, std_new)

    # Make sure the change is inplace.
    assert vector.closest(E_new, inplace=True) is None
    compare_unitful(vector.E, E_new)
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


def test_cumulative():
    E = np.array([0., 1.5, 3., 4.5, 6., 7.5])
    values = np.array([0., 0., 1., 2., 3., 4.])
    std = 0.1*values

    expect = np.cumsum(values)
    expect_std = np.sqrt(np.cumsum(std**2))

    vec = om.Vector(values=values, E=E, std=std)

    vec_cum = vec.cumulative(inplace=False)
    compare_unitful(vec_cum.E, E)
    assert_equal(vec_cum.values, expect)
    assert_equal(vec_cum.std, expect_std)

    vec_cum = vec.cumulative(factor=2., inplace=False)
    compare_unitful(vec_cum.E, E)
    assert_equal(vec_cum.values, 2.*expect)
    assert_equal(vec_cum.std, 2.*expect_std)

    vec_cum = vec.cumulative(factor='de', inplace=False)
    compare_unitful(vec_cum.E, E)
    assert_equal(vec_cum.values, (E[1]-E[0])*expect)
    assert_equal(vec_cum.std, (E[1]-E[0])*expect_std)

    assert vec.cumulative(inplace=True) is None
    compare_unitful(vec.E, E)
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


class TestCut:
    def test_unitless(self):
        E = np.arange(-1, 11, 1)
        values = np.arange(33, 45, 1)
        vector = om.Vector(values=values, E=E)

        vector.cut(Emin=0, Emax=8, inplace=False)
        compare_unitful(vector.E, E)
        assert_allclose(vector.values, values)

        vector.cut(Emin=0, Emax=8, inplace=True)

        Ecut = np.arange(0, 9, 1)
        valcut = np.arange(34, 43)
        compare_unitful(vector.E, Ecut)
        assert_allclose(vector.values, valcut)

    def test_unit(self):
        E = np.arange(-1, 11, 1)
        values = np.arange(33, 45, 1)
        vector = om.Vector(values=values, E=E)

        vector = vector.cut(Emin='0.002 MeV', Emax='8000 eV')

        Ecut = np.arange(2, 9, 1)
        valcut = np.arange(36, 43)
        compare_unitful(vector.E_true, Ecut)
        assert_allclose(vector.values, valcut)

    def test_std(self):
        E = np.arange(11)
        values = np.arange(21, 32)
        print(E, values)
        std = 0.1*values
        vec = om.Vector(values=values, E=E, std=std)
        cut = vec.cut(Emin=2, Emax=8)
        compare_unitful(cut.E_true, np.arange(2, 9))
        assert_allclose(cut.values, values[2:9])
        assert_allclose(cut.std, std[2:9])
        vec.cut(Emin=2, Emax=8, inplace=True)
        assert_vector(vec, cut)


def test_drop_nan():
    E = np.arange(10)
    values = np.arange(20, 30, dtype=float)
    values[[0, 5, 6]] = np.NaN
    vec = om.Vector(E=E, values=values)
    vec2 = vec.drop_nan()
    compare_unitful(vec2.E, [1, 2, 3, 4, 7, 8, 9])
    assert_allclose(vec2.values, [21, 22, 23, 24, 27, 28, 29])
    compare_unitful(vec.E, E)
    assert_allclose(vec.values, values)
    vec.drop_nan(inplace=True)
    assert_vector(vec, vec2)

    E = np.arange(10)
    values = np.arange(20, 30, dtype=float)
    std = 0.1*values
    values[[0, 5, 6]] = np.NaN
    vec = om.Vector(E=E, values=values, std=std)
    vec2 = vec.drop_nan()
    compare_unitful(vec2.E, [1, 2, 3, 4, 7, 8, 9])
    assert_allclose(vec2.values, [21, 22, 23, 24, 27, 28, 29])
    assert_allclose(vec2.std, [2.1, 2.2, 2.3, 2.4, 2.7, 2.8, 2.9])

    compare_unitful(vec.E, E)
    assert_allclose(vec.values, values)
    assert_allclose(vec.std, std)
    vec.drop_nan(inplace=True)
    assert_vector(vec, vec2)


class TestHasEqualBinning:
    def test_vector(self):
        E = np.linspace(0, 1)
        values = np.random.random(E.shape)
        vec1 = om.Vector(E=E, values=values)
        vec2 = om.Vector(E=E, values=0.5*values)
        assert vec1.has_equal_binning(vec2, error=False)
        vec3 = om.Vector(E=0.99*E, values=values)

        assert not vec1.has_equal_binning(vec3, error=False)
        vec4 = om.Vector(E=E, values=values)
        vec4.E[5] *= 2
        assert not vec1.has_equal_binning(vec4, error=False)
        vec5 = om.Vector(E=E[:10], values=values[:10])
        assert not vec1.has_equal_binning(vec5, error=False)
        assert not vec1.has_equal_binning(E, error=False)

    def test_raises(self):
        E = np.linspace(0, 1)
        values = np.random.random(E.shape)
        vec1 = om.Vector(E=E, values=values)
        vec2 = om.Vector(E=E, values=0.5*values)
        assert vec1.has_equal_binning(vec2, error=True)
        vec3 = om.Vector(E=0.99*E, values=values)

        with pytest.raises(ValueError):
            vec1.has_equal_binning(vec3, error=True)

        vec4 = om.Vector(E=E, values=values)
        vec4.E[5] *= 2
        with pytest.raises(ValueError):
            vec1.has_equal_binning(vec4, error=True)

        vec5 = om.Vector(E=E[:10], values=values[:10])
        with pytest.raises(ValueError):
            vec1.has_equal_binning(vec5, error=True)
        with pytest.raises(TypeError):
            vec1.has_equal_binning(E, error=True)


def test_sum():
    E = np.linspace(0, 1)
    values = np.random.random(E.shape)
    vec = om.Vector(E=E, values=values)
    assert_allclose(vec.sum(), values.sum())


def test_is_equidistant():
    E = np.linspace(0, 5)
    V = E
    vec = om.Vector(V, E)
    assert_equal(vec.is_equidistant(), True)
    E[25] += 2
    vec = om.Vector(V, E)
    assert_equal(vec.is_equidistant(), False)



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
