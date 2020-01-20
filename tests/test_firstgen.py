import pytest
import numpy as np
import ompy as om


@pytest.fixture()
def firstgen():
    return om.FirstGeneration()


def test_multiplicity_estimation(firstgen):
    firstgen.multiplicity_estimation = 'total'
    firstgen.multiplicity_estimation = 'toTal'
    firstgen.multiplicity_estimation = 'statistical'
    with pytest.raises(ValueError):
        firstgen.multiplicity_estimation = 'duck'


@pytest.mark.parametrize(
        "Ex,fg_Ex",
        [(np.arange(0.0, 5.0), np.arange(0.0, 5.0)),
         (np.arange(0.2, 5.2, 1), np.arange(0.2, 5.2, 1))])
def test_shape(firstgen, Ex, fg_Ex):
    values = np.ones((5, 5))
    values = np.tril(values)
    mat = om.Matrix(values=values)
    mat.Ex = Ex
    mat.verify_integrity()
    first = firstgen.apply(mat)
    assert first.Ex.shape == fg_Ex.shape
    np.testing.assert_allclose(first.Ex, fg_Ex)


@pytest.mark.parametrize(
        "Ex,fg_Ex",
        [(np.arange(-1.0, 4.0), np.arange(0.0, 4.0)),
         (np.arange(-1.2, 2.9, 1), np.arange(0.8, 2.9, 1))])
def test_negatives(firstgen, Ex, fg_Ex):
    values = np.ones((5, 5))
    values = np.tril(values)
    mat = om.Matrix(values=values)
    mat.Ex = Ex
    mat.verify_integrity()
    with pytest.raises(ValueError):
        firstgen.apply(mat)
