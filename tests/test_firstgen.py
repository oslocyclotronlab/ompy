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
        [(np.arange(-1.0, 4.0), np.arange(0.0, 4.0)),
         (np.arange(-1.2, 2.9, 1), np.arange(0.8, 2.9, 1))])
def test_cutoff(firstgen, Ex, fg_Ex):
    mat = om.all_generations_trivial((5,5))
    mat.Ex = Ex
    mat.verify_integrity()
    first = firstgen.apply(mat)
    assert first.Ex.shape == fg_Ex.shape
    np.testing.assert_allclose(first.Ex, fg_Ex)


def test_row_normalization(firstgen):
    mat = om.all_generations_trivial((5, 5))
    norm = firstgen.row_normalized(mat)
    np.testing.assert_allclose(norm.sum(axis=1), np.ones(5))
    assert norm[0, 0] == 1.0
    assert norm[1, 0] == norm[1, 1] == 0.5


def test_multiplicity_total(firstgen):
    firstgen.multiplicity_estimation = 'total'
    mat = om.all_generations_trivial((3, 3))
    multiplicities = firstgen.multiplicity_total(mat)
    np.testing.assert_allclose(multiplicities,
                               np.array([1.0, 1.8, 2.142857]))
