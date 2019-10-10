import pytest
import ompy as om
import numpy as np


def test_attributes():
    nld = om.Vector(np.linspace(0, 6, 100))
    discretes = \
        om.load_levels_discrete('example_data/discrete_levels_Dy164.txt',
                                nld.E)
    normalizer = om.Normalizer()

    with pytest.raises(ValueError):
        normalizer.normalize((0, 1), (0, 1))

    with pytest.raises(ValueError):
        normalizer.normalize((0, 1), (0, 1), nld=nld)

    normalizer.nld = nld
    with pytest.raises(ValueError):
        normalizer.normalize((0, 1), (0, 1))
    with pytest.raises(ValueError):
        normalizer.normalize((0, 1), (0, 1), nld=nld)

    normalizer.nld = None
    with pytest.raises(ValueError):
        normalizer.normalize((0, 1), (0, 1), discrete=discretes)
    normalizer.normalize((0, 1), (0, 1), nld=nld, discrete=discretes)
    normalizer.nld = nld
    normalizer.normalize((0, 1), (0, 1), discrete=discretes)
    normalizer.discrete = discretes
    normalizer.normalize((0, 1), (0, 1))
    normalizer.nld = None
    normalizer.normalize((0, 1), (0, 1), nld=nld)

    normalizer.discrete = None
    with pytest.raises(ValueError):
        normalizer.discrete = 'data/discrete_levels.txt'
    normalizer.nld = nld
    normalizer.discrete = 'data/discrete_levels.txt'
    normalizer.normalize((0, 1), (0, 1), nld=nld)

    normalizer.discrete = None
    normalizer.normalize((0, 1), (0, 1), nld=nld,
                         discrete='data/discrete_levels.txt')
