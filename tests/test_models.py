import pytest
import ompy as om
import numpy as np
from numpy.testing import assert_equal, assert_almost_equal
from typing import Tuple


def test_set_mass():

    normpar = om.NormalizationParameters(name="Test")

    normpar.A = 16
    normpar.Z = 8

    normpar.spincutModel = 'EB09_CT'
    normpar.spincutPars = {'mass': 16}

    with pytest.warns(UserWarning):
        normpar.A = 17

    with pytest.warns(UserWarning):
        normpar.spincutPars = {'mass': 18}

    # We expect this to NOT trigger a warning.
    with pytest.warns(None) as record:
        A = normpar.A
        assert len(record) == 0

    # We do not expect this to trigger a warning.
    with pytest.warns(None) as record:
        spinpar = normpar.spincutPars
        assert len(record) == 0

    # Neither should this
    with pytest.warns(None) as record:
        normpar.spincutPars = {'sigma': 2.9}
        assert len(record) == 0

if __name__ == "__main__":
    test_set_A()