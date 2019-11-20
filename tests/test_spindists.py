import pytest
import ompy as om
import numpy as np

@pytest.mark.parametrize(
                "spinpars",
                 [{"NLDa": 25.160, "Eshift": 0.120},
                 {"mass": 240, "Eshift": 0.120},
                 {"mass": 240, "NLDa": 25.160}]) # noqa
def test_missing_parameter(spinpars):
    E = 4
    Js = 20
    with pytest.raises(TypeError):
        spindist = om.SpinFunctions(E, Js,
                                    model="EB05", pars=spinpars).distibution()
        spindist /= spindist.sum(axis=1)[:, np.newaxis]
