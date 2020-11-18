import pytest
import ompy as om
import numpy as np
from numpy.testing import assert_allclose

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
                                    model="EB05", pars=spinpars).distribution()
        spindist /= spindist.sum(axis=1)[:, np.newaxis]


# tests comparing to rhobin / and or derived values
def y_interpol(x, x0, x1, y0, y1):
    return y0 + (x - x0) * (y1 - y0) / (x1 - x0)
testdata = [("EB05", [4.2, 6.534],
             {"mass": 240, "NLDa": 25.506, "Eshift": 0.162, "Sn": 6.534},
             [7.521, 8.387]),
            ("EB09_emp", [4.2, 6.534],
             {"mass": 240, "Pa_prime": 1.336},
             [4.841, 5.239]),
            ("EB09_CT", [4.2, 6.534],
             {"mass": 240},
             [4.803, 4.803]),
            ("const", [4.2, 6.534],  # not in robin, but just sigma=const
             {"sigma": 645.456},
             [645.456, 645.456]),
            ("Disc_and_EB05", [1.5, 4.2, 6.534],   # combination of the above
             {"mass": 240, "NLDa": 25.506, "Eshift": 0.162, "Sn": 6.534,
              "sigma2_disc": [1.5, 3.6]},
             [np.sqrt(3.6),
              np.sqrt(y_interpol(4.2, 1.5, 6.534, 3.6, 8.387**2)), 8.387])
            ]


@pytest.mark.parametrize("model, Ex, pars, expected", testdata)
def test_sigma2(model, Ex, pars, expected):
    J = None
    spinfunc = om.SpinFunctions(Ex, J, model=model, pars=pars)
    spincut = np.sqrt(spinfunc.get_sigma2())
    assert_allclose(spincut, expected, atol=0.01)
