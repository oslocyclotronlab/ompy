import pytest
from numpy.testing import assert_allclose
from ompy.normalizer_nld import NormalizerNLD


# expected values from d2rho
testdata = [(2.2, 0., {"sigma": 8.43}, 6.506E+07),
            (2.2, 0.5, {"sigma": 8.43}, 3.27e7),
            (5, 1, {"sigma": 7.}, 6.736E+06),
            (5, 5, {"sigma": 7.}, 2.441E+06),
            ]
@pytest.mark.parametrize("D0, Jtarget, spincutPars, expected", testdata)
def test_nldSn_from_D0(D0, Jtarget, spincutPars, expected):
    spincutModel = "const"
    Sn = 321  # dummy value
    nld = NormalizerNLD.nldSn_from_D0(D0, Sn, Jtarget, spincutModel,
                                      spincutPars)
    assert_allclose(nld[1], expected, rtol=0.01)


@pytest.mark.parametrize("expected, Jtarget, spincutPars, nldSn", testdata)
def test_D0_from_nldSn(nldSn, Jtarget, spincutPars, expected) -> float:
    """ Wrapper for ompy.NormalizerNLD.D0_from_nldSn with given nld(Sn)

    Args:
        nldSn: Level density at Sn
        pars: Normalization parameters (spin cut model ...).
            Can be provided through ompy.NormalizationParameters.asdict().

    Returns:
        D0: Average s-wave resonance spacing
    """

    def nld_model_dummy(x):
        return nldSn

    D0 = NormalizerNLD.D0_from_nldSn(nld_model_dummy, Sn=321, Jtarget=Jtarget,
                                     spincutModel="const",
                                     spincutPars=spincutPars)
    assert_allclose(D0, expected, rtol=0.01)
