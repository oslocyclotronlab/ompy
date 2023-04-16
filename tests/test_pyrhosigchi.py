import os
import pytest
from numpy.testing import assert_almost_equal
import ompy as om

DATAPATH = os.path.abspath(__file__)


def get_path(path):
    return os.path.abspath(os.path.join(DATAPATH, path))


def test_pyrhosigchi():

    fg = om.Matrix(path=get_path("../test_data/Ni63_firstgen.m"))
    fgerr = om.Matrix(path=get_path("../test_data/Ni63_firstgenerr.m"))

    fg.rebin(axis='Ex', mids=fg.Eg)
    fgerr.rebin(axis='Ex', mids=fgerr.Eg)

    Egmin = 1345.0
    Exmin = 2545.0
    Exmax = 6845.0
    nit = 51

    nld, gsf = om.pyrhosigchi.pyrhosigchi(fg, fgerr, Exmin,
                                          Exmax, Egmin, nit=nit)

    nld_correct = \
        om.Matrix(path=get_path("../test_data/Ni63_rho.m"))
    gsf_correct = \
        om.Matrix(path=get_path("../test_data/Ni63_sig.m"))

    nld_correct = om.Vector(E=nld_correct.Eg, values=nld_correct.values[0, :],
                            std=nld_correct.values[1, :], units="keV")
    gsf_correct = om.Vector(E=gsf_correct.Eg, values=gsf_correct.values[0, :],
                            std=gsf_correct.values[1, :], units="keV")

    nld_correct.cut(Emin=nld.E[0], Emax=nld.E[-1], inplace=True)
    gsf_correct.cut(Emax=gsf.E[-1], inplace=True)

    assert_almost_equal(nld.values, nld_correct.values, decimal=4)
    assert_almost_equal(gsf.values, gsf_correct.values, decimal=4)

if __name__ == "__main__":
    test_pyrhosigchi()
