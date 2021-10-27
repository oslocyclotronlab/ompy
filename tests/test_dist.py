import pytest
import ompy as om
import numpy as np
import pymc3 as pm
from numpy.testing import assert_equal, assert_allclose

import matplotlib.pyplot as plt


@pytest.mark.parametrize(
    "lam,mu",
    [(10., 1.),
     (5., 1.),
     (100., 2.),
     (50., 0.5)])
def test_fermi_dirac(lam, mu):

    prob = om.FermiDirac.dist(lam=lam, mu=mu)

    samples = prob.random(size=1000000)

    hist, bins = np.histogram(samples, bins=100, density=True)

    x = []
    for n in range(len(bins) - 1):
        x.append(0.5*(bins[n+1] + bins[n]))
    x = np.array(x)
    y = (lam/np.log(1 + np.exp(lam*mu)))/(np.exp(lam*(x - mu)) + 1)

    # We use a fairly large tolerance since error should be stocastic
    assert_allclose(hist, y, atol=0.1)


@pytest.mark.parametrize(
    "lam,mu",
    [(10., 1.),
     (5., 1.),
     (100., 2.),
     (50., 0.5)])
def test_fermi_dirac_logp(lam, mu):

    lam = 10.
    mu = 1.

    prob = om.FermiDirac.dist(lam=lam, mu=mu)

    x = np.linspace(0, 3, 1001)
    y = lam/np.log(1 + np.exp(lam*mu)) * 1/(np.exp(lam*(x - mu)) + 1)

    y_r = np.exp(prob.logp(x).eval())
    assert_allclose(y, y_r)


@pytest.mark.parametrize(
    "lam,mu",
    [(10., 1.),
     (5., 1.),
     (100., 2.),
     (50., 0.5)])
def test_fermi_dirac_logcdf(lam, mu):

    prob = om.FermiDirac.dist(lam=lam, mu=mu)
    x = np.linspace(0, 3, 1001)

    y = 1 - np.log(1 + np.exp(-lam*(x - mu)))/np.log(1 + np.exp(lam*mu))
    y_r = np.exp(prob.logcdf(x).eval())
    assert_allclose(y, y_r)
