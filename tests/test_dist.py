import pytest
import ompy as om
import numpy as np
import pymc3 as pm


def test_fermi_dirac():

    with pm.Model() as model:
        D = om.FermiDirac("D", lam=10., mu=1.0)

    print(model.vars)
    assert(False)
