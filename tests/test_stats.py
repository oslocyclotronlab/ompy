import pytest
import ompy as om
import numpy as np
import scipy.stats as stats
from numpy.testing import assert_allclose


def test_normal():
    N = 10000
    x = np.random.rand(N)

    norm_native = np.array([om.stats.normal(xi) for xi in x])
    norm_scipy = stats.norm.ppf(x)

    assert_allclose(norm_native, norm_scipy)


def test_truncnorm():
    N = 10000
    x = np.random.rand(N)

    a = -5.
    b = 25.

    truncnorm_native = np.array([om.stats.truncnorm(xi, a, b) for xi in x])
    truncnorm_scipy = stats.truncnorm.ppf(x, a, b)

    assert_allclose(truncnorm_native, truncnorm_scipy)