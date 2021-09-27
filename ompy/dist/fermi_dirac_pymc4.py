import pymc3 as pm
from typing import List, Tuple

import aesara.tensor as at
from aesara.tensor.random.op import RandomVariable

from pymc3.distributions.continuous import (PositiveContinuous,
                                            assert_negative_support)
from pymc3.distributions.dist_math import bound

"""
TODO:
    - Add tests according to pymc3 standard, 
        see https://github.com/pymc-devs/pymc3/blob/main/docs/source/developer_guide_implementing_distribution.md  # noqa
"""


class FermiDiracRV(RandomVariable):
    name: str = "fermidirac"

    # Minimum dim. is a scalar (0 = scalar, 1 = vector, etc.)
    ndim_supp: int = 0

    # Number of parameters for the RV
    ndim_params: List[int] = [0, 0]

    # Datatype, floatX is continious
    dtype: str = "floatX"

    # Print name
    _print_name: Tuple[str, str] = ("FermiDirac", "\\operatorname{FermiDirac}")

    @classmethod
    def rng_fn(cls,
               rng: np.random.RandomState,
               lam: np.ndarray,
               mu: np.ndarray, size: Tuple[int, ...]
               ) -> np.ndarray:
        q = rng.uniform(size=size)
        N = lam/np.log(1 + np.exp(lam*mu))
        return mu - np.log(1 - np.exp(lam*(1 - q)/lam))/lam


fermidirac = FermiDiracRV()


class FermiDirac(PositiveContinuous):
    """
    Fermi-Dirac distribution.

    The pdf of this distribution is

    .. math::

        f(x \mid \lambda \mu) =
            \frac{\lambda}{\lambda\mu - \ln(1 + e^{-\lambda\mu})}
            \frac{1}{e^{\lambda(x - \mu)} + 1}

    .. plot::


    ========  ============================
    Support   :math:`x \in [0, \infty)`
    Mean
    Variance
    ========  ============================


    Parameters
    ----------
    lam: float
        Rate of decay at mu (lam > 0)
    mu: float
        Decay position
    """
    rv_op = fermidirac

    @classmethod
    def dist(cls, lam, mu, *args, **kwargs):
        lam = at.as_tensor_variable(floatX(lam))
        mu = at.as_tensor_variable(floatX(mu))

        assert_negative_support(lam, "lam", "FermiDirac")

        return super().dist([lam, mu], *args, **kwargs)

    def logp(value, lam, mu):
        """
        Calculate log-probability of Fermi-Dirac distribution at spesified
        value.

        Parameters
        ----------
        value: numeric
            Value(s) for which log-probability is calculated. If the log
            probabilities for multiple values are desired the values must be
            provided in a numpy array or Aesara tensor

        Returns
        -------
        TensorVariable
        """

        N = lam/at.log(1 + at.exp(lam*mu))
        logp = at.log(N) - at.log(at.exp(lam*(value - mu)) + 1)
        return bound(logp, value >= 0, lam > 0)

    def logcdf(value, lam, mu):
        """
        Compute the log of cumulative distribution function for the Fermi-Dirac
        distribution at the specified value.
        Parameters
        ----------
        value: numeric or np.ndarray or aesara.tensor
            Value(s) for which log CDF is calculated. If the log CDF for
            multiple values are desired the values must be provided in a numpy
            array or Aesara tensor.
        Returns
        -------
        TensorVariable
        """

        N = lam/at.log(1 + at.exp(lam*mu))
        V = (at.exp(lam*(value - mu)) + 1)/(at.exp(lam*mu) + 1)
        logcdf = at.log(N) + at.log(lam*value - at.log(V))
        return bound(logcdf, value >= 0, lam > 0)
