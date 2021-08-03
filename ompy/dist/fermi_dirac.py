import numpy as np

import theano.tensor as tt
from pymc3.distributions.continuous import (
    PositiveContinuous, 
    assert_negative_support,
    draw_values,
    generate_samples)
from pymc3.distributions.dist_math import bound
from pymc3.theanof import floatX
#from mpmath import polylog

class FermiDirac(PositiveContinuous):
    """
    Fermi-Dirac distribution.

    The pdf of this distribution is

    .. math::

        f(x \mid \lambda \mu) = 
            \frac{\lambda}{\lambda\mu + \ln(1 + e^{-\lambda\mu})}
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

    def __init__(self, lam, mu, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.lam = tt.as_tensor_variable(floatX(lam))
        self.mu = tt.as_tensor_variable(floatX(mu))
        assert_negative_support(lam, "lam", "FermiDirac")

        self.median = self._ppf(0.5, self.lam, self.mu)

    def _ppf(self, q, lam, mu):
        """
        Calculate the CDF for the Fermi-Dirac distribution.
        """
        N = np.log(1 + np.exp(lam*mu))
        return mu - np.log(np.exp(N*(1-q)) - 1)/lam

    def _random(self, lam, mu, size=None):
        """
        Draw a random number from the Fermi-Dirac distribution.
        """
        v = np.random.uniform(size=size)
        return self._ppf(v, lam, mu)

    def random(self, point=None, size=None):
        """
        Draw random values from Fermi-Dirac distribution.
        Parameters
        ----------
        point: dict, optional
            Dict of variable values on which random values are to be
            conditioned (uses default point if not specified).
        size: int, optional
            Desired size of random sample (returns one sample if not
            specified).
        Returns
        -------
        array
        """
        lam, mu = draw_values([self.lam, self.mu], point=point, size=size)
        return generate_samples(self._random, lam, mu, dist_shape=self.shape, size=size)

    def logp(self, value):
        """
        Calculate log-probability of Fermi-Dirac distribution at specified value.
        Parameters
        ----------
        value: numeric
            Value(s) for which log-probability is calculated. If the log probabilities for multiple
            values are desired the values must be provided in a numpy array or theano tensor
        Returns
        -------
        TensorVariable
        """

        lam = self.lam
        mu = self.mu

        N = lam/tt.log(1 + tt.exp(lam*mu))
        logp = tt.log(N) - tt.log(tt.exp(lam*(value - mu)) + 1)
        return bound(logp, value >= 0, lam > 0)

    def logcdf(self, value):
        """
        Compute the log of cumulative distribution function for the Fermi-Dirac distribution
        at the specified value.
        Parameters
        ----------
        value: numeric or np.ndarray or theano.tensor
            Value(s) for which log CDF is calculated. If the log CDF for multiple
            values are desired the values must be provided in a numpy array or theano tensor.
        Returns
        -------
        TensorVariable
        """

        lam = self.lam
        mu = self.mu

        logcdf = tt.log(1 - tt.log(1 + tt.exp(-lam*(value - mu)))/tt.log(1 + tt.exp(lam*mu)))
        return bound(logcdf, value >= 0, lam > 0)
