import numpy as np

import theano.tensor as tt
from pymc3.distributions.continuous import (
    PositiveContinuous,
    assert_negative_support,
    draw_values,
    generate_samples)
from pymc3.distributions.dist_math import bound
from pymc3.theanof import floatX


class FermiDirac(PositiveContinuous):
    """
    Fermi-Dirac distribution.

    The pdf of this distribution is

    .. math::

        f(x \mid \lambda \mu) =
            \frac{\lambda}{\lambda\mu + \ln(1 + e^{-\lambda\mu})}
            \frac{1}{e^{\lambda(x - \mu)} + 1}

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np

        x = np.linspace(0, 5., 1000)
        lam = [1.0, 10., 25.]
        mu = [0.5, 1.0, 2.0]

        def pdf(x, lam, mu):
            return (lam/np.log(1 + np.exp(lam*mu))*(1/(np.exp(lam*(x-mu)) + 1))

        for l in lam:
            for m in mu:
                pdf = pdf(x, l, m)
                plt.plot(x, pdf, label=f"$\lambda = {l}$, $\mu = {m}$")
        plt.xlabel("x", fontsize=12)
        plt.ylabel("f(x)", fontsize=12)
        plt.legend(loc=1)
        plt.show()

    ========  ============================
    Support   :math:`x \in [0, \infty)`
    ========  ============================


    Parameters
    ----------
    lam: float
        Rate of decay at mu (lam > 0)
    mu: float
        Decay position

    Examples
    --------
    .. code-block:: python

        with pm.Model():
            x = ompy.FermiDirac('x', lam=10.0, mu=1.2)

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
        return mu - np.log((1 + np.exp(lam*mu))**(1-q) - 1)/lam

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
        return generate_samples(self._random, lam, mu, dist_shape=self.shape,
                                size=size)

    def logp(self, value):
        """
        Calculate log-probability of Fermi-Dirac distribution at specified
        value.

        Parameters
        ----------
        value: numeric
            Value(s) for which log-probability is calculated. If the log
            probabilities for multiple values are desired the values must be
            provided in a numpy array or theano tensor

        Returns
        -------
        TensorVariable
        """

        # The formula is
        # p(x) = lam/ln(1 + exp(lam*mu)) * 1/(exp(lam*(x-mu)) + 1)
        # ln(p(x)) = ln(lam/ln(1 + exp(lam*mu))) - ln(exp(lam*(x-mu)) + 1)

        lam = self.lam
        mu = self.mu

        N = lam/tt.log(1 + tt.exp(lam*mu))
        logp = tt.log(N) - tt.log(tt.exp(lam*(value - mu)) + 1)
        return bound(logp, value >= 0, lam > 0)

    def logcdf(self, value):
        """
        Compute the log of cumulative distribution function for the Fermi-Dirac
        distribution at the specified value.

        Parameters
        ----------
        value: numeric or np.ndarray or theano.tensor
            Value(s) for which log CDF is calculated. If the log CDF for
            multiple values are desired the values must be provided in a numpy
            array or theano tensor.

        Returns
        -------
        TensorVariable
        """

        # The formula is CDF
        # P(x) = 1 - ln(1 + exp(-lam*(x-mu)))/ln(1 + exp(lam*mu))
        # ln(P(x)) = ln(1 - ln(1 + exp(-lam*(x-mu)))/ln(1 + exp(lam*mu)))

        lam = self.lam
        mu = self.mu

        logcdf = tt.log(1 - tt.log(1 + tt.exp(-lam*(value - mu))) /
                        tt.log(1 + tt.exp(lam*mu)))
        return bound(logcdf, value >= 0, lam > 0)
