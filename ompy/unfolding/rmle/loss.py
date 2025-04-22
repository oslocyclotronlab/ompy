from __future__ import annotations
import jax.numpy as jnp
from abc import ABC, abstractmethod
from .stubs import LossFn


class Loss(ABC):
    @abstractmethod
    def closure(self) -> LossFn:
        pass


class KullbackLeibler(Loss):
    def closure(self) -> LossFn:
        return kl


class L2(Loss):
    def closure(self) -> LossFn:
        def fn(nu, n):
            return jnp.sum((nu - n) ** 2)

        return fn


class L1(Loss):
    def closure(self) -> LossFn:
        def fn(nu, n):
            return jnp.sum(jnp.abs(nu - n))

        return fn


def kl(nu, n):
    """Compute the Kullback-Leibler divergence between two distributions.

    The KL divergence is a measure of the difference between two probability distributions.
    This implementation includes small epsilon terms to avoid numerical instabilities
    when taking logarithms of values close to zero.

    Args:
        nu: The first distribution (typically the model prediction)
        n: The second distribution (typically the observed data)

    Returns:
        The KL divergence between the distributions
    """
    eps = 1e-10
    return nu - n + n * jnp.log(n / (nu + eps) + eps)
