from __future__ import annotations
import jax
import jax.numpy as jnp
from .stubs import LossFn, ExpectationParameter, Data, LossSpace
from abc import ABC
from .utils import pytree_dataclass

def kl(nu: ExpectationParameter, n: Data) -> jnp.ndarray:
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

class Loss(ABC):
    space: LossSpace
    fn: LossFn

    def __call__(self, alpha: ExpectationParameter, x: Data) -> jnp.ndarray:
        return self.fn(alpha, x)

@pytree_dataclass
class KullbackLeibler(Loss):
    space = 'nu'
    fn = staticmethod(jax.jit(kl))


def l2(nu: ExpectationParameter, n: Data) -> jnp.ndarray:
    return (nu - n) ** 2

@pytree_dataclass
class L2(Loss):
    space = 'nu'
    fn = l2


def l1(nu: ExpectationParameter, n: Data) -> jnp.ndarray:
    return jnp.abs(nu - n)

@pytree_dataclass
class L1(Loss):
    pace = 'nu'
    fn = l1