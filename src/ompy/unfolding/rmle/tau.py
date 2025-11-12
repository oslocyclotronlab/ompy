import jax.numpy as jnp
from .stubs import ExpectationParameter
from typing import NamedTuple, Callable

def to_tau[T: ExpectationParameter](mu: T) -> T:
    return jnp.sqrt(mu)
    # return jnp.sqrt(mu)
    # return jnp.log(mu + 1e-10)


def from_tau[T: ExpectationParameter](tau: T) -> T:
    # return jnp.where(tau < 0, tau**2, tau)
    # Need to be careful with the linear term, as it allows for negative values
    return tau**2  # + 1e-3*tau
    # return tau**2
    # return jnp.exp(tau)


class TauMap[T: ExpectationParameter, P: ExpectationParameter](NamedTuple):
    to_tau: Callable[[T], P]
    from_tau: Callable[[P], T]

TAU_MAP = TauMap(to_tau, from_tau)