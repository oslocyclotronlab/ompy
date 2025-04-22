from __future__ import annotations
from typing import Callable, TypeAlias, Literal, Protocol, runtime_checkable
import jax.numpy as jnp


#  nu -> n -> loss
LossFn: TypeAlias = Callable[[jnp.ndarray, jnp.ndarray], float]

PenaltyFn: TypeAlias = Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], tuple[float, float]]
PenaltyTarget: TypeAlias = Literal['mu', 'eta', 'nu', 'mu_normalized', 'eta_normalized', 'nu_normalized']


@runtime_checkable
class Closureable[**P, T](Protocol):
    def closure(self) -> Callable[P, T]:
        ...