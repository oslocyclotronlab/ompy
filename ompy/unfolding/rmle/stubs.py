from __future__ import annotations
from typing import Callable, TypeAlias, Literal, Protocol, runtime_checkable
import jax.numpy as jnp

from jaxtyping import Float, Array

#  nu -> n -> loss

type LossSpace = Literal['nu', 'eta', 'mu', 'nu normalized', 'eta normalized', 'mu normalized']

type Data1D = Float[Array, "Eg"]
type Data2D = Float[Array, "Ein Eg"]
type Data = Data1D | Data2D

type Background1D = Float[Array, "Eg"]
type Background2D = Float[Array, "Ein Eg"]
type Background = Background1D | Background2D

type ExpectationParameter1D = Float[Array, "Eg"]
type ExpectationParameter2D = Float[Array, "Ein Eg"]
type ExpectationParameter = ExpectationParameter1D | ExpectationParameter2D

type Tau1D = ExpectationParameter1D
type Tau2D = ExpectationParameter2D
type Mu1D = ExpectationParameter1D
type Mu2D = ExpectationParameter2D
type Xi1D = ExpectationParameter1D
type Xi2D = ExpectationParameter2D
type Nu1D = ExpectationParameter1D
type Nu2D = ExpectationParameter2D
type Beta1D = ExpectationParameter1D
type Beta2D = ExpectationParameter2D
type Contaminants1D = tuple[ExpectationParameter1D, ...]
type Empty = tuple[()]
type State1D = tuple[Tau1D, Beta1D | None, Contaminants1D | Empty]
type LossFn1D = Callable[[ExpectationParameter1D, Data1D], float]
type LossFn2D = Callable[[ExpectationParameter2D, Data2D], float]
type LossFn = LossFn1D | LossFn2D
#PenaltyFn: TypeAlias = Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], tuple[float, float]]
#PenaltyTarget: TypeAlias = Literal['mu', 'eta', 'nu', 'mu_normalized', 'eta_normalized', 'nu_normalized']
type GegMatrix = Float[Array, "Eg_true Eg_measured"]
type GegDMatrix = Float[Array, "Eg_true Eg_observed"]
type DMatrix = Float[Array, "Ein Eg"]

type ContaminantLossFn1D = Callable[[ExpectationParameter1D, GegMatrix, GegDMatrix], float]

@runtime_checkable
class Closureable[**P, T](Protocol):
    def closure(self) -> Callable[P, T]:
        ...


@runtime_checkable
class Optimizer[Params, Grads, Updates, OptState](
    Protocol[Params, Grads, Updates, OptState]
):
    def init(self, params: Params) -> OptState:
        ...

    def update(
        self,
        grads: Grads,
        state: OptState,
        params: Params | None = None,
    ) -> tuple[Updates, OptState]:
        ...
