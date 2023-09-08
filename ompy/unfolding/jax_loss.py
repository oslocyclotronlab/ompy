from __future__ import annotations
from .. import Matrix, Vector
import jax
from jax import Array
import jax.numpy as jnp
from abc import ABC, abstractmethod
import inspect
from typing import Callable
"""
Loss
MLE
Neglog
KL

Reg:
L1
L2
ME
I>0

Without background:
Poisson, Poisson
With background:
Skellam, Poisson?

from om.unfolding import jloss
loss = jloss.kl + jloss.me
print(loss.name)  # KL + alpha*ME
"""


# Step 2: Create a base class to handle the addition of loss functions and regularizations
class BaseLoss(ABC):
    def __init__(self, name: str):
        self.name = name

    def __add__(self, other: BaseLoss):
        match (self, other):
            case CombinedLoss(), CombinedLoss():
                combined = CombinedLoss(*self.losses, *other.losses)
            case CombinedLoss(), _:
                combined = CombinedLoss(*self.losses, other)
            case _, CombinedLoss():
                combined = CombinedLoss(self, *other.losses)
            case _, _:
                combined = CombinedLoss(self, other)
        return combined

    @abstractmethod
    def get(self) -> Callable[[Array, Array, Array], Array]: ...

class CombinedLoss(BaseLoss):
    def __init__(self, *losses: BaseLoss):
        super().__init__(" + ".join(loss.name for loss in losses))
        self.losses: list[BaseLoss] = losses

    def get(self, n, R: Array , G: Array , G_ex: Array | None = None, bg: Array | None = None):
        loss_fns = [loss.get() for loss in self.losses]
        params = [inspect.signature(loss).parameters.keys() for loss in loss_fns]
        print(params)
        map_fn = make_map_fn(n, R, G, G_ex)
        def fn(mu):
            mu = mu**2
            nu = map_fn(mu)
            loss = 0.0
            for lossfn in loss_fns:
                lossfn += lossfn(mu, nu, n)
            return loss
        return fn


def make_map_fn(x, R, G, G_ex):
    if x.ndim == 1:
        def fn(mu):
            nu = G@R@mu
            return nu
    elif x.ndim == 2:
        def fn(mu):
            nu = G_ex@mu@(G@R)
            return nu
    else:
        raise ValueError("x must be 1D or 2D")
    return fn


# Step 1: Define a class for each loss function and regularization term
class KL(BaseLoss):
    def __init__(self):
        super().__init__("KL")

    def get(self):
        def loss(mu, nu, n):
            return jnp.sum(nu - n + n * (jnp.log(n + 1e-9) - jnp.log(nu + 1e-9)))
        return loss

class ME(BaseLoss):
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        super().__init__(f"{alpha}*ME")

    def get(self):
        alpha = self.alpha
        def fn(mu, nu, n):
            return alpha * jnp.sum(mu * jnp.log(mu + 1e-9))
        return fn

# Other loss functions and regularizations can be defined similarly

# Usage:
kl = KL()
me = ME(alpha=0.5)

loss = kl + me
print(loss.name)  # KL + 0.5*ME

y_true = jnp.array([1.0, 2.0, 3.0])
y_pred = jnp.array([2.0, 2.0, 4.0])

loss_value = loss(y_true, y_pred)
print(loss_value)
