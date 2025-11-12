from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import Iterable
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from ... import Index, Vector
from .stubs import LossSpace, LossFn, ExpectationParameter
from .utils import pytree_dataclass

class Penalty(ABC):
    """Abstract base class for penalty terms in the unfolding.

    The penalty term is added to the likelihood to regularize the unfolding.
    Subclasses must implement the closure() method which returns a function
    that computes the penalty value and auxiliary information.

    Attributes:
        _target: The target distribution that the penalty is applied to.
                Can be 'mu', 'eta', 'nu' or their normalized versions.
    """

    @abstractmethod
    def __call__(self, *args, **kwargs) -> tuple[float, float]:
        pass



@pytree_dataclass
class Entropy(Penalty):
    """Entropy penalty term for regularizing the unfolding.

    This penalty encourages smoother distributions by maximizing the entropy
    of the normalized target distribution. A higher entropy indicates a more
    uniform distribution. The regularization strength can also be negative, in
    which case the entropy will be minimized, encouraging a more peaked distribution.

    Args:
        alpha: The strength of the penalty term. Higher values give more weight
              to the entropy regularization relative to the likelihood.

    Attributes:
        alpha: The penalty strength coefficient.
        _target: Set to 'eta_normalized' since entropy is only meaningful for
                normalized probability distributions.
    """
    alpha: float
    target: LossSpace = "eta_normalized"

    def __call__(self, eta: ExpectationParameter, axis: int | None = None) -> tuple[float, float]:
        penalty = jnp.sum(entropy(eta), axis=axis)
        return self.alpha * penalty, penalty

class Sobolev(Penalty):
    """Sobolev penalty term for regularizing the unfolding.

    This penalty encourages smoothness by penalizing large derivatives in the target
    distribution. It computes the L2 norm of both first and second derivatives,
    weighted by masks that can be used to selectively apply the penalty.

    Args:
        alpha: The strength of the penalty term. Higher values give more weight
              to the smoothness regularization relative to the likelihood.
        mask: Optional mask to apply to both first and second derivatives.
              If None, no masking is applied (mask=1).
        D1_mask: Optional specific mask for first derivatives. If None, derived from mask.
        D2_mask: Optional specific mask for second derivatives. If None, derived from mask.
        fit_masks: If True and using array masks, automatically adjusts mask sizes
                  to match derivative dimensions by truncating.
        target: Which distribution to apply penalty to - either 'eta' or 'eta_normalized'.
        step: Step size for derivative calculations. Can be float, Index, Vector or array.
              Required for scale invariance of the Sobolev norm.

    Attributes:
        alpha: The penalty strength coefficient.
        step: The step size used in derivative calculations.
        mask: The base mask applied to derivatives.
        D1_mask: The mask applied to first derivatives.
        D2_mask: The mask applied to second derivatives.
        _target: The target distribution for the penalty.
    """

    def __init__(
        self,
        alpha: float,
        mask: jnp.ndarray | None = None,
        D1_mask: jnp.ndarray | None = None,
        D2_mask: jnp.ndarray | None = None,
        fit_masks: bool = True,
        target: LossSpace = "eta",
        step: float | Index | Vector | np.ndarray | None = None,
    ) -> LossFn:
        if mask is None:
            mask = 1
        else:
            mask = jnp.asarray(mask)
        if D1_mask is None:
            if not isinstance(mask, Iterable):
                D1_mask = mask
            else:
                if fit_masks:
                    D1_mask = mask[:-1]
                else:
                    D1_mask = mask
        if D2_mask is None:
            if not isinstance(mask, Iterable):
                D2_mask = mask
            else:
                if fit_masks:
                    D2_mask = mask[:-2]
                else:
                    D2_mask = mask

        self.alpha = alpha
        self.step = _resolve_step(step)
        self.mask = mask
        self.D1_mask = D1_mask
        self.D2_mask = D2_mask
        if target not in ["eta_normalized", "eta"]:
            raise ValueError(
                f"Invalid target: {target}. Must be one of 'eta_normalized' or 'eta'."
            )
        self.target = target

    def __call__(self, x: ExpectationParameter) -> tuple[float, float]:
        alpha = self.alpha
        step = self.step
        D1_mask = self.D1_mask
        D2_mask = self.D2_mask

        # Take the Sobolev norm of the distribution
        # Much faster than using finite difference matrices
        D1p = ((x[:-1] - x[1:]) / step) ** 2 * D1_mask
        D2p = ((x[:-2] - 2 * x[1:-1] + x[2:]) / step**2) ** 2 * D2_mask
        penalty = jnp.sum(D1p) + jnp.sum(D2p)
        return alpha * penalty, penalty

    def __repr__(self):
        return f"Sobolev(alpha={self.alpha}, step={self.step}, mask={self.mask}, D1_mask={self.D1_mask}, D2_mask={self.D2_mask}, target={self.target})"

def tree_flatten(obj):
    children = (obj.alpha, obj.step, obj.mask, obj.D1_mask, obj.D2_mask)
    aux_data = {"target": obj.target}
    return children, aux_data

def tree_unflatten(aux_data, children):
    alpha, step, mask, D1_mask, D2_mask = children
    return Sobolev(
        alpha=alpha,
        step=step,
        mask=mask,
        D1_mask=D1_mask,
        D2_mask=D2_mask,
        target=aux_data["target"]
    )

jax.tree_util.register_pytree_node(Sobolev, tree_flatten, tree_unflatten)

@pytree_dataclass
class Sparsity(Penalty):
    """Penalty term that encourages sparsity in the solution.

    This penalty uses a smooth approximation of the Heaviside step function to penalize
    values above a threshold C. The smoothing is controlled by parameter D, with higher
    values giving a sharper transition.

    Args:
        alpha: Overall scaling factor for the penalty term
        C: Threshold value that determines when penalty starts increasing. Default: 0.1
        D: Smoothing parameter controlling sharpness of transition. Default: 100
        target: Which distribution to apply penalty to - either 'mu_normalized' or 'mu'.
               Default: 'mu_normalized'
    """
    alpha: float
    threshold: float = 0.1
    smoothing: float = 100
    target: LossSpace = "mu_normalized"

    def __call__(self, x: ExpectationParameter) -> tuple[float, float]:
        penalty = onecost(x, self.threshold, self.smoothing)
        return self.alpha * penalty, penalty

    def plot(self, x: np.ndarray | None = None, ax: plt.Axes | None = None):
        if ax is None:
            fig, ax = plt.subplots()
        if x is None:
            # Capture a range covering the smooth transition
            C = self.threshold
            D = self.smoothing
            width = C / D
            x = np.linspace(C - 3 * width, C + 3 * width, 100)
        cost = self(x)[0]
        ax.plot(x, cost)
        ax.set_xlabel("Counts")
        ax.set_ylabel("Penalty")
        return ax
            
    



class SobolevGauss(Penalty):
    def __init__(
        self,
        alpha: float,
        sigma: Vector | jnp.ndarray,
        mask: jnp.ndarray | None = None,
        D1_mask: jnp.ndarray | None = None,
        D2_mask: jnp.ndarray | None = None,
        fit_masks: bool = True,
        target: LossSpace = "eta",
        step: float | Index | Vector | np.ndarray | None = None,
    ) -> LossFn:
        if mask is None:
            mask = 1
        else:
            mask = jnp.asarray(mask)
        if D1_mask is None:
            if not isinstance(mask, Iterable):
                D1_mask = mask
            else:
                if fit_masks:
                    D1_mask = mask[:-1]
                else:
                    D1_mask = mask
        if D2_mask is None:
            if not isinstance(mask, Iterable):
                D2_mask = mask
            else:
                if fit_masks:
                    D2_mask = mask[:-2]
                else:
                    D2_mask = mask

        # Get the discretization step size

        self.alpha = alpha
        self.step = _resolve_step(step)
        self.mask = mask
        self.D1_mask = D1_mask
        self.D2_mask = D2_mask
        self.sigma = jnp.asarray(sigma)
        if target not in ["eta_normalized", "eta"]:
            raise ValueError(
                f"Invalid target: {target}. Must be one of 'eta_normalized' or 'eta'."
            )
        self.target = target

    def __call__(self, x: ExpectationParameter) -> tuple[float, float]:
        alpha = self.alpha
        step = self.step
        D1_mask = self.D1_mask
        D2_mask = self.D2_mask
        sigma = self.sigma

        # Take the Sobolev norm of the distribution
        # Much faster than using finite difference matrices
        D1p = ((x[:-1] - x[1:]) / step / sigma[:-1]) ** 2 * D1_mask
        D2p = ((x[:-2] - 2 * x[1:-1] + x[2:]) / step**2 / sigma[1:-1]**2) ** 2 * D2_mask
        penalty = jnp.sum(D1p) + jnp.sum(D2p)
        return alpha * penalty, penalty

    def __repr__(self):
        return f"SobolevGauss(alpha={self.alpha}, step={self.step}, sigma={self.sigma}, mask={self.mask}, D1_mask={self.D1_mask}, D2_mask={self.D2_mask}, target={self.target})"


def tree_flatten_sobolevgauss(obj):
    children = (obj.alpha, obj.step, obj.mask, obj.D1_mask, obj.D2_mask, obj.sigma)
    aux_data = {"target": obj.target}
    return children, aux_data


def tree_unflatten_sobolevgauss(aux_data, children):
    alpha, step, mask, D1_mask, D2_mask, sigma = children
    return SobolevGauss(
        alpha=alpha,
        step=step,
        mask=mask,
        D1_mask=D1_mask,
        D2_mask=D2_mask,
        sigma=sigma,
        target=aux_data["target"]
    )


jax.tree_util.register_pytree_node(SobolevGauss, tree_flatten_sobolevgauss, tree_unflatten_sobolevgauss)


@partial(jax.tree_util.register_dataclass, 
         data_fields=(),
         meta_fields=('alpha', 'order', 'mask', 'step', 'target'))
class SobolevOrder(Penalty):
    """
    Generalized Sobolev penalty: penalize the L2 norm of the kᵗʰ derivative.

    Args:
        alpha: overal strength of the penalty
        order: which derivative order to penalize (k)
        mask: optional mask array of length len(x)-order to zero out regions
        step: scalar or array giving your grid spacing
    """
    def __init__(
        self,
        alpha: float,
        order: int = 2,
        mask: jnp.ndarray | None = None,
        step: float | Index | Vector | np.ndarray | None = None,
        target: LossSpace = "eta",
        **kwargs,  # pass through whatever you need for LossSpace etc.
    ):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.order = order
        self.target = target
        # set up mask
        if mask is None:
            self.mask = 1.0
        else:
            mask = jnp.asarray(mask)
            # ensure mask length matches your diff length if needed
            if mask.shape and mask.shape[0] != ...:  # len(x)-order
                mask = mask[: -order]
            self.mask = mask

        # resolve step exactly as you did before...
        self.step = _resolve_step(step)

    def __call__(self, x: ExpectationParameter) -> tuple[float, float]:
        # compute the k-th forward difference
        # start with the zeroth difference
        #Dk = x
        # apply first-difference self.order times
        #for _ in range(self.order):
        #    Dk = Dk[1:] - Dk[:-1]
        Dk = kth_diff(x, self.order)
        # scale for step-size
        Dk_scaled = Dk / (self.step ** self.order)
        # mask out if desired
        penalty_term = jnp.sum((Dk_scaled ** 2) * self.mask)
        return self.alpha * penalty_term, penalty_term

    def __repr__(self):
        return (
            f"Sobolev(alpha={self.alpha}, order={self.order}, "
            f"step={self.step}, mask={self.mask!r})"
        )

@partial(jax.jit, static_argnames=("order",))
def kth_diff(x, order: int):
    return jnp.diff(x, n=order)


def tree_flatten_sobolevorder(obj):
    children = (obj.alpha, obj.step, obj.mask, obj.order)
    aux_data = {"target": obj.target}
    return children, aux_data


def tree_unflatten_sobolevorder(aux_data, children):
    alpha, step, mask, order = children
    return SobolevOrder(
        alpha=alpha,
        step=step,
        mask=mask,
        order=order,
        target=aux_data["target"]
    )





# helper to resolve `step` exactly as in your original __init__
def _resolve_step(step):
    # copy your `match step: ...` logic here
    if step is None:
        warnings.warn("Step size not specified; using 1.0.")
        return 1.0

    
    match step:
        case None:
            warnings.warn(
                "Step size not specified. Sobolev norm will not be scale invariant."
            )
            step = 1.0
        case float() | int() | np.number():
            step = float(step)
        case Index() | Vector():
            step = step.dX
        case jnp.ndarray():
            if step.shape == ():
                step = step
            else:
                d = jnp.diff(step)
                if not jnp.allclose(d, d[0]):
                    warnings.warn("Step size is not constant, using the first value.")
                step = d[0]
        case np.ndarray():
            d = np.diff(step)
            if not np.allclose(d, d[0]):
                warnings.warn("Step size is not constant, using the first value.")
            step = d[0]
        case _:
            raise ValueError(
                f"Invalid step size: {step}. Must be a float, Index, Vector, or np.ndarray."
            )
    return step


class IdPenalty(Penalty):
    def __call__(self, x: ExpectationParameter) -> tuple[float, float]:
        return 0.0, 0.0


def slog(x):
    """Compute the logarithm of x, with a small epsilon term to avoid log(0).

    NOTE: JAX is unstable with this function. Use instead JAX's implementation.

    Args:
        x: The input value

    Returns:
        The logarithm of x, with a small epsilon term to avoid log(0)
    """
    return jnp.where(x <= 1e-5, 0.0, jnp.log(x))


def entropy(x):
    """Compute the entropy of a distribution.

    This function computes a modified entropy term by multiplying x by log(x + 1).
    A small offset of 1.0 is added before taking the logarithm to avoid numerical
    instabilities when x is close to zero.

    Args:
        x: Input array representing a probability distribution

    Returns:
        The entropy term x * log(x + 1) computed element-wise
    """
    return x * jnp.log(x + 1.0)


def onecost(mu, C, D=100):
    """Compute a smooth sparsity cost in mu-space.

    This function computes a sparsity penalty using a smooth approximation of the
    Heaviside step function. The smoothing ensures gradients don't vanish during
    optimization. The penalty increases when values in mu exceed the threshold C.

    Args:
        mu: Input array to compute sparsity cost on
        C: Threshold value that determines when penalty starts increasing
        D: Smoothing parameter that controls the sharpness of the transition.
           Higher values give a sharper transition.

    Returns:
        The summed sparsity cost across all elements of mu
    """

    # Smooth approximation of the Heaviside step function using a logistic function
    return 0.5 * (1 + 2 / 3.141592 * jnp.arctan((mu - C) / (C / D)))

