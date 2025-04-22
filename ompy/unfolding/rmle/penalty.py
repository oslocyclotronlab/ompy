from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import Iterable, Callable

import jax.numpy as jnp
import numpy as np

from ... import Index, Vector
from .stubs import PenaltyFn, PenaltyTarget


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
    def closure(self) -> PenaltyFn:
        pass

    @property
    def target(self) -> PenaltyTarget:
        return self._target


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

    def __init__(self, alpha: float):
        self.alpha = alpha
        self._target = "eta_normalized"

    def closure(self) -> PenaltyFn:
        alpha = self.alpha

        def fn(mu, eta, p):
            penalty = jnp.sum(entropy(p))
            return alpha * penalty, penalty

        return fn


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
        target: PenaltyTarget = "eta",
        step: float | Index | Vector | np.ndarray | None = None,
    ) -> PenaltyFn:
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
            case np.ndarray():
                d = np.diff(step)
                if not np.allclose(d, d[0]):
                    warnings.warn("Step size is not constant, using the first value.")
                step = d[0]
            case _:
                raise ValueError(
                    f"Invalid step size: {step}. Must be a float, Index, Vector, or np.ndarray."
                )

        self.alpha = alpha
        self.step = step
        self.mask = mask
        self.D1_mask = D1_mask
        self.D2_mask = D2_mask
        if target not in ["eta_normalized", "eta"]:
            raise ValueError(
                f"Invalid target: {target}. Must be one of 'eta_normalized' or 'eta'."
            )
        self._target = target

    def closure(self) -> PenaltyFn:
        alpha = self.alpha
        step = self.step
        D1_mask = self.D1_mask
        D2_mask = self.D2_mask

        match self.target:
            case "eta_normalized":

                def fn(mu, eta, p):
                    # Take the Sobolev norm of the distribution
                    # Much faster than using finite difference matrices
                    D1p = ((p[:-1] - p[1:]) / step) ** 2 * D1_mask
                    D2p = ((p[:-2] - 2 * p[1:-1] + p[2:]) / step**2) ** 2 * D2_mask
                    penalty = jnp.sum(D1p) + jnp.sum(D2p)
                    return alpha * penalty, penalty
            case "eta":

                def fn(mu, eta, p):
                    D1eta = ((eta[:-1] - eta[1:]) / step) ** 2 * D1_mask
                    D2eta = (
                        (eta[:-2] - 2 * eta[1:-1] + eta[2:]) / step**2
                    ) ** 2 * D2_mask
                    penalty = jnp.sum(D1eta) + jnp.sum(D2eta)
                    return alpha * penalty, penalty

        return fn


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

    def __init__(
        self,
        alpha: float,
        threshold: float = 0.1,
        smoothing: float = 100,
        target: PenaltyTarget = "mu_normalized",
    ):
        self.alpha = alpha
        self.threshold = threshold
        self.smoothing = smoothing
        self._target = target

    def closure(self) -> PenaltyFn:
        alpha = self.alpha
        threshold = self.threshold
        smoothing = self.smoothing
        match self.target:
            case "mu_normalized":

                def fn(mu, eta, p):
                    penalty = onecost(p, threshold, smoothing)
                    return alpha * penalty, penalty
            case "mu":

                def fn(mu, eta, p):
                    penalty = onecost(mu, threshold, smoothing)
                    return alpha * penalty, penalty
            case _:
                raise ValueError(
                    f"Invalid target: {self.target}. Must be one of 'mu_normalized' or 'mu'."
                )
        return fn



class SobolevGauss(Penalty):
    def __init__(
        self,
        alpha: float,
        sigma: Vector | jnp.ndarray,
        mask: jnp.ndarray | None = None,
        D1_mask: jnp.ndarray | None = None,
        D2_mask: jnp.ndarray | None = None,
        fit_masks: bool = True,
        target: PenaltyTarget = "eta",
        step: float | Index | Vector | np.ndarray | None = None,
    ) -> PenaltyFn:
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
            case np.ndarray():
                d = np.diff(step)
                if not np.allclose(d, d[0]):
                    warnings.warn("Step size is not constant, using the first value.")
                step = d[0]
            case _:
                raise ValueError(
                    f"Invalid step size: {step}. Must be a float, Index, Vector, or np.ndarray."
                )

        self.alpha = alpha
        self.step = step
        self.mask = mask
        self.D1_mask = D1_mask
        self.D2_mask = D2_mask
        self.sigma = jnp.asarray(sigma)
        if target not in ["eta_normalized", "eta"]:
            raise ValueError(
                f"Invalid target: {target}. Must be one of 'eta_normalized' or 'eta'."
            )
        self._target = target

    def closure(self) -> PenaltyFn:
        alpha = self.alpha
        step = self.step
        D1_mask = self.D1_mask
        D2_mask = self.D2_mask
        sigma = self.sigma

        match self.target:
            case "eta_normalized":

                def fn(mu, eta, p):
                    # Take the Sobolev norm of the distribution
                    # Much faster than using finite difference matrices
                    D1p = ((p[:-1] - p[1:]) / step / sigma[:-1]) ** 2 * D1_mask
                    D2p = ((p[:-2] - 2 * p[1:-1] + p[2:]) / step**2 / sigma[1:-1]**2) ** 2 * D2_mask
                    penalty = jnp.sum(D1p) + jnp.sum(D2p)
                    return alpha * penalty, penalty
            case "eta":

                def fn(mu, eta, p):
                    D1eta = ((eta[:-1] - eta[1:]) / step / sigma[:-1]) ** 2 * D1_mask
                    D2eta = (
                        (eta[:-2] - 2 * eta[1:-1] + eta[2:]) / step**2 / sigma[1:-1]**2
                    ) ** 2 * D2_mask
                    penalty = jnp.sum(D1eta) + jnp.sum(D2eta)
                    return alpha * penalty, penalty

        return fn


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
    return jnp.sum(0.5 * (1 + 2 / 3.141592 * jnp.arctan((mu - C) / (C / D))))


def total_penalty(penalties: tuple[PenaltyFn, ...], mu, eta, p) -> tuple[float, float]:
    """Compute the total penalty value for a list of penalties.

    This function takes a tuple of penalty functions and applies them to the given
    distributions (mu, eta, p). It returns the total penalty value and the total
    penalty without regularization strength applied (for L-curve analysis).

    Args:
        penalties: Tuple of penalty functions
        mu: Array representing the mu distribution
        eta: Array representing the eta distribution
        p: Array representing the p distribution

    Returns:
        total: Total penalty value
        partial: Total penalty value without regularization strength applied
    """
    total = 0.0
    partial = 0.0
    for penalty in penalties:
        tot, part = penalty(mu, eta, p)
        total += tot
        partial += part
    return total, partial
