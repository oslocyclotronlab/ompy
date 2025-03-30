from __future__ import annotations
from .unfolder import Unfolder
from .result1d import UnfoldedResult1DSimple, Cost1D, Parameters1D, ResultMeta1D, Result
from .result2d import UnfoldedResult2DSimple, Cost2D, Parameters2D, ResultMeta2D
from .stubs import Space
from .. import Matrix, Vector, OPTAX_AVAILABLE
from ..stubs import Plot1D, Axes, array1D
from .. import Index
import numpy as np
import time
from tqdm.autonotebook import tqdm
from dataclasses import dataclass, fields, asdict
import matplotlib.pyplot as plt
from typing import Any, TypedDict, Iterable, Callable, TypeAlias, Literal
from functools import partial
from pathlib import Path
from itertools import product
from typing_extensions import override
import os
from abc import ABC, abstractmethod
from jax.scipy import special
from numba import njit
from jax.experimental import io_callback
import threading
import queue
from matplotlib.colors import LogNorm
from dataclasses import field

try:
    from jax_tqdm import loop_tqdm
except ImportError:
    loop_tqdm = lambda func: func


def is_jupyter_notebook():
    return "JPY_PARENT_PID" in os.environ


if OPTAX_AVAILABLE:
    import optax


import jax
from jax import numpy as jnp
from jax import Array

"""
TODO
-[ ] Optimize vector
-[x] Hyperparameter search for NAG++
     Parameter transform, logistic or inverse hyperbolic tangent
-[x] GPU. Remember to set correct environment variables
-[ ] Test more optimizers
-[ ] Initial optimizer hyperparameter search
-[ ] KL + ME
-[x] Background
-[x] Why does taking SiRi into account worsen the result?
     Because I am stupid.
"""

LOGGING_QUEUE = queue.Queue()


# JAX Jit hates this function
def slog(x):
    return jnp.where(x <= 1e-5, 0.0, jnp.log(x))


def kl(nu, n):
    # return nu - n + n * jnp.log(n / (nu+1e-10) + 1e-10)
    # return (nu - n) + n * (slog(n) - slog(nu))
    eps = 1e-5
    # mask = (nu <= eps) | (n <= eps)
    return nu - n + n * jnp.log(n / (nu + 1e-10) + 1e-10)
    # return jnp.where(mask, 0.0, (nu - n) + n * (jnp.log(n) - jnp.log(nu)))
    # return (nu - n) + n * (jnp.log(n) - jnp.log(nu))


def entropy(mu):
    # mask = mu <= 1e-5
    # return jnp.where(mask, 0.0, mu * jnp.log(mu))
    return mu * jnp.log(mu + 1.0)
    # return -jnp.sum(mu * slog(mu))
    # return mu * jnp.log(mu)


def split_entropy(mu, lower: float, upper: float, midpoint: float):
    entropy_ = entropy(mu)
    return logistic_interpolation(entropy_, lower, upper, midpoint)


def difference_cost(n, nu):
    return (jnp.sum(n) - jnp.sum(nu)) ** 2


def sigmoid(x):
    return 1 / (1 + jnp.exp(-x * 1e-2))


def logistic_interpolation(t, lower, upper, midpoint):
    # d = find_d(C, A, B, k)
    return lower + (upper - lower) * sigmoid(t - midpoint)


def onecost(mu, C):
    # Smooth approximation of the Heaviside step function using a logistic function
    # return jnp.sum(0.5 * (1 + jnp.tanh((mu - C) / (1e-6 + C / 10))))
    return jnp.sum(0.5 * (1 + 2 / 3.141592 * jnp.arctan((mu - C) / (C / 100))))


def onecost_2(mu, C):
    # Smooth approximation of the Heaviside step function using a logistic function
    # return jnp.sum(0.5 * (1 + jnp.tanh((mu - C) / (1e-6 + C / 10))))
    return jnp.sum(0.5 * (1 + 2 / 3.141592 * jnp.arctan((mu - C) / (C / 10))))


def to_tau(mu):
    return jnp.sqrt(mu)
    # return jnp.sqrt(mu)
    # return jnp.log(mu + 1e-10)


def from_tau(tau):
    # return jnp.where(tau < 0, tau**2, tau)
    # Need to be careful with the linear term, as it allows for negative values
    return tau**2  # + 1e-3*tau
    # return tau**2
    # return jnp.exp(tau)


def cost(
    tau,
    R,
    G_ex,
    G_eg,
    y,
    # D1: jnp.ndarray,
    # D2: jnp.ndarray,
    bg=None,
    alpha=0.0,
    alpha_c=1.0,
    D1_mask: jnp.ndarray | int = 1,
    D2_mask: jnp.ndarray | int = 1,
):
    mu_ = from_tau(tau)
    if bg is not None:
        mu, beta = jnp.vsplit(mu_, 2)
        nu = G_ex @ mu @ R
        nu = nu + beta
        loglike = jnp.sum(kl(nu, y)) + jnp.sum(kl(beta, bg))
        cost = loglike
    else:
        mu = mu_
        eta = mu @ G_eg
        nu = G_ex @ eta @ R
        # loglike = jnp.sum(kl(nu, y))
        loglike_per_instance = jnp.sum(kl(nu, y), axis=1)
        loglike = jnp.mean(loglike_per_instance)

        # penalty = alpha*onecost(mu, alpha_c)**2
        # Normalize row wise
        # prop = mu / jnp.sum(mu, axis=0)[None, 1]
        # jax.debug.print("mu.shape {shape}", shape=mu.shape)
        # jax.debug.print("MU min: {min}, max: {max}, mean: {mean}", min=jnp.min(mu), max=jnp.max(mu), mean=jnp.mean(mu))
        distribution = mu / (jnp.sum(mu, axis=1, keepdims=True) + 1e-10)
        # jax.debug.print("min: {min}, max: {max}, mean: {mean}", min=jnp.min(distribution), max=jnp.max(distribution), mean=jnp.mean(distribution))
        # row_sums = jnp.sum(mu, axis=1)
        # jax.debug.print("Row sums: {row_sums}", row_sums=row_sums)
        # entropy = jnp.sum(special.entr(distribution))
        # entropy = -jnp.sum(distribution * jnp.log(distribution+1e-10))
        # D1p = (D1@distribution.T)**2 #* D1_mask
        # D2p = (D2@distribution.T)**2 #* D2_mask

        # Normalize eta to a probability distribution row wise
        # distribution = eta / (jnp.sum(eta, axis=1, keepdims=True) + 1e-10)
        # distribution = mu

        # D1p = jnp.einsum('ij,kj->ik', D1, distribution) ** 2 * D1_mask
        # D2p = jnp.einsum('ij,kj->ik', D2, distribution) ** 2 * D2_mask
        D1p = (distribution[:, :-1] - distribution[:, 1:]) ** 2 * D1_mask
        D2p = (
            distribution[:, :-2] - 2 * distribution[:, 1:-1] + distribution[:, 2:]
        ) ** 2 * D2_mask
        # penalty = jnp.sum(D1p) + jnp.sum(D2p)
        # jax.debug.print("Entropy: {entropy}", entropy=entropy)
        # jax.debug.print("KL:w: {kl}", kl=loglike)
        # jax.debug.print(" ")
        # penalty = jnp.sum(D1p) + jnp.sum(D2p)
        # penalty = jnp.sum(D1p) + jnp.sum(D2p)
        penalty_per_instance = jnp.sum(D1p, axis=1) + jnp.sum(D2p, axis=1)
        penalty = jnp.mean(penalty_per_instance)

        cost = loglike + alpha * penalty

    aux = {"loglike": loglike, "penalty": penalty}

    return cost, aux


def sigmoid(x, omega, sigma):
    return 1 / (1 + jnp.exp(-(x - omega) / sigma))


def cost_1d_(
    tau,
    R,
    y,
    bg=None,
    alpha=0.0,
    alpha_c=1.0,
    alpha_bg=0.0,
    alpha_xi_alpha=0.0,
    alpha_xi_beta=0.0,
    alpha_xi_c=10.0,
    D1: jnp.ndarray | None = None,
    D2: jnp.ndarray | None = None,
    D1_mask: jnp.ndarray | None = None,
    D2_mask: jnp.ndarray | None = None,
):  # , alpha=0.3e-1):
    # omega = jnp.round(tau[-1]).astype(int)  # Parameter for mask
    # sigma = jnp.round(tau[-2]).astype(int)
    # tau = tau[:-1]
    mu_ = from_tau(tau)
    # "mu" might be [prompt..., background...]
    if bg is not None:
        mu = mu_[: -len(bg)]
        beta = mu_[-len(bg) :]
        nu = R @ mu
        nu = nu + beta
        loglike = jnp.sum(kl(nu, y)) + jnp.sum(kl(beta, bg))
        beta_diff = jnp.diff(beta)
        beta_cost = alpha_bg * jnp.sum(beta_diff**2)
        loglike = loglike + beta_cost
    else:
        mu = mu_
        nu = R @ mu
        loglike = jnp.sum(kl(nu, y))
    # Rescaling seems to make the optimization much slower
    # penalty = alpha*onecost(mu, alpha_c)**2
    # penalty = alpha*onecost(mu / (1+jnp.abs(mu)), 0.01)**2
    # try a entropy penalty
    # prop = jax.nn.softmax(mu)
    prop = mu / jnp.sum(mu)
    # penalty = alpha * -jnp.sum(prop * jnp.log(prop+1e-10))
    if D1 is None or D2 is None:
        raise ValueError("D1 and D2 must be provided")
    D1p = (D1 @ prop) ** 2
    if D1_mask is not None:
        # D1_mask = sigmoid(jnp.arange(len(nu)), omega, sigma)
        D1p = D1p * D1_mask
    # D2p = (D2@prop)**2
    penalty = jnp.sum(D1p)  # + jnp.sum(D2p)
    cost = loglike + alpha * penalty

    aux = {"loglike": loglike, "penalty": penalty}

    return cost, aux
    # nu = jnp.log(nu + 1e-1)
    # n = jnp.log(n + 1e-1)
    # return jnp.sum(jnp.abs(nu - n)) + alpha*onecost(mu)**2 #- beta*jnp.sum(entropy(mu))# + difference_cost(n, nu)
    # total = jnp.sum(kl(nu, n)) #- alpha*jnp.sum(entropy(mu)) + beta*difference_cost(n, nu)


@njit
def derivative_matrix_order1(n):
    """
    Create a finite difference matrix for the first derivative.
    Matrix shape is (n-1, n)
    """
    D = np.zeros((n - 1, n))
    for i in range(n - 1):
        D[i, i] = 1.0
        D[i, i + 1] = -1.0
    return D


@njit
def derivative_matrix_order2(n):
    """
    Create a finite difference matrix for the second derivative.
    Matrix shape is (n-2, n)
    """
    D = np.zeros((n - 2, n))
    for i in range(n - 2):
        D[i, i] = 1.0
        D[i, i + 1] = -2.0
        D[i, i + 2] = 1.0
    return D


@njit
def setup_D_matrices(
    n: int,
    # d1_mask: jnp.ndarray | None = None,
    # d2_mask: jnp.ndarray | None = None,
):
    D1 = derivative_matrix_order1(n)
    D2 = derivative_matrix_order2(n)
    # if d1_mask is not None:
    #    D1 = jnp.asarray(D1).to_device(d1_mask.device)
    #    M1 = jnp.sqrt(d1_mask)
    #    D1 = M1*D1
    # if d2_mask is not None:
    #    D2 = jnp.asarray(D2).to_device(d2_mask.device)
    #    M2 = jnp.sqrt(d2_mask)
    #    D2 = M2*D2
    return D1, D2


def logging_callback(x):
    LOGGING_QUEUE.put(int(x))
    return x


# Worker function that runs in a separate thread and updates tqdm.
def progress_bar_worker(total_iterations: int):
    pbar = tqdm(total=total_iterations)
    last_value = 0
    while last_value < total_iterations:
        try:
            # Wait a bit for new iteration counts.
            i = LOGGING_QUEUE.get(timeout=0.1)
            if i < 0:
                break
            # Compute the delta and update the progress bar.
            delta = i - last_value
            if delta > 0:
                pbar.update(delta)
                last_value = i
        except queue.Empty:
            continue
    pbar.close()


def gaussian(x, A, mu, sigma):
    y = jnp.exp(-((x - mu) ** 2) / (2 * sigma**2))
    return A * y


def var_penalty(param: float, lower: float, upper: float) -> float:
    # Quadratic penalty outside the [lower, upper] interval.
    lower_penalty = jnp.where(param < lower, (param - lower) ** 2, 0.0)
    upper_penalty = jnp.where(param > upper, (param - upper) ** 2, 0.0)
    return lower_penalty + upper_penalty


def relaxed_one_hot(logits, temperature=0.01):
    return jax.nn.softmax(logits / temperature)


def cost_1d(
    tau: jnp.ndarray,
    GegD: jnp.ndarray,
    y: jnp.ndarray,
    G_eg: jnp.ndarray,
    unpacker: Callable[[jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]],
    contaminants: tuple[Callable[[jnp.ndarray], tuple[jnp.ndarray, float]]],
    bg=None,
    alpha=0.0,
    D1_mask: jnp.ndarray | None = None,
    D2_mask: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, dict]:

    aleph = from_tau(tau)
    mu, mu_contaminants = unpacker(aleph)

    def xi_nop(_):
        return mu, mu, 0.0

    def handle_contaminants(_):
        if (
            len(contaminants) == 0
        ):  # If contaminants is empty, return defaults to make jax jit happy
            return mu, mu, 0.0
        xi_mu_list, penalty_list = zip(
            *[c(x, G_eg, GegD) for c, x in zip(contaminants, mu_contaminants)]
        )

        # Sum over all contaminants
        xi_mu_sum = mu + sum(xi_mu_list)
        xi_penalty = sum(penalty_list)
        return mu, xi_mu_sum, xi_penalty

    mu, xi_mu_sum, xi_penalty = jax.lax.cond(
        len(contaminants) > 0, handle_contaminants, xi_nop, None
    )

    nu = xi_mu_sum @ GegD
    likelihood_body = kl(nu, y)
    loglike = jnp.sum(likelihood_body)

    # We only compute eta if needed
    def eta_nop(_):
        return 0.0

    def eta_body(_):
        # Map to eta space
        eta = mu @ G_eg
        # Rescale to get a proper probability distribution
        prop = eta / jnp.sum(eta + 1e-10)

        # Take the Sobolev norm of the distribution
        # Much faster than using finite difference matrices
        D1p = (prop[:-1] - prop[1:]) ** 2 * D1_mask
        D2p = (prop[:-2] - 2 * prop[1:-1] + prop[2:]) ** 2 * D2_mask
        penalty = jnp.sum(D1p) + jnp.sum(D2p)
        penalty = penalty + 5e-12*(jnp.sum(eta[659:809]))  # 5e-12

        return penalty

    penalty = jax.lax.cond(alpha == 0.0, eta_nop, eta_body, None)

    # Rescaling seems to make the optimization much slower
    # penalty = alpha*onecost(mu, alpha_c)**2
    # penalty = alpha*onecost(mu / (1+jnp.abs(mu)), 0.01)**2
    # try a entropy penalty
    # prop = jax.nn.softmax(mu)
    # Ops! Want to do this on eta for sobolev
    # If
    # prop = eta / jnp.sum(eta)

    # entropy_body = prop * jnp.log(prop+1e-10)
    # prop = eta / (jnp.sum(eta) + 1e-10)
    # prop = mu

    # if penalty_mask is not None:
    #    entropy_body = entropy_body * penalty_mask
    # entropy = -jnp.sum(entropy_body)

    # penalty = alpha * entropy
    # xi_penalty = xi_pos_penalty + xi_A_penalty

    # Orthogonality penalty
    # ortho_penalty = 1e-12*jnp.sum(eta @ xi_eta)**2

    cost = loglike + alpha * penalty + xi_penalty  # + ortho_penalty

    aux = {
        "loglike": loglike,
        "penalty": penalty,
        "xi_penalty": xi_penalty,
    }

    return cost, aux


def bound_variable(u, a, b):
    return a + (b - a) * jax.nn.sigmoid(u)


def cost_1d_v3(
    tau,
    R,
    y,
    bg=None,
    R_xi=None,
    xi_constraints=None,
    alpha=0.0,
    alpha_c=1.0,
    alpha_bg=0.0,
    alpha_xi_alpha=1.0,
    alpha_xi_beta=1.0,
    alpha_xi_c=10.0,
):
    """
    Unified cost function for unfolding with optional background and contaminants.

    Parameters
    ----------
    tau : array
        Parameters to optimize (includes all components)
    R : array
        Main response matrix
    y : array
        Observed spectrum
    bg : array, optional
        Background spectrum data
    R_xsi : list of arrays, optional
        Response matrices for contaminants
    alpha : float
        Regularization parameter for main spectrum
    alpha_c : float
        Regularization parameter for contaminants
    zeta : float
        Smoothness parameter for background
    """
    # Split parameters based on what components are present
    n_bg = len(bg) if bg is not None else 0
    n_cont = len(R_xi) if R_xi is not None else 0
    # n_bg is the length of the vector, and all vectors must be the same length
    n_cont *= n_bg

    # Handle main spectrum
    main_tau = tau[: -n_bg - n_cont] if (n_bg + n_cont) > 0 else tau
    mu = main_tau**2  # Non-negativity via square
    nu = mu@R

    # Add regularization for main spectrum if needed
    if alpha > 0:
        cost += alpha * onecost(mu, alpha_c) ** 2

    # Initialize total folded spectrum
    nu_total = nu.copy()

    # Initialize cost
    cost = 0.0

    # Add background component if present
    if bg is not None:
        bg_tau = tau[-n_bg - n_cont : -n_cont] if n_cont > 0 else tau[-n_bg:]
        beta = bg_tau**2  # Non-negativity
        nu_total = nu_total + beta
        # Likelihood of background
        cost += jnp.sum(kl(beta, bg))

        # Background smoothness
        if alpha_bg > 0:
            beta_diff = jnp.diff(beta)
            cost += alpha_bg * jnp.sum(beta_diff**2)

    # Handle contaminants
    for i in range(n_cont):
        # Unpack the xi vector
        mu_xi_i = tau[-n_bg * i - n_bg : -n_bg * i]
        nu_xi_i =  mu_xi_i@R_xi[i]
        nu_total = nu_total + nu_xi_i

        # Add penalty
        probs = jnp.softmax(mu_xi_i)
        # entropy
        penalty = -jnp.sum(probs * jnp.log(probs))
        cost += alpha_xi_alpha * penalty

        # We can't allow the peak to be outside of the constraints
        idx = jnp.arange(len(mu))
        left, right = xi_constraints[i]
        lower_mask = jax.nn.sigmoid((idx - left) / alpha_xi_c)
        upper_mask = jax.nn.sigmoid((right - idx) / alpha_xi_c)
        valid_region = lower_mask * upper_mask
        # penalize values outside boundaries
        penalty = jnp.sum(mu_xi_i**2 * (1 - valid_region))
        cost += alpha_xi_beta * penalty

    # Final likelihood using total folded spectrum
    cost += jnp.sum(kl(nu_total, y))

    return cost


def cost_components_from_result(result, eta, alpha=0, beta=0):
    return cost_components(
        result.raw.values,
        result.unfolded().values,
        result.R.values,
        eta=eta,
        G_eg=result.G.values,
        G_ex=result.G_ex,
        alpha=result.meta.kwargs["alpha"],
    )


def cost_components(n, mu, R, eta=None, G_eg=None, G_ex=None, alpha=0, beta=0):
    """Return the loss, regularization and validation components of the cost function"""
    nu = R @ mu
    loss = jnp.sum(kl(nu, n))
    regularization = onecost(mu) ** 2  # - beta*jnp.sum(entropy(eta))
    if eta is not None:
        eta_ = G_eg @ mu
        validation = jnp.sum(kl(eta_, eta))
        return loss, regularization, validation
    return loss, regularization


@dataclass(kw_only=True)
class RMLEResult2D(Cost1D, UnfoldedResult2DSimple):
    beta: Matrix | None = None

    def _save(self, path: Path, meta: dict[str, Any], exist_ok: bool = False):
        Cost1D._save(self, path, meta, exist_ok)
        UnfoldedResult2DSimple._save(self, path, meta, exist_ok)

    @classmethod
    def _load(cls, path: Path, meta: dict[str, Any]) -> dict[str, np.ndarray | Matrix]:
        a = Cost1D._load(path, meta)
        b = UnfoldedResult2DSimple._load(path, meta)
        return a | b


@dataclass(kw_only=True)
class RMLEResult1D(Cost1D, UnfoldedResult1DSimple):
    beta: Vector | None = None

    def _save(self, path: Path, meta: dict[str, Any], exist_ok: bool = False):
        UnfoldedResult1DSimple._save(self, path, meta, exist_ok)
        Cost1D._save(self, path, meta, exist_ok)

    @classmethod
    def _load(cls, path: Path, meta: dict[str, Any]) -> dict[str, np.ndarray | Vector]:
        a = Cost1D._load(path, meta)
        b = UnfoldedResult1DSimple._load(path, meta)
        return a | b


class OptimResult1D:
    def __init__(
        self,
        *,
        prototype: Vector,
        mu,
        total_cost,
        loglike,
        penalty,
        xi_penalty,
        beta=None,
        xi=None,
    ):
        self.mu = prototype.clone(values=np.asarray(mu))
        if beta is not None:
            self.beta = prototype.clone(values=np.asarray(beta))
        else:
            self.beta = None
        if xi is not None:
            self.xi = [prototype.clone(values=np.asarray(x)) for x in xi]
        else:
            self.xi = []
        self.total_cost = np.asarray(total_cost)
        self.loglike = np.asarray(loglike)
        self.penalty = np.asarray(penalty)
        self.xi_penalty = np.asarray(xi_penalty)
        self.total_cost.flags.writeable = True
        self.loglike.flags.writeable = True
        self.penalty.flags.writeable = True
        self.xi_penalty.flags.writeable = True

    @property
    def aux(self):
        return {
            "loglike": self.loglike,
            "penalty": self.penalty,
            "xi_penalty": self.xi_penalty,
        }



@dataclass(kw_only=True)
class Contaminant1D:
    """A helper class to specify contaminants
    TODO:
    - Allow the user to specify D_xi, G_egD_xi, G_exD_xi
    """

    E: Index
    initial: tuple[int, float]  # (index, amplitude)
    central_bounds: tuple[int, int]  # (lower, upper)
    temperature: float = 0.01
    amplitude_mu_penalty: float = 0.0
    amplitude_eta_penalty: float = 0.0
    amplitude_nu_penalty: float = 0.0
    amplitude_mu_bounds: tuple[float, float] | None = None
    amplitude_eta_bounds: tuple[float, float] | None = None
    amplitude_nu_bounds: tuple[float, float] | None = None

    def __post_init__(self):
        # The user can specify the central bounds as indexable expressions
        i = self.E.index_expression(self.central_bounds[0])
        j = self.E.index_expression(self.central_bounds[1])
        self.central_bounds = (i, j)
        # Same with the central value for the initial guess
        k = self.E.index_expression(self.initial[0])
        self.initial = (k, self.initial[1])
        # Check the bounds
        if self.amplitude_mu_penalty != 0 and self.amplitude_mu_bounds is None:
            raise ValueError(
                "amplitude_mu_bounds must be set if amplitude_mu_penalty != 0.0"
            )
        if self.amplitude_eta_penalty != 0 and self.amplitude_eta_bounds is None:
            raise ValueError(
                "amplitude_eta_bounds must be set if amplitude_eta_penalty != 0.0"
            )
        if self.amplitude_nu_penalty != 0 and self.amplitude_nu_bounds is None:
            raise ValueError(
                "amplitude_nu_bounds must be set if amplitude_nu_penalty != 0.0"
            )
        if self.amplitude_mu_bounds is not None:
            if self.amplitude_mu_bounds[0] > self.amplitude_mu_bounds[1]:
                raise ValueError(
                    "amplitude_mu_bounds must be a tuple of two numbers, the lower and upper bounds of the amplitude of mu"
                )
        if self.amplitude_eta_bounds is not None:
            if self.amplitude_eta_bounds[0] > self.amplitude_eta_bounds[1]:
                raise ValueError(
                    "amplitude_eta_bounds must be a tuple of two numbers, the lower and upper bounds of the amplitude of eta"
                )
        if self.amplitude_nu_bounds is not None:
            if self.amplitude_nu_bounds[0] > self.amplitude_nu_bounds[1]:
                raise ValueError(
                    "amplitude_nu_bounds must be a tuple of two numbers, the lower and upper bounds of the amplitude of nu"
                )

    def closure(self) -> Callable[[jnp.ndarray], tuple[jnp.ndarray, float]]:
        lower, upper = self.central_bounds
        T = self.temperature
        amplitude_mu_penalty = self.amplitude_mu_penalty
        amplitude_eta_penalty = self.amplitude_eta_penalty
        amplitude_nu_penalty = self.amplitude_nu_penalty

        # Validate and extract bounds for each amplitude penalty
        def get_bounds(penalty, bounds, name):
            if penalty != 0.0 and bounds is None:
                raise ValueError(f"{name}_bounds must be set if {name}_penalty != 0.0")
            if bounds is not None:
                return bounds
            # Default bounds when not used, for JAX tracer
            return (-1.0, -1.0)

        lower_mu, upper_mu = get_bounds(
            amplitude_mu_penalty, self.amplitude_mu_bounds, "amplitude_mu"
        )

        lower_eta, upper_eta = get_bounds(
            amplitude_eta_penalty, self.amplitude_eta_bounds, "amplitude_eta"
        )

        lower_nu, upper_nu = get_bounds(
            amplitude_nu_penalty, self.amplitude_nu_bounds, "amplitude_nu"
        )

        @jax.jit
        def func(
            mu: jnp.ndarray,  # mu of xi, not of the data
            G_eg: jnp.ndarray,
            G_egD: jnp.ndarray,
        ) -> tuple[jnp.ndarray, float]:
            # Enforce the central bounds
            mu = mu.at[:lower].set(0.0)
            mu = mu.at[upper:].set(0.0)
            # The one-hot removes the amplitude, so we must reapply it
            amplitude_mu = jnp.max(mu)
            # One-hot encoding ensures a single non-zero element
            mu = amplitude_mu * relaxed_one_hot(mu, temperature=T)

            def identity_penalty(_):
                return 0.0

            def mu_penalty(_):
                return amplitude_mu_penalty * var_penalty(
                    amplitude_mu, lower_mu, upper_mu
                )

            def eta_penalty(_):
                eta = mu @ G_eg
                amplitude_eta = jnp.max(eta)
                return amplitude_eta_penalty * var_penalty(
                    amplitude_eta, lower_eta, upper_eta
                )

            def nu_penalty(_):
                nu = mu @ G_egD
                # Here we only care about the amplitude within FE,
                # which is equivalent to being within the bounds
                amplitude_nu = jnp.max(nu[lower:upper])
                return amplitude_nu_penalty * var_penalty(
                    amplitude_nu, lower_nu, upper_nu
                )

            mu_cost = jax.lax.cond(
                amplitude_mu_penalty == 0.0, identity_penalty, mu_penalty, None
            )

            eta_cost = jax.lax.cond(
                amplitude_eta_penalty == 0.0, identity_penalty, eta_penalty, None
            )

            nu_cost = jax.lax.cond(
                amplitude_nu_penalty == 0.0, identity_penalty, nu_penalty, None
            )

            return mu, mu_cost + eta_cost + nu_cost

        return func

    def setup_initial(self, initial: jnp.ndarray | None = None) -> jnp.ndarray:
        if initial is None:
            mu = jnp.zeros(self.E.bins.shape)
        else:
            if len(initial) != len(self.E):
                raise ValueError(
                    f"Initial must be of length of data, got {len(initial)}"
                )
            mu = jnp.zeros_like(initial)
        # We map the amplitude to tau space since it will be inverted in the loop
        mu = mu.at[self.initial[0]].set(to_tau(self.initial[1]))
        return mu

    def __len__(self):
        return len(self.E)

    def _repr_html_(self):
        """Generates an HTML representation of the object for Jupyter notebooks."""
        table_rows = []

        # Define the attributes to be displayed
        attributes = [
            ("E (Index)", repr(self.E)),
            ("Initial (index, amplitude)", self.initial),
            ("Central Bounds (lower, upper)", self.central_bounds),
            ("Temperature", self.temperature),
            ("Amplitude μ Penalty", self.amplitude_mu_penalty),
            ("Amplitude η Penalty", self.amplitude_eta_penalty),
            ("Amplitude ν Penalty", self.amplitude_nu_penalty),
            ("Amplitude μ Bounds", self.amplitude_mu_bounds),
            ("Amplitude η Bounds", self.amplitude_eta_bounds),
            ("Amplitude ν Bounds", self.amplitude_nu_bounds),
            ("Total Length", len(self)),
        ]

        for key, value in attributes:
            value_str = str(value)
            table_rows.append(
                f"<tr><th style='text-align:left; padding:5px;'>{key}</th><td style='padding:5px;'>{value_str}</td></tr>"
            )

        return f"""
        <table border="1" cellpadding="4" cellspacing="0" style="border-collapse: collapse; border: 1px solid black;">
            <thead style="background-color: #f2f2f2;">
                <tr>
                    <th style="text-align:left; padding:5px;">Attribute</th>
                    <th style="text-align:left; padding:5px;">Value</th>
                </tr>
            </thead>
            <tbody>
                {''.join(table_rows)}
            </tbody>
        </table>
        """

    def plot(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        initial = from_tau(self.setup_initial())
        ax.plot(initial, label="Initial in $\\mu$ space")
        return ax


def setup_contaminants(
    contaminants: list[Contaminant1D], initial: jnp.ndarray, mask: jnp.ndarray
) -> tuple[list[jnp.ndarray], list[jnp.ndarray]]:
    x = initial
    mask = mask
    for contaminant in contaminants:
        xi_initial = contaminant.setup_initial(initial)
        x = jnp.concatenate([x, xi_initial])
        xi_mask = jnp.zeros_like(mask)
        mask = jnp.concatenate([mask, xi_mask])
    return x, mask


def closure_unpack(
    contaminants: list[Contaminant1D], raw: jnp.ndarray | int
) -> Callable[[jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]]:

    if isinstance(raw, int):
        N = raw
    else:
        N = len(raw)
    M = len(contaminants)

    @jax.jit
    def func(x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        # returns a tuple of (mu vector, contaminant vectors)
        mu = x[:N]
        contaminants = x[N:].reshape(M, N)
        return mu, contaminants

    return func


@dataclass(kw_only=True)
class OptimComponents:
    raw: jnp.ndarray
    initial: jnp.ndarray
    mask: jnp.ndarray
    background: jnp.ndarray | None = None
    _run_checks: bool = True

    def __post_init__(self):
        self.raw = jnp.asarray(self.raw)
        N = len(self.raw)
        if len(self.initial) != N:
            raise ValueError(
                f"Initial must be of length of data, got {len(self.initial)}"
            )
        self.initial = jnp.asarray(self.initial)

        if len(self.mask) != N:
            raise ValueError(f"Mask must be of length of data, got {len(self.mask)}")
        # Here we flip the mask because jax.set uses the opposite convention
        if self._run_checks:
            self.mask = ~jnp.asarray(self.mask)
        else:
            self.mask = self.mask

        if self.background is not None:
            if len(self.background) != N:
                raise ValueError(
                    f"Background must be of length of data, got {len(self.background)}"
                )
            self.background = jnp.asarray(self.background)

    def __len__(self):
        return len(self.raw)


@dataclass(kw_only=True)
class OptimComponentsList:
    components: list[OptimComponents]
    same_mask: bool = False
    same_background: bool = False

    @classmethod
    def from_data(
        cls,
        data: list[Vector],
        initial: list[Vector],
        mask: list[np.ndarray],
        background: list[Vector] | None = None,
    ):
        N = len(data)
        # The mask is proably the same for all vectors
        mask0 = mask[0]
        all_same = all(all(m == mask0) for m in mask)
        if all_same:
            mask = ~jnp.asarray(mask0)
        else:
            mask = [~jnp.asarray(m) for m in mask]
            if len(mask) != N:
                raise ValueError("Mask must be of the same length as data")
        if len(initial) != N:
            raise ValueError("Initial must be of the same length as data")

        same_bg = False
        if background is not None:
            if len(background) == 1:
                if len(background) != N:
                    raise ValueError("Background must be of the same length as data")
                same_bg = True
            else:
                if len(background[0]) != N:
                    raise ValueError("Background must be of the same length as data")
                background = [jnp.asarray(bg) for bg in background]
        components = []
        for i in range(N):
            bg = background
            if background is not None and not same_bg:
                bg = background[i]
            component = OptimComponents(
                raw=jnp.asarray(data[i]),
                initial=jnp.asarray(initial[i]),
                mask=mask if all_same else mask[i],
                background=bg,
                # We elide the checks because we already checked the length,
                # and so that the shared arrays keep their memory
                _run_checks=False,
            )
            components.append(component)

        return cls(components=components, same_mask=all_same, same_background=same_bg)

    @property
    def masks(self):
        return [c.mask for c in self.components]

    @property
    def mask(self):
        if not self.same_mask:
            raise ValueError("Masks are not the same")
        return self.components[0].mask

    @property
    def backgrounds(self):
        if not self.has_background:
            raise ValueError("Background is not set")
        return [c.background for c in self.components]

    @property
    def initial(self):
        return [c.initial for c in self.components]

    @property
    def raw(self):
        return [c.raw for c in self.components]

    @property
    def has_background(self):
        return self.components[0].background is not None

    def __len__(self):
        return len(self.components)


@dataclass(kw_only=True)
class OptimParams:
    # Regularisation parameters
    alpha: float = 0.0
    # Optimisation parameters
    iterations: int = 100
    lr: float = 0.001
    # - Adam parameters
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    # Hyper-hyper parameters
    leave_tqdm: bool = True
    disable_tqdm: bool = False

    def __post_init__(self):
        self.iterations = int(self.iterations)

    @classmethod
    def from_kwargs(cls, R_cb, kwargs):
        if "lr" in kwargs and kwargs["lr"] == "auto":
            kwargs["lr"] = richardson_rate(R_cb())
        # We pop all keys from the dict, ensuring they do not remain in kwargs
        keys = {
            "iterations",
            "lr",
            "beta1",
            "beta2",
            "eps",
            "alpha",
            "leave_tqdm",
            "disable_tqdm",
        }
        values = {k: kwargs.pop(k) for k in keys if k in kwargs}
        return cls(**values)


@dataclass(kw_only=True)
class DataParams:
    D: jnp.ndarray
    G_eg: jnp.ndarray
    G_ex: jnp.ndarray | None = None
    prototype: Vector
    E: jnp.ndarray = None
    contaminants: list[Contaminant1D] = field(default_factory=list)
    penalty_mask: jnp.ndarray | None = None
    D1_mask: jnp.ndarray | None = None
    D2_mask: jnp.ndarray | None = None

    def __post_init__(self):
        self.D = jnp.asarray(self.D)
        self.G_eg = jnp.asarray(self.G_eg)
        self.G_ex = jnp.asarray(self.G_ex)
        self.E = jnp.asarray(self.prototype.X)

        N = len(self.prototype)

        if self.contaminants is None:
            self.contaminants = []
        for i, contaminant in enumerate(self.contaminants):
            if len(contaminant) != N:
                raise ValueError(
                    f"Contaminant must be of length of data, got {len(contaminant)} for number {i}."
                )

        # Handle the penalty mask
        # If D1 or D2 are not specified, they are set to the penalty mask
        # If no mask is given, D1 and D2 are set to 1 since it makes JAX elide them.
        if self.penalty_mask is not None:
            self.penalty_mask = jnp.asarray(self.penalty_mask)
            if len(self.penalty_mask) != N:
                raise ValueError(
                    f"penalty_mask must be of length of data, got {len(self.penalty_mask)}"
                )
            if self.D1_mask is None:
                self.D1_mask = self.penalty_mask[1:]
            else:
                self.D1_mask = jnp.asarray(self.D1_mask)
            if self.D2_mask is None:
                self.D2_mask = self.penalty_mask[2:]
            else:
                self.D2_mask = jnp.asarray(self.D2_mask)
            if len(self.D1_mask) != len(self.penalty_mask) - 1:
                raise ValueError(
                    f"D1_mask must be of length of data - 1, got {len(self.D1_mask)}"
                )
            if len(self.D2_mask) != len(self.penalty_mask) - 2:
                raise ValueError(
                    f"D2_mask must be of length of data - 2, got {len(self.D2_mask)}"
                )
        else:
            if self.D1_mask is None:
                self.D1_mask = 1
            if self.D2_mask is None:
                self.D2_mask = 1


def richardson_rate(R: np.ndarray) -> float:
    # get the largest and smallest singular values
    s = np.linalg.svd(R, compute_uv=False)
    s_max = s.max()
    s_min = s.min()
    return 2 / (s_max + s_min)


class RMLE(Unfolder):
    @staticmethod
    @override
    def supports_background():
        return True

    @override
    def _unfold_vector(
        self,
        data: Vector,
        background: Vector | None,
        initial: Vector,
        D: Matrix,
        G_eg: Matrix,
        mask: np.ndarray,
        contaminants: list[Contaminant1D] | None = None,
        profile: bool = False,
        **kwargs,
    ) -> RMLEResult1D:
        """
        This mostly just packs arguments into structs and then passes them to the optimizer,
        then unpacks and packs the results into a RMLEResult1D.
        """

        value_and_grad = jax.jit(
            jax.value_and_grad(cost_1d, has_aux=True),
            static_argnames=("alpha", "contaminants", "unpacker"),
        )
        # These are simple structs that contain the data and the parameters
        # Turns out we got a lot to keep track of
        components = OptimComponents(
            raw=data, initial=initial, mask=mask, background=background
        )
        optim_params = OptimParams.from_kwargs(lambda: G_eg @ D, kwargs)
        data_params = DataParams(
            D=D,
            G_eg=G_eg,
            prototype=data,
            contaminants=contaminants,
            penalty_mask=kwargs.pop("penalty_mask", None),
            D1_mask=kwargs.pop("D1_mask", None),
            D2_mask=kwargs.pop("D2_mask", None),
        )

        start = time.time()
        if profile:
            print("Profiling...")
            with jax.profiler.trace(
                "/tmp/jax-trace-unfold-vec", create_perfetto_link=True
            ):
                result = unfold_adam_1d(
                    components,
                    value_and_grad=value_and_grad,
                    optim_params=optim_params,
                    data_params=data_params,
                    **kwargs,
                )
            print(f"Profiling took {time.time() - start} seconds")
        else:
            result = unfold_adam_1d(
                components,
                value_and_grad=value_and_grad,
                optim_params=optim_params,
                data_params=data_params,
                **kwargs,
            )

        elapsed = time.time() - start
        kwargs = (
            asdict(optim_params)
            | {"contaminants": contaminants}
            | {"penalty_mask": data_params.penalty_mask}
        )
        parameters = Parameters1D(
            D=D,
            G_eg=G_eg,
            raw=data,
            background=background,
            initial=initial,
            kwargs=kwargs,
            mask=np.asarray(mask),
        )  # Kwargs got popped by OptimParams.from_kwargs()
        meta = ResultMeta1D(
            time=elapsed, space=self.space, parameters=parameters, method=self.__class__
        )
        return RMLEResult1D(
            meta=meta,
            cost=result.total_cost,
            u=result.mu,
            beta=result.beta,
            aux=result.aux,
            xi=result.xi,
        )

    @override
    def _unfold_vectors(
        self,
        data: list[Vector],
        background: list[Vector] | None,
        initial: list[Vector],
        D: Matrix,
        G_eg: Matrix,
        mask: list[np.ndarray],
        profile: bool = False,
        contaminants: list[Contaminant1D] | None = None,
        **kwargs,
    ) -> list[RMLEResult1D]:
        value_and_grad = jax.jit(
            jax.value_and_grad(cost_1d, has_aux=True),
            static_argnames=("alpha", "contaminants", "unpacker"),
        )
        components = OptimComponentsList.from_data(data, initial, mask, background)
        optim_params = OptimParams.from_kwargs(lambda: G_eg @ D, kwargs)
        data_params = DataParams(
            D=D,
            G_eg=G_eg,
            prototype=data[0],
            contaminants=contaminants,
            penalty_mask=kwargs.pop("penalty_mask", None),
            D1_mask=kwargs.pop("D1_mask", None),
            D2_mask=kwargs.pop("D2_mask", None),
        )

        start = time.time()
        optim_results = unfold_adam_1d_list(
            components=components,
            value_and_grad=value_and_grad,
            optim_params=optim_params,
            data_params=data_params,
            **kwargs,
        )

        elapsed = time.time() - start

        results: list[RMLEResult1D] = []
        for i, result in enumerate(optim_results):
            parameters = Parameters1D(
                D=D,
                G_eg=G_eg,
                raw=data[i],
                background=background[i] if background is not None else None,
                initial=initial[i],
                kwargs=asdict(optim_params) | {"contaminants": contaminants},
                mask=np.asarray(mask),
            )
            meta = ResultMeta1D(
                time=elapsed,
                space=self.space,
                parameters=parameters,
                method=self.__class__,
            )
            results.append(
                RMLEResult1D(
                    meta=meta,
                    cost=result.total_cost,
                    u=result.mu,
                    beta=result.beta,
                    aux=result.aux,
                    xi=result.xi,
                )
            )

        return results

    def _unfold_matrix(
        self,
        data: Matrix,
        background: Matrix | None,
        initial: Matrix,
        D: Matrix,
        G_eg: Matrix,
        G_ex: Matrix | None,
        mask: np.ndarray,
        **kwargs,
    ) -> UnfoldedResult2DSimple:

        mask = jnp.asarray(mask)

        u = to_tau(initial.values)
        D_ = jnp.asarray(D)
        G_ex_ = jnp.asarray(G_ex)
        G_eg_ = jnp.asarray(G_eg)
        n = jnp.asarray(data.values)
        if background is None:
            bg = None
        else:
            bg = jnp.asarray(background.values)
        loss = jax.jit(cost, static_argnames=("alpha"))
        grad = jax.grad(cost)
        grad = jax.jit(grad, static_argnames=("alpha"))
        method = kwargs.pop("method", "adam")
        match method:
            case "adam":
                unfold = unfold_adam
            case "optax":
                unfold = unfold_optax
            case _:
                raise ValueError(f"Unknown method {method}")
        value_and_grad = jax.jit(
            jax.value_and_grad(cost, has_aux=True), static_argnames=("alpha")
        )

        if "lr" not in kwargs or kwargs["lr"] == "auto":
            kwargs["lr"] = self.richardson_rate()
        start = time.time()
        u, total_cost, aux = unfold(
            u,
            raw=n,
            bg=bg,
            R=R_,
            G_ex=G_ex_,
            G_eg=G_eg_,
            loss=loss,
            grad=grad,
            value_and_grad=value_and_grad,
            mask=mask,
            **kwargs,
        )
        elapsed = time.time() - start
        # If we have a background, we need to unpack u
        if bg is not None:
            mu, beta = jnp.vsplit(u, 2)
            beta = background.clone(values=np.asarray(beta))
        else:
            mu = u
            beta = None
        mu = data.clone(values=np.asarray(mu))

        # TODO Add Response coefficients as optimisation parameter
        # TODO Loop over Ex and make error
        # print("Approximating variance")
        # hessian = jax.jit(jax.jacfwd(jax.jacrev(cost)))
        # hessian = hessian(u[160], R_, n[160])
        parameters = Parameters2D(
            D=D,
            raw=data,
            background=background,
            initial=initial,
            G_eg=G_eg,
            G_ex=G_ex,
            kwargs=kwargs | {"method": method},
            mask=mask,
        )
        meta = ResultMeta2D(
            time=elapsed, space=self.space, parameters=parameters, method=self.__class__
        )
        return RMLEResult2D(meta=meta, cost=total_cost, u=mu, beta=beta, aux=aux)

    def grid_search(
        self, eta: Matrix, *args, unfkwargs: dict[str, Any] | None = None, **kwargs
    ) -> GridSearchResult:
        if unfkwargs is None:
            raise ValueError("unfkwarg must be provided")
        if len(args) == 0:
            raise ValueError("At least one hyperparameter must be provided")
        if len(args) > 2:
            raise ValueError("Up to two hyperparameters supported")
        if len(args) == 1:
            param, values = args[0]
            return self.grid_search_1D(eta, param, values, unfkwargs)
        if len(args) == 2:
            raise NotImplementedError

    def grid_search_1D(
        self, eta: Matrix, param: str, values: np.ndarray, unfkwargs: dict[str, Any]
    ) -> GridSearchResult1D:
        kw = unfkwargs.copy()
        results: list[RMLEResult1D] = []
        if "leave_tqdm" not in kw:
            kw["leave_tqdm"] = False
        bar = tqdm(enumerate(values), total=len(values))
        for i, value in bar:
            bar.set_postfix({param: value})
            kw[param] = value
            res = self.unfold(**kw)
            results.append(res)
        return GridSearchResult1D(hyperparameter=param, grid=values, results=results)

    def grid_search_2D(
        self,
        eta: Matrix,
        param1: str,
        values1: np.ndarray,
        param2: str,
        values2: np.ndarray,
        mask: np.ndarray,
        unfkwargs: dict[str, Any],
    ) -> GridSearchResult2D:
        kw = unfkwargs.copy()
        results: list[RMLEResult2D] = []
        values = list(product(values1, values2))
        bar = tqdm(enumerate(values), total=len(values))
        for i, (value1, value2) in bar:
            bar.set_postfix()
            kw[param1] = value1
            kw[param2] = value2
            res = self.unfold(**kw)
            results.append(res)
        return GridSearchResult2D(
            param1=param1, grid1=values1, param2=param2, grid2=values2, results=results
        )

    def tune_learning_rate(
        self, lr: np.ndarray | None = None, unfkwargs: dict[str, Any] | None = None
    ) -> tuple[GridSearchResult1D, float]:
        # Perform a grid search over the learning rates
        if lr is None:
            lr = np.logspace(-3, 1, 10)
        if unfkwargs is None:
            unfkwargs = {}
        if "iterations" not in unfkwargs:
            unfkwargs["iterations"] = 1000
        result = self.grid_search_1D(None, "lr", lr, unfkwargs)

        # Find the learning rate that gave the lowest cost
        min_cost = np.inf
        for i, res in enumerate(result.results):
            if res.cost[-1] < min_cost:
                min_cost = res.cost[-1]
                best = i

        return result, result.grid[best]


@dataclass(kw_only=True)
class GridSearchResult:
    pass


@dataclass(kw_only=True)
class GridSearchResult1D(GridSearchResult):
    hyperparameter: str
    grid: np.ndarray
    results: list[RMLEResult1D]

    def plot(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()

        costs = [res.cost[-1] for res in self.results]
        line = ax.plot(self.grid, costs, "-o")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(self.hyperparameter)
        ax.set_ylabel("Cost")
        ax.set_title("Grid Search Results")
        ax.grid(True)

        return ax, line


@dataclass(kw_only=True)
class GridSearchResult2D(GridSearchResult):
    param1: str
    grid1: np.ndarray
    param2: str
    grid2: np.ndarray
    results: list[RMLEResult2D]


Schedule: TypeAlias = Callable[[int], float]


class Scheduler(ABC):
    @abstractmethod
    def make(self) -> Schedule: ...

    def plot(self, x: np.ndarray | None = None, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        if x is None:
            x = np.arange(0, 1_000_000, 10_000)
        fn = np.vectorize(self.make())
        ax.plot(x, fn(x))
        return ax


class ConstScheduler(Scheduler):
    def __init__(self, lr):
        self.lr = lr

    def make(self):
        lr = self.lr
        return lambda t: lr


class ExponentialDecayScheduler(Scheduler):
    def __init__(self, initial_lr, decay_rate):
        self.initial_lr = initial_lr
        self.decay_rate = decay_rate

    def make(self):
        # Return a function that computes the learning rate given the epoch
        lr0 = self.initial_lr
        theta = self.decay_rate

        def lr_schedule(epoch):
            return lr0 * (theta**epoch)

        return lr_schedule

    @classmethod
    def from_point(cls, initial_lr, epoch, lr):
        # Calculate the necessary decay_rate
        decay_rate = (lr / initial_lr) ** (1 / epoch)
        return cls(initial_lr, decay_rate)


class StepScheduler(Scheduler):
    def __init__(self, initial_lr, drop_factor, drop_every):
        self.initial_lr = initial_lr
        self.drop_factor = drop_factor
        self.drop_every = drop_every

    def make(self):
        def lr_schedule(epoch):
            steps = epoch // self.drop_every
            return self.initial_lr - (self.drop_factor * steps)

        return lr_schedule


class StepsScheduler(Scheduler):
    def __init__(self, points):
        # Ensure points are sorted by epoch
        self.points = sorted(points)

    def make(self):
        def lr_schedule(epoch):
            for i in range(len(self.points) - 1):
                if epoch < self.points[i + 1][0]:
                    return self.points[i][1]
            return self.points[-1][1]

        return lr_schedule


class CosineAnnealingScheduler(Scheduler):
    def __init__(self, initial_lr, min_lr, T_max):
        """
        :param initial_lr: The initial learning rate.
        :param min_lr: The minimum learning rate.
        :param T_max: The maximum number of iterations (epochs) for the schedule.
        """
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.T_max = T_max

    def make(self):
        def lr_schedule(epoch):
            # Cosine Annealing formula
            return (
                self.min_lr
                + (self.initial_lr - self.min_lr)
                * (1 + np.cos(np.pi * epoch / self.T_max))
                / 2
            )

        return lr_schedule


def unfold_adam(
    u: Array,
    *,
    raw: Array,
    bg: Array,
    R: Array,
    G_eg: Array,
    G_ex: Array,
    loss,
    grad,
    value_and_grad,
    mask,
    iterations=10,
    lr=0.001,
    beta1=0.9,
    beta2=0.999,
    abs_tol=1e-3,
    rel_tol=1e-3,
    use_abs_tol: bool = False,
    use_rel_tol: bool = False,
    lr_scheduler: Scheduler | None = None,
    **kwargs,
):
    iterations = int(iterations)
    # Extract additional parameters
    alpha = kwargs.get("alpha", 0.0)

    # Invert mask
    mask = ~mask

    # Combine prompt and background if bg is provided
    u = to_tau(u)
    if bg is not None:
        mask = jnp.concatenate([mask, jnp.zeros_like(bg, dtype=bool)])
        u = jnp.concatenate([u, 1.0 + jnp.zeros_like(bg)])

    eps = 1e-8

    # Initialize Adam parameters
    mean = jnp.zeros_like(u)
    var = jnp.zeros_like(u)

    # Initialize total_cost
    total_cost = jnp.zeros(iterations)

    loglike = jnp.zeros(iterations)
    penalty = jnp.zeros(iterations)

    penalty_mask = kwargs.pop("penalty_mask", None)
    if penalty_mask is not None:
        if penalty_mask.shape == u.shape:
            penalty_mask = penalty_mask.T
        elif penalty_mask.shape == (u.shape[1], u.shape[0]):
            pass
        else:
            raise ValueError(
                f"Invalid penalty mask shape: {penalty_mask.shape}. "
                "Should be of shape {u.shape} or {(u.shape[1], u.shape[0])}"
            )

        # We precut the mask
        D1_mask = jnp.asarray(penalty_mask[1:].T).to_device(u.device)
        D2_mask = jnp.asarray(penalty_mask[2:].T).to_device(u.device)
    else:
        D1_mask = 1
        D2_mask = 1

    def scan_body(carry, i):
        """
        Single optimization step for Adam.
        """
        u, mean, var, total_cost, loglike, penalty = carry

        # Compute loss and gradient
        (tloss, aux), g = value_and_grad(
            u,
            R=R,
            G_ex=G_ex,
            G_eg=G_eg,
            y=raw,
            bg=bg,
            alpha=alpha,
            D1_mask=D1_mask,
            D2_mask=D2_mask,
        )

        # Update Adam moments
        mean = beta1 * mean + (1 - beta1) * g
        var = beta2 * var + (1 - beta2) * (g**2)

        # Correct bias
        mean_cor = mean / (1 - beta1 ** (i + 1))
        var_cor = var / (1 - beta2 ** (i + 1))

        # Compute update
        v = lr * mean_cor / (jnp.sqrt(var_cor) + eps)
        u = u - v

        # Apply mask
        u = u.at[mask].set(0)

        # Update total_cost
        total_cost = total_cost.at[i].set(tloss)

        loglike = loglike.at[i].set(aux["loglike"])
        penalty = penalty.at[i].set(aux["penalty"])

        return (u, mean, var, total_cost, loglike, penalty), tloss

    # Perform the optimization loop using jax.lax.scan
    (
        u_final,
        mean_final,
        var_final,
        total_cost_final,
        loglike_final,
        penalty_final,
    ), _ = jax.lax.scan(
        scan_body, (u, mean, var, total_cost, loglike, penalty), jnp.arange(iterations)
    )

    # Convert back from tau
    optimized_u = from_tau(u_final)
    return (
        optimized_u,
        total_cost_final,
        {"loglike": loglike_final, "penalty": penalty_final},
    )


def rolling_standard_deviation(cost_array, window_size):
    if len(cost_array) < window_size:
        return np.std(cost_array)  # If not enough data, use the entire array
    return np.std(cost_array[-window_size:])


def rolling_coefficient_of_variation(cost_array, window_size):
    if len(cost_array) < window_size:
        mean = np.mean(cost_array)
        std_dev = np.std(cost_array)
    else:
        recent_values = cost_array[-window_size:]
        mean = np.mean(recent_values)
        std_dev = np.std(recent_values)

    return std_dev / mean if mean != 0 else float("inf")


def exponential_moving_average(cost_array, alpha=0.1):
    if len(cost_array) == 0:
        return np.inf
    ema = [cost_array[0]]  # Start with the first cost
    for cost in cost_array[1:]:
        ema.append(alpha * cost + (1 - alpha) * ema[-1])
    return ema[-1]


def unfold_optax(*args, **kwargs):
    raise ImportError("Optax is not available on your system")


def requires_lr(f) -> bool:
    try:
        return "learning_rate" in f.__code__.co_varnames
    except AttributeError:
        if f == optax.nadam:
            return True
    return False  # Je ne sais pas, let the error be thrown


def requires_max_learning_rate(f) -> bool:
    try:
        return "max_learning_rate" in f.__code__.co_varnames
    except AttributeError:
        if f == optax.nadam:
            return False
    return False  # Je ne sais pas, let the error be thrown


def requires_values(f) -> bool:
    match f:
        case optax.polyak_sgd:
            return True
        case _:
            return False


if OPTAX_AVAILABLE:

    def unfold_optax(u, raw, bg, R, G_ex, loss, grad, value_and_grad, mask, **kwargs):

        # Initialize the Adam optimizer
        rename_key(kwargs, "lr", "learning_rate")
        num_iters = int(kwargs.pop("iterations", 1000))
        bar = tqdm(
            range(num_iters),
            disable=kwargs.pop("disable_tqdm", False),
            leave=kwargs.pop("leave_tqdm", True),
        )
        break_at_nan = kwargs.pop("break_at_nan", True)

        method = kwargs.pop("optimizer", optax.adam)
        alpha = kwargs.pop("alpha", 0.0)
        beta = kwargs.pop("beta", 0.0)
        optim_kwargs = kwargs.pop("optimizer_kwargs", {})
        rename_key(optim_kwargs, "lr", "learning_rate")
        if "learning_rate" in optim_kwargs and "learning_rate" in kwargs:
            raise ValueError(
                "Only provide 'learning_rate' in 'optimizer_kwargs' or 'kwargs', not both"
            )
        if (
            "learning_rate" not in optim_kwargs
            and "learning_rate" not in kwargs
            and requires_lr(method)
        ):
            optim_kwargs["learning_rate"] = 0.001
        elif "learning_rate" in kwargs:
            optim_kwargs["learning_rate"] = kwargs.pop("learning_rate")

        if requires_max_learning_rate(method):
            if "learning_rate" in optim_kwargs:
                rename_key(optim_kwargs, "learning_rate", "max_learning_rate")
            # Let optax throw the error for missing keyword

        # All keyword arguments should be handled
        if len(kwargs) > 0:
            raise ValueError(f"Unknown keyword arguments: {kwargs.keys()}")

        optimizer = method(**optim_kwargs)

        # Initialize the optimizer state
        state = optimizer.init(u)

        # Perform the optimization
        total_cost = np.zeros(num_iters)
        for i in bar:
            # Compute the gradient
            value, gradients = value_and_grad(
                u, R, G_ex, raw, bg, None, None, alpha=alpha
            )

            # Update the parameters and the optimizer state
            updates, state = optimizer.update(gradients, state, u, value=value)
            u = optax.apply_updates(u, updates)
            total_cost[i] = value  ##loss(u, R, G_ex, raw, bg, None, None, alpha=alpha)
            if break_at_nan and not np.isfinite(value):
                i = i + 1
                break

            std = rolling_standard_deviation(total_cost[:i], 10)
            cv = rolling_coefficient_of_variation(total_cost[:i], 10)
            ema = exponential_moving_average(total_cost[:i], 0.1)
            bar.set_postfix(
                {"cost": total_cost[i], "std": std, "cv": cv, "ema": ema}, refresh=True
            )

        return u**2, total_cost[:i]


def rename_key(kw, old_key, new_key, default_value=None):
    if old_key in kw and new_key in kw:
        raise ValueError(f"Only provide '{old_key}' or '{new_key}', not both")
    if old_key in kw:
        kw[new_key] = kw.pop(old_key)
    elif new_key not in kw and default_value is not None:
        kw[new_key] = default_value


@dataclass
class AdamParams:
    lr: float | Iterable[float] = 0.001  # learning rate
    beta1: float | Iterable[float] = 0.9  # decay rate for first moment estimate
    beta2: float | Iterable[float] = 0.999  # decay rate for second moment estimate
    iterations: int | Iterable[int] = 10  # maximum number of iterations

    def get_iterables(self) -> list[str]:
        iterables = []
        for field in fields(self):
            if is_iterable(getattr(self, field.name)):
                iterables.append(field.name)
        return list(iterables)


def is_iterable(x) -> bool:
    try:
        iter(x)
        return True
    except TypeError:
        return False


def unfold_adam_1d(
    components: OptimComponents,
    value_and_grad,
    optim_params: OptimParams,
    data_params: DataParams,
    **kwargs,
) -> OptimResult1D:
    # Combine the prompt and the background
    tau = to_tau(components.initial)
    # if bg is not None:
    #    mask = jnp.concatenate([mask, jnp.zeros_like(bg, dtype=bool)])
    #    u = jnp.concatenate([u, 1.0 + jnp.zeros_like(bg)])

    # Set up Xi for contamination
    x, mask = setup_contaminants(data_params.contaminants, tau, components.mask)

    # We have used all kwargs as we can. The rest are probably misspelled
    if len(kwargs) > 0:
        raise ValueError(f"Unknown keyword arguments: {kwargs.keys()}")

    # The optimization function is created from a closure of all constants
    # that we never vmap over.
    lower = make_lower(
        value_and_grad,
        optim_params,
        data_params,
    )

    # Mask must be a concrete type for the jax.jit to work
    mask = jnp.where(mask)

    x, total_cost, loglike, penalty, xi_penalty = lower(
        x, components.raw, components.background, mask
    )
    aleph = from_tau(x)
    mu, contaminants = closure_unpack(data_params.contaminants, data_params.E)(aleph)

    result = OptimResult1D(
        prototype=data_params.prototype,
        mu=mu,
        total_cost=total_cost,
        loglike=loglike,
        penalty=penalty,
        xi_penalty=xi_penalty,
        xi=contaminants,
    )
    return result


def unfold_adam_1d_list(
    components: OptimComponentsList,
    value_and_grad,
    optim_params: OptimParams,
    data_params: DataParams,
    **kwargs,
) -> list[OptimResult1D]:

    if components.same_mask:
        masks = [components.mask]
    else:
        masks = components.masks

    if components.has_background:
        if components.same_background:
            bg = components.background
        else:
            bg = jnp.stack(components.backgrounds)
    else:
        bg = None

    raw = jnp.stack(components.raw)
    # Combine the prompt and the background
    tau = [to_tau(x) for x in components.initial]

    x: list[jnp.ndarray] = []
    combined_masks: list[jnp.ndarray] = []
    if data_params.contaminants:
        for i in range(len(tau)):
            mask_i = masks[0] if components.same_mask else masks[i]
            # We extend the tau and masks to account for contaminant arrays
            x_i, mask_i = setup_contaminants(data_params.contaminants, tau[i], mask_i)
            x.append(x_i)
            # We only need to extend the mask if we are not using the same mask
            if not components.same_mask or i < 1:
                combined_masks.append(mask_i)
    else:
        combined_masks = masks
        x = tau

    if components.same_mask:
        mask = jnp.where(combined_masks[0])
    else:
        raise NotImplementedError(
            "Different masks are not supported for list of components"
        )
        mask = [
            jnp.concatenate([m, jnp.zeros_like(t)]) for m, t in zip(combined_masks, tau)
        ]
        mask = jnp.stack(mask)
        mask = jnp.where(mask)
    x = jnp.stack(x)

    # We have used all kwargs as we can. The rest are probably misspelled
    if len(kwargs) > 0:
        raise ValueError(f"Unknown keyword arguments: {kwargs.keys()}")

    lower = make_lower(
        value_and_grad,
        optim_params,
        data_params,
    )

    in_axes = (
        0,  # The initial values
        0,  # The raw data
        None if components.same_background or bg is None else 0,  # The background
        None if components.same_mask else 0,  # The mask
    )
    lower_vmap = jax.vmap(lower, in_axes=in_axes)
    x, total_cost, loglike, penalty, xi_penalty = lower_vmap(x, raw, bg, mask)

    unpacker = closure_unpack(data_params.contaminants, data_params.E)
    results: list[OptimResult1D] = []
    for i in range(len(x)):
        aleph = from_tau(x[i])
        mu, contaminants = unpacker(aleph)
        result = OptimResult1D(
            prototype=data_params.prototype,
            mu=mu,
            total_cost=total_cost[i],
            loglike=loglike[i],
            penalty=penalty[i],
            xi_penalty=xi_penalty[i],
            xi=contaminants,
        )
        results.append(result)
    return results


def make_lower(
    value_and_grad,
    optim_params: OptimParams,
    data_params: DataParams,
):
    # The outer scope captures constants
    # The inner scope captures variables that can be vmaped over
    iterations = optim_params.iterations
    lr = optim_params.lr
    beta1 = optim_params.beta1
    beta2 = optim_params.beta2
    eps = optim_params.eps
    alpha = optim_params.alpha
    G_eg = data_params.G_eg
    GegD = data_params.D @ G_eg
    unpacker = closure_unpack(data_params.contaminants, data_params.E)
    contaminant_closures = tuple(c.closure() for c in data_params.contaminants)
    leave_tqdm = optim_params.leave_tqdm
    D1_mask = data_params.D1_mask
    D2_mask = data_params.D2_mask

    @jax.jit
    def lower(initial, y, bg, mask):
        mean = jnp.zeros_like(initial)
        var = jnp.zeros_like(initial)

        loglike = jnp.zeros(iterations)
        penalty = jnp.zeros(iterations)
        xi_penalty = jnp.zeros(iterations)

        @loop_tqdm(iterations, leave=leave_tqdm)
        @jax.jit
        def body_fun(i, state):
            x, mean, var, loglike, penalty, xi_penalty = state
            (tloss, aux), g = value_and_grad(
                x,
                GegD=GegD,
                y=y,
                bg=bg,
                G_eg=G_eg,
                unpacker=unpacker,
                contaminants=contaminant_closures,
                alpha=alpha,
                D1_mask=D1_mask,
                D2_mask=D2_mask,
            )
            mean = beta1 * mean + (1 - beta1) * g
            var = beta2 * var + (1 - beta2) * jnp.square(g)
            mean_cor = mean / (1 - beta1 ** (i + 1))  # i + 1 because i starts from 0
            var_cor = var / (1 - beta2 ** (i + 1))
            v = lr * mean_cor / (jnp.sqrt(var_cor) + eps)
            x = x - v
            x = x.at[mask].set(0)
            loglike = loglike.at[i].set(aux["loglike"])
            penalty = penalty.at[i].set(aux["penalty"])
            xi_penalty = xi_penalty.at[i].set(aux["xi_penalty"])
            return x, mean, var, loglike, penalty, xi_penalty

        state = (initial, mean, var, loglike, penalty, xi_penalty)
        state = jax.lax.fori_loop(0, iterations, body_fun, state)
        x, mean, var, loglike, penalty, xi_penalty = state
        total_cost = loglike + penalty
        return x, total_cost, loglike, penalty, xi_penalty

    return lower


def kl_2(nu: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """Compute the KL divergence."""
    # Avoid division by zero and log of zero
    safe_nu = jnp.where(nu == 0, 1e-10, nu)
    safe_y = jnp.where(y == 0, 1e-10, y)
    return safe_nu * jnp.log(safe_nu / safe_y) - safe_nu + safe_y


def loglikelihood(
    mu: jnp.ndarray, R: jnp.ndarray, G_in: jnp.ndarray, y: jnp.ndarray
) -> jnp.ndarray:
    """Compute the log-likelihood."""
    nu = G_in @ (mu**2) @ R
    return jnp.sum(kl_2(nu, y))


def hessian_vector_product(
    mu: jnp.ndarray, v: jnp.ndarray, R: jnp.ndarray, G_in: jnp.ndarray, y: jnp.ndarray
) -> jnp.ndarray:
    """Compute the Hessian-vector product H * v."""
    # First derivative (gradient)
    grad = jax.grad(loglikelihood, argnums=0)
    # Corrected line: Pass the function `grad` instead of calling it
    hvp = jax.jvp(grad, (mu, R, G_in, y), (v,))[1]
    return hvp


def estimate_largest_eigenvalue(
    mu: jnp.ndarray,
    R: jnp.ndarray,
    G_in: jnp.ndarray,
    y: jnp.ndarray,
    num_iters: int = 100,
    tol: float = 1e-6,
    key: jax.random.PRNGKey = jax.random.PRNGKey(0),
) -> float:
    """Estimate the largest eigenvalue of the Hessian using Power Iteration."""
    # Initialize a random vector v with unit norm
    v = jax.random.normal(key, shape=mu.shape)
    v = v / jnp.linalg.norm(v)

    for _ in range(num_iters):
        # Compute H * v
        Hv = hessian_vector_product(mu, v, R, G_in, y)

        # Compute the norm of Hv
        Hv_norm = jnp.linalg.norm(Hv)
        if Hv_norm == 0:
            break

        # Normalize Hv to get the next iteration's vector
        v_new = Hv / Hv_norm

        # Check for convergence (cosine similarity)
        cosine_sim = jnp.dot(v, v_new)
        if jnp.abs(cosine_sim - 1.0) < tol:
            break

        v = v_new

    # Estimate of the largest eigenvalue
    eigenvalue = jnp.dot(v, hessian_vector_product(mu, v, R, G_in, y))
    return eigenvalue


import jax
import jax.numpy as jnp
from functools import partial


def hessian_vector_product(f, x, v):
    """Compute Hessian-vector product without materializing the full Hessian."""

    def grad_dot_v(x):
        return jnp.vdot(jax.grad(f)(x), v)

    return jax.grad(grad_dot_v)(x)


def power_iteration(hvp_func, x_shape, num_iterations=5, tol=1e-6):
    """
    Compute largest eigenvalue using power iteration.

    Args:
        hvp_func: Function that computes Hessian-vector product
        x_shape: Shape of the input vector
        num_iterations: Maximum number of iterations
        tol: Convergence tolerance

    Returns:
        largest_eigenvalue: Estimated largest eigenvalue
    """
    # Initialize random vector and normalize it
    key = jax.random.PRNGKey(0)
    v = jax.random.normal(key, x_shape)
    v = v / jnp.linalg.norm(v)

    def body_fun(carry):
        i, v, prev_eigenvalue, _ = carry
        # Compute Hv
        Hv = hvp_func(v)
        # Calculate Rayleigh quotient (approximate eigenvalue)
        eigenvalue = jnp.vdot(v, Hv)
        # Normalize the new vector
        v_new = Hv / jnp.linalg.norm(Hv)
        # Check convergence
        converged = jnp.abs(eigenvalue - prev_eigenvalue) < tol
        return (i + 1, v_new, eigenvalue, converged)

    def cond_fun(carry):
        i, _, _, converged = carry
        return jnp.logical_and(i < num_iterations, jnp.logical_not(converged))

    # Initial state
    init_state = (0, v, jnp.inf, False)

    # Run power iteration
    final_state = jax.lax.while_loop(cond_fun, body_fun, init_state)
    _, _, eigenvalue, _ = final_state

    return eigenvalue


def compute_largest_eigenvalue(loglikelihood, mu, R, G_in, y):
    """
    Compute the largest eigenvalue of the Hessian of the log-likelihood function.
    """

    # Create partial function with fixed parameters except mu
    def f(mu):
        return loglikelihood(mu, R, G_in, y)

    # Create Hessian-vector product function
    def hvp(v):
        return hessian_vector_product(f, mu, v)

    # Run power iteration
    eigenvalue = power_iteration(hvp, mu.shape)
    return eigenvalue


# Example usage
@jax.jit
def find_largest_eigenvalue(mu, R, G_in, y):
    return compute_largest_eigenvalue(loglikelihood, mu, R, G_in, y)


def loglikelihood_hessian(mu: Array, R: Array, G_in: Array, y: Array) -> Array:
    """Compute the Hessian of the log-likelihood term (KL divergence)."""

    def loglikelihood(mu):
        nu = G_in @ (mu**2) @ R
        return jnp.sum(kl(nu, y))

    return jax.hessian(loglikelihood)(mu)


def regularization_hessian(mu: Array, alpha: float, C: float) -> Array:
    """Compute the Hessian of the regularization term (onecost)."""

    def regularization(mu):
        return alpha * onecost(mu**2, C) ** 2

    return jax.hessian(regularization)(mu)


def get_max_eigenvalue(H: Array) -> float:
    """Compute the largest eigenvalue of a Hessian matrix."""
    # Using power iteration method for efficiency
    v = jnp.ones_like(H[0])
    for _ in range(10):  # Usually converges quickly
        v_new = H @ v
        v = v_new / jnp.linalg.norm(v_new)

    return jnp.dot(v, H @ v)


def estimate_lipschitz_constant(samples: Array, R: Array, y: Array) -> float:
    """
    Estimate Lipschitz constant using gradient norm bounding over a specified range.

    Args:
        theta_range: Array of theta values to evaluate over
        R: Response matrix
        G_ex: Extended response matrix
        n: Data vector
        alpha: Regularization parameter
    Returns:
        L: Estimated Lipschitz constant
    """
    L_estimates = []

    # Define gradient function with inlined cost (no bg case)
    @jax.jit
    def grad_f(mu):
        nu = R @ mu
        return jnp.sum(kl(nu, y))

    grad_func = jax.grad(grad_f)

    @jax.jit
    def estimate_lipschitz_constant_pair(theta1, theta2):
        grad1 = grad_func(theta1)
        grad2 = grad_func(theta2)
        grad_diff_norm = jnp.linalg.norm(grad1 - grad2)
        param_diff_norm = jnp.linalg.norm(theta1 - theta2)
        return grad_diff_norm / param_diff_norm

    # Compare all pairs of points in the range
    for i in tqdm(range(len(samples)), leave=False):
        theta1 = samples[i]
        for j in range(i + 1, len(samples)):
            theta2 = samples[j]
            L_estimates.append(estimate_lipschitz_constant_pair(theta1, theta2))

    return np.max(np.asarray(L_estimates))


def estimate_lipschitz_constant_reg(
    samples: Array, alpha: float = 0.0, C: float = 1.0
) -> float:
    """
    Estimate Lipschitz constant using gradient norm bounding over a specified range.

    Args:
        theta_range: Array of theta values to evaluate over
        R: Response matrix
        G_ex: Extended response matrix
        n: Data vector
        alpha: Regularization parameter
    Returns:
        L: Estimated Lipschitz constant
    """
    L_estimates = []

    # Define gradient function with inlined cost (no bg case)
    @jax.jit
    def grad_f(mu):
        return alpha * onecost(mu, C) ** 2

    grad_func = jax.grad(grad_f)

    @jax.jit
    def estimate_lipschitz_constant_pair(theta1, theta2):
        grad1 = grad_func(theta1)
        grad2 = grad_func(theta2)
        grad_diff_norm = jnp.linalg.norm(grad1 - grad2)
        param_diff_norm = jnp.linalg.norm(theta1 - theta2)
        return grad_diff_norm / param_diff_norm

    # Compare all pairs of points in the range
    for i in tqdm(range(len(samples)), leave=False):
        theta1 = samples[i]
        for j in range(i + 1, len(samples)):
            theta2 = samples[j]
            L_estimates.append(estimate_lipschitz_constant_pair(theta1, theta2))

    return np.max(np.asarray(L_estimates))


def estimate_learning_rate_1d(
    R: Matrix,
    y: Vector,
    mu: Vector,
    sample_points: int | list[Array] = 100,
    alpha: float = 0.0,
    C: float = 1.0,
) -> float:
    """
    Estimate optimal learning rate using Lipschitz constant over a specified range.

    Args:
        R: Response matrix
        y: Data vector
        sample_points: Number of points to sample over
        alpha: Regularization parameter
    Returns:
        lr: Estimated optimal learning rate
    """
    if isinstance(sample_points, int):
        samples = [
            np.abs(mu.values + mu.values * np.random.normal(0, 1))
            + np.random.normal(0, 1000, size=(y.shape[0]))
            for _ in range(sample_points)
        ]
    else:
        samples = sample_points
    R = jnp.array(R)
    y = jnp.array(y)
    # Estimate Lipschitz constant with inlined cost function
    L0 = estimate_lipschitz_constant(samples, R, y)
    L1 = estimate_lipschitz_constant_reg(samples, alpha=alpha, C=C)

    print(f"Estimated Lipschitz constant for loglikelihood: {L0}")
    print(f"Estimated Lipschitz constant for regularization: {L1}")

    # Compute learning rate as 1/L
    lr_1 = 1.0 / (L0 + L1 + 1e-10)  # Add small constant for numerical stability
    lr_2 = 2.0 / (L0 + L1 + 1e-10)
    return lr_1, lr_2


def model_loglikelihood(res: Result) -> float:
    y = res.raw
    nu = res.best_folded()
    loglike = jnp.sum(kl(jnp.asarray(nu), jnp.asarray(y)))
    return loglike
