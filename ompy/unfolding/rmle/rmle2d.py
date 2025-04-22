from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from typing import Any

from .tau import from_tau, to_tau
from ... import Matrix
from ...stubs import Path
from ..result1d import Cost1D
from ..result2d import UnfoldedResult2DSimple
from .loss import kl

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

def unfold(
    u: jnp.ndarray,
    *,
    raw: jnp.ndarray,
    bg: jnp.ndarray,
    R: jnp.ndarray,
    G_eg: jnp.ndarray,
    G_ex: jnp.ndarray,
    loss,
    grad,
    value_and_grad,
    mask,
    iterations=10,
    lr=0.001,
    beta1=0.9,
    beta2=0.999,
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

@dataclass(kw_only=True)
class OptimComponentsMatrix:
    raw: jnp.ndarray
    initial: jnp.ndarray
    mask: jnp.ndarray
    background: jnp.ndarray | None = None

    def __post_init__(self):
        self.raw = jnp.asarray(self.raw)
        self.initial = jnp.asarray(self.initial)
        self.mask = jnp.asarray(self.mask)
        if self.background is not None:
            self.background = jnp.asarray(self.background)

        if self.raw.shape != self.initial.shape:
            raise ValueError("Raw and initial must have the same shape")
        if self.raw.shape != self.mask.shape:
            raise ValueError("Raw and mask must have the same shape")
        if self.background is not None and self.raw.shape != self.background.shape:
            raise ValueError("Raw and background must have the same shape")

    def __len__(self):
        return len(self.raw)


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