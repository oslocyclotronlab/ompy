from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Literal

import matplotlib.pyplot as plt
import numpy as np
from tqdm.autonotebook import tqdm

from ... import JAX_WORKING, Matrix
from ...helpers import (
    estimate_memory_usage,
    warn_memory,
)
from ...numbalib import njit
from ...stubs import Axes, array2D, array3D
from ..result2d import Cost2D, UnfoldedResult2D
from .fics_1d import chi2_safe, fluctuation_cost
from ..utils import loop_tqdm

if JAX_WORKING:
    import jax
    from jax import numpy as jnp
else:
    jax = lambda x: x
    jax.jit = lambda x: x

if TYPE_CHECKING:
    from .fics import FICSKwargs


@njit
def unfold_matrix(
    R: array2D, raw: array2D, initial: array2D, iterations: int, lr: float
):
    u = initial
    u_all = np.empty((iterations, *raw.shape))
    cost = np.empty(iterations)
    fluctuations = np.empty(iterations)
    mask = raw > 0
    R = R.T
    f = u @ R
    for i in range(iterations):
        u += lr * (raw - f)
        f = u @ R
        u_all[i] = u
        cost[i] = chi2_safe(f, raw, mask)
        # fluctuations[i] = fluctuation_cost(u, 20)
    return u_all, cost, fluctuations, []


def unfold_matrix_jax(R, Gex, raw, initial, kw: FICSKwargs):
    lr = kw.lr
    iterations = kw.iterations
    if kw.enforce_positivity:
        if False:
            raw_sqrt = jnp.sqrt(raw)
            initial = jnp.sqrt(raw)

            @jax.jit
            def body(R, u, f):
                f_sqrt = jnp.sqrt(f)
                u_sqrt = jnp.sqrt(u)
                u_sqrt = u_sqrt + lr * (raw_sqrt - f_sqrt)
                u = u_sqrt**2
                f = jnp.matmul(u, R)
                return u, f
        else:

            @jax.jit
            def body(R, u, f):
                u = u + lr * (raw - f)
                u = jnp.maximum(0, u)
                f = jnp.matmul(u, R)
                return u, f

    else:

        @jax.jit
        def body(R, Gex, u, f):
            u = u + lr * (raw - f)
            f = Gex@u@R
            return u, f

    u = initial
    cost = np.empty((iterations, raw.shape[0]))
    fluctuations = np.empty_like(cost)
    kl_div = np.empty((iterations, raw.shape[0]))
    mask = raw > 0
    # R = R.T
    f = Gex @ u @ R
    if kw.disable_tqdm:
        tqdm_ = lambda x, **kwargs: x
    else:
        tqdm_ = loop_tqdm

    @tqdm_(iterations, leave=kw.leave_tqdm)
    def body_fun(i, state):
        u, f, cost, kl_div = state
        u, f = body(R, Gex, u, f)
        cost = cost.at[i].set(chi2_jax(f, raw, mask)    )
        kl_div = kl_div.at[i].set(kl_jax(f, raw).sum(axis=1))
        # fluctuations[i] = fluctuation_cost(u, 20)
        return u, f, cost, kl_div
    state = (u, f, cost, kl_div)
    state = jax.lax.fori_loop(0, iterations, body_fun, state)
    u_all, _, cost, kl_div = state
    return u_all, cost, fluctuations, kl_div


@jax.jit
def kl_jax(nu, n):
    return nu - n + n * jnp.log(n / (nu + 1e-10) + 1e-10)


@jax.jit
def chi2_jax(a, b, mask):
    diff = (a - b) ** 2 / a
    # Use elementwise multiplication with the mask and then sum
    return jnp.sum(diff * mask, axis=1)


def unfold_matrix_jax_block(R, raw, initial, kw: FICSKwargs):
    lr = kw.lr
    iterations = kw.iterations

    @jax.jit
    def body(R, u, f):
        u = u + lr * (raw - f)
        f = jnp.matmul(u, R)
        return u, f

    u = initial
    cost = np.empty((iterations, raw.shape[0]))
    fluctuations = np.empty(iterations)
    warn_memory(
        estimate_memory_usage((iterations, *raw.shape)), "Cube of unfolded data"
    )
    u_all = np.empty((iterations, *raw.shape))
    mask = raw > 0
    R = R.T
    f = u @ R
    for i in tqdm(range(iterations)):
        u, f = body(R, u, f)
        u_all[i] = u
        cost[i] = chi2_jax(f, raw, mask)
        # fluctuations[i] = fluctuation_cost(u, 20)
    return u_all, cost, fluctuations, []


@dataclass(kw_only=True)  # (frozen=True, slots=True)
class FICSResult2DMultiple(Cost2D, UnfoldedResult2D):
    u: array3D
    fluctuations: array2D
    kl: array2D

    def unfolded(self, i: Iterable[int]) -> Matrix:
        rows = self.u.shape[1]
        x = self.u[i, np.arange(rows)]  # type: ignore
        return self.raw.clone(values=x)

    def best(self, cost: Literal["kl", "chi2"] = "kl", **kwargs) -> Matrix:
        if cost == "kl":
            score = self.kl
        elif cost == "chi2":
            score = self.score(**kwargs)
        # return self.unfolded(i)
        i = np.argmin(score, axis=0)
        return self.raw.clone(values=self.u[-1])

    def score(self, w: float | None = None) -> array2D:
        if w is None:
            w = self.get_param("weight")
        assert w is not None
        score = (1 - w) * self.cost + w * self.fluctuations
        return score

    def _save(self, path: Path, exist_ok: bool = False):
        np.save(path / "cost.npy", self.cost)
        np.save(path / "fluctuations.npy", self.fluctuations)
        np.save(path / "kl.npy", self.kl)
        np.save(path / "u.npy", self.u)

    @classmethod
    def _load(cls, path: Path) -> dict[str, np.ndarray]:
        cov = np.load(path / "cost.npy")
        flu = np.load(path / "fluctuations.npy")
        kl = np.load(path / "kl.npy")
        u = np.load(path / "u.npy")
        return {"cost": cov, "fluctuations": flu, "kl": kl, "u": u}


@dataclass(kw_only=True)  # (frozen=True, slots=True)
class FICSResult2DSimple(Cost2D, UnfoldedResult2D):
    u: array2D
    fluctuations: array2D
    kl: array2D

    def best(self) -> Matrix:
        return self.raw.clone(values=self.u)

    def plot_cost(self, ax: Axes | None = None, **kwargs) -> Plot1D:
        if ax is None:
            fig, ax = plt.subplots(nrows=2, sharex=True, constrained_layout=True)
        assert ax is not None
        ax = np.atleast_1d(ax).ravel()
        lines = []
        cmap = kwargs.pop("cmap", "turbo")
        colormap = plt.get_cmap(cmap)
        N = self.cost.shape[1]
        colors = [colormap(i) for i in np.linspace(0, 1, N)]
        for i, c in enumerate(self.cost.T):
            ax[0].plot(c, color=colors[i], **kwargs)
            ax[1].plot(self.kl[i], color=colors[i], **kwargs)
        # Create a "fake" mappable for the colorbar
        index = self.raw.Y
        norm = plt.Normalize(index.min(), index.max())
        sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
        sm.set_array([])

        # Add the colorbar
        cbar = ax[0].figure.colorbar(sm, ax=ax, orientation="vertical")
        cbar.set_label(self.raw.get_xlabel())
        fig.supxlabel("Iteration")
        ax[0].set_ylabel("Cost")
        ax[1].set_ylabel("KL divergence")
        return ax, lines

    def _save(self, path: Path, exist_ok: bool = False):
        np.save(path / "cost.npy", self.cost)
        np.save(path / "fluctuations.npy", self.fluctuations)
        np.save(path / "kl.npy", self.kl)
        np.save(path / "u.npy", self.u)

    @classmethod
    def _load(cls, path: Path) -> dict[str, np.ndarray]:
        cov = np.load(path / "cost.npy")
        flu = np.load(path / "fluctuations.npy")
        kl = np.load(path / "kl.npy")
        u = np.load(path / "u.npy")
        return {"cost": cov, "fluctuations": flu, "kl": kl, "u": u}
