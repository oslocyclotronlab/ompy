from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Literal, TYPE_CHECKING

import numpy as np
from matplotlib.rcsetup import cycler
from tqdm import tqdm

from ... import Vector
from ...array import AsymmetricVector
from ...helpers import make_ax, maybe_set
from ...stubs import Axes, Lines, Path, Plots1D
from ... import JAX_AVAILABLE
from .confidence import make_ci
from .resampling import Resampling
from .stubs import CI_Method
from ..unfolder import Unfolder

if JAX_AVAILABLE:
    import jax
    import jax.numpy as jnp
else:
    jnp = np

#if TYPE_CHECKING:
from ..result1d import UnfoldedResult1D


def resample_vector(
    res: UnfoldedResult1D,
    N: int,
    base: Literal["raw", "nu"] = "nu",
    bootstrap_background: bool = True,
    background_base: Literal["raw", "beta"] = "beta",
    **kwargs,
) -> Resampling1D:
    A_boots: list[Vector] = []
    unfolded_boots: list[Vector] = []
    best = res.best()
    kwargs = res.meta.kwargs | kwargs
    unfolder = Unfolder.from_result_constructor(res)

    match base:
        case "raw":
            A = res.raw.copy()
        case "folded" | "nu":
            A = res.best_folded().copy()
        case _:
            raise ValueError(f"Unknown sample type {base}. Expected 'raw' or 'folded'")
    A_boots = A.sample(N)
    for i, boot in enumerate(A_boots):
        A_boots[i] = boot.astype("float32")

    bg: Vector | None = None
    if res.background is not None:
        bg = res.background.copy()
        assert bg is not None
        bg.values = np.where(bg <= 0, 0, bg.values)
    bgs: list[Vector] = []
    # To avoid Poisson(0)
    # A.values = np.where(A <= 0, 3, A.values)
    costs: list[np.ndarray] = []
    bg_boots = None
    if bg is not None:
        if bootstrap_background:
            match background_base:
                case "raw":
                    bg_boots = bg.sample(N)
                case "beta":
                    bg_boots = res.beta.sample(N)
        else:
            bg_boots = [bg] * N
    mean = np.maximum(best, np.mean(best))
    initials = [best.clone(values=np.random.uniform(0, 5 * mean)) for i in range(N)]
    mask_1d = res.meta.parameters.mask
    # Stack the mask to match the shape of the unfolded result
    # Stack the penalty mask to match the shape of the unfolded result

    mask = mask_1d
    start = time.time()
    costs: list[np.ndarray] = []
    auxs: list[dict[str, np.ndarray]] = []
    unfolded_betas: list[Vector] = []
    # TODO use unfolder.unfold_vectors()
    unf_res: list[UnfoldedResult1D] = unfolder.unfold_vectors(
        A_boots,
        initial=initials,
        background=bg_boots,
        mask=mask,
        **kwargs,
    )
    if False:
        for i in tqdm(range(N)):
            bg_boot = bg_boots[i] if bg_boots is not None else None
            res_ = unfolder.unfold(
                A_boots[i],
                initial=initials[i],
                background=bg_boot,
                mask=mask,
                penalty_mask=penalty_mask,
                **kwargs,
            )
            if has_cost(res_):
                costs.append(res_.cost)
            if hasattr(res_, "aux"):
                auxs.append(res_.aux)
            unfolded_boots.append(res_.best_mu())
            if hasattr(res_, "beta") and res_.beta is not None:
                unfolded_betas.append(res_.beta)
    elapsed = time.time() - start

    unfolded_boots = [res_.best_mu() for res_ in unf_res]
    costs = [res_.cost for res_ in unf_res]
    auxs = [res_.aux for res_ in unf_res]
    contaminants = [res_.xi for res_ in unf_res]

    bootstraped = Resampling1D(
        base=res,
        bootstraps=A_boots,
        unfolded=unfolded_boots,  # type: ignore
        backgrounds=bgs,
        costs=costs,
        initials=initials,
        kwargs=kwargs,
        betas=unfolded_betas,
        elapsed_time=elapsed,
        aux=auxs,
        contaminants=contaminants,
    )
    return bootstraped  # , bg_boots


@dataclass(kw_only=True)
class Resampling1D(Resampling[Vector]):
    base: UnfoldedResult1D
    bootstraps: list[Vector]
    unfolded: list[Vector]
    costs: np.ndarray
    initials: list[Vector]
    betas: list[Vector] | None = None
    kwargs: dict[str, Any] = field(default_factory=dict)
    ndim: Literal[1] = 1
    contaminants: list[list[Vector]] = field(default_factory=lambda: [[]])

    @classmethod
    def from_path(cls, path: str | Path, read_only: int | None = None) -> Resampling1D:
        return Resampling._load(Path(path), Vector, Resampling1D, read_only)  # type: ignore

    def plot_unfolded(self, ax: Axes | None = None, **kwargs) -> Plots1D:
        ax = make_ax(ax)
        mu: Vector = self.base.best()
        lines: list[Lines] = []
        _, l = mu.plot(ax=ax, label=r"$\hat{\mu}$")
        lines.append(l)
        for i in range(len(self)):
            u: Vector = self.unfolded[i]
            _, l = u.plot(ax=ax, color="k", alpha=1 / len(self))
            if i == 0:
                lines.append(l)
        return ax, lines

    def plot_backgrounds(self, ax: Axes | None = None, **kwargs) -> Plots1D:
        ax = make_ax(ax)
        if self.background is None:
            return ax, []
        _, l = self.background.plot(ax=ax, **kwargs)
        lines = [l]
        if self.backgrounds is None:
            return ax, lines
        color = kwargs.pop("color", "k")
        alpha = kwargs.pop("alpha", 1 / len(self))
        for i in range(len(self)):
            _, l = self.backgrounds[i].plot(ax=ax, color=color, alpha=alpha, **kwargs)
            if i == 0:
                lines.append(l)
        return ax, (l, lines[-1])

    def plot_cost(
        self,
        ax: Axes | None = None,
        start: float | int = 0,
        auxiliary: bool | Iterable[str] = True,
        relative: bool = False,
        plot_each: bool = False,
        **kwargs,
    ) -> Plots1D:
        ax = make_ax(ax)
        if len(self.costs) == 0:
            return ax, []
        if isinstance(self.costs[0], Iterable):
            return self._plot_cost_list(
                ax, start, auxiliary, relative, plot_each, **kwargs
            )
        else:
            return self._plot_cost_single(ax, start, auxiliary, relative, **kwargs)

    def _plot_cost_single(
        self,
        ax: Axes | None = None,
        start: float | int = 0,
        auxiliary: bool | Iterable[str] = True,
        relative: bool = False,
        **kwargs,
    ) -> Plots1D:
        if isinstance(start, float):
            start = int(start * len(self.costs))
        cost = self.costs[start:]

        if isinstance(auxiliary, str):
            auxiliary = [auxiliary]
        aux = {}
        if len(self.aux) > 0:
            if isinstance(auxiliary, bool):
                keys = self.aux.keys() if auxiliary else []
            else:
                keys = auxiliary
            aux = {k: self.aux[k][start:] for k in keys}

        if relative:
            cost /= cost[0]
            for k in aux:
                aux[k] /= aux[k][0]

        i = np.arange(start, len(self.costs))
        lines: list[Lines] = []
        (l,) = ax.plot(i, cost, **kwargs)
        lines.append(l)
        for k, v in aux.items():
            (l,) = ax.plot(i, v, label=k)
            lines.append(l)
        maybe_set(ax, xlabel="Iteration")
        maybe_set(ax, ylabel="Cost")
        ax.legend()
        return ax, lines

    def _plot_cost_list(
        self,
        ax: Axes | None = None,
        start: float | int = 0,
        auxiliary: bool | Iterable[str] = True,
        relative: bool = False,
        plot_each: bool = False,
        **kwargs,
    ) -> Plots1D:
        if isinstance(start, float):
            start = int(start * len(self.costs[0]))

        # Process costs
        costs = np.array([cost[start:] for cost in self.costs])
        if relative:
            costs = costs / costs[:, 0:1]

        # Process auxiliary data
        if isinstance(auxiliary, str):
            auxiliary = [auxiliary]
        aux = {}
        if len(self.aux) > 0:
            if isinstance(auxiliary, bool):
                keys = self.aux.keys() if auxiliary else []
            else:
                keys = auxiliary
            aux = {k: np.array([a[start:] for a in self.aux[k]]) for k in keys}
            if relative:
                for k in aux:
                    aux[k] = aux[k] / aux[k][:, 0:1]

        i = np.arange(start, len(self.costs[0]))
        lines: list[Lines] = []

        default_cycler = cycler(color=["r", "g", "b", "y", "m", "c"])
        if plot_each:
            # Plot individual trajectories
            for j, cost in enumerate(costs):
                color = kwargs.get("color", None)
                (l,) = ax.plot(i, cost, color=color, alpha=0.2, **kwargs)
                if j == 0:  # Only add first line to legend
                    lines.append(l)

            # Plot auxiliary data
            for k, v in aux.items():
                color = next(default_cycler)["color"]
                for traj in v:
                    (l,) = ax.plot(
                        i,
                        traj,
                        color=color,
                        alpha=0.2,
                        label=k if len(lines) == 1 else "",
                    )
                    if len(lines) == 1:  # Only add first line of each type to legend
                        lines.append(l)
        else:
            # Plot mean and std band
            mean_cost = np.mean(costs, axis=0)
            std_cost = np.std(costs, axis=0)
            (l,) = ax.plot(i, mean_cost, **kwargs)
            lines.append(l)
            ax.fill_between(i, mean_cost - std_cost, mean_cost + std_cost, alpha=0.3)

            # Plot auxiliary data
            for k, v in aux.items():
                color = next(default_cycler)["color"]
                mean_aux = np.mean(v, axis=0)
                std_aux = np.std(v, axis=0)
                (l,) = ax.plot(i, mean_aux, color=color, label=k)
                lines.append(l)
                ax.fill_between(
                    i, mean_aux - std_aux, mean_aux + std_aux, color=color, alpha=0.3
                )

        maybe_set(ax, xlabel="Iteration")
        maybe_set(ax, ylabel="Cost")
        ax.legend()
        return ax, lines

    def plot_initials(self, ax: Axes | None = None, **kwargs) -> Plots1D:
        ax = make_ax(ax)
        lines = []
        kwargs = {"color": "k", "alpha": 1 / 10} | kwargs
        x = self.base.best().X
        dx = self.base.best().dX
        x = x + dx / 2
        for initial in self.initials:
            l = ax.plot(initial.X, initial.values, "_", **kwargs)
            # l = initial.plot(ax=ax, **kwargs)
            # l = ax.plot(x, initial, '_', **kwargs)
            lines.extend(l)
        return ax, lines

    def _make_ci(
        self,
        bootstraps: np.ndarray,
        original: np.ndarray,
        alpha=0.05,
        summary=np.median,
        method: CI_Method = "standard",
    ) -> AsymmetricVector:
        x = self.base.raw.clone(values=summary(bootstraps, axis=0))
        lower, upper = make_ci(
            bootstraps, original=original, alpha=alpha, method=method
        )
        return AsymmetricVector.from_CI(
            x, lower=lower, upper=upper, clip=True, order="K"
        )

    def mu(
        self, alpha=0.05, summary=np.median, method: CI_Method = "standard"
    ) -> AsymmetricVector:
        return self._make_ci(
            self.ubox,
            original=self.base.best().values,
            alpha=alpha,
            summary=summary,
            method=method,
        )

    def eta(
        self, alpha=0.05, summary=np.median, method: CI_Method = "standard"
    ) -> AsymmetricVector:
        return self._make_ci(
            self.etabox,
            original=self.base.best_eta().values,
            alpha=alpha,
            summary=summary,
            method=method,
        )

    def nu(
        self, alpha=0.05, summary=np.median, method: CI_Method = "standard"
    ) -> AsymmetricVector:
        return self._make_ci(
            self.nubox,
            original=self.base.best_folded().values,
            alpha=alpha,
            summary=summary,
            method=method,
        )

    def bg(
        self, alpha=0.05, summary=np.median, method: CI_Method = "standard"
    ) -> AsymmetricVector | None:
        if self.backgrounds is None:
            return None
        box = np.stack([b.values for b in self.backgrounds])
        return self._make_ci(
            box,
            self.base.background.values,
            alpha=alpha,
            summary=summary,
            method=method,
        )

    def beta(
        self, alpha=0.05, summary=np.median, method: CI_Method = "standard"
    ) -> AsymmetricVector | None:
        if self.betas is None:
            return None
        box = np.stack([beta.values for beta in self.betas])
        return self._make_ci(
            box,
            self.base.best_beta().values,
            alpha=alpha,
            summary=summary,
            method=method,
        )

    def xi_eta(
        self,
        i: int | None = None,
        alpha=0.05,
        summary=np.median,
        method: CI_Method = "standard",
    ) -> list[AsymmetricVector] | AsymmetricVector:
        xibox = self.xi_eta_box()
        if i is None:
            ci: list[AsymmetricVector] = []
            for i in range(xibox.shape[1]):
                ci.append(
                    self._make_ci(
                        xibox[:, i, :],
                        self.base.best_xi_eta(i).values,
                        alpha=alpha,
                        summary=summary,
                        method=method,
                    )
                )
            return ci
        else:
            return self._make_ci(
                xibox[:, i, :],
                self.base.best_xi_eta(i).values,
                alpha=alpha,
                summary=summary,
                method=method,
            )

    def xi_nu(
        self,
        i: int | None = None,
        alpha=0.05,
        summary=np.median,
        method: CI_Method = "standard",
    ) -> list[AsymmetricVector] | AsymmetricVector:
        xibox = self.xi_nu_box()
        if i is None:
            ci: list[AsymmetricVector] = []
            for i in range(xibox.shape[1]):
                ci.append(
                    self._make_ci(
                        xibox[:, i, :],
                        self.base.best_xi_folded(i).values,
                        alpha=alpha,
                        summary=summary,
                        method=method,
                    )
                )
            return ci
        else:
            return self._make_ci(
                xibox[:, i, :],
                self.base.best_xi_folded(i).values,
                alpha=alpha,
                summary=summary,
                method=method,
            )

    def total_nu(
        self,
        alpha=0.05,
        summary=np.median,
        method: CI_Method = "standard",
    ) -> AsymmetricVector:
        x = self.nubox + self.xi_nu_box().sum(axis=1)
        y = self.base.best_folded().values + np.sum(
            [self.base.best_xi_folded(i) for i in range(self.xi_nu_box().shape[1])]
        )
        return self._make_ci(
            x,
            y,
            alpha=alpha,
            summary=summary,
            method=method,
        )

    @property
    def ubox(self) -> np.ndarray:
        if self._ubox is None:
            self._ubox = np.stack(self.unfolded)  # type: ignore
        return self._ubox

    @property
    def etabox(self) -> np.ndarray:
        if self._etabox is None:
            self._etabox = jnp.einsum("ji,kj->ki", self.G_eg.values, self.ubox)
        return self._etabox

    @property
    def nubox(self) -> np.ndarray:
        if self._nubox is None:
            self._nubox = jnp.einsum("ji,kj->ki", self.GegD.values, self.ubox)
        return self._nubox

    def xi_eta_box(self) -> np.ndarray:
        if self._xi_eta_box is None:
            box = []
            for sample in self.contaminants:
                box.append(jnp.stack([c.values for c in sample]))
            box = jnp.stack(box)
            # Contaminants is a list of lists of vectors
            # We want to apply the same function to each vector in the list
            einsum_op = lambda x: jnp.einsum("ji,kj->ki", self.G_eg.values, x)
            self._xi_eta_box = jax.vmap(einsum_op)(box)
        return self._xi_eta_box

    def xi_nu_box(self) -> np.ndarray:
        if self._xi_nu_box is None:
            box = []
            for sample in self.contaminants:
                box.append(jnp.stack([c.values for c in sample]))
            box = jnp.stack(box)
            # Contaminants is a list of lists of vectors
            # We want to apply the same function to each vector in the list
            einsum_op = lambda x: jnp.einsum("ji,kj->ki", self.GegD.values, x)
            self._xi_nu_box = jax.vmap(einsum_op)(box)
        return self._xi_nu_box
