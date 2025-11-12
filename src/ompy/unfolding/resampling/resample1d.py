from __future__ import annotations

import time
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Iterable, Literal, TYPE_CHECKING, cast

import numpy as np
from matplotlib.rcsetup import cycler
from tqdm import tqdm

from ... import Vector
from ...accel import jax_available
from ...array import AsymmetricVector
from ...helpers import make_ax, maybe_set
from ...stubs import Axes, Lines, Path, Plots1D
from .confidence import make_ci
from .resampling import Resampling
from .stubs import CI_Method
from .sampler import Sampler
from ..rmle.rmle1d import BackgroundModel as BackgroundModel1D
from ..unfolder import Unfolder

if jax_available():
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
    sampler = Sampler.from_result(res)
    kwargs = res.meta.kwargs | kwargs
    unfolder = Unfolder.from_result_constructor(res)

    A_boots = [vec.astype("float32") for vec in sampler.sample_data(N, base=base)]

    background_samples = sampler.sample_background(
        N,
        base=background_base,
        bootstrap=bootstrap_background,
    )
    bg_models = background_samples if background_samples else None
    best = res.best()
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
        background=bg_models,
        mask=mask,
        **kwargs,
    )

    elapsed = time.time() - start

    unfolded_boots = [res_.best_mu() for res_ in unf_res]
    costs = [res_.cost for res_ in unf_res]
    auxs = [res_.aux for res_ in unf_res]
    contaminants = [res_.contaminants for res_ in unf_res]
    unfolded_betas = [res_.beta for res_ in unf_res]

    bootstraped = Resampling1D(
        base=res,
        bootstraps=A_boots,
        unfolded=unfolded_boots,  # type: ignore
        backgrounds=bg_models,
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

    def contaminant_eta(
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
                        self.base.best_contaminant_eta(i).values,
                        alpha=alpha,
                        summary=summary,
                        method=method,
                    )
                )
            return ci
        else:
            return self._make_ci(
                xibox[:, i, :],
                self.base.best_contaminant_eta(i).values,
                alpha=alpha,
                summary=summary,
                method=method,
            )

    def contaminant_nu(
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
                        self.base.best_contaminant_folded(i).values,
                        alpha=alpha,
                        summary=summary,
                        method=method,
                    )
                )
            return ci
        else:
            return self._make_ci(
                xibox[:, i, :],
                self.base.best_contaminant_folded(i).values,
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
            [self.base.best_contaminant_folded(i) for i in range(self.xi_nu_box().shape[1])]
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


class Sampler1D(Sampler):
    ndim = 1

    @property
    def result(self) -> UnfoldedResult1D:
        return cast(UnfoldedResult1D, super().result)

    def sample_data(
        self,
        count: int,
        base: Literal["raw", "folded", "nu"] = "nu",
    ) -> list[Vector]:
        match base:
            case "raw":
                vec = self.result.raw.copy()
            case "folded" | "nu":
                vec = self.result.best_folded().copy()
            case _:
                raise ValueError(
                    "sample_data base must be 'raw', 'folded', or 'nu' for 1D results"
                )
        vec = vec.astype("float32")
        mask = self.result.meta.parameters.mask
        samples = vec.sample(count, mask=mask)
        return [sample.astype("float32") for sample in samples]

    def sample_background(
        self,
        count: int,
        *,
        base: Literal["raw", "beta"] = "beta",
        bootstrap: bool = True,
    ) -> list[BackgroundModel1D]:
        background = self.result.background
        if background is None or len(background.backgrounds) == 0:
            return []

        model_bgs = background.backgrounds
        if bootstrap:
            match base:
                case "raw":
                    draws = [sample(bg, count) for bg in model_bgs]
                case "beta":
                    beta = self.result.beta
                    if beta is None:
                        raise ValueError("Result does not include beta for background sampling")
                    beta_samples = beta.sample(count)
                    draws = [
                        [vec.values for vec in beta_samples]
                        for _ in model_bgs
                    ]
                case _:
                    raise ValueError("Background base must be 'raw' or 'beta'")
            bg_boots = list(zip(*[np.asarray(d) for d in draws]))
            loss = background.loss
            return [
                BackgroundModel1D(
                    loss=loss,
                    backgrounds=tuple(np.asarray(bg_) for bg_ in boot),
                )
                for boot in bg_boots
            ]

        base_draws = [np.asarray(bg.values) for bg in model_bgs]
        loss = background.loss
        return [
            BackgroundModel1D(
                loss=loss,
                backgrounds=tuple(np.asarray(draw) for draw in base_draws),
            )
            for _ in range(count)
        ]

    def sample_total(
        self,
        count: int,
        *,
        base: Literal["folded", "raw"] = "folded",
    ) -> list[Vector]:
        match base:
            case "folded":
                vec = self.result.folded_total()
            case "raw":
                vec = self.result.raw
            case _:
                raise ValueError("total sampling supports only 'folded' or 'raw' for 1D results")
        vec = vec.astype("float32")
        mask = self.result.meta.parameters.mask
        samples = vec.sample(count, mask=mask)
        return [sample.astype("float32") for sample in samples]


@partial(jax.jit, static_argnames=("N", "zero_value", "zero_limit"))
def sample(arr: jnp.ndarray, N: int, mask: np.ndarray | None = None, zero_value: int = 0,
                zero_limit: int = 0, key: jnp.ndarray | None = None, **kwargs) -> jnp.ndarray:
    """ Draw `N` poisson samples from the array.

    The `mask` specifies values to ignore. If not set, the mask is assumed to be
    all zero elements "after" the diagonal.
    
    Args:
        N (int): The number of samples to generate.
        mask (np.ndarray, optional): A boolean mask array to apply zeros to. If not provided, the last non-zero elements are used.
    
    Returns:
        Iterator[Self]: An iterator that yields `N` new instances of the array, with the sampled values.
    """

    if mask is None:
        idxs = jnp.arange(arr.shape[0])
        mask = idxs <= last_nonzero_index(arr)

    X = jnp.where((arr <= zero_limit) | ~mask, zero_value, arr)
    if key is None:
        key = jax.random.PRNGKey(0)
    return jax.random.poisson(key, X, (N, len(arr)))

@jax.jit
def last_nonzero_index(arr: jnp.ndarray) -> int:
    """
    Returns the index of the last non-zero value in a 1D array,
    or -1 if all values are zero.
    """
    # build an array that's [i if arr[i] != 0, else -1]
    idxs = jnp.where(arr != 0,
                     jnp.arange(arr.shape[0], dtype=jnp.int32),
                     -1)
    # the max of that is the last non-zero index (or -1)
    return idxs.max()
