from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, TYPE_CHECKING
from tqdm import tqdm
import numpy as np

from ...numbalib import njit
from .resampling import Resampling

from ..unfolder import Unfolder
from ... import Matrix, Vector, JAX_AVAILABLE
from ...array import AsymmetricVector
from ...stubs import Axes, Lines, Plots1D
from ...helpers import make_ax

if JAX_AVAILABLE:
    import jax
    import jax.numpy as jnp
else:
    jnp = np

if TYPE_CHECKING:
    from ..result2d import UnfoldedResult2D

def resample_matrix(
    res: UnfoldedResult2D, N: int, base: Literal["raw", "folded"] = "folded", **kwargs
) -> Resampling2D:
    """Create Bootstrap ensemble of `A` using `res` method"""
    best = res.best().astype("float32")
    R = (res.meta.space, res.R.T.astype("float32"))
    G = res.G.T.astype("float32")
    G_ex = res.G_ex.astype("float32")
    kwargs = res.meta.kwargs | kwargs
    unfolder = Unfolder.resolve_method(res.meta.method)(
        R=res.R.T.astype("float32"), G=G.T.astype("float32")
    )

    match base:
        case "raw":
            A = res.raw.as_numpy().copy().astype("float32")
        case "folded":
            A = res.best_folded().as_numpy().copy().astype("float32")
        case _:
            raise ValueError(f"Unknown sample type {base}. Expected 'raw' or 'folded'")
    # TODO Add background
    A_boots: list[Matrix] = A.sample(N)
    unfolded_boot: list[Matrix] = []
    costs: list[np.ndarray] = []
    initials: list[Matrix] = []
    backgrounds: list[Matrix] = []
    best = best
    disable_tqdm = kwargs.pop("disable_tqdm", True)
    background = res.background
    if background is not None:
        raise NotImplementedError("Background not implemented yet")
        background = background.astype("float32")
        background.values = np.where(background <= 0, 3, background.values)
        background[~mask] = 0

    start = time.time()
    for i in tqdm(range(N)):
        initial = best.clone(values=np.random.uniform(1, 3 * best))
        if background is not None:
            bg = background.clone(values=np.random.poisson(background))
            backgrounds.append(bg)
        else:
            bg = None
        unf = unfolder.unfold(
            A_boots[i],
            initial=initial,
            R=R,
            G=G,
            G_ex=G_ex,
            background=background,
            disable_tqdm=disable_tqdm,
            **kwargs,
        )
        unf.to_device("cpu")
        unf.as_numpy()
        if False and unf.cost[-1] > unf.cost[0]:
            raise RuntimeError("Unfolding diverged")
        unfolded_boot.append(unf.best())

        # We do a rescaling to make the cost fit into fewer bytes to take up less space
        costs.append((unf.cost / unf.cost.max()).astype("float16"))
        initials.append(initial)
    elapsed = time.time() - start
    bootstraped = Resampling2D(
        base=res,
        bootstraps=A_boots,
        unfolded=unfolded_boot,
        kwargs=kwargs,
        costs=costs,
        initials=initials,
        backgrounds=backgrounds if background is not None else None,
        elapsed_time=elapsed,
    )
    return bootstraped



@dataclass(kw_only=True)
class Resampling2D(Resampling[Matrix]):
    base: UnfoldedResult2D
    bootstraps: list[Matrix]
    unfolded: list[Matrix]
    initials: list[Matrix]
    kwargs: dict[str, Any] = field(default_factory=dict)
    ndim: Literal[2] = 2

    @classmethod
    def from_path(
        cls, path: str | Path, read_only: int | None = None
    ) -> Resampling2D:
        return Resampling._load(Path(path), Matrix, Resampling2D, read_only)

    def plot_unfolded(self, Ex: float | int, ax: Axes | None = None) -> Plots1D:
        ax = make_ax(ax)
        mu: Vector = self.base.best().loc[Ex, :]
        j = mu.last_nonzero()
        mu = mu.iloc[:j]
        lines: list[Lines] = []
        _, l = mu.plot(ax=ax, label=r"$\hat{\mu}$")
        lines.append(l)
        for i in range(len(self)):
            u: Vector = self.unfolded[i].loc[Ex, :j]
            _, l = u.plot(ax=ax, color="k", alpha=0.01)
            if i == 0:
                lines.append(l)
        return ax, lines

    def plot_folded(self, Ex: float | int, ax: Axes | None = None) -> Plots1D:
        ax = make_ax(ax)
        nu: Vector = self.base.best_folded().loc[Ex, :]
        j = nu.last_nonzero()
        nu = nu.iloc[:j]
        lines: list[Lines] = []
        _, l = nu.plot(ax=ax, label=r"$\hat{\mu}$")
        lines.append(l)
        for i in range(len(self)):
            u: Vector = ((self.base.R @ (self.unfolded[i]).T).T).loc[Ex, :j]
            _, l = u.plot(ax=ax, color="k", alpha=0.01)
            if i == 0:
                lines.append(l)
        return ax, lines

    def plot_eta(self, Ex: float | int, ax: Axes | None = None, **kwargs) -> Plots1D:
        ax = make_ax(ax)
        eta: Vector = (self.base.best() @ self.G_eg).loc[Ex, :]
        j = eta.last_nonzero()
        eta = eta.iloc[:j]
        lines: list[Lines] = []
        N = len(self)
        alpha = kwargs.pop("alpha", 1 / (N * 0.5))
        for i in range(N):
            u: Vector = ((self.base.G @ (self.unfolded[i]).T).T).loc[Ex, :j]
            _, l = u.plot(
                ax=ax, color="k", alpha=alpha, label=r"$\hat{\eta}_\mathrm{boot}$"
            )
            if i == 0:
                lines.append(l)
        _, l = eta.plot(ax=ax, label=r"$\hat{\eta}$")
        lines.append(l)
        return ax, lines

    def eta_vec(
        self, Ex: float | int, alpha=0.05, summary=np.median
    ) -> AsymmetricVector:
        i = self.base.raw.X_index.index_expression(Ex, strict=False)
        j = last_nonzero(self.etabox[:, i, :])
        eta = summary(self.etabox[:, i, :j], axis=0)
        eta = self.base.raw.iloc[i, :j].clone(values=eta)
        lower = np.percentile(self.etabox[:, i, :j], 100 * alpha / 2, axis=0)
        upper = np.percentile(self.etabox[:, i, :j], 100 * (1 - alpha / 2), axis=0)
        eta = AsymmetricVector.from_CI(eta, lower=lower, upper=upper, clip=True)
        return eta

    def nu_vec(
        self, Ex: float | int, alpha=0.05, summary=np.median
    ) -> AsymmetricVector:
        i = self.base.raw.X_index.index_expression(Ex, strict=False)
        nubox = self.nubox
        j = last_nonzero(nubox[:, i, :])
        nu = summary(nubox[:, i, :j], axis=0)
        nu = self.base.raw.iloc[i, :j].clone(values=nu)
        lower = np.percentile(nubox[:, i, :j], 100 * alpha / 2, axis=0)
        upper = np.percentile(nubox[:, i, :j], 100 * (1 - alpha / 2), axis=0)
        nu = AsymmetricVector.from_CI(nu, lower=lower, upper=upper, clip=True)
        return nu

    def eta_mat(self, summary=np.median) -> Matrix:
        eta = summary(self.etabox, axis=0)
        eta = self.base.raw.clone(values=eta)
        return eta

    def eta_ci(
        self,
        alpha=0.05,
        summary=np.median,
        as_matrix: bool = True,
    ) -> tuple[Matrix, Matrix] | tuple[np.ndarray, np.ndarray]:
        a_low = 100 * alpha / 2
        lower = np.percentile(self.etabox, a_low, axis=0)
        a_high = 100 * (1 - alpha / 2)
        upper = np.percentile(self.etabox, a_high, axis=0)
        if as_matrix:
            lower = self.base.raw.clone(
                values=lower, name=f"Lower {100 * (1 - alpha):.0f}% PI"
            )
            upper = self.base.raw.clone(
                values=upper, name=f"Upper {100 * (1 - alpha):.0f}% PI"
            )
        return lower, upper

    def nu_mat(self, summary=np.median) -> Matrix:
        nu = summary(self.nubox, axis=0)
        nu = self.base.raw.clone(values=nu)
        return nu

    def mu_mat(self, summary=np.median) -> Matrix:
        mu = summary(self.ubox, axis=0)
        mu = self.base.raw.clone(values=mu)
        return mu

    def mu_vec(
        self, Ex: float | int, alpha=0.05, summary=np.median
    ) -> AsymmetricVector:
        i = self.base.raw.X_index.index_expression(Ex, strict=False)
        j = last_nonzero(self.ubox[:, i, :])
        mu = summary(self.ubox[:, i, :j], axis=0)
        mu = self.base.raw.iloc[i, :j].clone(values=mu)
        lower = np.percentile(self.ubox[:, i, :j], 100 * alpha / 2, axis=0)
        upper = np.percentile(self.ubox[:, i, :j], 100 * (1 - alpha / 2), axis=0)
        mu = AsymmetricVector.from_CI(mu, lower=lower, upper=upper, clip=True)
        return mu

    def get_eta(self, i: int) -> Matrix:
        return self.base.raw.clone(values=self.etabox[i, :, :], name=f"eta {i}")

    @property
    def ubox(self) -> np.ndarray:
        if self._ubox is None:
            self._ubox = np.stack(self.unfolded)  # type: ignore
        return self._ubox

    @property
    def etabox(self) -> np.ndarray:
        if self.base.meta.space != "RG":
            warnings.warn(
                f"eta is only properly defined for the RG space, not {self.base.meta.space}."
            )
        if self._etabox is None:
            self._etabox = gmul(self.ubox, self.G_eg.values, self.G_ex.values)
        return self._etabox

    @property
    def nubox(self) -> np.ndarray:
        if self._nubox is None:
            self._nubox = gmul(self.ubox, self.GegD.values.T, self.G_ex.values)
        return self._nubox


def gmul(X, A, B=None):
    if B is None:
        return np.einsum("ijk,kl->ijl", X, A)
    else:
        raise NotImplementedError("Numpy einsum just stalls. Use jax.")
        # For the daredevils, this is the einsum
        return np.einsum("ij,kjl,lm->kim", B, X, A)
    
if JAX_AVAILABLE:

    @jax.jit
    def _gmul(X, A, B=None):
        if B is None:
            return jnp.einsum("ijk,kl->ijl", X, A)
        else:
            return jnp.einsum("ij,kjl,lm->kim", B, X, A)

    def gmul(X, A, B=None):
        x = _gmul(X, A, B)
        return np.asarray(x)



@njit
def last_nonzero(box: np.ndarray) -> int:
    S = np.sum(box, axis=0)
    for i in range(len(S) - 1, -1, -1):
        if S[i] > 0:
            return i
    return 0