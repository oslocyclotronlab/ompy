from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ... import Vector
from ...helpers import append_label
from ...numbalib import njit
from ...stubs import Axes, Plots1D, array1D
from ..result1d import (
    Cost1D,
    UnfoldedResult1DMultiple,
)


@njit
def unfold_vector(
    R: array1D, raw: array1D, initial: array1D, iterations: int, lr: float
):
    u = initial
    u_all = np.empty((iterations, len(u)))
    cost = np.empty(iterations)
    kl_cost = np.empty_like(cost)
    fluctuations = np.empty(iterations)
    mask = raw > 0
    f = u @ R
    for i in range(iterations):
        u += lr * (raw - f)
        f = u @ R
        u_all[i] = u
        cost[i] = chi2_safe_1d(raw, f, mask)
        fluctuations[i] = fluctuation_cost(u, 20, mask)
        kl_cost[i] = kl(f, raw).sum()
    return u_all, cost, fluctuations, kl_cost


@njit
def unfold_vector_pos(
    R: array1D, raw: array1D, initial: array1D, iterations: int, lr: float
):
    assert np.all(initial >= 0), "Initial values must be positive"
    assert np.all(raw >= 0), "Raw values must be positive"
    u = initial
    u_all = np.empty((iterations, len(u)))
    cost = np.empty(iterations)
    kl_cost = np.empty_like(cost)
    fluctuations = np.empty(iterations)
    raw_sqrt = np.sqrt(raw)
    u_sqrt = np.sqrt(u)

    f = R @ u
    for i in range(iterations):
        f_sqrt = np.sqrt(f)
        u_sqrt = np.sqrt(u)
        u_sqrt += lr * (raw_sqrt - f_sqrt)
        u = u_sqrt**2
        f = R @ u
        u_all[i] = u
        cost[i] = chi2(f, raw)
        fluctuations[i] = fluctuation_cost(u, 20)
        kl_cost[i] = kl(f, raw).sum()
    return u_all, cost, fluctuations, kl_cost


@njit
def chi2(a, b):
    return np.sum((a - b) ** 2 / a)


@njit
def chi2_safe_1d(a, b, mask):
    s = 0.0
    for i in range(a.shape[0]):
        if mask[i]:
            s += (a[i] - b[i]) ** 2 / a[i]
    return s


@njit
def chi2_safe(a, b, mask):
    s = 0.0
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            if mask[i, j]:
                s += (a[i, j] - b[i, j]) ** 2 / a[i, j]
    return s


@njit
def kl(nu, n):
    return nu - n + n * np.log(n / (nu + 1e-10) + 1e-10)


@njit
def fluctuation_cost(x, sigma: float, mask):
    smoothed = gaussian_filter_1d(x, sigma)
    diff = 0.0
    for i in range(x.shape[0]):
        if mask[i]:
            diff += np.abs(((smoothed[i] - x[i]) / smoothed[i]))
    return diff


@njit
def gaussian_filter_1d(x, sigma):
    """
    1D Gaussian filter with standard deviation sigma.
    """
    k = int(4.0 * sigma + 0.5)
    w = np.zeros(2 * k + 1)
    for i in range(-k, k + 1):
        w[i + k] = np.exp(-0.5 * i**2 / sigma**2)
    w /= np.sum(w)

    # Handle edge cases of input signal
    y = np.zeros_like(x)
    for i in range(len(x)):
        for j in range(-k, k + 1):
            if i + j >= 0 and i + j < len(x):
                y[i] += x[i + j] * w[j + k]
    return y


@dataclass(kw_only=True)  # (frozen=True, slots=True)
class FICSResult1D(Cost1D, UnfoldedResult1DMultiple):
    fluctuations: array1D
    kl: array1D

    def best(self, min: int = 0, w: float | None = None) -> Vector:
        score = self.score(w)
        i = max(min, np.argmin(score))  # type: ignore
        return self.unfolded(i)

    def plot_cost(
        self,
        ax: list[Axes] | None = None,
        start: int | float = 0,
        legend: bool = True,
        yscale: str = "log",
        **kwargs,
    ) -> Plots1D:
        if ax is None:
            fig, ax = plt.subplots(nrows=4, sharex=True, constrained_layout=True)
        else:
            fig = ax[0].figure
        ax = np.atleast_1d(ax).ravel()
        if len(ax) < 4:
            raise ValueError("Not enough axes. Expected 4.")

        if isinstance(start, float):
            start = int(start * len(self.cost))
        x = np.arange(start, len(self.cost))

        score = self.score(kwargs.pop("w", None))
        lines = []
        root_label = kwargs.pop("label", None)
        label = append_label("cost", root_label)
        (line,) = ax[0].plot(x, self.cost[start:], label=label, **kwargs)
        label = append_label("fluctuations", root_label)
        (line,) = ax[1].plot(x, self.fluctuations[start:], label=label, **kwargs)
        label = append_label("score", root_label)
        ax[2].plot(x, score[start:], label=label, **kwargs)
        label = append_label("KL divergence", root_label)
        ax[3].plot(x, self.kl[start:], label=label, **kwargs)
        lines.append(line)
        fig.supylabel("Cost")
        fig.supxlabel("Iteration")
        if legend:
            for a in ax:
                a.legend()
        for a in ax:
            a.set_yscale(yscale)
        return ax, lines

    def score(self, w: float | None = None) -> array1D:
        w = self.get_param("weight") if w is None else w
        cost = (1 - w) * self.cost + w * self.fluctuations
        return cost

    def _save(self, path: Path, exist_ok: bool = False):
        np.save(path / "cost.npy", self.cost)
        np.save(path / "fluctuations.npy", self.fluctuations)
        np.save(path / "kl.npy", self.kl)

    @classmethod
    def _load(cls, path: Path) -> dict[str, np.ndarray]:
        cov = np.load(path / "cost.npy")
        flu = np.load(path / "fluctuations.npy")
        kl = np.load(path / "kl.npy")
        return {"cost": cov, "fluctuations": flu, "kl": kl}
