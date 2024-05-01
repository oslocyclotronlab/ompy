from __future__ import annotations
import numpy as np
from ..numbalib import njit, prange
from .. import Vector, Matrix, Response, zeros_like, JAX_AVAILABLE, JAX_WORKING
from .unfolder import Unfolder
from .result1d import Cost1D, UnfoldedResult1DMultiple, ResultMeta1D, Parameters1D, UnfoldedResult1D
from .result2d import Cost2D, UnfoldedResult2D, ResultMeta2D, Parameters2D
from .stubs import Space
from ..stubs import array1D, array2D, array3D, Plots1D, Axes
from dataclasses import dataclass, fields, asdict
import time
from typing_extensions import override
from tqdm.autonotebook import tqdm
import logging
from ..helpers import readable_time, warn_memory, estimate_memory_usage, make_ax, maybe_set, append_label
from typing import Iterable, Literal
import matplotlib.pyplot as plt
from pathlib import Path

LOG = logging.getLogger(__name__)

if JAX_AVAILABLE:
    from jax import numpy as jnp
    import jax
else:
    jax = lambda x: x
    jax.jit = lambda x: x



@dataclass
class GuttormsenKwargs:
    iterations: int
    weight: float
    lr: float
    save_block: bool = True
    disable_tqdm: bool = False
    enforce_positivity: bool = True
    leave_tqdm: bool = True


class Guttormsen(Unfolder):
    """ Unfolding algorithm from Guttormsen et al. 1998

    This algorithm is only valid for 1D histograms with uniform binning.
    The algorithm is described in the paper:

    Guttormsen, K. A., Kjeldsen, H. K., & Nielsen, J. B. (1998).
    Unfolding of multidimensional histograms.
    Nuclear Instruments and Methods in Physics Research Section A: Accelerators, Spectrometers, Detectors and Associated Equipment, 400(1), 1–8. https://doi.org/10.1016/S0168-9002(97)00459-6

    Parameters
    ----------
    R: Matrix
        The unsmoothed response matrix
    G: Matrix
        The gaussian smoothing matrix
    iterations: int
        The number of iterations to perform
    """

    def __init__(self,
                 R: Matrix,
                 G: Matrix,
                 iterations: int = 10,
                 weight: float = 1e-3,
                 use_JAX: bool | None = None,
                 save_block: bool = False,
                 enforce_positivity: bool = False):
        super().__init__(R, G)
        self.iterations = iterations
        self.weight = weight  # Fluctuation weight
        # We prefer to use GPUs, but fall back to CPU if not available
        # If the user specifies GPU, but GPU is not available, raise an error
        if use_JAX is None:
            self.use_JAX = JAX_AVAILABLE and JAX_WORKING
        elif use_JAX and not JAX_WORKING:
            raise ValueError("JAX is not working. Cannot use GPU. Specify 'use_JAX=False' to use CPU.")
        else:
            self.use_JAX = use_JAX
        self.lr = 1  # Learning rate. Unused
        self.save_block = save_block  # Save block of unfolded matrices
        self.enforce_positivity = enforce_positivity

    def handle_kwargs(self, kwargs) -> GuttormsenKwargs:
        supported = [f.name for f in fields(GuttormsenKwargs)]
        described = {k: v for k, v in kwargs.items() if k in supported}
        superfluous = {k: v for k, v in kwargs.items() if k not in supported}
        defaults = dict(iterations=self.iterations,
                        weight=self.weight,
                        lr=self.lr,
                        save_block=self.save_block,
                        enforce_positivity=self.enforce_positivity)
        kw = GuttormsenKwargs(**(defaults | described))
        assert kw.lr == 1, "Learning rate must be 1. Bug in code or in me?"

        LOG.debug(f"Unfolding up to {kw.iterations} iterations")
        LOG.debug(f"Fluctuation weight of {kw.weight}")
        LOG.debug(f"Learning rate of {kw.lr}")
        LOG.debug(f"Enforcing positive values: {kw.enforce_positivity}")
        if superfluous:
            LOG.warning(f"Unused kwargs: {superfluous}")
        return kw

    def _unfold_vector(self,
                       R: Matrix,
                       data: Vector,
                       background: Vector | None,
                       initial: Vector,
                       space: Space,
                       G: Matrix | None = None,
                       **kwargs) -> GuttormsenResult1D:
        kw = self.handle_kwargs(kwargs)
        LOG.debug("Unfolding vector with Guttormsen method")
        LOG.debug("Unfolding to space: %s", space)
        data_raw = data
        if background is not None:
            LOG.debug("Background is just subtracted from data.")
            data = data - background
        start = time.time()


        data = data.astype('float32')
        initial = data.astype('float32')
        if kw.enforce_positivity:
            fn = _unfold_vector_pos
        else:
            fn = _unfold_vector
        uall, cost, fluctuations, kl = fn(R.values, data.values,
                                        initial.values,
                                        kw.iterations, kw.lr)
        elapsed = time.time() - start
        kw_ = asdict(kw)
        kw_.pop('disable_tqdm')
        kw_.pop('leave_tqdm')
        parameters = Parameters1D(raw=data_raw,
                                  background=background,
                                  initial=initial,
                                  G=G,
                                  R=R,
                                  kwargs=kw_)

        meta = ResultMeta1D(time=elapsed,
                            space=space,
                            parameters=parameters,
                            method=self.__class__)
        return GuttormsenResult1D(meta=meta,
                                  u=uall,
                                  cost=cost,
                                  fluctuations=fluctuations,
                                  kl=kl)

    @override
    def _unfold_matrix(self, R: Matrix, data: Matrix,
                       background: Matrix | None, initial: Matrix,
                       use_previous: bool, space: Space, G: Matrix,
                       G_ex: Matrix, **kwargs) -> GuttormsenResult2DSimple | GuttormsenResult2DMultiple:
        LOG.debug("Unfolding matrix with Guttormsen method")
        kw = self.handle_kwargs(kwargs)
        LOG.debug("Unfolding to space: %s", space)
        raw = data
        if background is not None:
            data = data - background

        start = time.time()
        if self.use_JAX:
            LOG.debug("Using JAX version")
            Rj = jnp.array(R.values)
            dataj = jnp.array(data.values)
            initialj = jnp.array(initial.values)
            Gexj = jnp.array(G_ex.values)
            if self.save_block:
                LOG.debug("Saving block of unfolded matrices")
                uall, cost, fluctuations, kl_div = _unfold_matrix_jax_block(Rj, dataj, initialj, kw)
            else:
                uall, cost, fluctuations, kl_div = _unfold_matrix_jax(Rj, Gexj, dataj, initialj, kw)
        else:
            fn = _unfold_matrix
            uall, cost, fluctuations = fn(R.values, data.values,
                                          initial.values, kw.iterations, kw.lr)
        elapsed = time.time() - start
        LOG.debug(f"Unfolding took {readable_time(elapsed)} seconds")

        kw_ = asdict(kw) | {'save_block': self.save_block}
        kw_.pop('disable_tqdm')
        kw_.pop('leave_tqdm')
        parameters = Parameters2D(R=R,
                                  raw=raw,
                                  background=background,
                                  initial=initial,
                                  G=G,
                                  kwargs=kw_,
                                  G_ex=G_ex)
        meta = ResultMeta2D(time=elapsed,
                            space=space,
                            parameters=parameters,
                            method=self.__class__)
        if self.save_block:
            rescls = GuttormsenResult2DMultiple
        else:
            rescls = GuttormsenResult2DSimple
        return rescls(meta=meta,
                      u=uall,
                      cost=cost,
                      fluctuations=fluctuations,
                      kl=kl_div)

    @override
    def supports_background(self) -> bool:
        return True 


@njit
def _unfold_vector(R: array1D, raw: array1D, initial: array1D, iterations: int,
                   lr: float):
    u = initial
    u_all = np.empty((iterations, len(u)))
    cost = np.empty(iterations)
    kl_cost = np.empty_like(cost)
    fluctuations = np.empty(iterations)
    f = R @ u
    for i in range(iterations):
        u += lr * (raw - f)
        f = R @ u
        u_all[i] = u
        cost[i] = chi2(f, raw)
        fluctuations[i] = fluctuation_cost(u, 20)
        kl_cost[i] = kl(f, raw).sum()
    return u_all, cost, fluctuations, kl_cost


@njit
def _unfold_vector_pos(R: array1D, raw: array1D, initial: array1D, iterations: int,
                   lr: float):
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
def _unfold_matrix(R: array2D, raw: array2D, initial: array2D, iterations: int,
                   lr: float):
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
        #fluctuations[i] = fluctuation_cost(u, 20)
    return u_all, cost, fluctuations, []


@njit
def chi2(a, b):
    return np.sum((a - b)**2 / a)


@njit
def chi2_safe(a, b, mask):
    s = 0.0
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            if mask[i, j]:
                s += (a[i, j] - b[i, j])**2 / a[i, j]
    return s

@njit
def kl(nu, n):
    return nu - n + n * np.log(n / (nu+1e-10) + 1e-10)


def _unfold_matrix_jax(R, Gex, raw, initial, kw: GuttormsenKwargs):
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
            fp = jnp.matmul(u, R)
            f = jnp.matmul(Gex, fp)
            return u, f

    u = initial
    cost = np.empty((iterations, raw.shape[0]))
    fluctuations = np.empty_like(cost)
    kl_div = np.empty((iterations, raw.shape[0]))
    mask = raw > 0
    R = R.T
    f = Gex @ u @ R
    if kw.disable_tqdm:
        tqdm_ = lambda x: x
    else:
        if not kw.leave_tqdm:
            tqdm_ = lambda x: tqdm(x, leave=False)
        else:
            tqdm_ = tqdm

    for i in tqdm_(range(iterations)):
        u, f = body(R, Gex, u, f)
        cost[i] = chi2_jax(f, raw, mask)
        kl_div[i] = kl_jax(f, raw).sum(axis=1)
        #fluctuations[i] = fluctuation_cost(u, 20)
    u_all = u
    return u_all, cost, fluctuations, kl_div

@jax.jit
def kl_jax(nu, n):
    return nu - n + n * jnp.log(n / (nu+1e-10) + 1e-10)



@jax.jit
def chi2_jax(a, b, mask):
    diff = (a - b)**2 / a
    # Use elementwise multiplication with the mask and then sum
    return jnp.sum(diff * mask, axis=1)

def _unfold_matrix_jax_block(R, raw, initial, kw: GuttormsenKwargs):
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
    warn_memory(estimate_memory_usage((iterations, *raw.shape)),
                "Cube of unfolded data")
    u_all = np.empty((iterations, *raw.shape))
    mask = raw > 0
    R = R.T
    f = u @ R
    for i in tqdm(range(iterations)):
        u, f = body(R, u, f)
        u_all[i] = u
        cost[i] = chi2_jax(f, raw, mask)
        #fluctuations[i] = fluctuation_cost(u, 20)
    return u_all, cost, fluctuations, []


@njit
def fluctuation_cost(x, sigma: float):
    smoothed = gaussian_filter_1d(x, sigma)
    diff = np.abs(((smoothed - x) / smoothed))
    return diff.sum()

def compton_subtraction(res: UnfoldedResult1D | UnfoldedResult2D,
                        response: Response,
                        space='eta',
                        use_eff: bool = False) -> Vector | Matrix:
    if space == 'eta':
        u = res.best_eta()
    elif space == 'mu':
        u = res.best()
    else:
        raise ValueError(f"Invalid space: {space}")
    return compton_subtraction_(u, res.raw, response, use_eff=use_eff)


def compton_subtraction_(unfolded: Vector, raw: Vector, response: Response,
                         use_eff: bool = False):
    G = response.gaussian_like(unfolded).T
    eff = response.interpolation.Eff(unfolded.Eg)

    f = response.fold_componentwise(unfolded)
    fe, se, de, ap, compton0 = f.FE, f.SE, f.DE, f.AP, f.compton
    # Need to smooth AP to correct for commutator
    ap = G@ap

    # The discrete structures: w
    w = se + de + ap
    # Total, without compton: v
    v = fe + w
    # Assume everything left over from the raw spectrum is the compton.
    # Incredible bad assumption, doesn't take into account the noise
    # of the raw spectrum, nor, more importantly, the errors made in the unfolding.
    # More succintly: the peaks FE, SE, DE, AP, can only be assumed to be correct
    # when the unfolding is correct, but if the unfolding were correct, there
    # would be no need for the this compton subtraction method to be used!
    compton = raw - v
    ax, _ = compton0.plot(label='compton 0')
    compton.plot(ax=ax, label='compton 1')
    # We know the compton is smooth, so smooth it.
    # Assume this is to correct for the noise, but this is too 
    # ad-hoc. 
    compton = G @ compton
    compton.plot(ax=ax, label='compton 2')
    ax.legend()

    # The raw spectrum minus the modeled compton, and the folded discrete structures
    # is the unfolded spectrum. I don't like this either, since now the
    # noise of the raw spectrum infects the unfolded spectrum, which we *also*
    # know must be smooth.
    unf = (raw - compton - w) / fe

    if use_eff:
        unf = unf / eff

    ax0, _ = unf.plot(label='unf')
    compton.plot(ax=ax0, label='compton')
    fe.plot(ax=ax0, label='fe')
    se.plot(ax=ax0, label='se')
    de.plot(ax=ax0, label='de')
    ap.plot(ax=ax0, label='uap')
    raw.plot(ax=ax0, label='raw')
    unfolded.plot(ax=ax0, label='unfolded')
    ax0.legend()

    return ax, ax0


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


@dataclass(kw_only=True)  #(frozen=True, slots=True)
class GuttormsenResult1D(Cost1D, UnfoldedResult1DMultiple):
    fluctuations: array1D
    kl: array1D

    def best(self, min: int = 0, w: float | None = None) -> Vector:
        score = self.score(w)
        i = max(min, np.argmin(score))  # type: ignore
        return self.unfolded(i)

    def plot_cost(self, ax: list[Axes] | None = None, start: int | float = 0,
                  legend: bool = True, yscale: str = 'log', **kwargs) -> Plots1D:
        if ax is None:
            fig, ax = plt.subplots(nrows=4, sharex=True, constrained_layout=True)
        ax = np.atleast_1d(ax).ravel()
        if len(ax) < 4:
            raise ValueError("Not enough axes. Expected 4.")

        if isinstance(start, float):
            start = int(start*len(self.cost))
        x = np.arange(start, len(self.cost))

        score = self.score(kwargs.pop('w', None))
        lines = []
        root_label = kwargs.pop('label', None)
        label = append_label('cost', root_label)
        line, = ax[0].plot(x, self.cost[start:], label=label, **kwargs)
        label = append_label('fluctuations', root_label)
        line, = ax[1].plot(x, self.fluctuations[start:], label=label, **kwargs)
        label = append_label('score', root_label)
        ax[2].plot(x, score[start:], label=label, **kwargs)
        label = append_label('KL divergence', root_label)
        ax[3].plot(x, self.kl[start:], label=label, **kwargs)
        lines.append(line)
        fig.supylabel('Cost')
        fig.supxlabel('Iteration')
        if legend:
            for a in ax:
                a.legend()
        for a in ax:
            a.set_yscale(yscale)
        return ax, lines

    def score(self, w: float | None = None) -> array1D:
        w = self.get_param("weight") if w is None else w
        cost = (1-w)*self.cost + w*self.fluctuations
        return cost

    def _save(self, path: Path, exist_ok: bool = False):
        np.save(path / 'cost.npy', self.cost)
        np.save(path / 'fluctuations.npy', self.fluctuations)
        np.save(path / 'kl.npy', self.kl)

    @classmethod
    def _load(cls, path: Path) -> dict[str, np.ndarray]:
        cov = np.load(path / 'cost.npy')
        flu = np.load(path / 'fluctuations.npy')
        kl = np.load(path / 'kl.npy')
        return {'cost': cov, 'fluctuations': flu, 'kl': kl}




@dataclass(kw_only=True)  #(frozen=True, slots=True)
class GuttormsenResult2DMultiple(Cost2D, UnfoldedResult2D):
    u: array3D
    fluctuations: array2D
    kl: array2D

    def unfolded(self, i: Iterable[int]) -> Matrix:
        rows = self.u.shape[1]
        x = self.u[i, np.arange(rows)]  # type: ignore
        return self.raw.clone(values=x)

    def best(self, cost: Literal['kl', 'chi2'] = 'kl',
             **kwargs) -> Matrix:
        if cost == 'kl':
            score = self.kl
        elif cost == 'chi2':
            score = self.score(**kwargs)
        #return self.unfolded(i)
        i = np.argmin(score, axis=0)
        return self.raw.clone(values=self.u[-1])

    def score(self, w: float | None = None) -> array2D:
        if w is None:
            w = self.get_param("weight")
        assert w is not None
        score = (1 - w)*self.cost + w * self.fluctuations
        return score

    def _save(self, path: Path, exist_ok: bool = False):
        np.save(path / 'cost.npy', self.cost)
        np.save(path / 'fluctuations.npy', self.fluctuations)
        np.save(path / 'kl.npy', self.kl)
        np.save(path / 'u.npy', self.u)

    @classmethod
    def _load(cls, path: Path) -> dict[str, np.ndarray]:
        cov = np.load(path / 'cost.npy')
        flu = np.load(path / 'fluctuations.npy')
        kl = np.load(path / 'kl.npy')
        u = np.load(path / 'u.npy')
        return {'cost': cov, 'fluctuations': flu, 'kl': kl, 'u': u}


@dataclass(kw_only=True)  #(frozen=True, slots=True)
class GuttormsenResult2DSimple(Cost2D, UnfoldedResult2D):
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
        cmap = kwargs.pop('cmap', 'turbo')
        colormap = plt.get_cmap(cmap)
        N = self.cost.shape[1]
        colors = [colormap(i) for i in np.linspace(0, 1, N)]
        for i, c in enumerate(self.cost.T):
            ax[0].plot(c, color=colors[i],  **kwargs)
            ax[1].plot(self.kl[i], color=colors[i], **kwargs)
        # Create a "fake" mappable for the colorbar
        index = self.raw.Y
        norm = plt.Normalize(index.min(), index.max())
        sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
        sm.set_array([])

        # Add the colorbar
        cbar = ax[0].figure.colorbar(sm, ax=ax, orientation='vertical')
        cbar.set_label(self.raw.get_xlabel())
        fig.supxlabel("Iteration")
        ax[0].set_ylabel("Cost")
        ax[1].set_ylabel("KL divergence")
        return ax, lines

    def _save(self, path: Path, exist_ok: bool = False):
        np.save(path / 'cost.npy', self.cost)
        np.save(path / 'fluctuations.npy', self.fluctuations)
        np.save(path / 'kl.npy', self.kl)
        np.save(path / 'u.npy', self.u)

    @classmethod
    def _load(cls, path: Path) -> dict[str, np.ndarray]:
        cov = np.load(path / 'cost.npy')
        flu = np.load(path / 'fluctuations.npy')
        kl = np.load(path / 'kl.npy')
        u = np.load(path / 'u.npy')
        return {'cost': cov, 'fluctuations': flu, 'kl': kl, 'u': u}
