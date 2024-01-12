
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap, value_and_grad
from jax.example_libraries import optimizers
from tqdm.autonotebook import tqdm, trange
from .product import index_map, nld_T_product
import numpy as np
from .. import Matrix, Vector
from ..array.ufunc import exeg_to_efeg
from ..stubs import array1D, array2D, Axes
from dataclasses import dataclass
from typing import Any, TypeVar
from ..helpers import make_ax
from .jax_kl import kl_div  # TODO: jax.scipy.special.kl_div is not released yet
from scipy.stats import gaussian_kde
from matplotlib.colorbar import Colorbar

"""
Instead of solving P(Ex, Eg) \propto rho(Ex-Eg)*T(Eg),
we can solve P(Ef, Eg) \propto rho(Ef)*T(Eg), making the multiplication
much more friendly to vectorization. Equivalent to going from (Ex, Eg) -> (Ef, Eg)
and solving the problem there. This works very well.
Convergence within ~500 steps, taking a fraction of a second in gpu and a second or two on cpu.
TODO:
    -[ ] Bookkeeping for bootstraps
    -[ ] Grid search for Ex/Eg bounds. Compare loss.
         - how to propagate that uncertainty?
    -[ ] Simpler result class to only save nld, gsf and FG.
    -[ ] Does choice of loss impact normalization group?
    -[ ] Generalize exeg_to_efeg ++
    -[ ] Allow for different loss functions
    -[ ] Check sensitivity to initial conditions
    -[ ] Check sensitivity to optimizer
    -[ ] Check sensitivity to linear/log space
    -[ ] P.sum(Ex) is always equal to FG.sum(Ex), but P.sum(Eg) is never. Why?
    -[ ] Don't we actually want to solve P = rho*T where G'@P@G = FG?
         -At least indicate the smoothing as uncertainty in X direction
         -> FG = G'@rho * T@G
         -> Should normalization take G into account? I.e. at low Ex, discrete
            levels are fit taking into account large Ex width.
            Or does that happen automatically?
    - Gaussian mixture for peak removal.
    -[ ] The area under gaussian peaks where the number of levels is known (1, 2?)
         is simply the number of levels. Can be used to normalize?
         Same as cumulative fit?
    -[ ] The relative residuals show that the model "underestimates" gaussian peaks,
         overestimates the continuum of diagonals, perhaps to compensate?
         Maybe aggressive smoothing on diagonals in (Ef, Eg) to ensure continuum
    -[ ] Regularize by enforcing G'@rho and T@G. Other regularizations?
    -[ ] Put the error plot into its own class
"""



@dataclass
class DecompositionResult:
    rho: Vector
    T: Vector
    N: int
    optimizer: Any
    FG: Matrix
    P: Matrix
    loss: np.ndarray

    def plot_loss(self, ax: Axes | None = None, **kwargs):
        ax = make_ax(ax)
        ax.plot(self.loss, **kwargs)
        return ax

    def plot(self, ax: Axes | None = None, **kwargs):
        ax: np.ndarray = make_ax(ax, ncols=2, constrained_layout=True)  # type: ignore
        self.rho.plot(ax=ax[0], **kwargs)
        self.T.plot(ax=ax[1], **kwargs)
        return ax

    def plot_compare(self, ax: Axes | None = None, abs_kwargs: dict | None = None,
                     rel_kwargs: dict | None = None, **kwargs):
        ax: np.ndarray = make_ax(ax, nrows=2, ncols=2, constrained_layout=True,
                                 sharex=True, sharey=True,
                                 figsize=(10, 10))  # type: ignore
        ax = np.ravel(ax)
        self.FG.plot(ax=ax[0], **kwargs)
        self.P.plot(ax=ax[1], **kwargs)
        err = self.FG - self.P
        rel_err = err / self.FG


        abs_kwargs = {} if abs_kwargs is None else abs_kwargs
        rel_kwargs = {} if rel_kwargs is None else rel_kwargs

        cbar = error_plot(err, ax=ax[2], **abs_kwargs)
        cbar.ax.set_ylabel('abs err')
        cbar = error_plot(rel_err, ax=ax[3], **rel_kwargs)
        cbar.ax.set_ylabel('rel err')


        xlabel = ax[0].get_xlabel()
        ylabel = ax[0].get_ylabel()
        for i in range(4):
            ax[i].set_xlabel('')
            ax[i].set_ylabel('')

        ax[0].figure.supxlabel(xlabel)
        ax[0].figure.supylabel(ylabel)
        return ax

    def plot_compare_2(self, ax: Axes | None = None, abs_kwargs: dict | None = None,
                     rel_kwargs: dict | None = None, **kwargs):

        ax: np.ndarray = make_ax(ax, nrows=2, ncols=2, constrained_layout=True,
                                 sharex=True, sharey=True,
                                 figsize=(10, 10))  # type: ignore
        ax = np.ravel(ax)
        FG = exeg_to_efeg(self.FG, cut=True)
        P = exeg_to_efeg(self.P, cut=True)
        ef0 = P.Ef[0]
        FG = FG.loc[f'>{ef0}':, :]
        FG.plot(ax=ax[0], **kwargs)
        P.plot(ax=ax[1], **kwargs)
        err = FG - P
        rel_err = err / FG

        abs_kwargs = {} if abs_kwargs is None else abs_kwargs
        rel_kwargs = {} if rel_kwargs is None else rel_kwargs

        cbar = error_plot(err, ax=ax[2], **abs_kwargs)
        cbar.ax.set_ylabel('abs err')
        cbar = error_plot(rel_err, ax=ax[3], **rel_kwargs)
        cbar.ax.set_ylabel('rel err')


        xlabel = ax[0].get_xlabel()
        ylabel = ax[0].get_ylabel()
        for i in range(4):
            ax[i].set_xlabel('')
            ax[i].set_ylabel('')

        ax[0].figure.supxlabel(xlabel)
        ax[0].figure.supylabel(ylabel)
        return ax

def error_plot(mat: Matrix, ax, **kwargs):
    # Need to remove 0 because of the zeros representing "non-data"
    y = mat[np.isfinite(mat) & (mat.values != 0)].ravel()
    vmin, vmax = IQR_range(y, kwargs.pop('factor', 1.5))
    v_abs = max(abs(vmin), abs(vmax))
    default = dict(cmap='RdBu_r', vmin=-v_abs, vmax=v_abs)
    kwargs = default | kwargs
    vmin, vmax = kwargs['vmin'], kwargs['vmax']

    ax, (_, cbar) = mat.plot(ax=ax, **kwargs)
    IQR_cbar(y, cbar, vmin, vmax)
    return cbar

    #TODO @property nld and gsf

def IQR_cbar(data, ax, vmin=None, vmax=None, **kwargs):
    if isinstance(ax, Colorbar):
        ax = ax.ax

    y = data[np.isfinite(data)].ravel()
    vmin_, vmax_ = IQR_range(y)
    if vmin is None:
        vmin = vmin_
    if vmax is None:
        vmax = vmax_

    y = y[(y > vmin) & (y < vmax)]
    kde = gaussian_kde(y)
    x = np.linspace(vmin, vmax, 1000)
    y = kde(x)
    y /= y.max()
    y*= 0.8

    default = {'color': 'k'}
    kwargs = default | kwargs
    ax.plot(y, x, **kwargs)

    nans = sum(~np.isfinite(data))
    data = data[np.isfinite(data)]
    lower = sum(data < vmin) / len(data)
    higher = sum(data > vmax) / len(data)

    print(f'{higher*100:.1f}%')

    ax.text(0.5, 1.05, f'{higher*100:.1f}%', transform=ax.transAxes, ha='center', va='bottom')

    # Add a label to the bottom of the colorbar using 'text'
    # The coordinates (0.5, -0.05) place the text just below the colorbar
    ax.text(0.5, -0.05, f'{lower*100:.1f}%', transform=ax.transAxes, ha='center', va='top')


def IQR_range(data, factor=1.5):
    # Calculate the IQR
    data = data[np.isfinite(data)]
    q25, q75 = np.percentile(data, [25, 75])
    iqr = q75 - q25
    
    # Define color limits to be a certain factor beyond the IQR
    vmin = q25 - factor * iqr
    vmax = q75 + factor * iqr
    return vmin, vmax


# Define the loss function
@jit
def loss_fn(params, P, mask):
    rho, T = params
    P_hat = jnp.outer(rho, T)
    #loss = (P - P_hat)**2
    #loss = (P - P_hat)**2 / P
    loss = kl_div(P, P_hat)
    return jnp.sum(jnp.where(mask, loss, 0.0))


def setup(FG: Matrix) -> tuple[array2D, array1D, array1D, array1D]:
    assert FG.Ex_index.is_uniform(), "Ex must be uniform"
    assert FG.Eg_index.is_uniform(), "Eg must be uniform"
    dEx = FG.Ex_index.step(0)
    dEg = FG.Eg_index.step(0)
    if not np.isclose(dEx, dEg):
        raise ValueError("Ex and Eg must have the same bin width")
    P = exeg_to_efeg(FG, cut=True)
    return P.values, P.Ef, P.Eg, FG.Ex


# Begin the optimization
def optimize(FG: Matrix, N: int = 500, optimizer=optimizers.rmsprop_momentum(1e-5),
             normalize: bool = True, disable_tqdm: bool = False):
    if normalize:
        FG = FG / FG.sum()
    #FG = row_normalize(FG)  # type: ignore

    P, Ef, Eg, Ex = setup(FG)
    #imap = jnp.array(index_map(Ex, Eg, Ef)).T
    Ef = jnp.array(Ef, dtype='float32')
    Eg = jnp.array(Eg, dtype='float32')
    Ex = jnp.array(Ex, dtype='float32')

    # Initial parameters
    # If they are too high, the optimizer will fail
    #rho_0 = jnp.exp(0.05*Ef / 1000)
    #T_0 = jnp.exp(0.06*Eg / 1000)
    a = np.sqrt(np.mean(P.sum(axis=1)))
    b = np.sqrt(np.mean(P.sum(axis=0)))
    a = 1e-8
    b = 1e-8
    rho_0 = a*jnp.ones_like(Ef)
    T_0 = b*jnp.ones_like(Eg)

    #rho_0 = jnp.log(rho_0)
    #v = jnp.array(np.random.rand(len(nld)))
    #w = jnp.array(np.random.rand(len(gsf)))
    params_init = (rho_0, T_0)

    # Optimizer setup
    opt_init, opt_update, get_params = optimizer
    opt_state = opt_init(params_init)

    # Loss function gradient
    loss_and_grad = jit(value_and_grad(loss_fn))

    #P = np.log(P)
    # lift optional mask to args
    mask = np.isfinite(P) & (P > 1e-10)
    mask = jnp.array(mask)
    #P = P_data
    P = jnp.array(P)

    # Update step
    @jit
    def step(i, opt_state):
        params = get_params(opt_state)
        loss, g = loss_and_grad(params, P, mask)
        return loss, opt_update(i, g, opt_state)

    losses = np.zeros(N)

    trange_ = trange
    if disable_tqdm:
        trange_ = range
    for i in trange_(N):  # Arbitrary number of steps
        loss, opt_state = step(i, opt_state)
        losses[i] = loss

    # There seems to be a small floating point error in jnp to np
    # Ensure exact binwidths

    Eg = np.linspace(Eg[0], Eg[-1], len(Eg))
    Ef = np.linspace(Ef[0], Ef[-1], len(Ef))

    rho, T = get_params(opt_state)
    rho = Vector(E=Ef, values=np.array(rho), ylabel='unormalized nld')
    T = Vector(E=Eg, values=np.array(T), xlabel=r'$E_\gamma$', ylabel='unormalized T')

    P = nld_T_product(rho, T, Ex=np.asarray(Ex))

    return DecompositionResult(rho=rho, T=T, N=N, optimizer=optimizer, FG=FG, P=P, loss=losses)

