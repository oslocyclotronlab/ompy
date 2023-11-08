import jax
import jax.numpy as jnp
from jax import grad, jit, vmap, value_and_grad
from jax.example_libraries import optimizers
from tqdm.autonotebook import tqdm, trange
from .product import index_map, nld_T_product
import numpy as np
from .. import Matrix, Vector
from ..stubs import array1D, array2D
from dataclasses import dataclass
from typing import Any, TypeVar

"""
Instead of solving P(Ex, Eg) \propto rho(Ex-Eg)*T(Eg),
we can solve P(Ef, Eg) \propto rho(Ef)*T(Eg), making the multiplication
much more friendly to vectorization. Equivalent to going from (Ex, Eg) -> (Ef, Eg)
and solving the problem there
"""

@dataclass
class DecompositionResult:
    nld: Vector
    gsf: Vector
    N: int
    optimizer: Any
    FG: Matrix
    P: Matrix


# Calculate P based on nld and gsf, vectorized over Ex and Eg
@jit
def compute(nld, gsf, Ef, Eg, Ex):
    """

    Turns out to be faster to compute index in loop instead of
    using lookup table. Probably memory access time is the bottleneck.

    """
    # def single_compute(k, gsf_value):
    #    return jnp.where(k >= 0, nld[k] + gsf_value, 0)
    #return vmap(vmap(single_compute, (0, None)), (0, None))(Ef_indices, gsf)
    #imap = Ef_indices
    # def single_compute(i, j):
    #    k = imap[i, j]
    #    return jnp.where(k >= 0, nld[k] + gsf[j], 0)
    #return vmap(vmap(single_compute, (0, None)), (None, 0))(jnp.arange(imap.shape[0])
    #                                                        , jnp.arange(imap.shape[1]))
    def single_compute(ex, eg, j):  # 'j' is the current index in 'Eg'
        ef = ex - eg
        k = jnp.searchsorted(Ef, ef, side='right') - 1
        return jnp.where(k >= 0, nld[k] + gsf[j], 0)  # corrected to multiplication

    # 'j' will automatically take on the value of the current index in 'Eg'
    return vmap(vmap(single_compute, (None, 0, 0)), (0, None, None))(Ex, Eg, jnp.arange(Eg.shape[0]))
    
def logsumexp(x):
    c = jnp.max(x)
    return c + jnp.log(jnp.nansum(jnp.exp(x - c)))

def kl_divergence_log_space(log_p, log_q):
    softmax_log_p = jnp.exp(log_p - logsumexp(log_p))
    return jnp.nansum(softmax_log_p * (log_p - log_q))
 
# Define the loss function
def loss_fn(params, Ef, Eg, Ex, P_data):
    nld, gsf = params
    P_pred = compute(nld, gsf, Ef, Eg, Ex)
    #P_pred = jnp.exp(P_pred)
    #P_pred = jnp.exp(P_pred)
    #P_data = jnp.exp(P_data)
    #P_pred_safe = jnp.clip(P_pred, a_min=1e-15, a_max=None)
    P_data_safe = jnp.clip(P_data, a_min=1e-15, a_max=None)
    #P_pred_safe = jnp.clip(P_pred, a_min=-10, a_max=None)
    #P_data_safe = jnp.clip(P_data, a_min=-10, a_max=None)
    return jnp.nansum((P_data - P_pred) ** 2 / P_data_safe)  # L2 loss
    #return kl_divergence_log_space(P_pred, P_data)
    #return jnp.sum(P_data_safe * jnp.log(P_data_safe / P_pred_safe) - P_data_safe + P_pred_safe)
    #return jnp.sum((P_data_safe - P_pred_safe)**2 / P_data_safe)


def setup(FG: Matrix) -> tuple[array1D, array1D, array1D]:
    assert FG.Ex_index.is_uniform(), "Ex must be uniform"
    assert FG.Eg_index.is_uniform(), "Eg must be uniform"
    dEx = FG.Ex_index.step(0)
    dEg = FG.Eg_index.step(0)
    if not np.isclose(dEx, dEg):
        raise ValueError("Ex and Eg must have the same bin width")
    dEf = dEx
    Ex, Eg = FG.Ex, FG.Eg
    Ef_min = Ex.min() - Eg.max()  # minimum possible final energy
    Ef_min = max(Ef_min, 0)  # Final energy cannot be negative
    Ef_max = Ex.max() - Eg.min()  # maximum possible final energy
    Ef = np.arange(Ef_min, Ef_max + dEf, dEf)
    return Ex, Eg, Ef

T = TypeVar('T', bound=np.ndarray)
def row_normalize(A: T) -> T:
    return A / A.sum(axis=1, keepdims=True)


# Begin the optimization
def optimize(FG: Matrix, N: int = 10_000, optimizer=optimizers.adam(1e-3),
             pre_normalize: bool = True, disable_tqdm: bool = False):
    if pre_normalize:
        FG = row_normalize(FG)  # type: ignore

    Ex, Eg, Ef = setup(FG)
    #imap = jnp.array(index_map(Ex, Eg, Ef)).T
    Ef = jnp.array(Ef, dtype='float32')
    Eg = jnp.array(Eg, dtype='float32')
    Ex = jnp.array(Ex, dtype='float32')

    # Initial parameters
    nld_0 = 0.1+jnp.ones_like(Ef)
    gsf_0 = 0.5+jnp.ones_like(Eg)
    #v = jnp.array(np.random.rand(len(nld)))
    #w = jnp.array(np.random.rand(len(gsf)))
    params_init = (nld_0, gsf_0)

    # Optimizer setup
    opt_init, opt_update, get_params = optimizer
    opt_state = opt_init(params_init)

    # Loss function gradient
    loss_and_grad = jit(value_and_grad(loss_fn))

    P = np.log(FG)
    P[~np.isfinite(P)] = -15
    #P = P_data
    P = jnp.array(P)

    # Update step
    @jit
    def step(i, opt_state):
        params = get_params(opt_state)
        loss, g = loss_and_grad(params, Ef, Eg, Ex, P)
        return loss, opt_update(i, g, opt_state)

    losses = np.zeros(N)

    trange_ = trange
    if disable_tqdm:
        trange_ = range
    for i in trange_(N):  # Arbitrary number of steps
        loss, opt_state = step(i, opt_state)
        losses[i] = loss

    nld1, gsf1 = get_params(opt_state)
    nld = Vector(E=np.asarray(Ef), values=np.exp(nld1))
    gsf = Vector(E=np.asarray(Eg), values=np.exp(gsf1))

    P = nld_T_product(nld, gsf, Ex=np.asarray(Ex))

    return DecompositionResult(nld=nld, gsf=gsf, N=N, optimizer=optimizer, FG=FG, P=P)

