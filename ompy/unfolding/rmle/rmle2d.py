from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from typing import Any, Callable, Self

from .tau import from_tau, to_tau
from ... import Matrix
from ...stubs import Path
from ..result1d import Cost1D
from ..result2d import UnfoldedResult2DSimple
from .loss import kl, Loss, LossFn, KullbackLeibler
from .penalty import total_penalty
from .stubs import PenaltyFn, Optimizer
from jaxtyping import Float, Array
from ..utils import scan_tqdm
from functools import partial
import optax

type Data = Float[Array, "Ein Eg"]
type ExpectationParameter = Float[Array, "Ein Eg"]
type Mu = ExpectationParameter
type Beta = ExpectationParameter
type Contaminants = tuple[ExpectationParameter, ...]
type Empty = tuple[()]
type State = tuple[Mu, Beta | Empty, Contaminants | Empty]

type LossFn = Callable[[ExpectationParameter, Data], jnp.ndarray]

def cost(
    state: State,
    GegD: Float[Array, "Eg_true Eg_observed"],
    G_ex: Float[Array, "Ein_true Ein_observed"],
    G_eg: Float[Array, "Eg_true Eg_measured"],
    y: Data,
    loss: LossFn,
    loss_background: LossFn,
    penalties: tuple[PenaltyFn, ...],
    penalties_background: tuple[PenaltyFn, ...],
    backgrounds: tuple[jnp.ndarray, ...] = ()
) -> tuple[float, dict]:
    """

    There may be no or N backgrounds. We model the background as observations of
    the same process (beta) as the data.
        B_i ~ Poisson(beta)      [Observed backgrounds] for i in 1..N
          Y ~ Poisson(nu + beta) [Observed data]
          B ~ Poisson(beta)      [Latent background in Y] 
    There may be no or M contaminants (xi_j for j in 1..M). We can't separate the contaminants from the
    data except by very strict modelling. This is up to the user. 
    The observed data is then
        Y ~ Poisson(nu + beta + sum(xi_j))

    The penalties are applied *row-wise* because each row should be viewed
    as a normalized distribution, probability of observing a E_g given E_x: P(E_g | E_x).
    The penalties on the background is shared between all backgrounds, but independent of
    the penalties on the foreground.
    # TODO
    - Implemenet backgroudn
      - Background should optionally be folded
    - implement contaminants
    - Implement user specified summary instead of just mean
    """
    tau, beta_tau, contaminants = state
    mu = from_tau(tau)

    # The left handed product is the same for both nu and eta
    tmp = G_ex@mu
    nu = tmp@GegD

    def background_body(_) -> tuple[jnp.ndarray, float, float, float]:
        print("We have backgrounds: ", backgrounds)
        if not backgrounds:
            # We need this path to make JAX happy
            return nu, 0.0, 0.0, 0.0
    
        # Beta is shared for all backgrounds
        beta = from_tau(beta_tau)
        # loss_bg: num_bg x Ex x Eg -> Ex x Eg
        loss_bg = jnp.sum(loss_background(beta, bg) for bg in backgrounds)
        # penalty: num_bg, num_bg
        penalty, penalty_term = zip(*[penalty(beta) for penalty in penalties_background])
        alpha = nu + beta
        # loss_bg: Ex x Eg -> Ex -> ()
        loss_bg = jnp.mean(jnp.sum(loss_bg, axis=1))
        penalty_bg = jnp.mean(jnp.sum(penalty, axis=1))
        penalty_terms = jnp.mean(jnp.sum(penalty_term, axis=1))
        return alpha, loss_bg, penalty_bg, penalty_terms

    nu, loss_bg, penalty_bg, penalty_bg_terms = jax.lax.cond(len(backgrounds) > 0, background_body, lambda _: (nu, 0.0, 0.0, 0.0), None)

    likelihood_body = loss(nu, y)
    loglike_per_instance = jnp.sum(likelihood_body, axis=1)
    loglike = jnp.mean(loglike_per_instance)

    def eta_nop(_):
        return 0.0, 0.0

    def eta_body(_):
        print("We have penalties: ", penalties)
        eta = tmp@G_eg
        distribution = eta / (jnp.sum(eta, axis=1, keepdims=True) + 1e-10)
        # Penalties must be taken for each row, then summarized by e.g. the mean
        total, partial = total_penalty(penalties, mu, eta, distribution, axis=1)
        return total, partial

    penalty, penalty_terms = jax.lax.cond(len(penalties) > 0, eta_body, eta_nop, None)

    cost = loglike + loss_bg + penalty_bg + penalty

    aux = {"loglike": loglike, "penalty": penalty_terms, "penalty_bg": penalty_bg, "loss_bg": loss_bg}

    return cost, aux

def unfold(*,
    data: OptimizationData,
    components: OptimizationComponents,
    settings: OptimizationSettings,
) -> OptimizationResult:

    run_optimization = make_lower(data, components, settings)

    params, loss_state = run_optimization()
    # Convert back from tau
    tau, beta_tau, contaminants = params
    loglike, penalty, total_cost = loss_state
    mu = from_tau(tau)
    beta = from_tau(beta_tau) if beta_tau else None

    result = OptimizationResult(
        prototype=data.prototype,
        mu=mu,
        beta=beta,
        total_cost=total_cost,
        loglike=loglike,
        penalty=penalty,
    )

    return result

def make_lower(
        data: OptimizationData,
        components: OptimizationComponents,
        settings: OptimizationSettings,
):
    iterations = settings.iterations

    G_eg = data.G_eg  
    G_ex = data.G_ex
    D = data.D
    raw = data.raw
    backgrounds = tuple(background for background in data.backgrounds)
    GegD = D@G_eg

    mask = components.mask
    loss = components.loss
    if hasattr(loss, "closure"):
        loss = loss.closure()

    loss_background = components.loss_background
    if hasattr(loss_background, "closure"):
        loss_background = loss_background.closure()

    penalties = components.penalties
    penalties_background = components.penalties_background

    mu_initial = to_tau(components.initial)
    
    beta_initial = mu_initial if backgrounds else ()
    contaminant_initial = ()

    params_init = (mu_initial, beta_initial, contaminant_initial)

    optimizer = settings.optimizer

    opt_state_init = optimizer.init(params_init)
    cost_closure = partial(cost, GegD=GegD, G_eg=G_eg, G_ex=G_ex, y=raw,
                            loss=loss, loss_background=loss_background,
                            penalties=penalties,
                            penalties_background=penalties_background,
                            backgrounds=backgrounds)

    value_and_grad = jax.jit(jax.value_and_grad(cost_closure, has_aux=True))

    #@jax.jit
    def lower(params_init, opt_state_init, iterations: int):
        print(iterations)

        #@jax.jit
        @scan_tqdm(iterations, leave=settings.leave_tqdm)
        def body(state, i):
            params, opt_state, loss_state = state
            (loss, aux), grads = value_and_grad(params)
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            # Apply mask
            u = params[0]
            u = u.at[mask].set(0)
            params = (u, params[1], params[2])

            # Update loss state
            loglike, penalty, total_cost = loss_state
            total_cost = total_cost.at[i].set(loss)

            loglike = loglike.at[i].set(aux["loglike"])
            penalty = penalty.at[i].set(aux["penalty"])
            loss_state = (loglike, penalty, total_cost)

            return (params, opt_state, loss_state), loss

        loglike = jnp.zeros(iterations)
        penalty = jnp.zeros(iterations)
        total_cost = jnp.zeros(iterations)

        loss_state = (loglike, penalty, total_cost)

        (params, opt_state, loss_state), loss = jax.lax.scan(
            body,
            (params_init, opt_state_init, loss_state),
            xs=jnp.arange(iterations)
        )
        return params, loss_state
    return lambda : lower(params_init, opt_state_init, iterations)


def _adam():
    # Update Adam moments
    mean = beta1 * mean + (1 - beta1) * g
    var = beta2 * var + (1 - beta2) * (g**2)

    # Correct bias
    mean_cor = mean / (1 - beta1 ** (i + 1))
    var_cor = var / (1 - beta2 ** (i + 1))

    # Compute update
    v = lr * mean_cor / (jnp.sqrt(var_cor) + eps)
    u = u - v


@dataclass(kw_only=True)
class OptimizationComponents:
    """
    A dataclass for storing optimization components.

    Optimization components are components that are parameters of the optimization,
    but not of the optimizer or optimization process. The OptimizationSettings class is for the optimizer parameters.

    This class holds the raw data, initial guess, mask, and optional background for
    optimization. All arrays must have the same length as the raw data.
    """
    initial: Float[Array, "Ein Eg"]
    # The mask of the data to optimize
    mask: Float[Array, "Ein Eg"]
    loss: LossFn | Loss = KullbackLeibler()
    loss_background: LossFn | Loss = KullbackLeibler()
    penalties: tuple[PenaltyFn, ...] = ()
    penalties_background: tuple[PenaltyFn, ...] = ()

    def __post_init__(self):
        self.initial = into_array(self.initial)
        self.mask = into_array(self.mask)

        if self.initial.shape != self.mask.shape:
            raise ValueError("Initial and mask must have the same shape")

    def __len__(self):
        return len(self.initial)

@dataclass(kw_only=True)
class OptimizationSettings:
    """
    Settings for optimization using Adam optimizer and other optimization settings.

    This class holds parameters for the Adam optimizer and other optimization settings.
    """
    # Optimisation parameters
    optimizer: Optimizer
    iterations: int = 10
    # General parameters
    leave_tqdm: bool = True
    disable_tqdm: bool = False

    def __post_init__(self):
        self.iterations = int(self.iterations)

    @classmethod
    def from_kwargs(cls, optimizer: Optimizer, **kwargs) -> Self:
        # We pop all keys from the dict, ensuring they do not remain in kwargs
        keys = {
            "iterations",
            "leave_tqdm",
            "disable_tqdm",
        }
        values = {k: kwargs.pop(k) for k in keys if k in kwargs}
        return cls(optimizer=optimizer, **values)

@dataclass(kw_only=True)
class OptimizationData:
    # Input data
    raw: Float[Array, "Ein Eg"]
    backgrounds: tuple[Float[Array, "Ein Eg"], ...] = ()
    # Model of the folding process
    D: Float[Array, "Ein Eg"]
    G_eg: Float[Array, "Ein Eg"]
    G_ex: Float[Array, "Ein Eg"]
    # Prototype of the unfolded spectrum
    prototype: Matrix
    # Contaminants
    contaminants: tuple[Float[Array, "Ein Eg"], ...] = ()

    def __post_init__(self):
        self.D = into_array(self.D)
        self.G_eg = into_array(self.G_eg)
        self.G_ex = into_array(self.G_ex)
        self.raw = into_array(self.raw)
        self.backgrounds = tuple(into_array(background) for background in self.backgrounds)

        for i, background in enumerate(self.backgrounds):
            if self.raw.shape != background.shape:
                raise ValueError(f"Raw and background #{i} must have the same shape")

@dataclass(kw_only=True)
class OptimizationResult:
    prototype: Matrix
    mu: Matrix
    beta: Matrix | None = None
    total_cost: jnp.ndarray
    loglike: jnp.ndarray
    penalty: jnp.ndarray

    def __post_init__(self):
        self.mu = self.prototype.clone(values=self.mu)
        if self.beta is not None:
            self.beta = self.prototype.clone(values=self.beta)


@dataclass(kw_only=True)
class RMLEResult2D(Cost1D, UnfoldedResult2DSimple):
    def _save(self, path: Path, meta: dict[str, Any], exist_ok: bool = False):
        Cost1D._save(self, path, meta, exist_ok)
        UnfoldedResult2DSimple._save(self, path, meta, exist_ok)

    @classmethod
    def _load(cls, path: Path, meta: dict[str, Any]) -> dict[str, np.ndarray | Matrix]:
        a = Cost1D._load(path, meta)
        b = UnfoldedResult2DSimple._load(path, meta)
        return a | b


def into_array(x: Matrix | np.ndarray | jnp.ndarray) -> jnp.ndarray:
    if hasattr(x, "values"):
        x = x.values
    return jnp.asarray(x)