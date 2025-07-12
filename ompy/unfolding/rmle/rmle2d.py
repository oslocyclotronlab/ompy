from __future__ import annotations

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np
from typing import Any, Self, TYPE_CHECKING, Iterable

from .tau import from_tau, to_tau
from ... import Matrix
from ...stubs import Path
from ..result1d import Cost1D
from ..result2d import UnfoldedResult2DSimple
from .loss import Loss, LossFn, KullbackLeibler
from jaxtyping import Float, Array
from ..utils import scan_tqdm
from functools import partial
from .stubs import Background2D, Beta2D, Mu2D, Tau2D, GegDMatrix, GexMatrix, State2D, Data2D
from .lossmodel import ModelLoss
from .utils import pytree_dataclass, into_array
from .tau import TauMap, TAU_MAP
from .contaminant2d import Contaminant2D

try:
    import optax
except ImportError:
    pass

type PenaltyFn = Any

if TYPE_CHECKING:
    from .rmle1d import Settings

@pytree_dataclass
class BackgroundModel2D:
    loss: Loss = ModelLoss()
    backgrounds: tuple[Background2D, ...] = ()
    do_fold: bool = True


    def cost(self, beta: Beta2D) -> tuple[float, float, tuple[float, ...]]:
        row_wise_loss = sum(tuple(jnp.sum(self.loss.loss(beta, bg), axis=1) for bg in self.backgrounds))
        loss = jnp.mean(row_wise_loss)
        # Penalties here
        #penalty, penalty_term = zip(*[penalty(beta) for penalty in penalties_background])
        #loss = jnp.mean(jnp.sum(loss, axis=1))
        return loss

    def __len__(self) -> int:
        return len(self.backgrounds)

    def __getitem__(self, idx: int) -> Background2D:
        return self.backgrounds[idx]


def cost(
    state: State2D,
    GegD: Float[Array, "Eg_true Eg_observed"],
    G_ex: Float[Array, "Ein_true Ein_observed"],
    G_eg: Float[Array, "Eg_true Eg_measured"],
    y: Data2D,
    loss: ModelLoss,
    background: BackgroundModel2D,
    tau_map: TauMap[Mu2D, Tau2D],
    efficiency: Float[Array, "Eg"] | None,
    contaminant_models: tuple[Contaminant2D, ...]
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


    loss_contaminants = 0.0
    if len(contaminants) > 0:
        print("has contaminants", len(contaminants))
        for contaminant, model in zip(contaminants, contaminant_models):
            #contaminant_nu, tmp = model.loss(from_tau(contaminant))
            #loss_contaminants += tmp
            #nu = nu + contaminant_nu
            contaminant_mu, tmp = model.loss(contaminant)
            mu = mu + contaminant_mu
            loss_contaminants += tmp

    # The left handed product is the same for both nu and eta
    tmp = G_ex@mu
    nu = tmp@GegD

    if len(background) > 0:
        print("has background")
        beta = from_tau(beta_tau)
        if background.do_fold:
            beta = G_ex@beta@G_eg
            if efficiency is not None:
                beta = beta / efficiency[None, :]
        loss_bg = background.cost(beta)
        nu = nu + beta
    else:
        print("no background")
        loss_bg = 0.0


    likelihood_body = loss(nu, y)
    loglike_per_instance = jnp.sum(likelihood_body, axis=1)
    loglike = jnp.mean(loglike_per_instance)

    #def eta_body(_):
    #    print("We have penalties: ", penalties)
    #    eta = tmp@G_eg
    #    distribution = eta / (jnp.sum(eta, axis=1, keepdims=True) + 1e-10)
    #    # Penalties must be taken for each row, then summarized by e.g. the mean
    #    total, partial = total_penalty(penalties, mu, eta, distribution, axis=1)
    #    return total, partial

    #penalty, penalty_terms = jax.lax.cond(len(penalties) > 0, eta_body, eta_nop, None)
    penalty_terms = 0.0
    penalty_bg = 0.0

    cost = loglike + loss_bg + penalty_terms + loss_contaminants

    aux = {"loglike": loglike, "penalty": penalty_terms, "penalty_bg": penalty_bg, "loss_bg": loss_bg}

    return cost, aux

def unfold(*,
    data: OptimizationData,
    components: OptimizationComponents,
    settings: Settings,
) -> OptimizationResult:

    run_optimization = make_lower(data, components, settings)

    params, aux = run_optimization()
    # Convert back from tau
    mu, beta, contaminants = params

    result = OptimizationResult(
        prototype=data.prototype,
        mu=mu,
        beta=beta,
        contaminants=contaminants,
        aux=aux
    )

    return result

def make_lower(
        data: OptimizationData,
        components: OptimizationComponents,
        settings: Settings,
):
    iterations = settings.iterations

    G_eg = data.G_eg  
    G_ex = data.G_ex
    D = data.D
    raw = data.raw
    GegD = D@G_eg

    mask = components.mask

    mu_initial = settings.tau_map.to_tau(components.initial)

    if data.has_background:
        # Mean of the backgrounds is the best estimate
        if len(data.background) == 1:
            beta_initial = data.background[0]
        else:
            beta_initial = jnp.mean(data.background.backgrounds, axis=0)
        beta_initial = settings.tau_map.to_tau(beta_initial)
    else:
        beta_initial = ()

    
    #contaminant_initial = tuple(settings.tau_map.to_tau(contaminant.setup_initial())
    #                            for contaminant in data.contaminants)
    contaminant_initial = tuple(contaminant.setup_initial()
                                for contaminant in data.contaminants)

    params_init = (mu_initial, beta_initial, contaminant_initial)

    optimizer = settings.optimizer

    opt_state_init = optimizer.init(params_init)
    cost_closure = partial(cost, GegD=GegD, G_eg=G_eg, G_ex=G_ex, y=raw,
                            loss=components.loss,
                            background=data.background,
                            tau_map=settings.tau_map,
                            efficiency=data.efficiency,
                            contaminant_models=data.contaminants)

    value_and_grad = jax.jit(jax.value_and_grad(cost_closure, has_aux=True))

    #@jax.jit
    def lower(params_init, opt_state_init, iterations: int):

        @jax.jit
        @scan_tqdm(iterations, leave=settings.leave_tqdm)
        def body(state, i):
            params, opt_state, loss_state = state

            #jax.debug.print(opt_state[0])
            
            (loss, aux), grads = value_and_grad(params)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            # Apply mask
            u = params[0]
            u = u.at[mask].set(0)
            params = (u, params[1], params[2])

            # Update loss state
            loss_state = jax.tree.map(
                lambda state, val: state.at[i].set(val),
                loss_state,
                {k: v for k, v in aux.items() if k in loss_state}
            )

            return (params, opt_state, loss_state), loss

        # TODO Need a way to get all losses to construct the arrays
        losses = ('loglike', 'penalty', 'loss_bg')
        loss_state = {loss: jnp.zeros(iterations) for loss in losses}

        (params, opt_state, loss_state), loss = jax.lax.scan(
            body,
            (params_init, opt_state_init, loss_state),
            xs=jnp.arange(iterations)
        )
        mu_tau, beta_tau, contaminants = params
        mu = settings.tau_map.from_tau(mu_tau)
        beta = settings.tau_map.from_tau(beta_tau) if len(beta_tau) > 0 else ()
        #contaminants = tuple(settings.tau_map.from_tau(contaminant) for contaminant in contaminants)
        contaminants = tuple(model.transform_out(contaminant) for contaminant, model in zip(contaminants, data.contaminants))
        return (mu, beta, contaminants), loss_state
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
    loss: ModelLoss = ModelLoss()

    def __post_init__(self):
        self.initial = into_array(self.initial)
        self.mask = into_array(self.mask)

        if self.initial.shape != self.mask.shape:
            raise ValueError("Initial and mask must have the same shape")

    def __len__(self):
        return len(self.initial)


@dataclass(kw_only=True)
class OptimizationData:
    # Input data
    raw: Data2D
    background: BackgroundModel2D = BackgroundModel2D()
    # Model of the folding process
    D: Float[Array, "Ein Eg"]
    G_eg: Float[Array, "Ein Eg"]
    G_ex: Float[Array, "Ein Eg"]
    # Prototype of the unfolded spectrum
    prototype: Matrix
    # Contaminants
    contaminants: tuple[Contaminant2D, ...] = ()
    efficiency: Float[Array, "Eg"] | None = None

    def __post_init__(self):
        self.D = into_array(self.D)
        self.G_eg = into_array(self.G_eg)
        self.G_ex = into_array(self.G_ex)
        self.raw = into_array(self.raw)

        if not isinstance(self.background, BackgroundModel2D):
            if not isinstance(self.background, Iterable):
                self.background = BackgroundModel2D(backgrounds=(into_array(self.background),))
            else:
                self.background = BackgroundModel2D(backgrounds=tuple(into_array(bg) for bg in self.background))

        for i, background in enumerate(self.background.backgrounds):
            if self.raw.shape != background.shape:
                raise ValueError(f"Raw and background #{i} must have the same shape")

        if self.efficiency is not None:
            self.efficiency = into_array(self.efficiency)
            print(self.efficiency.shape)
            if self.efficiency.shape[0] != self.raw.shape[1]:
                raise ValueError("Efficiency must have the same shape as the raw data"
                                 f"Efficiency: {self.efficiency.shape}, Raw: {self.raw.shape}")

        if not isinstance(self.contaminants, Iterable):
            self.contaminants = tuple(self.contaminants)

    @property
    def has_background(self) -> bool:
        return len(self.background.backgrounds) > 0

@dataclass(kw_only=True)
class OptimizationResult:
    prototype: Matrix
    mu: Matrix
    beta: Matrix | None = None
    contaminants: tuple[Matrix, ...] = ()
    aux: dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self):
        self.mu = self.prototype.clone(values=self.mu, name='mu')
        if len(self.beta) > 0:
            self.beta = self.prototype.clone(values=self.beta, name='beta')
        else:
            self.beta = None

        for key, value in self.aux.items():
            self.aux[key] = np.asarray(value)

        #contaminants = []
        #for contaminant in self.contaminants:
        #    contaminants.append(self.prototype.clone(values=contaminant, name='contaminant'))
        #self.contaminants = tuple(contaminants)


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

