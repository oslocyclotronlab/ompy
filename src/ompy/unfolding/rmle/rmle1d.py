from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterable, Self
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jaxtyping import Array, Float
from functools import partial
from ... import Vector
from ...stubs import Path
from ..result1d import Cost1D, UnfoldedResult1DSimple
from ..utils import loop_tqdm
from .contaminant1d import Contaminant1D
from .stubs import Optimizer, Background1D, Mu1D, Tau1D, Mask1D
from .stubs import Beta1D as Beta
from .stubs import Contaminants1D as Contaminants
from .stubs import Data1D as Data
from .stubs import ExpectationParameter1D as ExpectationParameter
from .stubs import Nu1D, DMatrix, GegMatrix, GegDMatrix
from .stubs import State1D as State
from .tau import TAU_MAP, TauMap
from .utils import pytree_dataclass, into_array
from .lossmodel import ModelLoss

if TYPE_CHECKING:
    from .contaminant1d import Contaminant1D, ContaminantModel1D

"""
User can supply the background either as array, iterable of arrays, or as a
BackgroundModel object.

        #total, partial = total_penalty(penalties, mu, eta, prop)
        # penalty = penalty + 5e-12*(jnp.sum(eta[659:809]))  # 5e-12


    # Rescaling seems to make the optimization much slower
    # penalty = alpha*onecost(mu, alpha_c)**2
    # penalty = alpha*onecost(mu / (1+jnp.abs(mu)), 0.01)**2
    # try a entropy penalty
    # prop = jax.nn.softmax(mu)
    # Ops! Want to do this on eta for sobolev
    # If
    # prop = eta / jnp.sum(eta)

    # entropy_body = prop * jnp.log(prop+1e-10)
    # prop = eta / (jnp.sum(eta) + 1e-10)
    # prop = mu

    # entropy = -jnp.sum(entropy_body)

    # penalty = alpha * entropy
    # xi_penalty = xi_pos_penalty + xi_A_penalty

    # Orthogonality penalty
    # ortho_penalty = 1e-12*jnp.sum(eta @ xi_eta)**2
"""


@dataclass(kw_only=True, frozen=True)
class BackgroundModel:
    loss: ModelLoss = ModelLoss()
    backgrounds: tuple[Background1D, ...] = ()
    do_fold: bool = False

    def __init__(self, backgrounds: Background1D | tuple[Background1D, ...] | list[Background1D] | None = None,
                 loss: ModelLoss = ModelLoss(), do_fold: bool = False):
        # Use object.__setattr__ to bypass frozen dataclass restrictions
        object.__setattr__(self, 'loss', loss)
        object.__setattr__(self, 'do_fold', do_fold)
        if backgrounds is None:
            object.__setattr__(self, 'backgrounds', ())
        elif isinstance(backgrounds, Iterable):
            object.__setattr__(self, 'backgrounds', tuple(into_array(bg) for bg in backgrounds))
        else:
            object.__setattr__(self, 'backgrounds', (into_array(backgrounds),))

    def cost(self, beta: Beta) -> tuple[float, float, tuple[float, ...]]:
        loss = sum(jnp.sum(self.loss.loss(beta, bg)) for bg in self.backgrounds)
        penalty_terms = tuple(
            jnp.sum(penalty(beta)[0]) for penalty in self.loss.penalty
        )
        penalty = sum(penalty_terms)
        return loss, penalty

    def __len__(self):
        return len(self.backgrounds)

    def __eq__(self, other: BackgroundModel) -> bool:
        if len(self) != len(other):
            return False
        same_bg = all(
            jnp.all(bg == other.backgrounds[i]) for i, bg in enumerate(self.backgrounds)
        )
        return self.loss == other.loss and same_bg

    def clone(self, backgrounds: Background1D | tuple[Background1D, ...] | list[Background1D] | None = None,
              loss: ModelLoss | None = None, do_fold: bool | None = None) -> Self:
        if backgrounds is None:
            backgrounds = self.backgrounds
        if loss is None:
            loss = self.loss
        if do_fold is None:
            do_fold = self.do_fold
        return BackgroundModel(backgrounds=backgrounds, loss=loss, do_fold=do_fold)

def flatten_background_model(model: BackgroundModel):
    children = (model.loss, model.backgrounds)  # arrays/dynamic values
    aux_data = {'do_fold': model.do_fold}  # static values
    return children, aux_data


def unflatten_background_model(aux_data, children):
    loss, backgrounds = children
    return BackgroundModel(loss=loss, backgrounds=backgrounds, do_fold=aux_data['do_fold'])


jax.tree_util.register_pytree_node(BackgroundModel, 
                                  flatten_background_model,
                                  unflatten_background_model)


def cost_contaminants(
    nu: Nu1D,
    contaminants: Contaminants,
    models: tuple[ContaminantModel1D, ...],
) -> tuple[Nu1D, float]:
    # This path will not run, but JAX needs it
    if not contaminants:
        return nu, 0.0

    xi_nu, cost_list = zip(
        *[model.cost(tau) for tau, model in zip(contaminants, models)]
    )
    nu_sum = nu + sum(xi_nu)
    cost_sum = sum(cost_list)
    return nu_sum, cost_sum


def cost(
    state: State,
    GegD: Float[Array, "Eg_true Eg_observed"],
    G_eg: Float[Array, "Eg_true Eg_measured"],
    y: Data,
    contaminant_models: tuple[ContaminantModel1D, ...],
    loss: ModelLoss,
    background: BackgroundModel,
    tau_map: TauMap[Mu1D, Tau1D],
) -> tuple[jnp.ndarray, dict]:
    tau, beta_tau, contaminants = state

    mu = tau_map.from_tau(tau)

    if len(contaminant_models) > 0:
        c = contaminant_models[0]
        mu_c, contaminant_loss = c.loss(contaminants[0])
        mu = mu + mu_c
    else:
        contaminant_loss = 0.0

    nu = mu @ GegD

    if background.backgrounds:
        beta = tau_map.from_tau(beta_tau)
        if background.do_fold:
            beta = beta @ GegD
        loss_bg, penalty_bg = background.cost(beta)
        nu = nu + beta
    else:
        loss_bg, penalty_bg = (0.0, 0.0)

    likelihood_body = loss.loss(nu, y)
    loglike = jnp.sum(likelihood_body)

    eta_penalties = ()
    eta_normalized_penalties = ()
    mu_penalties = ()
    mu_normalized_penalties = ()
    for penalty in loss.penalty:
        if penalty.target == "eta":
            eta_penalties = eta_penalties + (penalty,)
        elif penalty.target == "eta_normalized":
            eta_normalized_penalties = eta_normalized_penalties + (penalty,)
        elif penalty.target == "mu":
            mu_penalties = mu_penalties + (penalty,)
        elif penalty.target == "mu_normalized":
            mu_normalized_penalties = mu_normalized_penalties + (penalty,)
        else:
            raise ValueError(f"Unknown penalty target: {penalty.target}")

    total_penalty = 0.0
    total_penalty_magnitude = 0.0
    if mu_penalties:
        total_penalty += sum(penalty(mu) for penalty in mu_penalties)
        total_penalty_magnitude += sum(penalty(mu) for penalty in mu_penalties)

    if mu_normalized_penalties:
        mu_norm = mu / jnp.sum(mu + 1e-10)
        # penalty, penalty_magnitude = zip(*[penalty(mu_norm) for penalty in mu_normalized_penalties])
        # total_penalty += sum(penalty)
        # total_penalty_magnitude += sum(penalty_magnitude)
        for penalty in mu_normalized_penalties:
            p, p_mag = penalty(mu_norm)
            total_penalty += p.sum()
            total_penalty_magnitude += p_mag.sum()

    if eta_penalties or eta_normalized_penalties:
        # Map to eta space
        eta = mu @ G_eg
        if eta_penalties:
            penalty, penalty_magnitude = zip(
                *[penalty(eta) for penalty in eta_penalties]
            )
            total_penalty += sum(penalty)
            total_penalty_magnitude += sum(penalty_magnitude)
        if eta_normalized_penalties:
            # Rescale to get a proper probability distribution
            eta_norm = eta / jnp.sum(eta + 1e-10)

            penalty, penalty_magnitude = zip(
                *[penalty(eta_norm) for penalty in eta_normalized_penalties]
            )
            total_penalty += sum(penalty)
            total_penalty_magnitude += sum(penalty_magnitude)

    cost = loglike + total_penalty + contaminant_loss + loss_bg + penalty_bg

    aux = {
        "loglike": loglike,
        "penalty": total_penalty_magnitude,
        "contaminant_loss": contaminant_loss,
    }

    return cost, aux


def unfold(
    dynamic: DynamicData,
    settings: Settings,
    static: StaticData,
) -> OptimResult1D:
    # if bg is not None:
    #    mask = jnp.concatenate([mask, jnp.zeros_like(bg, dtype=bool)])
    #    u = jnp.concatenate([u, 1.0 + jnp.zeros_like(bg)])

    # Set up Xi for contamination
    # x, mask = setup_contaminants(data_params.contaminants, tau, components.mask)

    # The optimization function is created from a closure of all constants
    # that we never vmap over.
    lower = make_lower(
        settings,
        static,
        dynamic,
    )

    if dynamic.background.backgrounds:
        # Mean is the best guess
        beta = sum(dynamic.background.backgrounds) / len(dynamic.background.backgrounds)
    else:
        beta = None

    if static.contaminants:
        contaminants = tuple(c.setup_initial() for c in static.contaminants)
    else:
        contaminants = ()

    state = (
        dynamic.initial,
        beta,
        contaminants,
    )

    if settings.profile:
        print("Profiling...")
        print("Doing a warmup")
        lower(state)
        print("Starting profile")
        start = time.time()
        with jax.profiler.trace("/tmp/jax-trace-unfold-vec", create_perfetto_link=True):
            state, total_cost, loglike, penalty = lower(state)
        print(f"Profiling took {time.time() - start} seconds")
    else:
        state, total_cost, loglike, penalty = lower(state)

    mu, beta, contaminants = state
    if contaminants:
        p = static.contaminants[0].transform_out(contaminants[0])
        print(p)

    # Reconstitute the contaminants as vectors
    contaminant_vecs: list[Vector] = [
        model.into_vector(vector=static.prototype, params=c)
        for c, model in zip(contaminants, static.contaminants)
    ]

    result = OptimResult1D(
        prototype=static.prototype,
        mu=mu,
        total_cost=total_cost,
        loglike=loglike,
        penalty=penalty,
        beta=beta,
        xi_penalty=0,
        xi=contaminant_vecs,
    )
    return result


def make_lower(
    settings: Settings,
    data: StaticData,
    dynamic: DynamicData,
):
    """Create a closure for the optimization loop that unfolds the spectrum.

    This function creates and returns a closure that performs the main optimization loop
    using Adam optimization. The closure captures constant parameters and data that don't
    change during optimization.

    Args:
        optim_params: Optimization parameters like learning rate, iterations etc.
        data_params: Data parameters including response matrices and prototype vector

    Returns:
        A jitted function that takes initial values, raw data, background and mask and returns:
            - x: The optimized parameters in tau space
            - total_cost: Array of total cost values for each iteration
            - loglike: Array of log-likelihood values for each iteration
            - penalty: Array of penalty values for each iteration
            - xi_penalty: Array of contaminant penalty values for each iteration
    """
    # The outer scope captures constants
    # The inner scope captures variables that can be vmaped over
    iterations = settings.iterations
    G_eg = data.G_eg
    GegD = data.D @ G_eg
    # Contaminant models that have its matrices unspecified inherit the main matrices
    # no the user must provide the contaminant models
    leave_tqdm = settings.leave_tqdm
    tau_map = settings.tau_map

    cost_closure = partial(
        cost,
        GegD=GegD,
        G_eg=G_eg,
        contaminant_models=data.contaminants,
        loss=data.loss,
        background=dynamic.background,
        tau_map=tau_map,
        y=dynamic.raw,
    )
    value_and_grad = jax.jit(jax.value_and_grad(cost_closure, has_aux=True))

    if settings.print_jaxpr:
        jaxpr = jax.make_jaxpr(cost_closure)((dynamic.initial, dynamic.initial, ()))
        print(jaxpr)

    optimizer = settings.optimizer
    mask = jnp.asarray(dynamic.mask)

    type LoopState = tuple[State, optax.OptState, jnp.ndarray, jnp.ndarray]

    @jax.jit
    def lower(
        initial: State,
    ) -> tuple[State, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        loglike = jnp.zeros(iterations)
        penalty = jnp.zeros(iterations)
        # Map into tau
        mu_tau = tau_map.to_tau(initial[0])
        beta_tau = tau_map.to_tau(initial[1]) if initial[1] is not None else None
        initial = (mu_tau, beta_tau, initial[2])

        opt_state_init = optimizer.init(initial)

        @loop_tqdm(iterations, leave=leave_tqdm)
        @jax.jit
        def body_fun(i: int, state: LoopState) -> LoopState:
            params, opt_state, loglike, penalty = state
            (_, aux), g = value_and_grad(params)
            updates, opt_state = optimizer.update(g, opt_state, params=params)
            params = optax.apply_updates(params, updates)
            # Everything that is masked is zeroed out now to prevent
            # gradients from being applied
            unfolded = params[0].at[mask].set(0)
            params = (unfolded, params[1], params[2])

            loglike = loglike.at[i].set(aux["loglike"])
            penalty = penalty.at[i].set(aux["penalty"])
            return params, opt_state, loglike, penalty

        loop_state: LoopState = (initial, opt_state_init, loglike, penalty)
        res = jax.lax.fori_loop(0, iterations, body_fun, loop_state)
        state, _, loglike, penalty = res
        total_cost = loglike + penalty

        # The states are in tau space, so we need to map them back
        mu = tau_map.from_tau(state[0])
        beta = tau_map.from_tau(state[1]) if state[1] is not None else None

        return (mu, beta, state[2]), total_cost, loglike, penalty

    return lower


@dataclass(kw_only=True)
class RMLEResult1D(Cost1D, UnfoldedResult1DSimple):
    """Result class for 1D Regularized Maximum Likelihood Estimation (RMLE) unfolding.

    This class stores the results of RMLE unfolding, including the unfolded spectrum,
    cost function components, and optional background components. It inherits from
    both Cost1D and UnfoldedResult1DSimple to provide functionality for saving/loading
    results and accessing cost metrics.

    Attributes:
        beta: Optional background component vector
    """

    def _save(self, path: Path, meta: dict[str, Any], exist_ok: bool = False):
        UnfoldedResult1DSimple._save(self, path, meta, exist_ok)
        Cost1D._save(self, path, meta, exist_ok)

    @classmethod
    def _load(cls, path: Path, meta: dict[str, Any]) -> dict[str, np.ndarray | Vector]:
        a = Cost1D._load(path, meta)
        b = UnfoldedResult1DSimple._load(path, meta)
        return a | b


class OptimResult1D:
    """Container class for optimization results from 1D unfolding.

    This class stores the results of the optimization process, including the unfolded
    spectrum (mu), background components (beta), contaminants (xi), and various cost
    metrics like total cost, likelihood, and penalties.

    Args:
        prototype (Vector): Template vector to clone for creating result vectors
        mu: Unfolded spectrum values
        total_cost: Total cost/loss value from optimization
        loglike: Log-likelihood component of the cost
        penalty: Regularization penalty component
        xi_penalty: Penalty term for contaminants
        beta: Optional background component values
        xi: Optional list of contaminant component values

    Attributes:
        mu (Vector): Unfolded spectrum
        beta (Vector | None): Background component if provided
        xi (list[Vector]): List of contaminant components
        total_cost (ndarray): Total optimization cost
        loglike (ndarray): Log-likelihood term
        penalty (ndarray): Regularization penalty
        xi_penalty (ndarray): Contaminant penalty
    """

    def __init__(
        self,
        *,
        prototype: Vector,
        mu,
        total_cost,
        loglike,
        penalty,
        xi_penalty,
        beta=None,
        xi=None,
    ):
        self.mu = prototype.clone(values=np.asarray(mu))
        if beta is not None:
            self.beta = prototype.clone(values=np.asarray(beta))
        else:
            self.beta = None
        if xi is not None:
            self.xi = [prototype.clone(values=np.asarray(x)) for x in xi]
        else:
            self.xi = ()
        self.total_cost = np.asarray(total_cost)
        self.loglike = np.asarray(loglike)
        self.penalty = np.asarray(penalty)
        self.xi_penalty = np.asarray(xi_penalty)
        self.total_cost.flags.writeable = True
        self.loglike.flags.writeable = True
        self.penalty.flags.writeable = True
        self.xi_penalty.flags.writeable = True

    @property
    def aux(self):
        return {
            "loglike": self.loglike,
            "penalty": self.penalty,
            "xi_penalty": self.xi_penalty,
        }


@dataclass(kw_only=True)
class DynamicData:
    """A dataclass for storing optimization components.

    This class holds data that would be resampled during the
    MC unceratinty procedure, and hence vmaped over during
    the unfolding. It contains the raw data, initial guess, mask, and optional background for
    optimization. All arrays must have the same length as the raw data.

    Attributes:
        raw: The raw data array to be unfolded
        initial: Initial guess for the unfolded spectrum
        mask: Boolean mask array indicating which values to optimize (True = optimize)
        background: Optional background spectrum to subtract from raw data
        _run_checks: Whether to run validation checks in __post_init__
    """

    raw: Data
    initial: ExpectationParameter
    mask: Mask1D
    background: BackgroundModel = BackgroundModel()
    _run_checks: bool = True

    def __post_init__(self):
        self.raw = jnp.asarray(self.raw)
        N = len(self.raw)
        if len(self.initial) != N:
            raise ValueError(
                f"Initial must be of length of data, got {len(self.initial)}"
            )
        self.initial = jnp.asarray(self.initial)

        if len(self.mask) != N:
            raise ValueError(f"Mask must be of length of data, got {len(self.mask)}")
        # Here we flip the mask because jax.set uses the opposite convention
        if self._run_checks:
            self.mask = ~jnp.asarray(self.mask)

        if not isinstance(self.background, BackgroundModel):
            if not isinstance(self.background, Iterable):
                self.background = BackgroundModel(
                    backgrounds=(jnp.asarray(self.background.values),)
                )
            else:
                bgs = []
                for bg in self.background:
                    # Each element may be a BackgroundModel or an arraylike
                    if len(bg) == 0:
                        continue

                    if isinstance(bg, BackgroundModel):
                        bgs.extend(bg.backgrounds)
                    else:
                        bgs.append(jnp.asarray(bg))
                self.background = BackgroundModel(backgrounds=tuple(bgs))

        if self._run_checks:
            for i, bg in enumerate(self.background.backgrounds):
                if len(bg) != N:
                    raise ValueError(
                        f"Background must be of length of data, got {len(bg)}"
                        + (f"for number {i}." if len(bg) > 1 else "")
                    )

    def __len__(self):
        return len(self.raw)


@dataclass(kw_only=True)
class Settings:
    """Parameters for optimization using Adam optimizer.

    This class holds parameters for the Adam optimizer and other optimization settings.

    Attributes:
        iterations: Number of optimization iterations to run
        lr: Learning rate for Adam optimizer. Can be set to "auto" to use Richardson rate.
        beta1: Adam optimizer beta1 parameter for first moment estimates
        beta2: Adam optimizer beta2 parameter for second moment estimates
        eps: Small constant for numerical stability in Adam
        leave_tqdm: Whether to leave the progress bar after completion
        disable_tqdm: Whether to disable the progress bar
        penalties: Tuple of penalty functions to apply during optimization
    """

    optimizer: Optimizer = optax.adam(1e-1)
    tau_map: TauMap = TAU_MAP
    iterations: int = 100

    # Hyper-hyper parameters
    leave_tqdm: bool = True
    disable_tqdm: bool = False
    print_jaxpr: bool = False
    profile: bool = False

    def __post_init__(self):
        self.iterations = int(self.iterations)

    @classmethod
    def from_kwargs(cls, kwargs):
        # We pop all keys from the dict, ensuring they do not remain in kwargs
        keys = {
            "leave_tqdm",
            "disable_tqdm",
            "iterations",
            "optimizer",
            "print_jaxpr",
            "tau_map",
            "profile",
        }
        values = {k: kwargs.pop(k) for k in keys if k in kwargs}
        return cls(**values)


@dataclass(kw_only=True)
class StaticData:
    D: DMatrix
    G_eg: GegMatrix
    G_ex: GegDMatrix | None = None
    prototype: Vector
    E: jnp.ndarray | None = None
    contaminants: tuple[Contaminant1D, ...] = ()
    loss: ModelLoss = ModelLoss()

    def __post_init__(self):
        self.D = jnp.asarray(self.D)
        self.G_eg = jnp.asarray(self.G_eg)
        if self.G_ex is not None:
            self.G_ex = jnp.asarray(self.G_ex)
        self.E = jnp.asarray(self.prototype.X)

        N = len(self.prototype)

        if not isinstance(self.contaminants, Iterable):
            self.contaminants = (self.contaminants,)
        if not isinstance(self.contaminants, tuple):
            self.contaminants = tuple(self.contaminants)

        for i, contaminant in enumerate(self.contaminants):
            if len(contaminant) != N:
                raise ValueError(
                    f"Contaminant must be of length of data, got {len(contaminant)} for number {i}."
                )
