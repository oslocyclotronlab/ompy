from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Iterable

import jax
import jax.numpy as jnp
import numpy as np

from ... import Vector
from ...stubs import Path
from ..result1d import Cost1D, UnfoldedResult1DSimple
from .contaminant1d import Contaminant1D, setup_contaminants
from .loss import kl
from .penalty import total_penalty
from .stubs import Closureable, LossFn, PenaltyFn
from .tau import from_tau, to_tau
from ..utils import closure_unpack, loop_tqdm, richardson_rate

if TYPE_CHECKING:
    from .contaminant1d import Contaminant1D
    from .penalty import Penalty


def cost(
    tau: jnp.ndarray,
    GegD: jnp.ndarray,
    y: jnp.ndarray,
    G_eg: jnp.ndarray,
    unpacker: Callable[[jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]],
    contaminants: tuple[Callable[[jnp.ndarray], tuple[jnp.ndarray, float]]],
    penalties: tuple[PenaltyFn, ...],
    loss: LossFn,
    bg=None,
) -> tuple[jnp.ndarray, dict]:
    aleph = from_tau(tau)
    mu, mu_contaminants = unpacker(aleph)

    def xi_nop(_):
        return mu, mu, 0.0

    def handle_contaminants(_):
        if (
            len(contaminants) == 0
        ):  # If contaminants is empty, return defaults to make jax jit happy
            return mu, mu, 0.0
        xi_mu_list, penalty_list = zip(
            *[c(x, G_eg, GegD) for c, x in zip(contaminants, mu_contaminants)]
        )

        # Sum over all contaminants
        xi_mu_sum = mu + sum(xi_mu_list)
        xi_penalty = sum(penalty_list)
        return mu, xi_mu_sum, xi_penalty

    mu, xi_mu_sum, xi_penalty = jax.lax.cond(
        len(contaminants) > 0, handle_contaminants, xi_nop, None
    )

    nu = xi_mu_sum @ GegD
    likelihood_body = loss(nu, y)
    loglike = jnp.sum(likelihood_body)

    # We only compute eta if needed
    def eta_nop(_):
        return 0.0, 0.0

    def eta_body(_):
        # Map to eta space
        eta = mu @ G_eg
        # Rescale to get a proper probability distribution
        prop = eta / jnp.sum(eta + 1e-10)

        total, partial = total_penalty(penalties, mu, eta, prop)
        # penalty = penalty + 5e-12*(jnp.sum(eta[659:809]))  # 5e-12

        return total, partial

    penalty, partial = jax.lax.cond(len(penalties) > 0, eta_body, eta_nop, None)

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

    cost = loglike + penalty + xi_penalty  # + ortho_penalty

    aux = {
        "loglike": loglike,
        "penalty": partial,
        "xi_penalty": xi_penalty,
    }

    return cost, aux


def unfold(
    components: OptimComponents,
    value_and_grad,
    optim_params: OptimParams,
    data_params: DataParams,
    **kwargs,
) -> OptimResult1D:
    # Combine the prompt and the background
    tau = to_tau(components.initial)
    # if bg is not None:
    #    mask = jnp.concatenate([mask, jnp.zeros_like(bg, dtype=bool)])
    #    u = jnp.concatenate([u, 1.0 + jnp.zeros_like(bg)])

    # Set up Xi for contamination
    x, mask = setup_contaminants(data_params.contaminants, tau, components.mask)

    # We have used all kwargs as we can. The rest are probably misspelled
    if len(kwargs) > 0:
        raise ValueError(f"Unknown keyword arguments: {kwargs.keys()}")

    # The optimization function is created from a closure of all constants
    # that we never vmap over.
    lower = make_lower(
        value_and_grad,
        optim_params,
        data_params,
    )

    # Mask must be a concrete type for the jax.jit to work
    mask = jnp.where(mask)

    x, total_cost, loglike, penalty, xi_penalty = lower(
        x, components.raw, components.background, mask
    )
    aleph = from_tau(x)
    mu, contaminants = closure_unpack(data_params.contaminants, data_params.E)(aleph)

    result = OptimResult1D(
        prototype=data_params.prototype,
        mu=mu,
        total_cost=total_cost,
        loglike=loglike,
        penalty=penalty,
        xi_penalty=xi_penalty,
        xi=contaminants,
    )
    return result


def make_lower(
    value_and_grad,
    optim_params: OptimParams,
    data_params: DataParams,
):
    """Create a closure for the optimization loop that unfolds the spectrum.

    This function creates and returns a closure that performs the main optimization loop
    using Adam optimization. The closure captures constant parameters and data that don't
    change during optimization.

    Args:
        value_and_grad: Function that computes both value and gradient of the cost function
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
    iterations = optim_params.iterations
    lr = optim_params.lr
    beta1 = optim_params.beta1
    beta2 = optim_params.beta2
    eps = optim_params.eps
    G_eg = data_params.G_eg
    GegD = data_params.D @ G_eg
    unpacker = closure_unpack(data_params.contaminants, data_params.E)
    contaminant_closures = tuple(into_closure(c) for c in data_params.contaminants)
    leave_tqdm = optim_params.leave_tqdm
    penalties = tuple(into_closure(p) for p in optim_params.penalties)
    loss = into_closure(optim_params.loss)

    @jax.jit
    def lower(initial, y, bg, mask):
        mean = jnp.zeros_like(initial)
        var = jnp.zeros_like(initial)

        loglike = jnp.zeros(iterations)
        penalty = jnp.zeros(iterations)
        xi_penalty = jnp.zeros(iterations)

        @loop_tqdm(iterations, leave=leave_tqdm)
        @jax.jit
        def body_fun(i, state):
            x, mean, var, loglike, penalty, xi_penalty = state
            (tloss, aux), g = value_and_grad(
                x,
                GegD=GegD,
                y=y,
                bg=bg,
                G_eg=G_eg,
                unpacker=unpacker,
                contaminants=contaminant_closures,
                penalties=penalties,
                loss=loss,
            )
            mean = beta1 * mean + (1 - beta1) * g
            var = beta2 * var + (1 - beta2) * jnp.square(g)
            mean_cor = mean / (1 - beta1 ** (i + 1))  # i + 1 because i starts from 0
            var_cor = var / (1 - beta2 ** (i + 1))
            v = lr * mean_cor / (jnp.sqrt(var_cor) + eps)
            x = x - v
            x = x.at[mask].set(0)
            loglike = loglike.at[i].set(aux["loglike"])
            penalty = penalty.at[i].set(aux["penalty"])
            xi_penalty = xi_penalty.at[i].set(aux["xi_penalty"])
            return x, mean, var, loglike, penalty, xi_penalty

        state = (initial, mean, var, loglike, penalty, xi_penalty)
        state = jax.lax.fori_loop(0, iterations, body_fun, state)
        x, mean, var, loglike, penalty, xi_penalty = state
        total_cost = loglike + penalty
        return x, total_cost, loglike, penalty, xi_penalty

    return lower


def into_closure[**P, T](x: Callable[P, T] | Closureable[P, T]) -> Callable[P, T]:
    if hasattr(x, "closure"):
        return x.closure()
    else:
        return x


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

    beta: Vector | None = None

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
            self.xi = []
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
class OptimComponents:
    """A dataclass for storing optimization components.

    This class holds the raw data, initial guess, mask, and optional background for
    optimization. All arrays must have the same length as the raw data.

    Attributes:
        raw: The raw data array to be unfolded
        initial: Initial guess for the unfolded spectrum
        mask: Boolean mask array indicating which values to optimize (True = optimize)
        background: Optional background spectrum to subtract from raw data
        _run_checks: Whether to run validation checks in __post_init__
    """

    raw: jnp.ndarray
    initial: jnp.ndarray
    mask: jnp.ndarray
    background: jnp.ndarray | None = None
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
        else:
            self.mask = self.mask

        if self.background is not None:
            if len(self.background) != N:
                raise ValueError(
                    f"Background must be of length of data, got {len(self.background)}"
                )
            self.background = jnp.asarray(self.background)

    def __len__(self):
        return len(self.raw)


@dataclass(kw_only=True)
class OptimParams:
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

    # Optimisation parameters
    iterations: int = 100
    lr: float = 0.001
    # - Adam parameters
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    # Hyper-hyper parameters
    leave_tqdm: bool = True
    disable_tqdm: bool = False
    # Penalties to use in the loop
    penalties: tuple[PenaltyFn | Penalty, ...] = ()
    loss: LossFn = kl

    def __post_init__(self):
        self.iterations = int(self.iterations)
        if not isinstance(self.penalties, Iterable):
            self.penalties = (self.penalties,)
        self.penalties = tuple(self.penalties)

    @classmethod
    def from_kwargs(cls, R_cb, kwargs):
        if "lr" in kwargs and kwargs["lr"] == "auto":
            kwargs["lr"] = richardson_rate(R_cb())
        # We pop all keys from the dict, ensuring they do not remain in kwargs
        keys = {
            "iterations",
            "lr",
            "beta1",
            "beta2",
            "eps",
            "alpha",
            "leave_tqdm",
            "disable_tqdm",
            "penalties",
            "loss",
        }
        values = {k: kwargs.pop(k) for k in keys if k in kwargs}
        return cls(**values)


@dataclass(kw_only=True)
class DataParams:
    D: jnp.ndarray
    G_eg: jnp.ndarray
    G_ex: jnp.ndarray | None = None
    prototype: Vector
    E: jnp.ndarray = None
    contaminants: list[Contaminant1D] = field(default_factory=list)

    def __post_init__(self):
        self.D = jnp.asarray(self.D)
        self.G_eg = jnp.asarray(self.G_eg)
        if self.G_ex is not None:
            self.G_ex = jnp.asarray(self.G_ex)
        self.E = jnp.asarray(self.prototype.X)

        N = len(self.prototype)

        if self.contaminants is None:
            self.contaminants = []
        for i, contaminant in enumerate(self.contaminants):
            if len(contaminant) != N:
                raise ValueError(
                    f"Contaminant must be of length of data, got {len(contaminant)} for number {i}."
                )
