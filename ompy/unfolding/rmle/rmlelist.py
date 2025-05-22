from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import Iterable
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
import optax

from ... import Vector
from .rmle1d import DynamicData, OptimResult1D, BackgroundModel, cost, State
from ..utils import loop_tqdm

if TYPE_CHECKING:
    from .rmle1d import StaticData, Settings


def unfold(
    dynamic: DynamicDataList,
    settings: Settings,
    static: StaticData,
) -> list[OptimResult1D]:


    if dynamic.same_mask:
        masks = [dynamic.mask]
    else:
        masks = dynamic.masks


    lower = make_lower(
        settings,
        static,
    )

    if dynamic.has_background:
        if not dynamic.same_background:
            raise NotImplementedError("Different backgrounds are not supported for list of components")
        # Mean is the best guess
        bg = dynamic.components[0].background
        beta = sum(bg.backgrounds)/len(bg.backgrounds)
    else:
        beta = ()


    state = (
        dynamic.initial,
        beta,
        (),
    )
    
    

    in_axes = (
        0,  # The initial values
        0,  # The raw data
        #None if dynamic.same_background or  is None else 0,  # The background
        #None if dynamic.same_mask else 0,  # The mask
    )
    lower_vmap = jax.vmap(lower, in_axes=in_axes)
    #states, total_costs, loglikes, penalties = lower_vmap(state, dynamic.)

    results: list[OptimResult1D] = []
    for i in range(len(states)):
        mu, beta, contaminants = states[i]
        result = OptimResult1D(
            prototype=static.prototype,
            mu=mu,
            beta=beta,
            total_cost=total_costs[i],
            loglike=loglikes[i],
            penalty=penalties[i],
            xi_penalty=0,
            xi=contaminants,
        )
        results.append(result)
    return results

    

def make_lower(
    settings: Settings,
    data: StaticData,
    dynamic: DynamicDataList,
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
    contaminants = tuple(model.set_matrices(G_eg, GegD) for model in data.contaminants)
    # no the user must provide the contaminant models
    leave_tqdm = settings.leave_tqdm
    tau_map = settings.tau_map

    # If there is a shared background, we can bake it in here
    if not dynamic.same_background:
        raise NotImplementedError("Different backgrounds are not supported for list of components")

    cost_closure = partial(
        cost,
        GegD=GegD,
        G_eg=G_eg,
        contaminant_models=0,#contaminants,
        loss=data.loss,
        background=dynamic.background,
        tau_map=tau_map,
    )
    value_and_grad = jax.jit(jax.value_and_grad(cost_closure, has_aux=True))

    if settings.print_jaxpr:
        jaxpr = jax.make_jaxpr(cost_closure)((dynamic.initial, dynamic.initial, ()))
        print(jaxpr)

    optimizer = settings.optimizer
    if not dynamic.same_mask:
        raise NotImplementedError("Different masks are not supported for list of components")
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
            updates, opt_state = optimizer.update(g, opt_state)
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
class DynamicDataList:
    components: list[DynamicData]
    same_mask: bool = False
    same_background: bool = False

    @classmethod
    def from_data(
        cls,
        data: list[Vector],
        initial: list[Vector],
        mask: list[np.ndarray],
        background: tuple[BackgroundModel, ...] | Iterable[BackgroundModel | Vector | jnp.ndarray] | BackgroundModel | Vector | jnp.ndarray = (),
    ):
        N = len(data)
        # The mask is proably the same for all vectors
        mask0 = mask[0]
        all_same = all(all(m == mask0) for m in mask)
        if all_same:
            mask = ~jnp.asarray(mask0)
        else:
            mask = [~jnp.asarray(m) for m in mask]
            if len(mask) != N:
                raise ValueError("Mask must be of the same length as data")
        if len(initial) != N:
            raise ValueError("Initial must be of the same length as data")

        # User can provide:
        # - (A) A single background Vector/array for all
        # - (B) A tuple of background Vectors/arrays
        # In those cases we convert it to (BackgroundModel(backgrounds=(bg,)))
        # They can also provide:
        # - (C) A single BackgroundModel for all
        # - (D) A tuple of BackgroundModels
        # In those cases we just convert it to a tuple
        # This code ensures we are left with the type tuple[BackgroundModel, ...]
        if isinstance(background, BackgroundModel):
            # (C)
            background = (background,)
        elif isinstance(background, Iterable):
            # (B) and (D)
            backgrounds = ()
            for bg in background:
                if not isinstance(bg, BackgroundModel):
                    # (B)
                    bg = BackgroundModel(backgrounds=(jnp.asarray(bg),))
                # (D) and (B)*
                backgrounds = backgrounds + (bg,)
            background = backgrounds
        else:
            # (A) here we just try to convert it to a jnp.array and let if fail for the user
            background = (BackgroundModel(backgrounds=(jnp.asarray(background),)),)

        # We now have a tuple of BackgroundModels
        # Ensure compatibility. We could let jax fail later, but the
        # error message will be cryptic
        for i, model in enumerate(background):
            for j, bg in enumerate(model.backgrounds):
                if len(bg) != len(data[0]):
                    raise ValueError(f"Background {j} of model {i} must be of the same length as data.\n"
                                     f"Got len(bg) = {len(bg)} and len(data) = {len(data[0])}")
        # Try to compress the background tuple if possible
        if len(background) > 1:
            seen = (background[0], )
            for i, model in enumerate(background[1:]):
                if model not in seen:
                    seen = seen + (model, )
            background = seen

        same_bg = len(background) == 1

        # We now either have (A) a single shared background model, or (B) a tuple of background models
        # If (B), then it must be the same length as the number of vectors
        if not same_bg and len(background) != N:
            raise ValueError(f"Background tuple must be the same length as the number of vectors.\n"
                             f"Got len(background) = {len(background)} and len(data) = {N}")

        components = []
        for i in range(N):
            bg = background
            if background and not same_bg:
                # Now we known [i] exists by the check above
                bg = background[i]
            component = DynamicData(
                raw=jnp.asarray(data[i]),
                initial=jnp.asarray(initial[i]),
                mask=mask if all_same else mask[i],
                background=bg,
                # We elide the checks because we already checked the length,
                # and so that the shared arrays keep their memory
                _run_checks=False,
            )
            components.append(component)

        return cls(components=components, same_mask=all_same, same_background=same_bg)

    @property
    def masks(self):
        return [c.mask for c in self.components]

    @property
    def mask(self):
        if not self.same_mask:
            raise ValueError("Masks are not the same")
        return self.components[0].mask

    @property
    def backgrounds(self):
        if not self.has_background:
            raise ValueError("Background is not set")
        return [c.background for c in self.components]

    @property
    def initial(self):
        return [c.initial for c in self.components]

    @property
    def raw(self):
        return [c.raw for c in self.components]

    @property
    def has_background(self):
        return len(self.components[0].background) > 0

    def __len__(self):
        return len(self.components)
