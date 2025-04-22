from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from ... import Vector
from .contaminant1d import setup_contaminants
from .rmle1d import make_lower, OptimComponents, OptimResult1D
from .utils import closure_unpack
from .tau import from_tau, to_tau

if TYPE_CHECKING:
    from .rmle1d import DataParams, OptimParams, OptimResult1D


def unfold(
    components: OptimComponentsList,
    value_and_grad,
    optim_params: OptimParams,
    data_params: DataParams,
    **kwargs,
) -> list[OptimResult1D]:
    if components.same_mask:
        masks = [components.mask]
    else:
        masks = components.masks

    if components.has_background:
        if components.same_background:
            bg = components.background
        else:
            bg = jnp.stack(components.backgrounds)
    else:
        bg = None

    raw = jnp.stack(components.raw)
    # Combine the prompt and the background
    tau = [to_tau(x) for x in components.initial]

    x: list[jnp.ndarray] = []
    combined_masks: list[jnp.ndarray] = []
    if data_params.contaminants:
        for i in range(len(tau)):
            mask_i = masks[0] if components.same_mask else masks[i]
            # We extend the tau and masks to account for contaminant arrays
            x_i, mask_i = setup_contaminants(data_params.contaminants, tau[i], mask_i)
            x.append(x_i)
            # We only need to extend the mask if we are not using the same mask
            if not components.same_mask or i < 1:
                combined_masks.append(mask_i)
    else:
        combined_masks = masks
        x = tau

    if components.same_mask:
        mask = jnp.where(combined_masks[0])
    else:
        raise NotImplementedError(
            "Different masks are not supported for list of components"
        )
        mask = [
            jnp.concatenate([m, jnp.zeros_like(t)]) for m, t in zip(combined_masks, tau)
        ]
        mask = jnp.stack(mask)
        mask = jnp.where(mask)
    x = jnp.stack(x)

    # We have used all kwargs as we can. The rest are probably misspelled
    if len(kwargs) > 0:
        raise ValueError(f"Unknown keyword arguments: {kwargs.keys()}")

    lower = make_lower(
        value_and_grad,
        optim_params,
        data_params,
    )

    in_axes = (
        0,  # The initial values
        0,  # The raw data
        None if components.same_background or bg is None else 0,  # The background
        None if components.same_mask else 0,  # The mask
    )
    lower_vmap = jax.vmap(lower, in_axes=in_axes)
    x, total_cost, loglike, penalty, xi_penalty = lower_vmap(x, raw, bg, mask)

    unpacker = closure_unpack(data_params.contaminants, data_params.E)
    results: list[OptimResult1D] = []
    for i in range(len(x)):
        aleph = from_tau(x[i])
        mu, contaminants = unpacker(aleph)
        result = OptimResult1D(
            prototype=data_params.prototype,
            mu=mu,
            total_cost=total_cost[i],
            loglike=loglike[i],
            penalty=penalty[i],
            xi_penalty=xi_penalty[i],
            xi=contaminants,
        )
        results.append(result)
    return results


@dataclass(kw_only=True)
class OptimComponentsList:
    components: list[OptimComponents]
    same_mask: bool = False
    same_background: bool = False

    @classmethod
    def from_data(
        cls,
        data: list[Vector],
        initial: list[Vector],
        mask: list[np.ndarray],
        background: list[Vector] | None = None,
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

        same_bg = False
        if background is not None:
            if len(background) == 1:
                if len(background) != N:
                    raise ValueError("Background must be of the same length as data")
                same_bg = True
            else:
                if len(background[0]) != N:
                    raise ValueError("Background must be of the same length as data")
                background = [jnp.asarray(bg) for bg in background]
        components = []
        for i in range(N):
            bg = background
            if background is not None and not same_bg:
                bg = background[i]
            component = OptimComponents(
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
        return self.components[0].background is not None

    def __len__(self):
        return len(self.components)
