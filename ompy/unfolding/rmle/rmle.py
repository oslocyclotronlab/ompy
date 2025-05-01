from __future__ import annotations
from typing_extensions import override
import time
import jax
import jax.numpy as jnp
import numpy as np
from dataclasses import asdict
from typing import TYPE_CHECKING
import optax
from typing import Any
from .rmle1d import (
    RMLEResult1D,
    OptimComponents,
    OptimParams,
    DataParams,
    cost as cost1d,
    unfold as unfold1d,
)
from .rmlelist import unfold as unfold_list, OptimComponentsList
from .rmle2d import (
    unfold as unfold_matrix,
    OptimizationSettings,
    OptimizationComponents,
    OptimizationData,
    RMLEResult2D,
)
from ..unfolder import Unfolder
from ..result1d import Parameters1D, ResultMeta1D
from ..result2d import Parameters2D, ResultMeta2D
from .loss import Loss, LossFn, KullbackLeibler
from ... import Vector, Matrix

if TYPE_CHECKING:
    from .contaminant1d import Contaminant1D
    from .contaminant2d import Contaminant2D
    from .penalty import PenaltyFn

def jit_cost1d() -> jax.core.Callable:
    return jax.jit(
        jax.value_and_grad(cost1d, has_aux=True),
        static_argnames=("contaminants", "unpacker", "penalties", "loss"),
    )

type Optimizer = Any


class RMLE(Unfolder):
    @staticmethod
    @override
    def supports_background():
        return True

    @override
    def _unfold_vector(
        self,
        data: Vector,
        background: Vector | None,
        initial: Vector,
        D: Matrix,
        G_eg: Matrix,
        mask: np.ndarray,
        contaminants: list[Contaminant1D] | None = None,
        profile: bool = False,
        **kwargs,
    ) -> RMLEResult1D:
        """
        This mostly just packs arguments into structs and then passes them to the optimizer,
        then unpacks and packs the results into a RMLEResult1D.
        """

        value_and_grad = jit_cost1d()

        # These are simple structs that contain the data and the parameters
        # Turns out we got a lot to keep track of
        components = OptimComponents(
            raw=data, initial=initial, mask=mask, background=background
        )
        optim_params = OptimParams.from_kwargs(lambda: G_eg @ D, kwargs)
        data_params = DataParams(
            D=D,
            G_eg=G_eg,
            prototype=data,
            contaminants=contaminants,
        )

        start = time.time()
        if profile:
            print("Profiling...")
            with jax.profiler.trace(
                "/tmp/jax-trace-unfold-vec", create_perfetto_link=True
            ):
                result = unfold1d(
                    components,
                    value_and_grad=value_and_grad,
                    optim_params=optim_params,
                    data_params=data_params,
                    **kwargs,
                )
            print(f"Profiling took {time.time() - start} seconds")
        else:
            result = unfold1d(
                components,
                value_and_grad=value_and_grad,
                optim_params=optim_params,
                data_params=data_params,
                **kwargs,
            )

        elapsed = time.time() - start
        kwargs = (
            asdict(optim_params)
            | {"contaminants": contaminants}
        )
        parameters = Parameters1D(
            D=D,
            G_eg=G_eg,
            raw=data,
            background=background,
            initial=initial,
            kwargs=kwargs,
            mask=np.asarray(mask),
        )  # Kwargs got popped by OptimParams.from_kwargs()
        meta = ResultMeta1D(
            time=elapsed, space=self.space, parameters=parameters, method=self.__class__
        )
        return RMLEResult1D(
            meta=meta,
            cost=result.total_cost,
            u=result.mu,
            beta=result.beta,
            aux=result.aux,
            xi=result.xi,
        )

    @override
    def _unfold_vectors(
        self,
        data: list[Vector],
        background: list[Vector] | None,
        initial: list[Vector],
        D: Matrix,
        G_eg: Matrix,
        mask: list[np.ndarray],
        profile: bool = False,
        contaminants: list[Contaminant1D] | None = None,
        **kwargs,
    ) -> list[RMLEResult1D]:
        value_and_grad = jit_cost1d()
        components = OptimComponentsList.from_data(data, initial, mask, background)
        optim_params = OptimParams.from_kwargs(lambda: G_eg @ D, kwargs)
        data_params = DataParams(
            D=D,
            G_eg=G_eg,
            prototype=data[0],
            contaminants=contaminants,
        )

        start = time.time()
        optim_results = unfold_list(
            components=components,
            value_and_grad=value_and_grad,
            optim_params=optim_params,
            data_params=data_params,
            **kwargs,
        )

        elapsed = time.time() - start

        results: list[RMLEResult1D] = []
        for i, result in enumerate(optim_results):
            parameters = Parameters1D(
                D=D,
                G_eg=G_eg,
                raw=data[i],
                background=background[i] if background is not None else None,
                initial=initial[i],
                kwargs=asdict(optim_params) | {"contaminants": contaminants},
                mask=np.asarray(mask),
            )
            meta = ResultMeta1D(
                time=elapsed,
                space=self.space,
                parameters=parameters,
                method=self.__class__,
            )
            results.append(
                RMLEResult1D(
                    meta=meta,
                    cost=result.total_cost,
                    u=result.mu,
                    beta=result.beta,
                    aux=result.aux,
                    xi=result.xi,
                )
            )

        return results

    def _unfold_matrix(
        self,
        data: Matrix,
        background: tuple[Matrix, ...],
        initial: Matrix,
        D: Matrix,
        G_eg: Matrix,
        G_ex: Matrix | None,
        mask: np.ndarray,
        optimizer: Optimizer = optax.adam(0.001),
        contaminants: tuple[Contaminant2D, ...] = (),
        loss: LossFn | Loss = KullbackLeibler(),
        loss_background: LossFn | Loss = KullbackLeibler(),
        penalties: tuple[PenaltyFn, ...] = (),
        penalties_background: tuple[PenaltyFn, ...] = (),
        **kwargs,
    ) -> RMLEResult2D:


        components = OptimizationComponents(
            initial=initial,
            mask=mask,
            loss=loss,
            loss_background=loss_background,
            penalties=penalties,
            penalties_background=penalties_background,
        )

        settings = OptimizationSettings.from_kwargs(optimizer=optimizer, **kwargs)
        optim_data = OptimizationData(
            raw=data,
            backgrounds=background,
            D=D,
            G_eg=G_eg,
            G_ex=G_ex,
            prototype=data,
            contaminants=contaminants,
        )
        start = time.time()
        result = unfold_matrix(
            data=optim_data,
            components=components,
            settings=settings,
        )
        elapsed = time.time() - start

        # TODO Add Response coefficients as optimisation parameter
        parameters = Parameters2D(
            D=D,
            raw=data,
            background=background,
            initial=initial,
            G_eg=G_eg,
            G_ex=G_ex,
            kwargs=kwargs | {"optimizer": optimizer},
            mask=mask,
        )
        meta = ResultMeta2D(
            time=elapsed, space=self.space, parameters=parameters, method=self.__class__
        )
        return RMLEResult2D(meta=meta, cost=result.total_cost, u=result.mu, aux={})
