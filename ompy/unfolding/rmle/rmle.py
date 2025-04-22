from __future__ import annotations
from typing_extensions import override
import time
import jax
import jax.numpy as jnp
import numpy as np
from dataclasses import asdict
from typing import TYPE_CHECKING

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
    cost as cost2d,
    unfold as unfold_matrix,
    OptimComponentsMatrix,
    RMLEResult2D,
)
from ..unfolder import Unfolder
from ..result1d import Parameters1D, ResultMeta1D
from ..result2d import Parameters2D, ResultMeta2D
from ... import Vector, Matrix

if TYPE_CHECKING:
    from .contaminant1d import Contaminant1D


def jit_cost1d() -> jax.core.Callable:
    return jax.jit(
        jax.value_and_grad(cost1d, has_aux=True),
        static_argnames=("contaminants", "unpacker", "penalties", "loss"),
    )


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
        background: Matrix | None,
        initial: Matrix,
        D: Matrix,
        G_eg: Matrix,
        G_ex: Matrix | None,
        mask: np.ndarray,
        **kwargs,
    ) -> RMLEResult2D:
        components = OptimComponentsMatrix(
            raw=data, initial=initial, mask=mask, background=background
        )

        optim_params = OptimParams.from_kwargs(lambda: G_eg @ D, kwargs)
        G_ex_ = jnp.asarray(G_ex)
        G_eg_ = jnp.asarray(G_eg)
        n = jnp.asarray(data.values)
        if background is None:
            bg = None
        else:
            bg = jnp.asarray(background.values)
        loss = jax.jit(cost_2d, static_argnames=("alpha"))
        grad = jax.grad(cost)
        grad = jax.jit(grad, static_argnames=("alpha"))
        method = "adam"
        value_and_grad = jax.jit(
            jax.value_and_grad(cost2d, has_aux=True), static_argnames=("alpha")
        )

        if "lr" not in kwargs or kwargs["lr"] == "auto":
            kwargs["lr"] = self.richardson_rate()
        start = time.time()
        u, total_cost, aux = unfold_matrix(
            u,
            raw=n,
            bg=bg,
            R=R_,
            G_ex=G_ex_,
            G_eg=G_eg_,
            loss=loss,
            grad=grad,
            value_and_grad=value_and_grad,
            mask=mask,
            **kwargs,
        )
        elapsed = time.time() - start
        # If we have a background, we need to unpack u
        if bg is not None:
            mu, beta = jnp.vsplit(u, 2)
            beta = background.clone(values=np.asarray(beta))
        else:
            mu = u
            beta = None
        mu = data.clone(values=np.asarray(mu))

        # TODO Add Response coefficients as optimisation parameter
        # TODO Loop over Ex and make error
        # print("Approximating variance")
        # hessian = jax.jit(jax.jacfwd(jax.jacrev(cost)))
        # hessian = hessian(u[160], R_, n[160])
        parameters = Parameters2D(
            D=D,
            raw=data,
            background=background,
            initial=initial,
            G_eg=G_eg,
            G_ex=G_ex,
            kwargs=kwargs | {"method": method},
            mask=mask,
        )
        meta = ResultMeta2D(
            time=elapsed, space=self.space, parameters=parameters, method=self.__class__
        )
        return RMLEResult2D(meta=meta, cost=total_cost, u=mu, beta=beta, aux=aux)
