from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, fields
from typing import TYPE_CHECKING

import numpy as np
from typing_extensions import override

from ...accel import jax_available, jax_working
from ... import Matrix, Vector
from ...helpers import (
    readable_time,
)
from ..result1d import (
    Parameters1D,
    ResultMeta1D,
)
from ..result2d import Parameters2D, ResultMeta2D
from ..stubs import Space
from ..unfolder import Unfolder
from .fics_1d import FICSResult1D, unfold_vector, unfold_vector_pos
from .fics_2d import (
    FICSResult2DMultiple,
    FICSResult2DSimple,
    unfold_matrix,
    unfold_matrix_jax,
    unfold_matrix_jax_block,
)
from .fics_1d_jax import unfold_vector_wrapper, unfold_vectors_wrapper

LOG = logging.getLogger(__name__)


if TYPE_CHECKING:
    from ...detector import Detector

_HAS_JAX = jax_available()
_HAS_JAX_WORKING = jax_working()

if _HAS_JAX_WORKING:
    import jax.numpy as jnp
else:
    jnp = np


@dataclass
class FICSKwargs:
    iterations: int
    weight: float
    lr: float
    save_block: bool = True
    disable_tqdm: bool = False
    enforce_positivity: bool = True
    leave_tqdm: bool = True


class FICS(Unfolder):
    """Unfolding algorithm from Guttormsen et al. 1998

    This algorithm is only valid for 1D histograms with uniform binning.
    The algorithm is described in the paper:

    Guttormsen, K. A., Kjeldsen, H. K., & Nielsen, J. B. (1998).
    Unfolding of multidimensional histograms.
    Nuclear Instruments and Methods in Physics Research Section A: Accelerators, Spectrometers, Detectors and Associated Equipment, 400(1), 1–8. https://doi.org/10.1016/S0168-9002(97)00459-6

    Parameters
    ----------
    R: Matrix
        The unsmoothed response matrix
    G: Matrix
        The gaussian smoothing matrix
    iterations: int
        The number of iterations to perform
    """

    def __init__(
        self,
        D_eg: Matrix | None = None,
        G_eg: Matrix | None = None,
        G_ex: Matrix | None = None,
        detector: Detector | None = None,
        space: Space = "mu",
        warn_int_data: bool = True,
        iterations: int = 10,
        weight: float = 1e-3,
        use_JAX: bool | None = None,
        save_block: bool = False,
        enforce_positivity: bool = False,
    ):
        super().__init__(D_eg=D_eg, G_eg=G_eg, G_ex=G_ex, detector=detector, space=space, warn_int_data=warn_int_data)
        self.iterations = iterations
        self.weight = weight  # Fluctuation weight
        # We prefer to use GPUs, but fall back to CPU if not available
        # If the user specifies GPU, but GPU is not available, raise an error
        if use_JAX is None:
            self.use_JAX = _HAS_JAX and _HAS_JAX_WORKING
        elif use_JAX and not _HAS_JAX_WORKING:
            raise ValueError(
                "JAX is not working. Cannot use GPU. Specify 'use_JAX=False' to use CPU."
            )
        else:
            self.use_JAX = use_JAX
        self.lr = 1  # Learning rate.
        self.save_block = save_block  # Save block of unfolded matrices
        self.enforce_positivity = enforce_positivity

    def handle_kwargs(self, kwargs) -> FICSKwargs:
        supported = [f.name for f in fields(FICSKwargs)]
        described = {k: v for k, v in kwargs.items() if k in supported}
        superfluous = {k: v for k, v in kwargs.items() if k not in supported}
        defaults = dict(
            iterations=self.iterations,
            weight=self.weight,
            lr=self.lr,
            save_block=self.save_block,
            enforce_positivity=self.enforce_positivity,
        )
        kw = FICSKwargs(**(defaults | described))
        LOG.debug(f"Unfolding up to {kw.iterations} iterations")
        LOG.debug(f"Fluctuation weight of {kw.weight}")
        LOG.debug(f"Learning rate of {kw.lr}")
        LOG.debug(f"Enforcing positive values: {kw.enforce_positivity}")
        if superfluous:
            LOG.warning(f"Unused kwargs: {superfluous}")
        return kw

    def optimal_lr(self, tol: float | None = None) -> float:
        # kappa = np.linalg.cond(self.R.values, tol)
        # get the largest and smallest singular values
        R = self.D @ self.G_eg
        s = np.linalg.svd(R.values, compute_uv=False)
        s_max = s.max()
        s_min = s.min()
        # return 1 - 2 / (kappa + 1)
        return 2 / (s_max + s_min)

    def _unfold_vector(
        self,
        data: Vector,
        background: tuple[Vector, ...],
        initial: Vector,
        D: Matrix,
        G_eg: Matrix,
        mask: np.ndarray,
        **kwargs,
    ) -> FICSResult1D:
        kw = self.handle_kwargs(kwargs)
        LOG.debug("Unfolding vector with FICS method")
        data_raw = data
        if background:
            # We use the mean as the parameter
            bg = np.mean(background, axis=0)
            LOG.debug("Background is just subtracted from data.")
            data = data - bg
        start = time.time()

        data = data.astype("float32")
        initial = data.astype("float32")
        if kw.enforce_positivity:
            fn = unfold_vector_pos
        else:
            fn = unfold_vector

        # TODO Abstract this away
        D = D.as_numpy().astype("float32")
        G_eg = G_eg.as_numpy().astype("float32")
        data = data.as_numpy().astype("float32")
        initial = initial.as_numpy().astype("float32")
        R = D @ G_eg
        uall, cost, fluctuations, kl = fn(
            R.values, data.values, initial.values, kw.iterations, kw.lr
        )
        elapsed = time.time() - start
        kw_ = asdict(kw)
        kw_.pop("disable_tqdm")
        kw_.pop("leave_tqdm")
        parameters = Parameters1D(
            raw=data_raw,
            background=background,
            initial=initial,
            G_eg=G_eg,
            D_eg=D,
            kwargs=kw_,
        )

        meta = ResultMeta1D(
            time=elapsed, parameters=parameters, space=self.space, method=self.__class__
        )
        return FICSResult1D(
            meta=meta, u=uall, cost=cost, fluctuations=fluctuations, kl=kl
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
        **kwargs,
    ) -> list[FICSResult1D]:
        kw = self.handle_kwargs(kwargs)
        data_raw = data
        if background:
            # We use the mean as the parameter
            bg = np.mean(background, axis=0)
            data = [d - bg for d in data]
        data = [d.astype("float32") for d in data]
        initial = [i.astype("float32") for i in initial]
        D = D.as_numpy().astype("float32")
        G_eg = G_eg.as_numpy().astype("float32")
        start = time.time()
        R = D @ G_eg
        res = unfold_vectors_wrapper(R, data, initial, kw.iterations, kw.lr)
        elapsed = time.time() - start

        kw = asdict(kw)
        kw.pop("disable_tqdm")
        kw.pop("leave_tqdm")
        results = []
        for i in range(len(data_raw)):
            parameters = Parameters1D(
                raw=data_raw[i],
                background=background,
                initial=initial[i],
                G_eg=G_eg,
                D_eg=D,
                kwargs=kw,
            )
            meta = ResultMeta1D(
                time=elapsed, parameters=parameters, space=self.space, method=self.__class__
            )
            results.append(
                FICSResult1D(
                    meta=meta,
                    u=res[i][0],
                    cost=res[i][1],
                    fluctuations=res[i][2],
                    kl=res[i][3]
                    )
                )

        return results
        

    @override
    def _unfold_matrix(
        self,
        data: Matrix,
        background: Matrix | None,
        initial: Matrix,
        D: Matrix,
        G_eg: Matrix,
        G_ex: Matrix,
        mask: np.ndarray,
        **kwargs,
    ) -> FICSResult2DSimple | FICSResult2DMultiple:
        LOG.debug("Unfolding matrix with Guttormsen method")
        kw = self.handle_kwargs(kwargs)
        LOG.debug("Unfolding to space: %s", self.space)
        raw = data
        if background:
            data = data - background
        data[mask] = 0

        R = D @ G_eg
        start = time.time()
        if self.use_JAX:
            LOG.debug("Using JAX version")
            Rj = jnp.array(R.values)
            dataj = jnp.array(data.values)
            initialj = jnp.array(initial.values)
            Gexj = jnp.array(G_ex.values)
            if self.save_block:
                LOG.debug("Saving block of unfolded matrices")
                uall, cost, fluctuations, kl_div = unfold_matrix_jax_block(
                    Rj, dataj, initialj, kw
                )
            else:
                uall, cost, fluctuations, kl_div = unfold_matrix_jax(
                    Rj, Gexj, dataj, initialj, kw
                )
        else:
            fn = unfold_matrix
            uall, cost, fluctuations = fn(
                R.values, data.values, initial.values, kw.iterations, kw.lr
            )
        elapsed = time.time() - start
        LOG.debug(f"Unfolding took {readable_time(elapsed)} seconds")

        kw_ = asdict(kw) | {"save_block": self.save_block}
        kw_.pop("disable_tqdm")
        kw_.pop("leave_tqdm")
        parameters = Parameters2D(
            raw=raw,
            background=background,
            initial=initial,
            D_eg=D,
            G_eg=G_eg,
            G_ex=G_ex,
            kwargs=kw_,
            mask=mask,
        )
        meta = ResultMeta2D(
            time=elapsed, space=self.space, parameters=parameters, method=self.__class__
        )
        if self.save_block:
            rescls = FICSResult2DMultiple
        else:
            rescls = FICSResult2DSimple
        return rescls(
            meta=meta, u=uall, cost=cost, fluctuations=fluctuations, kl=kl_div
        )

    @override
    def supports_background(self) -> bool:
        return True
