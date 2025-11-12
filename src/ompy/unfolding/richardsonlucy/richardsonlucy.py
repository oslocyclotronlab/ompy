from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, asdict, fields
from typing import TYPE_CHECKING, Literal

import logging
import numpy as np
from numpy.typing import NDArray
from typing_extensions import override

from ... import Matrix, Vector
from ..result import Parameters1D, Parameters2D, ResultMeta1D, ResultMeta2D
from ..result1d import UnfoldedResult1DSimple
from ..result2d import UnfoldedResult2DSimple
from ..stubs import Space
from ..unfolder import Unfolder
from .backends import (
    BackendNotAvailableError,
    best_available_backend,
    get_backend,
)

if TYPE_CHECKING:
    from ...detector import Detector

LOG = logging.getLogger(__name__)


@dataclass(kw_only=True)
class RichardsonLucyKwargs:
    iterations: int = 50
    tolerance: float | None = None
    epsilon: float = 1e-12
    store_history: bool = False
    disable_tqdm: bool = True  # Placeholder to maintain parity with other algorithms
    leave_tqdm: bool = True


@dataclass(kw_only=True)
class RichardsonLucyResult1D(UnfoldedResult1DSimple):
    iterations: int = 0
    converged: bool = False
    history: NDArray[np.float64] | None = None
    loss: NDArray[np.float64] | None = None


@dataclass(kw_only=True)
class RichardsonLucyResult2D(UnfoldedResult2DSimple):
    iterations: int = 0
    converged: bool = False
    history: NDArray[np.float64] | None = None
    loss: NDArray[np.float64] | None = None


def _aggregate_vector_background(
    background: Sequence[Vector] | None, prototype: Vector
) -> tuple[NDArray[np.float64] | None, int, Vector | None]:
    if not background:
        return None, 0, None
    arrays = [np.asarray(bg.values, dtype=np.float64) for bg in background]
    stacked = np.vstack(arrays)
    mean = stacked.mean(axis=0)
    stored = prototype.clone(values=mean.astype(prototype.values.dtype, copy=False))
    return mean, len(arrays), stored


def _aggregate_matrix_background(
    background: Sequence[Matrix] | None, prototype: Matrix
) -> tuple[NDArray[np.float64] | None, int, Matrix | None]:
    if not background:
        return None, 0, None
    if isinstance(background, Matrix):
        return np.asarray(background.values, dtype=np.float64), 1, background
    arrays = [np.asarray(mat.values, dtype=np.float64) for mat in background]
    stacked = np.stack(arrays, axis=0)
    mean = stacked.mean(axis=0)
    stored = prototype.clone(values=mean.astype(prototype.values.dtype, copy=False))
    return mean, len(arrays), stored


def _ensure_mask_vector(mask: NDArray[np.bool_] | None, length: int) -> NDArray[np.bool_]:
    if mask is None:
        return np.ones(length, dtype=bool)
    if mask.dtype != bool:
        return mask.astype(bool)
    return mask


def _ensure_mask_matrix(mask: NDArray[np.bool_] | None, shape: tuple[int, int]) -> NDArray[np.bool_]:
    if mask is None:
        return np.ones(shape, dtype=bool)
    if mask.dtype != bool:
        return mask.astype(bool)
    return mask


class RichardsonLucy(Unfolder):
    r"""Richardson–Lucy deconvolution with configurable computational backends.

    The algorithm assumes Poisson-distributed counts and seeks the latent
    spectrum :math:`\mu` that best explains the observed data :math:`n`.
    With response kernel :math:`R = D_{\text{eg}} G_{\text{eg}}`,
    optional excitation smoothing :math:`G_{\text{ex}}`, and background
    :math:`B`, the forward model is

    .. math::

       \nu = G_{\text{ex}}\,\mu\,R + B

    and the Poisson log-likelihood reads

    .. math::

       \mathcal{L}(\mu) = \sum_i \left( n_i \log \nu_i - \nu_i \right).

    The Richardson–Lucy fixed-point iteration updates :math:`\mu`
    multiplicatively:

    .. math::

       \mu^{(k+1)} = \mu^{(k)} \odot
       \frac{ G_{\text{ex}}^\top \left( \frac{n}{\nu^{(k)}} \right) R^\top }
            { G_{\text{ex}}^\top \mathbf{1} \, R^\top },
       \qquad
       \nu^{(k)} = G_{\text{ex}}\,\mu^{(k)} R + B,

    where :math:`\mathbf{1}` is an array of ones shaped like :math:`n`,
    and division and multiplication are element-wise. When :math:`G_{\text{ex}}`
    is absent (1D unfolding) the numerator and denominator reduce to
    :math:`R^\top`.

    This implementation handles background averaging, mask enforcement,
    convergence logging, and delegates numerical kernels to NumPy, Numba,
    or JAX/Optax backends depending on availability or user request.
    """

    def __init__(
        self,
        D_eg: Matrix | None = None,
        G_eg: Matrix | None = None,
        G_ex: Matrix | None = None,
        efficiency: Vector | None = None,
        *,
        detector: Detector | None = None,
        space: Space = "mu",
        warn_int_data: bool = True,
        iterations: int = 50,
        tolerance: float | None = None,
        epsilon: float = 1e-12,
        store_history: bool = False,
        backend: Literal["auto", "numpy", "numba", "jax"] = "auto",
    ) -> None:
        super().__init__(
            D_eg=D_eg,
            G_eg=G_eg,
            G_ex=G_ex,
            efficiency=efficiency,
            detector=detector,
            space=space,
            warn_int_data=warn_int_data,
        )
        self.iterations = iterations
        self.tolerance = tolerance
        self.epsilon = epsilon
        self.store_history = store_history
        self._backend_preference = backend
        self.backend_name = backend
        LOG.debug(
            "Initializing RichardsonLucy with backend=%s iterations=%s tolerance=%s epsilon=%s",
            backend,
            iterations,
            tolerance,
            epsilon,
        )
        try:
            self._resolve_backend()
        except BackendNotAvailableError as exc:
            LOG.exception("Failed to resolve backend preference=%s", backend)
            raise ValueError(str(exc)) from exc

    def _handle_kwargs(self, kwargs: dict) -> RichardsonLucyKwargs:
        supported = {f.name for f in fields(RichardsonLucyKwargs)}
        provided = {k: kwargs.pop(k) for k in list(kwargs.keys()) if k in supported}
        defaults = asdict(
            RichardsonLucyKwargs(
                iterations=self.iterations,
                tolerance=self.tolerance,
                epsilon=self.epsilon,
                store_history=self.store_history,
            )
        )
        defaults.update(provided)
        kw = RichardsonLucyKwargs(**defaults)
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise ValueError(f"Unknown keyword arguments for RichardsonLucy: {unknown}")
        return kw

    @staticmethod
    @override
    def supports_background() -> bool:
        return True

    def _resolve_backend(self) -> None:
        if self._backend_preference == "auto":
            backend = best_available_backend()
            self.backend_name = backend.name
            self._backend = backend
            LOG.info("RichardsonLucy auto-selected backend '%s'", backend.name)
        else:
            self.backend_name = self._backend_preference
            self._backend = get_backend(self._backend_preference)
            LOG.info("RichardsonLucy using user-selected backend '%s'", self.backend_name)

    def _select_backend(self) -> None:
        try:
            self._resolve_backend()
        except BackendNotAvailableError as exc:  # pragma: no cover - guard
            LOG.exception("Backend '%s' is not available", self._backend_preference)
            raise ValueError(str(exc)) from exc

    @override
    def _unfold_vector(
        self,
        data: Vector,
        background: Sequence[Vector] | None,
        initial: Vector,
        D: Matrix,
        G_eg: Matrix,
        mask: NDArray[np.bool_],
        efficiency: Vector | None = None,
        **kwargs,
    ) -> RichardsonLucyResult1D:
        kw = self._handle_kwargs(kwargs)
        self._select_backend()

        start = time.perf_counter()
        response = (D @ G_eg).values.astype(np.float64, copy=False)
        data_values = np.array(data.values, dtype=np.float64, copy=True)
        initial_values = np.array(initial.values, dtype=np.float64, copy=True)

        background_values, background_count, stored_background = _aggregate_vector_background(
            background, data
        )

        mask_bool = _ensure_mask_vector(mask, len(data_values))
        LOG.debug(
            "Unfolding vector with backend '%s': size=%d iterations=%d background_samples=%d",
            self.backend_name,
            data_values.size,
            kw.iterations,
            background_count,
        )

        final, iterations_performed, converged, history, loglike = self._backend.run_vector(
            response,
            data_values,
            initial_values,
            mask_bool,
            background=background_values,
            iterations=kw.iterations,
            tolerance=kw.tolerance,
            epsilon=kw.epsilon,
            store_history=kw.store_history,
            show_progress=not kw.disable_tqdm,
            leave=kw.leave_tqdm,
            description=f"Richardson-Lucy 1D ({self.backend_name})",
        )
        elapsed = time.perf_counter() - start
        LOG.info(
            "Vector unfolding finished: backend=%s iterations=%d elapsed=%.3fs converged=%s",
            self.backend_name,
            iterations_performed,
            elapsed,
            converged,
        )

        result_vector = initial.clone(values=final.astype(initial.values.dtype, copy=False))
        #result_vector.values[~mask_bool] = 0.0

        # Convert optional arrays
        history_np = None if history is None or history.size == 0 else history
        loglike_np = loglike if loglike.size else np.array([0.0], dtype=np.float64)

        parameters = Parameters1D(
            raw=data,
            background=stored_background,
            initial=initial,
            D_eg=D,
            G_eg=G_eg,
            kwargs=asdict(kw)
            | {
                "iterations_performed": iterations_performed,
                "converged": converged,
                "background_samples": background_count,
                "backend": self.backend_name,
            },
            mask=mask_bool,
            efficiency=efficiency,
        )

        meta = ResultMeta1D(
            time=elapsed,
            space=self.space,
            parameters=parameters,
            method=self.__class__,
        )

        return RichardsonLucyResult1D(
            meta=meta,
            u=result_vector,
            iterations=iterations_performed,
            converged=converged,
            history=history_np,
            loss=loglike_np,
        )

    @override
    def _unfold_vectors(
        self,
        data: list[Vector],
        background: list[Sequence[Vector]] | list[Vector] | None,
        initial: list[Vector],
        D: Matrix,
        G_eg: Matrix,
        mask: list[NDArray[np.bool_]] | None,
        efficiency: Vector | None = None,
        **kwargs,
    ) -> list[RichardsonLucyResult1D]:
        kw = self._handle_kwargs(kwargs)
        self._select_backend()
        kw_dict = asdict(kw)
        results: list[RichardsonLucyResult1D] = []

        for idx, vec in enumerate(data):
            if background:
                bg_entry = background[idx]
                if isinstance(bg_entry, Vector):
                    background_seq: Sequence[Vector] | None = (bg_entry,)
                else:
                    background_seq = tuple(bg_entry)
            else:
                background_seq = None

            mask_array = mask[idx] if mask is not None else None

            LOG.debug(
                "Unfolding vector #%d/%d with backend '%s'",
                idx + 1,
                len(data),
                self.backend_name,
            )

            result = self._unfold_vector(
                data=vec,
                background=background_seq,
                initial=initial[idx],
                D=D,
                G_eg=G_eg,
                mask=mask_array if mask_array is not None else np.ones(len(vec), dtype=bool),
                efficiency=efficiency,
                **kw_dict,
            )
            results.append(result)
        return results

    @override
    def _unfold_matrix(
        self,
        data: Matrix,
        background: Sequence[Matrix] | None,
        initial: Matrix,
        D: Matrix,
        G_eg: Matrix,
        G_ex: Matrix,
        mask: NDArray[np.bool_],
        efficiency: Vector | None = None,
        **kwargs,
    ) -> RichardsonLucyResult2D:
        kw = self._handle_kwargs(kwargs)
        self._select_backend()

        start = time.perf_counter()
        response = (D @ G_eg).values.astype(np.float64, copy=False)
        data_values = np.array(data.values, dtype=np.float64, copy=True)
        initial_values = np.array(initial.values, dtype=np.float64, copy=True)

        background_values, background_count, stored_background = _aggregate_matrix_background(
            background, data
        )

        mask_bool = _ensure_mask_matrix(mask, data_values.shape)
        mask_bool = ~mask_bool
        LOG.debug(
            "Unfolding matrix with backend '%s': shape=%s iterations=%d background_samples=%d",
            self.backend_name,
            data_values.shape,
            kw.iterations,
            background_count,
        )

        final, iterations_performed, converged, history, loglike = self._backend.run_matrix(
            response,
            data_values,
            initial_values,
            mask_bool,
            background=background_values,
            iterations=kw.iterations,
            tolerance=kw.tolerance,
            epsilon=kw.epsilon,
            store_history=kw.store_history,
            G_ex=G_ex.values.astype(np.float64, copy=False) if G_ex is not None else None,
            show_progress=not kw.disable_tqdm,
            leave=kw.leave_tqdm,
            description=f"Richardson-Lucy 2D ({self.backend_name})",
        )
        elapsed = time.perf_counter() - start

        result_matrix = initial.clone(values=final.astype(initial.values.dtype, copy=False))
        #result_matrix.values[~mask_bool] = 0.0

        LOG.info(
            "Matrix unfolding finished: backend=%s shape=%s iterations=%d elapsed=%.3fs converged=%s",
            self.backend_name,
            result_matrix.shape,
            iterations_performed,
            elapsed,
            converged,
        )

        history_np = None if history is None or history.size == 0 else history
        loglike_np = loglike if loglike.size else np.array([0.0], dtype=np.float64)

        parameters = Parameters2D(
            raw=data,
            background=stored_background,
            initial=initial,
            D_eg=D,
            G_eg=G_eg,
            G_ex=G_ex,
            kwargs=asdict(kw)
            | {
                "iterations_performed": iterations_performed,
                "converged": converged,
                "background_samples": background_count,
                "backend": self.backend_name,
            },
            mask=mask_bool,
            efficiency=efficiency,
        )

        meta = ResultMeta2D(
            time=elapsed,
            space=self.space,
            parameters=parameters,
            method=self.__class__,
        )

        return RichardsonLucyResult2D(
            meta=meta,
            u=result_matrix,
            iterations=iterations_performed,
            converged=converged,
            history=history_np,
            loss=loglike_np,
        )
