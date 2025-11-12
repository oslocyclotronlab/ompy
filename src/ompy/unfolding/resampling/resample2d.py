from __future__ import annotations

import gc
import hashlib
import logging
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypeAlias

import numpy as np
from tqdm.autonotebook import tqdm

from ...accel import h5py_available, jax_available
from ...array import AsymmetricVector, Matrix
from ...ensemble import EnsembleMatrix
from ...numbalib import njit
from ...version import FULLVERSION
from ..rmle.rmle2d import BackgroundModel2D
from ..unfolder import Unfolder
from .resampling import Resampling
from .sampler import Sampler

BackgroundInput: TypeAlias = BackgroundModel2D | tuple[Matrix, ...]

_HAS_JAX = jax_available()

LOG = logging.getLogger(__name__)

if _HAS_JAX:
    import jax
    import jax.numpy as jnp
else:
    jnp = np

if h5py_available():
    import h5py  # type: ignore
else:
    h5py = None  # type: ignore

if TYPE_CHECKING:
    from ..result2d import UnfoldedResult2D


def resample_matrix(
    res: UnfoldedResult2D,
    N: int,
    base: Literal["raw", "folded", "nu"] = "folded",
    bootstrap_background: bool = True,
    background_base: Literal["raw", "beta"] = "beta",
    *,
    checkpoint: bool = True,
    checkpoint_path: str | Path | None = None,
    checkpoint_interval: int = 10,
    **kwargs,
) -> Resampling2D:
    """Bootstrap a set of unfolded 2D spectra from an existing result.

    The routine mirrors the newer 1D sampler:
    1. draw `N` Poisson realisations from the chosen baseline matrix,
    2. optionally sample background matrices,
    3. unfold every replica, gathering diagnostics, and
    4. pack everything into a `Resampling2D` container.

    Parameters
    ----------
    res:
        Reference unfolding result that provides the configuration and best fit.
    N:
        Number of bootstrap replications to generate.
    base:
        Which matrix to sample from (`"raw"`, `"folded"`/`"nu"`). Defaults to `"folded"`.
    bootstrap_background:
        If `True`, background matrices are drawn for every bootstrap.
    background_base:
        Source used when sampling the background (`"raw"` or `"beta"`).
    checkpoint:
        Enable periodic checkpointing so work can resume after interruption.
    checkpoint_path:
        Destination file for checkpoint data. Defaults to
        `<cwd>/<method>_resample2d_checkpoint.pkl`.
    checkpoint_interval:
        Number of completed bootstrap iterations between checkpoint writes.
    kwargs:
        Extra options forwarded to the unfolder (e.g. optimiser settings).
    """

    if checkpoint_interval <= 0:
        raise ValueError("checkpoint_interval must be a positive integer.")

    LOG.info(
        "Starting resample_matrix for %s: N=%d, base=%s, checkpoint=%s, interval=%d",
        res.__class__.__name__,
        N,
        base,
        checkpoint,
        checkpoint_interval,
    )

    best = res.best().astype("float32")
    kwargs = res.meta.kwargs | kwargs
    for forbidden in ("mask", "initial", "background", "data"):
        kwargs.pop(forbidden, None)

    unfolder = Unfolder.from_result_constructor(res)
    source_matrix = _select_bootstrap_source(res, base)

    mask = res.meta.parameters.mask
    unfold_mask: Any = mask if mask is not None else "last nonzero"

    sampler = Sampler2D.from_result(res)
    bootstraps = sampler.sample_data(N, base=base)
    background_inputs = sampler.sample_background(
        N,
        base=background_base,
        bootstrap=bootstrap_background,
    )
    initials = _make_initials(best, N)

    digest = _compute_result_digest(res)
    if checkpoint_path is None:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        checkpoint_path = (
            Path.cwd()
            / f"{res.meta.method}_resample2d_checkpoint_{timestamp}.h5"
        )
    LOG.debug("Using checkpoint path %s", checkpoint_path)
    checkpoint_path = Path(checkpoint_path)

    (
        start_index,
        elapsed,
        bootstraps,
        initials,
        background_inputs,
        unfolded_boot,
        costs,
        auxs,
        betas,
        contaminants,
    ) = _initialize_checkpoint_state(
        checkpoint=checkpoint,
        checkpoint_path=checkpoint_path,
        digest=digest,
        total=N,
        bootstraps=bootstraps,
        initials=initials,
        backgrounds=background_inputs,
        bootstrap_prototype=source_matrix,
        initial_prototype=best,
        background_prototype=res.raw,
    )

    if start_index >= N:
        result = _build_resampling(
            res=res,
            bootstraps=bootstraps,
            unfolded=unfolded_boot,
            costs=costs,
            initials=initials,
            backgrounds=background_inputs,
            betas=betas,
            aux=auxs,
            contaminants=contaminants,
            kwargs=kwargs,
            elapsed=elapsed,
        )
        if checkpoint and checkpoint_path.exists():
            checkpoint_path.unlink(missing_ok=True)
            LOG.debug("Removed checkpoint %s (already complete)", checkpoint_path)
        return result

    iterator = range(start_index, N)
    iterator = tqdm(iterator, initial=start_index, total=N)

    since_last_checkpoint = 0
    kwargs['leave_tqdm'] = False

    for i in iterator:
        iter_start = time.time()
        background = () if background_inputs is None else background_inputs[i]
        result = unfolder.unfold_matrix(
            bootstraps[i],
            initial=initials[i],
            background=background,
            mask=unfold_mask,
            **kwargs,
        )
        result.to_device("cpu")
        result.as_numpy()
        

        unfolded_boot.append(result.best())
        costs.append(np.asarray(result.cost))
        aux_dict = result.aux if result.aux is not None else {}
        auxs.append({k: np.asarray(v) for k, v in aux_dict.items()})
        betas.append(result.beta)
        contaminants.append(list(result.contaminants or ()))

        del result
        gc.collect()
        jax.clear_caches()

        elapsed += time.time() - iter_start
        since_last_checkpoint += 1

        if checkpoint and since_last_checkpoint >= checkpoint_interval:
            _write_checkpoint(
                checkpoint_path=checkpoint_path,
                digest=digest,
                total=N,
                completed=len(unfolded_boot),
                elapsed=elapsed,
                bootstraps=bootstraps,
                bootstrap_prototype=source_matrix,
                initials=initials,
                initial_prototype=best,
                backgrounds=background_inputs,
                background_prototype=res.raw,
                unfolded=unfolded_boot,
                costs=costs,
                betas=betas,
                aux=auxs,
                contaminants=contaminants,
            )
            LOG.debug(
                "Checkpoint saved at iteration %d (elapsed %.2fs)",
                len(unfolded_boot),
                elapsed,
            )
            since_last_checkpoint = 0

    if checkpoint and checkpoint_path.exists():
        checkpoint_path.unlink(missing_ok=True)
        LOG.debug("Removed checkpoint %s after completion", checkpoint_path)

    result = _build_resampling(
        res=res,
        bootstraps=bootstraps,
        unfolded=unfolded_boot,
        costs=costs,
        initials=initials,
        backgrounds=background_inputs,
        betas=betas,
        aux=auxs,
        contaminants=contaminants,
        kwargs=kwargs,
        elapsed=elapsed,
    )
    LOG.info("Completed resample_matrix for %s (elapsed %.2fs)", res.__class__.__name__, elapsed)
    return result


def _select_bootstrap_source(
    res: "UnfoldedResult2D", base: Literal["raw", "folded", "nu"]
) -> Matrix:
    match base:
        case "raw":
            return res.raw.copy().astype("float32")
        case "folded" | "nu":
            return res.best_folded().copy().astype("float32")
        case _:
            raise ValueError(
                f"Unknown sample type {base}. Expected 'raw', 'folded', or 'nu'."
            )


def _sample_data(matrix: Matrix, N: int) -> list[Matrix]:
    # Draw Poisson realisations and ensure float32 dtype for downstream JAX compatibility.
    samples = matrix.sample(N)
    return [sample.astype("float32") for sample in samples]


def _prepare_background_inputs(
    res: "UnfoldedResult2D",
    N: int,
    *,
    bootstrap_background: bool,
    background_base: Literal["raw", "beta"],
) -> list[BackgroundInput] | None:
    background = res.background
    if background is None:
        return None

    if isinstance(background, BackgroundModel2D):
        arrays, loss, do_fold = _extract_model_background_arrays(background)
        if not arrays:
            return None
        sampled = _sample_model_backgrounds(
            res,
            arrays,
            N,
            bootstrap_background=bootstrap_background,
            background_base=background_base,
        )
        return _build_background_models(sampled, loss, do_fold)

    matrices = _coerce_background_matrices(background, res)
    if len(matrices) == 0:
        return None
    return _sample_matrix_backgrounds(
        res,
        matrices,
        N,
        bootstrap_background=bootstrap_background,
        background_base=background_base,
    )


def _extract_model_background_arrays(
    background: BackgroundModel2D,
) -> tuple[tuple[np.ndarray, ...], Any, bool]:
    arrays = tuple(np.asarray(bg) for bg in background.backgrounds)
    loss = getattr(background, "loss", None)
    do_fold = getattr(background, "do_fold", True)
    return arrays, loss, do_fold


def _sample_model_backgrounds(
    res: "UnfoldedResult2D",
    arrays: tuple[np.ndarray, ...],
    N: int,
    *,
    bootstrap_background: bool,
    background_base: Literal["raw", "beta"],
) -> list[tuple[np.ndarray, ...]]:
    if not bootstrap_background:
        return [
            tuple(np.asarray(bg, dtype=np.float32) for bg in arrays) for _ in range(N)
        ]

    match background_base:
        case "raw":
            per_source = [
                [
                    sample.values.astype("float32")
                    for sample in res.raw.clone(
                        values=np.asarray(bg, dtype=np.float32)
                    ).sample(N)
                ]
                for bg in arrays
            ]
        case "beta":
            beta = res.beta
            if beta is None:
                raise ValueError(
                    "Result does not include beta; cannot bootstrap background from beta."
                )
            beta_matrix = beta.astype("float32")
            per_source = [
                [
                    sample.values.astype("float32")
                    for sample in beta_matrix.sample(N)
                ]
                for _ in arrays
            ]
        case _:
            raise ValueError(
                f"Unknown background base {background_base}. Expected 'raw' or 'beta'."
            )

    aggregated: list[tuple[np.ndarray, ...]] = []
    for i in range(N):
        aggregated.append(
            tuple(
                np.asarray(per_source[j][i], dtype=np.float32)
                for j in range(len(per_source))
            )
        )
    return aggregated


def _build_background_models(
    sampled: list[tuple[np.ndarray, ...]],
    loss: Any,
    do_fold: bool,
) -> list[BackgroundModel2D]:
    models: list[BackgroundModel2D] = []
    for sample in sampled:
        kwargs: dict[str, Any] = {"backgrounds": sample, "do_fold": do_fold}
        if loss is not None:
            kwargs["loss"] = loss
        models.append(BackgroundModel2D(**kwargs))
    return models


def _coerce_background_matrices(
    background: Any, res: "UnfoldedResult2D"
) -> tuple[Matrix, ...]:
    def to_matrix(value: Any) -> Matrix:
        if isinstance(value, Matrix):
            return value.astype("float32")
        array = np.asarray(value, dtype=np.float32)
        return res.raw.clone(values=array)

    if isinstance(background, Matrix):
        return (background.astype("float32"),)
    if isinstance(background, (tuple, list)):
        return tuple(to_matrix(bg) for bg in background)
    return ()


def _sample_matrix_backgrounds(
    res: "UnfoldedResult2D",
    matrices: tuple[Matrix, ...],
    N: int,
    *,
    bootstrap_background: bool,
    background_base: Literal["raw", "beta"],
) -> list[tuple[Matrix, ...]]:
    if not bootstrap_background:
        return [tuple(bg.astype("float32") for bg in matrices) for _ in range(N)]

    match background_base:
        case "raw":
            per_source = [
                [
                    sample.astype("float32")
                    for sample in bg.astype("float32").sample(N)
                ]
                for bg in matrices
            ]
        case "beta":
            beta = res.beta
            if beta is None:
                raise ValueError(
                    "Result does not include beta; cannot bootstrap background from beta."
                )
            beta_matrix = beta.astype("float32")
            per_source = [
                [
                    sample.astype("float32")
                    for sample in beta_matrix.sample(N)
                ]
                for _ in matrices
            ]
        case _:
            raise ValueError(
                f"Unknown background base {background_base}. Expected 'raw' or 'beta'."
            )

    aggregated: list[tuple[Matrix, ...]] = []
    for i in range(N):
        aggregated.append(tuple(per_source[j][i] for j in range(len(per_source))))
    return aggregated


def _compute_result_digest(res: "UnfoldedResult2D") -> str:
    h = hashlib.sha256()
    raw = res.raw.values
    h.update(raw.tobytes())
    h.update(str(res.meta.method).encode())
    h.update(str(res.meta.space).encode())
    h.update(np.asarray(res.meta.time, dtype=np.float64).tobytes())
    return h.hexdigest()


def _initialize_checkpoint_state(
    *,
    checkpoint: bool,
    checkpoint_path: Path,
    digest: str,
    total: int,
    bootstraps: list[Matrix],
    initials: list[Matrix],
    backgrounds: list[BackgroundInput] | None,
    bootstrap_prototype: Matrix,
    initial_prototype: Matrix,
    background_prototype: Matrix,
) -> tuple[
    int,
    float,
    list[Matrix],
    list[Matrix],
    list[BackgroundInput] | None,
    list[Matrix],
    list[np.ndarray],
    list[dict[str, np.ndarray]],
    list[Matrix | None],
    list[list[Matrix]],
]:
    if not checkpoint:
        LOG.debug("Checkpoint disabled; starting fresh")
        return (
            0,
            0.0,
            list(bootstraps),
            list(initials),
            backgrounds,
            [],
            [],
            [],
            [],
            [],
        )
    loaded = _load_checkpoint_data(
        checkpoint_path,
        digest,
        total,
        bootstrap_prototype,
        initial_prototype,
        background_prototype,
    )
    if loaded is None:
        LOG.debug("Creating new checkpoint at %s", checkpoint_path)
        _write_checkpoint(
            checkpoint_path=checkpoint_path,
            digest=digest,
            total=total,
            completed=0,
            elapsed=0.0,
            bootstraps=bootstraps,
            bootstrap_prototype=bootstrap_prototype,
            initials=initials,
            initial_prototype=initial_prototype,
            backgrounds=backgrounds,
            background_prototype=background_prototype,
            unfolded=[],
            costs=[],
            betas=[],
            aux=[],
            contaminants=[],
        )
        return (
            0,
            0.0,
            list(bootstraps),
            list(initials),
            backgrounds,
            [],
            [],
            [],
            [],
            [],
        )

    LOG.debug(
        "Resuming from checkpoint %s (completed=%d, elapsed=%.2fs)",
        checkpoint_path,
        loaded[0],
        loaded[1],
    )
    return loaded


def _write_checkpoint(
    *,
    checkpoint_path: Path,
    digest: str,
    total: int,
    completed: int,
    elapsed: float,
    bootstraps: list[Matrix],
    bootstrap_prototype: Matrix,
    initials: list[Matrix],
    initial_prototype: Matrix,
    backgrounds: list[BackgroundInput] | None,
    background_prototype: Matrix,
    unfolded: list[Matrix],
    costs: list[np.ndarray],
    betas: list[Matrix | None],
    aux: list[dict[str, np.ndarray]],
    contaminants: list[list[Matrix]],
) -> None:
    if not h5py_available():
        raise RuntimeError("h5py is required for checkpointing.")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    LOG.debug(
        "Writing checkpoint to %s (completed=%d, elapsed=%.2fs)",
        checkpoint_path,
        completed,
        elapsed,
    )
    with h5py.File(tmp_path, "w") as f:
        f.attrs["version"] = FULLVERSION
        f.attrs["base_digest"] = digest
        f.attrs["total"] = int(total)
        f.attrs["completed"] = int(completed)
        f.attrs["elapsed"] = float(elapsed)

        _save_matrix_dataset(f, "bootstraps", bootstraps, bootstrap_prototype)
        _save_matrix_dataset(f, "initials", initials, initial_prototype)
        _save_matrix_dataset(f, "unfolded", unfolded, initial_prototype)

        _save_costs(f, costs)
        _save_aux(f, aux)
        _save_betas(f, betas, initial_prototype)
        _save_contaminants(f, contaminants, background_prototype)
        _save_backgrounds(f, backgrounds, background_prototype)

    tmp_path.replace(checkpoint_path)


def _load_checkpoint_data(
    checkpoint_path: Path,
    digest: str,
    total: int,
    bootstrap_prototype: Matrix,
    initial_prototype: Matrix,
    background_prototype: Matrix,
) -> tuple[
    int,
    float,
    list[Matrix],
    list[Matrix],
    list[BackgroundInput] | None,
    list[Matrix],
    list[np.ndarray],
    list[dict[str, np.ndarray]],
    list[Matrix | None],
    list[list[Matrix]],
] | None:
    if not checkpoint_path.exists():
        return None
    try:
        with h5py.File(checkpoint_path, "r") as f:
            if f.attrs.get("base_digest", "") != digest:
                warnings.warn("Checkpoint belongs to another result. Ignoring it.")
                LOG.debug("Digest mismatch for checkpoint %s", checkpoint_path)
                return None
            if int(f.attrs.get("total", -1)) != total:
                warnings.warn(
                    "Checkpoint total iterations differ from requested N. Ignoring checkpoint."
                )
                LOG.debug(
                    "Total mismatch for checkpoint %s (stored %s, requested %s)",
                    checkpoint_path,
                    f.attrs.get("total"),
                    total,
                )
                return None
            completed = int(f.attrs.get("completed", 0))
            elapsed = float(f.attrs.get("elapsed", 0.0))
            bootstraps = _load_matrix_dataset(f, "bootstraps", bootstrap_prototype)
            initials = _load_matrix_dataset(f, "initials", initial_prototype)
            backgrounds = _load_backgrounds(f, background_prototype)
            unfolded = _load_matrix_dataset(f, "unfolded", initial_prototype)
            costs = _load_costs(f)
            aux = _load_aux(f)
            betas = _load_betas(f, initial_prototype)
            contaminants = _load_contaminants(f, background_prototype)
            if completed != len(unfolded):
                warnings.warn(
                    "Checkpoint data is inconsistent (completed != len(unfolded)). "
                    "Restarting from scratch."
                )
                LOG.debug(
                    "Checkpoint %s inconsistent (completed=%d, stored=%d)",
                    checkpoint_path,
                    completed,
                    len(unfolded),
                )
                return None
            return (
                completed,
                elapsed,
                bootstraps,
                initials,
                backgrounds,
                unfolded,
                costs,
                aux,
                betas,
                contaminants,
            )
    except Exception as exc:
        warnings.warn(f"Failed to read checkpoint {checkpoint_path}: {exc}")
        return None


def _save_matrix_dataset(
    h5: h5py.File,
    name: str,
    matrices: list[Matrix],
    prototype: Matrix,
) -> None:
    group = h5.require_group(name)
    for key in list(group.keys()):
        del group[key]
    shape = prototype.values.shape
    dtype_str = np.dtype(prototype.values.dtype).str
    group.attrs["shape"] = shape
    group.attrs["dtype"] = dtype_str
    if len(matrices) == 0:
        data = np.empty((0,) + shape, dtype=np.dtype(dtype_str))
    else:
        data = np.stack([mat.values for mat in matrices], axis=0)
    group.create_dataset("values", data=data, compression="gzip")


def _load_matrix_dataset(
    h5: h5py.File,
    name: str,
    prototype: Matrix,
) -> list[Matrix]:
    if name not in h5:
        return []
    group = h5[name]
    if "values" not in group:
        return []
    data = group["values"][...]
    return [prototype.clone(values=data[i]) for i in range(data.shape[0])]


def _save_costs(h5: h5py.File, costs: list[np.ndarray]) -> None:
    group = h5.require_group("costs")
    for key in list(group.keys()):
        del group[key]
    group.attrs["count"] = len(costs)
    for i, cost in enumerate(costs):
        group.create_dataset(str(i), data=np.asarray(cost), compression="gzip")


def _load_costs(h5: h5py.File) -> list[np.ndarray]:
    if "costs" not in h5:
        return []
    group = h5["costs"]
    count = int(group.attrs.get("count", 0))
    return [group[str(i)][...] for i in range(count)]


def _save_aux(h5: h5py.File, aux: list[dict[str, np.ndarray]]) -> None:
    group = h5.require_group("aux")
    for key in list(group.keys()):
        del group[key]
    group.attrs["count"] = len(aux)
    for i, mapping in enumerate(aux):
        sub = group.create_group(str(i))
        for key, value in mapping.items():
            sub.create_dataset(key, data=np.asarray(value), compression="gzip")


def _load_aux(h5: h5py.File) -> list[dict[str, np.ndarray]]:
    if "aux" not in h5:
        return []
    group = h5["aux"]
    count = int(group.attrs.get("count", 0))
    result: list[dict[str, np.ndarray]] = []
    for i in range(count):
        sub = group[str(i)]
        mapping = {key: sub[key][...] for key in sub.keys()}
        result.append(mapping)
    return result


def _save_betas(
    h5: h5py.File,
    betas: list[Matrix | None],
    prototype: Matrix,
) -> None:
    group = h5.require_group("betas")
    for key in list(group.keys()):
        del group[key]
    group.attrs["total"] = len(betas)
    mask = np.array([beta is not None for beta in betas], dtype=np.int8)
    group.create_dataset("mask", data=mask)
    group.attrs["shape"] = prototype.values.shape
    group.attrs["dtype"] = np.dtype(prototype.values.dtype).str
    values = np.zeros(
        (len(betas),) + prototype.values.shape,
        dtype=prototype.values.dtype,
    )
    for i, beta in enumerate(betas):
        if beta is not None:
            values[i] = beta.values
    group.create_dataset("values", data=values, compression="gzip")


def _load_betas(h5: h5py.File, prototype: Matrix) -> list[Matrix | None]:
    if "betas" not in h5:
        return []
    group = h5["betas"]
    total = int(group.attrs.get("total", 0))
    mask = (
        group["mask"][...].astype(bool)
        if "mask" in group
        else np.ones(total, dtype=bool)
    )
    values = (
        group["values"][...]
        if "values" in group
        else np.empty((total,) + prototype.values.shape, dtype=prototype.values.dtype)
    )
    result: list[Matrix | None] = []
    for i in range(total):
        if mask[i]:
            result.append(prototype.clone(values=values[i]))
        else:
            result.append(None)
    return result


def _save_contaminants(
    h5: h5py.File,
    contaminants: list[list[Matrix]],
    prototype: Matrix,
) -> None:
    group = h5.require_group("contaminants")
    for key in list(group.keys()):
        del group[key]
    counts = np.asarray([len(inner) for inner in contaminants], dtype=np.int32)
    group.create_dataset("counts", data=counts)
    total = int(counts.sum())
    if total:
        values = np.stack(
            [mat.values for inner in contaminants for mat in inner],
            axis=0,
        )
    else:
        values = np.empty((0,) + prototype.values.shape, dtype=prototype.values.dtype)
    group.create_dataset("values", data=values, compression="gzip")


def _load_contaminants(
    h5: h5py.File,
    prototype: Matrix,
) -> list[list[Matrix]]:
    if "contaminants" not in h5:
        return []
    group = h5["contaminants"]
    counts = (
        group["counts"][...] if "counts" in group else np.empty(0, dtype=np.int32)
    )
    values = (
        group["values"][...]
        if "values" in group
        else np.empty((0,) + prototype.values.shape, dtype=prototype.values.dtype)
    )
    idx = 0
    result: list[list[Matrix]] = []
    for count in counts:
        inner: list[Matrix] = []
        for _ in range(int(count)):
            inner.append(prototype.clone(values=values[idx]))
            idx += 1
        result.append(inner)
    return result


def _save_backgrounds(
    h5: h5py.File,
    backgrounds: list[BackgroundInput] | None,
    prototype: Matrix,
) -> None:
    group = h5.require_group("backgrounds")
    for key in list(group.keys()):
        del group[key]
    if backgrounds is None:
        group.attrs["kind"] = "none"
        return
    if len(backgrounds) == 0 or isinstance(backgrounds[0], tuple):
        group.attrs["kind"] = "tuple"
        counts = np.asarray([len(bg) for bg in backgrounds], dtype=np.int32)
        group.create_dataset("counts", data=counts)
        total = int(counts.sum())
        if total:
            values = np.stack(
                [mat.values for group_ in backgrounds for mat in group_], axis=0
            )
        else:
            values = np.empty((0,) + prototype.values.shape, dtype=prototype.values.dtype)
        group.create_dataset("values", data=values, compression="gzip")
        group.attrs["shape"] = prototype.values.shape
        group.attrs["dtype"] = np.dtype(prototype.values.dtype).str
        return
    if isinstance(backgrounds[0], BackgroundModel2D):
        group.attrs["kind"] = "model"
        counts = np.asarray(
            [len(model.backgrounds) for model in backgrounds], dtype=np.int32
        )
        group.create_dataset("counts", data=counts)
        total = int(counts.sum())
        if total:
            values = np.stack(
                [np.asarray(bg) for model in backgrounds for bg in model.backgrounds],
                axis=0,
            )
        else:
            values = np.empty((0,) + prototype.values.shape, dtype=prototype.values.dtype)
        group.create_dataset("values", data=values, compression="gzip")
        group.create_dataset(
            "do_fold",
            data=np.asarray([model.do_fold for model in backgrounds], dtype=np.int8),
        )
        loss_classes = np.asarray(
            [
                f"{model.loss.__class__.__module__}:{model.loss.__class__.__qualname__}"
                for model in backgrounds
            ],
            dtype="S256",
        )
        group.create_dataset("loss", data=loss_classes)
        group.attrs["shape"] = prototype.values.shape
        group.attrs["dtype"] = np.dtype(prototype.values.dtype).str
        return
    raise TypeError("Unsupported background type for checkpointing.")


def _load_backgrounds(
    h5: h5py.File,
    prototype: Matrix,
) -> list[BackgroundInput] | None:
    if "backgrounds" not in h5:
        return None
    group = h5["backgrounds"]
    kind = group.attrs.get("kind", "none")
    if kind == "none":
        return None
    counts = group["counts"][...] if "counts" in group else np.empty(0, dtype=np.int32)
    values = (
        group["values"][...]
        if "values" in group
        else np.empty((0,) + prototype.values.shape, dtype=prototype.values.dtype)
    )
    idx = 0
    if kind == "tuple":
        result: list[BackgroundInput] = []
        for count in counts:
            mats = []
            for _ in range(int(count)):
                mats.append(prototype.clone(values=values[idx]))
                idx += 1
            result.append(tuple(mats))
        return result
    if kind == "model":
        do_fold = (
            group["do_fold"][...].astype(bool)
            if "do_fold" in group
            else np.ones(len(counts), dtype=bool)
        )
        loss_data = group["loss"][...] if "loss" in group else np.empty(len(counts), dtype="S1")
        result_models: list[BackgroundInput] = []
        for i, count in enumerate(counts):
            arrs = []
            for _ in range(int(count)):
                arrs.append(values[idx])
                idx += 1
            backgrounds = tuple(np.asarray(arr) for arr in arrs)
            loss_cls = loss_data[i].decode() if len(loss_data) > 0 else ""
            loss = _instantiate_loss(loss_cls)
            result_models.append(
                BackgroundModel2D(
                    loss=loss,
                    backgrounds=backgrounds,
                    do_fold=bool(do_fold[i]),
                )
            )
        return result_models
    raise ValueError(f"Unknown background kind {kind}")


def _instantiate_loss(loss_path: str):
    from ..rmle.lossmodel import ModelLoss

    if not loss_path:
        return ModelLoss()
    module, _, qualname = loss_path.partition(":")
    if not qualname:
        return ModelLoss()
    import importlib

    mod = importlib.import_module(module)
    obj = mod
    for part in qualname.split("."):
        obj = getattr(obj, part)
    return obj()


class Sampler2D(Sampler):
    ndim = 2

    def sample_data(
        self,
        count: int,
        base: Literal["raw", "folded", "nu"] = "folded",
    ) -> list[Matrix]:
        source = _select_bootstrap_source(self.result, base).astype("float32")
        LOG.debug("Sampling %d data replicas from %s base", count, base)
        return [sample.astype("float32") for sample in source.sample(count)]

    def sample_background(
        self,
        count: int,
        *,
        base: Literal["raw", "beta"] = "beta",
        bootstrap: bool = True,
    ) -> list[BackgroundInput]:
        backgrounds = _prepare_background_inputs(
            self.result,
            count,
            bootstrap_background=bootstrap,
            background_base=base,
        )
        if backgrounds is None:
            return []
        LOG.debug(
            "Sampling %d background replicas (base=%s, bootstrap=%s)",
            count,
            base,
            bootstrap,
        )
        return backgrounds

    def sample_total(
        self,
        count: int,
        *,
        base: Literal["folded", "raw"] = "folded",
    ) -> list[Matrix]:
        match base:
            case "folded":
                matrix = self.result.folded_total()
            case "raw":
                matrix = self.result.raw
            case _:
                raise ValueError(
                    "total sampling supports only 'folded' or 'raw' for 2D results"
                )
        matrix = matrix.astype("float32")
        LOG.debug("Sampling %d total replicas in %s space", count, base)
        return [sample.astype("float32") for sample in matrix.sample(count)]


def resample_background(
    result: "UnfoldedResult2D",
    count: int,
    *,
    base: Literal["raw", "beta"] = "beta",
    bootstrap: bool = True,
) -> list[BackgroundInput]:
    """Convenience wrapper to sample background spectra directly."""
    return Sampler.from_result(result).sample_background(
        count, base=base, bootstrap=bootstrap
    )


def sample_total(
    result: "UnfoldedResult2D",
    count: int,
    *,
    base: Literal["folded", "raw"] = "folded",
) -> list[Matrix]:
    """Sample the predicted total spectrum of a result."""
    return Sampler.from_result(result).sample_total(count, base=base)


def _make_initials(best: Matrix, N: int) -> list[Matrix]:
    mean = np.maximum(best.values, np.mean(best.values))
    # Start each bootstrap far from the solution to probe optimiser stability.
    return [
        best.clone(values=np.random.uniform(0.0, 5.0 * mean).astype("float32"))
        for _ in range(N)
    ]

def _build_resampling(
    *,
    res: UnfoldedResult2D,
    bootstraps: list[Matrix],
    unfolded: list[Matrix],
    costs: list[np.ndarray],
    initials: list[Matrix],
    backgrounds: list[BackgroundInput] | None,
    betas: list[Matrix | None],
    aux: list[dict[str, np.ndarray]],
    contaminants: list[list[Matrix]],
    kwargs: dict[str, Any],
    elapsed: float,
) -> Resampling2D:
    betas_field = betas if any(beta is not None for beta in betas) else None
    return Resampling2D(
        base=res,
        bootstraps=bootstraps,
        unfolded=unfolded,
        costs=costs,
        initials=initials,
        backgrounds=backgrounds,
        betas=betas_field,
        kwargs=kwargs,
        elapsed_time=elapsed,
        aux=aux,
        contaminants=contaminants,
    )


@dataclass(kw_only=True)
class Resampling2D(Resampling[Matrix]):
    base: UnfoldedResult2D
    bootstraps: list[Matrix]
    unfolded: list[Matrix]
    costs: np.ndarray | list[np.ndarray]
    initials: list[Matrix]
    betas: list[Matrix | None] | None = None
    kwargs: dict[str, Any] = field(default_factory=dict)
    ndim: Literal[2] = 2
    contaminants: list[list[Matrix]] = field(default_factory=lambda: [[]])

    @classmethod
    def from_path(cls, path: str | Path, read_only: int | None = None) -> Resampling2D:
        return Resampling._load(Path(path), Matrix, Resampling2D, read_only)

    def eta_vec(
        self, Ex: float | int, alpha=0.05, summary=np.median
    ) -> AsymmetricVector:
        i = self.base.raw.X_index.index_expression(Ex, strict=False)
        j = last_nonzero(self.etabox[:, i, :])
        eta = summary(self.etabox[:, i, :j], axis=0)
        eta = self.base.raw.iloc[i, :j].clone(values=eta)
        lower = np.percentile(self.etabox[:, i, :j], 100 * alpha / 2, axis=0)
        upper = np.percentile(self.etabox[:, i, :j], 100 * (1 - alpha / 2), axis=0)
        eta = AsymmetricVector.from_CI(eta, lower=lower, upper=upper, clip=True)
        return eta

    def nu_vec(
        self, Ex: float | int, alpha=0.05, summary=np.median
    ) -> AsymmetricVector:
        i = self.base.raw.X_index.index_expression(Ex, strict=False)
        nubox = self.nubox
        j = last_nonzero(nubox[:, i, :])
        nu = summary(nubox[:, i, :j], axis=0)
        nu = self.base.raw.iloc[i, :j].clone(values=nu)
        lower = np.percentile(nubox[:, i, :j], 100 * alpha / 2, axis=0)
        upper = np.percentile(nubox[:, i, :j], 100 * (1 - alpha / 2), axis=0)
        nu = AsymmetricVector.from_CI(nu, lower=lower, upper=upper, clip=True)
        return nu

    def eta(self) -> EnsembleMatrix:
        return EnsembleMatrix(self.etabox, template=self.base.raw)

    def eta_mat(self, summary=np.median) -> Matrix:
        eta = summary(self.etabox, axis=0)
        eta = self.base.raw.clone(values=eta)
        return eta

    def eta_ci(
        self,
        alpha=0.05,
        summary=np.median,
        as_matrix: bool = True,
    ) -> tuple[Matrix, Matrix] | tuple[np.ndarray, np.ndarray]:
        a_low = 100 * alpha / 2
        lower = np.percentile(self.etabox, a_low, axis=0)
        a_high = 100 * (1 - alpha / 2)
        upper = np.percentile(self.etabox, a_high, axis=0)
        if as_matrix:
            lower = self.base.raw.clone(
                values=lower, name=f"Lower {100 * (1 - alpha):.0f}% PI"
            )
            upper = self.base.raw.clone(
                values=upper, name=f"Upper {100 * (1 - alpha):.0f}% PI"
            )
        return lower, upper

    def nu_mat(self, summary=np.median) -> Matrix:
        nu = summary(self.nubox, axis=0)
        nu = self.base.raw.clone(values=nu)
        return nu

    def mu_mat(self, summary=np.median) -> Matrix:
        mu = summary(self.ubox, axis=0)
        mu = self.base.raw.clone(values=mu)
        return mu

    def mu_vec(
        self, Ex: float | int, alpha=0.05, summary=np.median
    ) -> AsymmetricVector:
        i = self.base.raw.X_index.index_expression(Ex, strict=False)
        j = last_nonzero(self.ubox[:, i, :])
        mu = summary(self.ubox[:, i, :j], axis=0)
        mu = self.base.raw.iloc[i, :j].clone(values=mu)
        lower = np.percentile(self.ubox[:, i, :j], 100 * alpha / 2, axis=0)
        upper = np.percentile(self.ubox[:, i, :j], 100 * (1 - alpha / 2), axis=0)
        mu = AsymmetricVector.from_CI(mu, lower=lower, upper=upper, clip=True)
        return mu

    def get_eta(self, i: int) -> Matrix:
        return self.base.raw.clone(values=self.etabox[i, :, :], name=f"eta {i}")

    @property
    def ubox(self) -> np.ndarray:
        if self._ubox is None:
            self._ubox = np.stack(self.unfolded)
        return self._ubox

    @property
    def etabox(self) -> np.ndarray:
        if self.base.meta.space != "RG":
            warnings.warn(
                f"eta is only properly defined for the RG space, not {self.base.meta.space}."
            )
        if self._etabox is None:
            self._etabox = gmul(self.ubox, self.G_eg.values, self.G_ex.values)
        return self._etabox

    @property
    def nubox(self) -> np.ndarray:
        if self._nubox is None:
            self._nubox = gmul(self.ubox, self.GegD.values.T, self.G_ex.values)
        return self._nubox


def gmul(X, A, B=None):
    if B is None:
        return np.einsum("ijk,kl->ijl", X, A)
    else:
        raise NotImplementedError("Numpy einsum just stalls. Use jax.")
        # For the daredevils, this is the einsum
        return np.einsum("ij,kjl,lm->kim", B, X, A)


if _HAS_JAX:

    @jax.jit
    def _gmul(X, A, B=None):
        if B is None:
            return jnp.einsum("ijk,kl->ijl", X, A)
        else:
            return jnp.einsum("ij,kjl,lm->kim", B, X, A)

    def gmul(X, A, B=None):
        x = _gmul(X, A, B)
        return np.asarray(x)


@njit
def last_nonzero(box: np.ndarray) -> int:
    S = np.sum(box, axis=0)
    for i in range(len(S) - 1, -1, -1):
        if S[i] > 0:
            return i
    return 0
