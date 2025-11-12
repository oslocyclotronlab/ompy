from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal, Self

import operator

import numpy as np
from numpy.typing import NDArray

from matplotlib import pyplot as plt

from .._version import version as __version__
from ..accel import h5py_available, jax_working
from ..rendering.html import collapse, table
from ..stubs import Axes, Colorbar, Figure, Pathlike, QuadMesh
from ..version import warn_version
from .error_vector import AsymmetricVector
from .vector import Vector
from .index import Index
from .matrix import ColorBy, Matrix
from .matrixmetadata import MatrixMetadata

if h5py_available():
    from .filehandling import dict_to_hdf5, hdf5_to_dict, save_hdf5_2D
else:  # pragma: no cover - exercised when h5py missing
    from .filehandling import save_hdf5_2D  # type: ignore[assignment]
    dict_to_hdf5 = hdf5_to_dict = None  # type: ignore[assignment]

if jax_working():
    import jax.numpy as jnp
else:
    jnp = np


def _infer_slice(
    parent: Index, child: Index | Sequence[float] | NDArray[Any] | None
) -> slice:
    if child is None or child is parent:
        return slice(None)
    if isinstance(child, Index):
        bins = child.bins
    else:
        bins = jnp.asarray(child, dtype=parent.bins.dtype)
    if bins.size == 0:
        return slice(0, 0)
    if bins.shape == parent.bins.shape and jnp.array_equal(bins, parent.bins):
        return slice(None)
    start = parent.index(bins[0])
    stop = start + len(bins)
    return slice(start, stop)


def _resolve_reducer(
    reducer: str | Callable[..., NDArray[Any]],
    axis: int,
    kwargs: dict[str, Any] | None = None,
) -> Callable[[NDArray[Any]], NDArray[Any]]:
    kwargs = {} if kwargs is None else dict(kwargs)
    if isinstance(reducer, str):
        reducer = reducer.lower()
        if reducer == "sum":
            return lambda arr: arr.sum(axis=axis, **kwargs)
        if reducer == "mean":
            return lambda arr: arr.mean(axis=axis, **kwargs)
        if reducer == "median":
            return lambda arr: jnp.median(arr, axis=axis, **kwargs)
        if reducer == "max":
            return lambda arr: arr.max(axis=axis, **kwargs)
        if reducer == "min":
            return lambda arr: arr.min(axis=axis, **kwargs)
        raise ValueError(f"Unknown reducer {reducer!r}")

    def _wrapped(arr: NDArray[Any]) -> NDArray[Any]:
        try:
            return reducer(arr, axis=axis, **kwargs)
        except TypeError:
            if kwargs:
                try:
                    return reducer(arr, **kwargs)
                except TypeError:
                    return reducer(arr)
            return reducer(arr)

    return _wrapped


class ErrorMatrix(Matrix):
    @classmethod
    @abstractmethod
    def add_error(cls, matrix: Matrix, *args, **kwargs) -> ErrorMatrix: ...

    @classmethod
    def from_matrix(cls, matrix: Matrix, *args, **kwargs) -> ErrorMatrix:
        return cls.add_error(matrix, *args, **kwargs)

    def save(self, path: str, *args, **kwargs) -> None:
        raise NotImplementedError()

    @classmethod
    def from_path(cls, path: str, *args, **kwargs) -> ErrorMatrix:
        raise NotImplementedError()

    def get_CI(self) -> tuple[NDArray[Any], NDArray[Any]]:
        return self.values - self.lerr, self.values + self.uerr


class AsymmetricMatrix(ErrorMatrix):
    def __init__(
        self,
        *,
        lerr: Iterable[float],
        uerr: Iterable[float],
        copy: bool = False,
        **kwargs: Any,
    ):
        kwargs["copy"] = copy
        super().__init__(**kwargs)
        dtype_values = self.values.dtype
        self.lerr: NDArray[Any] = jnp.asarray(lerr, dtype=dtype_values)
        self.uerr: NDArray[Any] = jnp.asarray(uerr, dtype=dtype_values)
        if copy:
            self.lerr = self.lerr.copy()
            self.uerr = self.uerr.copy()
        if self.lerr.shape != self.values.shape:
            raise ValueError(
                "lerr must have the same shape as values. "
                f"Got {self.lerr.shape} and {self.values.shape}"
            )
        if self.uerr.shape != self.values.shape:
            raise ValueError(
                "uerr must have the same shape as values. "
                f"Got {self.uerr.shape} and {self.values.shape}"
            )

    @classmethod
    def add_error(
        cls,
        matrix: Matrix,
        lerr: NDArray[Any],
        uerr: NDArray[Any],
        *,
        metadata: MatrixMetadata | None = None,
        copy: bool = False,
        **kwargs: Any,
    ) -> AsymmetricMatrix:
        kwargs.setdefault("dtype", matrix.values.dtype)
        metadata = metadata if metadata is not None else matrix.metadata
        return cls(
            X=matrix.X_index,
            Y=matrix.Y_index,
            values=matrix.values,
            lerr=lerr,
            uerr=uerr,
            metadata=metadata,
            copy=copy,
            **kwargs,
        )

    @classmethod
    def from_CI(
        cls,
        matrix: Matrix,
        lower: NDArray[Any],
        upper: NDArray[Any],
        *,
        clip: bool = False,
        **kwargs: Any,
    ) -> AsymmetricMatrix:
        lerr = matrix.values - lower
        uerr = upper - matrix.values
        if jnp.any(lerr < 0) or jnp.any(uerr < 0):
            if clip:
                lerr = jnp.maximum(lerr, 0)
                uerr = jnp.maximum(uerr, 0)
            else:
                raise ValueError(
                    "CI must bound the matrix values. "
                    "Set clip=True to truncate negative errors."
                )
        return cls(
            X=matrix.X_index,
            Y=matrix.Y_index,
            values=matrix.values,
            lerr=lerr,
            uerr=uerr,
            metadata=matrix.metadata,
            copy=kwargs.pop("copy", False),
            **kwargs,
        )

    def clone(
        self,
        X: Index | None = None,
        Y: Index | None = None,
        values: NDArray[Any] | None = None,
        lerr: NDArray[Any] | None = None,
        uerr: NDArray[Any] | None = None,
        metadata: MatrixMetadata | None = None,
        copy: bool = False,
        dtype: Any | None = None,
        **kwargs: Any,
    ) -> AsymmetricMatrix:
        X = self.X_index if X is None else X
        Y = self.Y_index if Y is None else Y
        values = self.values if values is None else values
        metadata = self.metadata if metadata is None else metadata
        metadata = metadata.update(**kwargs)
        if dtype is None:
            dtype = values.dtype
        x_slice = _infer_slice(self.X_index, X)
        y_slice = _infer_slice(self.Y_index, Y)
        if lerr is None:
            lerr = self.lerr[x_slice, y_slice]
        if uerr is None:
            uerr = self.uerr[x_slice, y_slice]
        return type(self)(
            X=X,
            Y=Y,
            values=values,
            lerr=lerr,
            uerr=uerr,
            metadata=metadata,
            copy=copy,
            dtype=dtype,
        )

    def project(
        self,
        axis: int | str,
        reducer: str | Callable[..., NDArray[Any]] = "sum",
        *,
        reducer_kwargs: dict[str, Any] | None = None,
    ) -> AsymmetricVector:
        axis_idx = self.axis_to_int(axis, allow_both=False)
        reducer_fn = _resolve_reducer(reducer, axis_idx, reducer_kwargs)
        values = reducer_fn(self.values)
        lower = reducer_fn(self.values - self.lerr)
        upper = reducer_fn(self.values + self.uerr)

        index = self.Y_index if axis_idx == 0 else self.X_index
        base_vector = super().meta_into_vector(index=index, values=values)
        lerr = jnp.maximum(values - lower, 0)
        uerr = jnp.maximum(upper - values, 0)
        return AsymmetricVector(
            X=base_vector.X_index,
            values=base_vector.values,
            lerr=lerr,
            uerr=uerr,
            metadata=base_vector.metadata,
            copy=False,
            order="K",
        )

    def plot(
        self,
        ax: Sequence[Axes] | Axes | None = None,
        *,
        n_sigma: float = 1.0,
        scale: str | None = None,
        vmin: float | None = None,
        vmax: float | None = None,
        add_cbar: bool = True,
        cbarkwargs: dict[str, Any] | None = None,
        titles: Sequence[str] | None = None,
        figsize: tuple[float, float] | None = None,
        **kwargs: Any,
    ) -> tuple[Figure, list[tuple[Axes, tuple[QuadMesh, Colorbar | None]]]]:

        lower_values = self.values - self.lerr
        upper_values = self.values + self.uerr
        stacked = jnp.stack((lower_values, self.values, upper_values))

        vmin_eff = vmin if vmin is not None else float(jnp.nanmin(stacked))
        vmax_eff = vmax if vmax is not None else float(jnp.nanmax(stacked))

        if vmin is None and (scale == 'log' or scale is None):
            vmin_eff = 1 # We realistically only care about 1 count

        if ax is None:
            figsize = figsize if figsize is not None else (15.0, 4.5)
            fig, axes = plt.subplots(1, 3, sharex=True, sharey=True, figsize=figsize)
        elif isinstance(ax, Sequence):
            axes = list(ax)
            if len(axes) != 3:
                raise ValueError("Expected a sequence of exactly three axes.")
            fig = axes[0].figure
        else:
            raise ValueError("Pass ax=None or a sequence of three axes.")

        titles = list(titles) if titles is not None else [
            "Median",
            "Lower",
            "Upper",
        ]

        results: list[tuple[Axes, tuple[QuadMesh, Colorbar | None]]] = []

        median_ax, median_artists = super().plot(
            axes[0],
            scale=scale,
            vmin=vmin_eff,
            vmax=vmax_eff,
            add_cbar=add_cbar,
            cbarkwargs=cbarkwargs,
            **kwargs,
        )
        axes[0].set_title(titles[0])
        results.append((median_ax, median_artists))

        lower_matrix = Matrix(
            X=self.X_index,
            Y=self.Y_index,
            values=lower_values,
            metadata=self.metadata,
        )
        lower_ax, lower_artists = lower_matrix.plot(
            axes[1],
            scale=scale,
            vmin=vmin_eff,
            vmax=vmax_eff,
            add_cbar=False,
            cbarkwargs=None,
            **kwargs,
        )
        axes[1].set_title(titles[1])
        results.append((lower_ax, lower_artists))

        upper_matrix = Matrix(
            X=self.X_index,
            Y=self.Y_index,
            values=upper_values,
            metadata=self.metadata,
        )
        upper_ax, upper_artists = upper_matrix.plot(
            axes[2],
            scale=scale,
            vmin=vmin_eff,
            vmax=vmax_eff,
            add_cbar=False,
            cbarkwargs=None,
            **kwargs,
        )
        axes[2].set_title(titles[2])
        results.append((upper_ax, upper_artists))

        fig = axes[0].figure
        fig.tight_layout()
        return fig, results

    def to_matrix(self, *, copy: bool = True) -> Matrix:
        return Matrix(
            X=self.X_index,
            Y=self.Y_index,
            values=self.values,
            metadata=self.metadata,
            copy=copy,
        )

    def _wrap_locator_result(
        self,
        *,
        axis: int | None,
        key: tuple[slice | int | None, slice | int | None],
        values: NDArray[Any] | float,
        result: Matrix | Vector | float,
    ) -> Matrix | Vector | float:
        if isinstance(result, float):
            return result

        match axis:
            case None if isinstance(result, Matrix):
                lerr = self.lerr.__getitem__(key)
                uerr = self.uerr.__getitem__(key)
                return type(self)(
                    X=result.X_index,
                    Y=result.Y_index,
                    values=result.values,
                    lerr=lerr,
                    uerr=uerr,
                    metadata=result.metadata,
                    copy=False,
                )
            case 0 | 1 if isinstance(result, Vector):
                lerr = self.lerr.__getitem__(key)
                uerr = self.uerr.__getitem__(key)
                return AsymmetricVector(
                    X=result.X_index,
                    values=result.values,
                    lerr=lerr,
                    uerr=uerr,
                    metadata=result.metadata,
                    copy=False,
                    order="K",
                )
            case _:
                return result

    def to_hdf5(
        self,
        path: Pathlike,
        *,
        exist_ok: bool = False,
        compression: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Save the asymmetric matrix, including error bands, to HDF5."""
        if not h5py_available():
            raise ImportError("h5py is not installed; cannot save to HDF5.")

        path = Path(path)
        if path.suffix.lower() not in {".h5", ".hdf5"}:
            path = path.with_suffix(".h5")

        dataset_kwargs = dict(kwargs)
        if compression is not None:
            dataset_kwargs["compression"] = compression

        save_hdf5_2D(self, path, exist_ok=exist_ok, **dataset_kwargs)

        dataset_kwargs.setdefault("compression", "gzip")

        import h5py

        with h5py.File(path, "a") as f:
            for name, data in (("lerr", self.lerr), ("uerr", self.uerr)):
                if name in f:
                    del f[name]
                f.create_dataset(name, data=data, **dataset_kwargs)
            f.attrs["error_type"] = "asymmetric"
            f.attrs["error_sigma_scale"] = 1.0

    def save(
        self,
        path: Pathlike,
        *,
        exist_ok: bool = False,
        compression: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Alias for :meth:`to_hdf5` to mirror the :class:`Matrix` API."""
        self.to_hdf5(path, exist_ok=exist_ok, compression=compression, **kwargs)

    @classmethod
    def from_hdf5(cls, path: Pathlike) -> AsymmetricMatrix:
        """Load an asymmetric matrix (values + errors) from HDF5."""
        if not h5py_available():
            raise ImportError("h5py is not installed; cannot load from HDF5.")

        if hdf5_to_dict is None:
            raise ImportError("hdf5 helper utilities not available.")

        import h5py

        path = Path(path)
        with h5py.File(path, "r") as f:
            version = f.attrs.get("version")
            if version is not None:
                warn_version(version)
            meta = hdf5_to_dict(f, "meta/")
            X_index = Index.from_dict(hdf5_to_dict(f, "X_index/"))
            Y_index = Index.from_dict(hdf5_to_dict(f, "Y_index/"))
            values = jnp.array(f["values"])
            try:
                lerr = jnp.array(f["lerr"])
                uerr = jnp.array(f["uerr"])
            except KeyError as exc:
                raise KeyError(
                    "HDF5 file does not contain required 'lerr'/'uerr' datasets."
                ) from exc

        return cls(X=X_index, Y=Y_index, values=values, lerr=lerr, uerr=uerr, **meta)

    @classmethod
    def from_path(cls, path: Pathlike, *_, **__) -> AsymmetricMatrix:
        """Load an asymmetric matrix from disk."""
        suffix = Path(path).suffix.lower()
        if suffix not in {".h5", ".hdf5"}:
            raise ValueError(
                f"Unsupported file extension {suffix!r}. Expected '.h5' or '.hdf5'."
            )
        return cls.from_hdf5(path)