"""Ensemble vector for uncertainty quantification through Monte Carlo sampling."""
from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any, TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from .._version import version as __version__
from ..accel import h5py_available, jax_working
from ..array.index import Index
from ..array.vector import Vector
from ..array.vectormetadata import VectorMetadata
from ..rendering.html import collapse, table
from ..stubs import Pathlike
from ..version import warn_version
from .array import EnsembleArray, ArrayType
from .base import Ensemble
from .meta import EnsembleMeta

# Forward references for type hints
if TYPE_CHECKING:
    from ..array.error_vector import AsymmetricVector

if h5py_available():
    from ..array.filehandling import dict_to_hdf5, hdf5_to_dict
else:  # pragma: no cover - exercised when h5py missing
    dict_to_hdf5 = hdf5_to_dict = None  # type: ignore[assignment]

if jax_working():
    import jax.numpy as jnp
else:
    jnp = np


def _resolve_summary(
    summary: str | Callable[..., NDArray[Any]],
    axis: int,
    kwargs: dict[str, Any] | None = None,
) -> Callable[[NDArray[Any]], NDArray[Any]]:
    """Resolve a summary statistic function."""
    kwargs = {} if kwargs is None else dict(kwargs)
    if isinstance(summary, str):
        summary = summary.lower()
        if summary == "mean":
            return lambda data: jnp.mean(data, axis=axis, **kwargs)
        if summary == "median":
            return lambda data: jnp.median(data, axis=axis, **kwargs)
        if summary == "std":
            return lambda data: jnp.std(data, axis=axis, **kwargs)
        if summary == "min":
            return lambda data: jnp.min(data, axis=axis, **kwargs)
        if summary == "max":
            return lambda data: jnp.max(data, axis=axis, **kwargs)
        raise ValueError(f"Unknown summary {summary!r}")

    def _wrapped(data: NDArray[Any]) -> NDArray[Any]:
        try:
            return summary(data, axis=axis, **kwargs)
        except TypeError:
            if kwargs:
                try:
                    return summary(data, **kwargs)
                except TypeError:
                    return summary(data)
            return summary(data)

    return _wrapped


class EnsembleVector[Array: ArrayType](EnsembleArray[Vector, Array]):
    """Container for a Monte Carlo ensemble of aligned vectors.
    
    The ``EnsembleVector`` stores multiple :class:`Vector` instances that share
    the same shape and index structure. It enables efficient uncertainty quantification
    through Monte Carlo sampling and provides statistical summaries of the ensemble.
    
    All arithmetic operations (``+``, ``-``, ``*``, ``/``, ``**``) are supported and
    applied element-wise to each member. To apply :class:`Vector` methods to all members,
    use the ``.each`` property, or the ``.map()`` and ``.apply()`` methods.
    
    Parameters
    ----------
    members : Sequence[Vector] or ndarray
        Either a sequence of :class:`Vector` instances or a 2D array of shape
        ``(n_members, n_x)`` containing the values of each ensemble member.
        If an array is provided, ``template`` must also be specified.
    template : Vector, optional
        A :class:`Vector` instance providing the index structure and metadata
        for the ensemble. Required when ``members`` is an ndarray.
    copy : bool, default: False
        If ``True``, copy the input data. Otherwise, store references where possible.
    metadata : VectorMetadata, optional
        Metadata to override in the template vector.
    
    Attributes
    ----------
    size : int
        Number of ensemble members.
    shape : tuple[int]
        Shape ``(n_x,)`` of each member vector.
    data : ndarray
        The underlying 2D array of shape ``(n_members, n_x)``.
    template : Vector
        Template vector containing shared index structure and metadata.
    
    Examples
    --------
    Create an ensemble from a list of vectors:
    
    >>> import numpy as np
    >>> from ompy import Vector
    >>> from ompy.ensemble import EnsembleVector
    >>> 
    >>> # Create some example vectors
    >>> X = np.linspace(0, 10, 101)
    >>> vectors = []
    >>> for i in range(100):
    ...     values = np.random.rand(101) * (i + 1)
    ...     vectors.append(Vector(X=X, values=values))
    >>> 
    >>> # Create ensemble
    >>> ensemble = EnsembleVector(members=vectors)
    >>> print(f"Ensemble has {ensemble.size} members")
    >>> print(f"Each member has shape {ensemble.shape}")
    
    Perform arithmetic operations:
    
    >>> # Operations are applied to each member
    >>> scaled = ensemble * 2.0
    >>> 
    >>> # Get sum of each member
    >>> sums = ensemble.sum()  # Returns list of floats
    
    Apply Vector methods to the entire ensemble:
    
    >>> # Rebin all members using .each
    >>> rebinned = ensemble.each.rebin(factor=2.0)
    >>> print(f"Rebinned shape: {rebinned.shape}")
    >>> 
    >>> # Use .apply() for custom transformations
    >>> result = ensemble.apply(lambda v: v.rebin(factor=2.0))
    >>> 
    >>> # Use .map() to get list of results
    >>> sums = ensemble.map(lambda v: v.sum())
    
    Compute statistical summaries:
    
    >>> # Get median of ensemble
    >>> median_vector = ensemble.median()
    >>> 
    >>> # Get mean
    >>> mean_vector = ensemble.mean()
    >>> 
    >>> # Get percentiles
    >>> p16, p84 = ensemble.percentile([16, 84])
    
    Convert to error vector with confidence intervals:
    
    >>> # Create AsymmetricVector with 68% confidence interval
    >>> error_vector = ensemble.summarize(alpha=0.68)
    >>> # error_vector.values contains the median
    >>> # error_vector.lerr contains lower error (median - 16th percentile)
    >>> # error_vector.uerr contains upper error (84th percentile - median)
    
    Iterate over members:
    
    >>> # Access individual members
    >>> first_member = ensemble[0]  # Returns a Vector
    >>> 
    >>> # Iterate over all members
    >>> for i, member in enumerate(ensemble):
    ...     print(f"Member {i} sum: {member.sum()}")
    
    Save and load ensembles:
    
    >>> # Save to HDF5
    >>> ensemble.save("my_ensemble.h5")
    >>> 
    >>> # Load from HDF5
    >>> loaded = EnsembleVector.from_path("my_ensemble.h5")
    
    Notes
    -----
    - All ensemble members must have identical shapes and compatible indices.
    - Arithmetic operations maintain the ensemble structure and return new
      :class:`EnsembleVector` instances.
    - When a :class:`Vector` method is called on the ensemble, it is delegated
      to each member. Methods returning :class:`Vector` objects return a new
      :class:`EnsembleVector`; methods returning scalars return a list of results.
    - The template vector defines the shared structure (index, metadata) but
      its values may differ from the actual member values.
    
    See Also
    --------
    AsymmetricVector : Vector with asymmetric error bars.
    Vector : Base vector class for 1D data.
    EnsembleMatrix : Ensemble container for matrices.
    
    """

    def __init__(
        self,
        members: Sequence[Vector] | Array,
        *,
        template: Vector | None = None,
        copy: bool = False,
        metadata: VectorMetadata | None = None,
        _meta: EnsembleMeta | None = None,  # Internal: explicit meta
    ):
        if isinstance(members, (np.ndarray, jnp.ndarray)):
            if members.ndim != 2:
                raise ValueError(
                    f"Ensemble array must be 2D (n, x). Got shape {members.shape}."
                )
            if template is None:
                raise ValueError("template must be provided when constructing from ndarray.")
            self._data = members.copy() if copy else members
            self._template = template.clone(copy=copy)
            if self._data.shape[1] != len(self._template):
                raise ValueError(
                    "Template length does not match ensemble members. "
                    f"Expected {len(self._template)}, got {self._data.shape[1]}."
                )
        else:
            vectors = list(members)
            if not vectors:
                raise ValueError("EnsembleVector requires at least one member.")
            first = vectors[0]
            for i, vec in enumerate(vectors[1:], start=1):
                if len(vec) != len(first):
                    raise ValueError(
                        f"Vector {i} has length {len(vec)}, expected {len(first)}."
                    )
                if not first.is_compatible_with(vec):
                    raise ValueError(f"Vector {i} index is incompatible with the ensemble.")
            self._data = jnp.stack([jnp.array(vec.values, copy=copy) for vec in vectors], axis=0)
            template = first
            if metadata is not None:
                template = template.clone(metadata=metadata, copy=False)
            self._template = template.clone(copy=copy)

        if metadata is not None:
            self._template = self._template.clone(metadata=metadata, copy=False)
        
        # Initialize alignment metadata
        if _meta is not None:
            # Explicit meta provided (internal use - preserve lineage)
            if _meta.n != self._data.shape[0]:
                raise ValueError(
                    f"meta.n ({_meta.n}) != data size ({self._data.shape[0]})"
                )
            self._meta = _meta
        else:
            # Create new meta with fresh token (new lineage)
            self._meta = EnsembleMeta.create(n=self._data.shape[0])
        
        # Initialize locators for indexing
        self.iloc = EnsembleVectorIndexLocator(self)
        self.vloc = EnsembleVectorValueLocator(self, strict=True)
        self.loc = EnsembleVectorValueLocator(self, strict=False)

    @property
    def shape(self) -> tuple[int]:
        """Shape of each member vector."""
        return (self._data.shape[1],)

    def _is_compatible_member(self, other: Any) -> bool:
        """Check if other is a Vector."""
        return isinstance(other, Vector)

    def _ensure_member_compat(self, vector: Vector) -> None:
        if not self._template.is_compatible_with(vector):
            raise ValueError("Index is incompatible with the ensemble template.")
        if len(vector) != self.shape[0]:
            raise ValueError(
                f"Vector length {len(vector)} does not match ensemble members {self.shape[0]}."
            )

    def _ensure_ensemble_compat(self, other: EnsembleVector) -> None:
        if self.shape != other.shape:
            raise ValueError(
                f"Ensemble member shapes do not match: {self.shape} vs {other.shape}."
            )
        if self.size != other.size:
            raise ValueError(
                f"Ensemble sizes do not match: {self.size} vs {other.size}."
            )
        if not self._template.is_compatible_with(other._template):
            raise ValueError("Indices between ensembles are incompatible.")

    def _validate_operand_shape(self, operand: NDArray[Any]) -> None:
        """Validate array operand is 1D with correct length."""
        if operand.ndim == 1:
            if operand.shape[0] != self.shape[0]:
                raise ValueError(
                    f"Operand has length {operand.shape[0]}, expected {self.shape[0]}."
                )
        elif operand.ndim != 0:
            raise ValueError("Operand must be scalar or 1D array matching member length.")

    def __getitem__(self, index: int | slice) -> Vector | EnsembleVector:
        """Access ensemble member(s) by index.
        
        Parameters
        ----------
        index : int or slice
            - If int: Index of the member to retrieve (0-based)
            - If slice: Slice of members to retrieve
        
        Returns
        -------
        Vector or EnsembleVector
            - If int: The requested ensemble member as a :class:`Vector` instance
            - If slice: New :class:`EnsembleVector` with sliced members
        
        Examples
        --------
        >>> ensemble = EnsembleVector(members=vectors)
        >>> 
        >>> # Get single member
        >>> first = ensemble[0]
        >>> isinstance(first, Vector)
        True
        >>> 
        >>> # Get slice of members
        >>> subset = ensemble[4:9]  # Returns new EnsembleVector with 5 members
        >>> isinstance(subset, EnsembleVector)
        True
        >>> subset.size
        5
        
        """
        if isinstance(index, slice):
            # Slice the data and return new EnsembleVector
            sliced_data = self._data[index]
            # Create new meta with updated size
            from .meta import EnsembleMeta
            new_meta = EnsembleMeta(
                n=len(sliced_data),
                token=self._meta.token,
                stage=self._meta.stage
            )
            return type(self)(
                members=sliced_data,
                template=self._template,
                copy=False,
                _meta=new_meta
            )
        else:
            # Return single member
            return self._template.clone(values=self._data[index], copy=False)

    def __iter__(self) -> Iterable[Vector]:
        """Enable iteration over ensemble members.
        
        Yields
        ------
        Vector
            Each ensemble member as a :class:`Vector` instance.
        
        Examples
        --------
        >>> ensemble = EnsembleVector(members=vectors)
        >>> 
        >>> # Iterate using for loop
        >>> for member in ensemble:
        ...     print(len(member))
        >>> 
        >>> # Convert to list
        >>> member_list = list(ensemble)
        >>> 
        >>> # Use in comprehension
        >>> sums = [vec.sum() for vec in ensemble]
        
        """
        for i in range(self.size):
            yield self[i]

    def __repr__(self) -> str:
        """String representation of the EnsembleVector.
        
        Returns
        -------
        str
            A concise string showing the ensemble size and member length.
        
        Examples
        --------
        >>> ensemble = EnsembleVector(members=vectors)
        >>> print(ensemble)
        EnsembleVector(size=100, length=101)
        
        """
        return f"EnsembleVector(size={self.size}, length={self.shape[0]})"

    def _repr_html_(self) -> str:
        """Generate HTML representation for Jupyter notebook display.
        
        Shows ensemble information and the template vector structure.
        Since all members share the same index and metadata, we display
        the template information along with the number of ensemble members.
        
        Returns
        -------
        str
            HTML string for rich display in Jupyter notebooks.
        
        """
        # Ensemble-specific information
        ensemble_info = [
            ("Ensemble size", str(self.size)),
            ("Member length", str(self.shape[0])),
            ("Array type", self._data.__class__.__name__),
        ]
        
        # Template information
        template_info = [
            ("Array type", self._template.values.__class__.__name__),
            ("Total counts (template)", f"{self._template.sum():.3g}"),
        ]
        
        # Create metadata table if available
        metadata_html = ""
        if (
            hasattr(self._template, "metadata")
            and hasattr(self._template.metadata, "misc")
            and len(self._template.metadata.misc) > 0
        ):
            metadata_items = [
                (key, str(val)) for key, val in self._template.metadata.misc.items()
            ]
            metadata_html = f"""
            <div class="metadata-section">
                <h4>Template Metadata:</h4>
                {table(metadata_items, color="#f0f0f0")}
            </div>
            """
        
        # Main HTML structure
        html = f"""
        <div class="ensemble-container" style="margin: 10px 0;">
            <h3 style="margin: 0 0 10px 0; color: #2c3e50;">EnsembleVector</h3>
            <div class="ensemble-info" style="margin-bottom: 15px;">
                <h4 style="margin: 5px 0;">Ensemble Information:</h4>
                {table(ensemble_info, color="#e6f7ff")}
            </div>
            <div class="meta-section" style="margin-bottom: 15px;">
                {collapse(self._meta._repr_html_(), "Ensemble Metadata")}
            </div>
            <div class="template-info" style="margin-bottom: 15px;">
                <h4 style="margin: 5px 0;">Template Information:</h4>
                {table(template_info, color="#e6f7ff")}
            </div>
            <div class="indices-section" style="margin-top: 10px;">
                {collapse(self._template.X_index._repr_html_(), "X index")}
            </div>
            
            {metadata_html}
        </div>
        """
        
        return html

    def summarize(
        self,
        *,
        summary: str | Callable[..., NDArray[Any]] = "median",
        alpha: float = 0.68,
        summary_kwargs: dict[str, Any] | None = None,
        percentile_kwargs: dict[str, Any] | None = None,
        clip: bool = False,
    ) -> AsymmetricVector:  # type: ignore[name-defined]
        """Convert ensemble to an asymmetric error vector with confidence intervals.
        
        Parameters
        ----------
        summary : str or callable, default: "median"
            Function to compute the central value.
        alpha : float, default: 0.68
            Confidence level for the interval, between 0 and 1.
        summary_kwargs : dict, optional
            Keyword arguments for the summary function.
        percentile_kwargs : dict, optional
            Keyword arguments for the percentile computation.
        clip : bool, default: False
            If ``True``, clip error bars to ensure they don't go negative.
        
        Returns
        -------
        AsymmetricVector
            Vector with central values and asymmetric error bars.
        
        """
        # Import here to avoid circular imports
        from ..array.error_vector import AsymmetricVector
        
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0, 1].")
        
        # Compute central value
        summary_fn = _resolve_summary(summary, axis=0, kwargs=summary_kwargs)
        central_values = summary_fn(self._data)
        summary_vector = self._template.clone(values=central_values, copy=True)
        
        # Compute confidence interval
        lower_q = (1 - alpha) / 2 * 100
        upper_q = (1 + alpha) / 2 * 100
        percentile_kwargs = {} if percentile_kwargs is None else dict(percentile_kwargs)
        lower = jnp.percentile(self._data, lower_q, axis=0, **percentile_kwargs)
        upper = jnp.percentile(self._data, upper_q, axis=0, **percentile_kwargs)
        return AsymmetricVector.from_CI(summary_vector, lower, upper, clip=clip, order='K')

    def to_hdf5(
        self,
        path: Pathlike,
        *,
        exist_ok: bool = False,
        compression: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Persist the full ensemble (members + template) to HDF5.
        
        Parameters
        ----------
        path : str or Path
            Path to the HDF5 file.
        exist_ok : bool, default: False
            If ``False``, raises an error if the file already exists.
        compression : str, optional
            Compression algorithm (e.g., "gzip", "lzf").
        **kwargs
            Additional keyword arguments passed to ``h5py.create_dataset``.
        
        """
        if not h5py_available():
            raise ImportError("h5py is not installed; cannot save EnsembleVector.")
        if dict_to_hdf5 is None:
            raise ImportError("hdf5 helper utilities not available.")

        import h5py

        path = Path(path)
        if path.suffix.lower() not in {".h5", ".hdf5"}:
            path = path.with_suffix(".h5")
        if not exist_ok and path.exists():
            raise FileExistsError(f"{path} already exists")

        dataset_kwargs = dict(kwargs)
        if compression is not None:
            dataset_kwargs["compression"] = compression
        dataset_kwargs.setdefault("compression", "gzip")

        template_payload = {
            "X_index": self._template.X_index.to_dict(),
            "meta": asdict(self._template.metadata),
            "values": self._template.values,
        }

        with h5py.File(path, "w") as f:
            f.attrs["version"] = __version__
            f.attrs["kind"] = "EnsembleVector"
            f.attrs["n_members"] = self.size
            f.attrs["meta_token"] = self._meta.token
            f.attrs["meta_stage"] = str(self._meta.stage) if self._meta.stage else ""
            f.create_dataset("members", data=self._data, **dataset_kwargs)
            f.create_group("template")
            dict_to_hdf5(f, template_payload, "template/")

    @classmethod
    def from_hdf5(cls, path: Pathlike) -> EnsembleVector:
        """Load an ensemble from an HDF5 file.
        
        Parameters
        ----------
        path : str or Path
            Path to the HDF5 file to load.
        
        Returns
        -------
        EnsembleVector
            The loaded ensemble with all members, index, and metadata restored.
        
        """
        if not h5py_available():
            raise ImportError("h5py is not installed; cannot load EnsembleVector.")
        if hdf5_to_dict is None:
            raise ImportError("hdf5 helper utilities not available.")

        import h5py

        path = Path(path)
        with h5py.File(path, "r") as f:
            version = f.attrs.get("version")
            if version is not None:
                warn_version(version)
            members = jnp.array(f["members"])
            meta_token = f.attrs.get("meta_token", 0)  # Backward compatible
            meta_stage_str = f.attrs.get("meta_stage", "")  # Backward compatible
            template_dict = hdf5_to_dict(f, "template/")

        X_index = Index.from_dict(template_dict["X_index"])
        metadata = template_dict.get("meta", {})
        template_values = template_dict.get("values")
        template = Vector(
            X=X_index,
            values=template_values,
            **metadata,
        )
        
        # Reconstruct meta with stage
        from ..pipeline import Stage
        from .meta import EnsembleMeta
        meta_stage = Stage.from_any(meta_stage_str) if meta_stage_str else None
        meta = EnsembleMeta(n=len(members), token=meta_token, stage=meta_stage)
        
        return cls(members=members, template=template, copy=True, _meta=meta)

    @classmethod
    def from_path(cls, path: Pathlike) -> EnsembleVector:
        """Load an ensemble from a file path with automatic format detection.
        
        Parameters
        ----------
        path : str or Path
            Path to the file containing the ensemble.
        
        Returns
        -------
        EnsembleVector
            The loaded ensemble.
        
        """
        suffix = Path(path).suffix.lower()
        if suffix not in {".h5", ".hdf5"}:
            raise ValueError(
                f"Unsupported file extension {suffix!r}. Expected '.h5' or '.hdf5'."
            )
        return cls.from_hdf5(path)


class EnsembleVectorIndexLocator:
    """Integer-based indexing for EnsembleVector (accessed via .iloc).
    
    Applies integer-based indexing to all members of the ensemble.
    When slicing results in a Vector, returns a new EnsembleVector.
    When indexing returns a scalar, returns a stacked numpy array.
    
    """
    def __init__(self, ensemble: EnsembleVector):
        self.ensemble = ensemble
    
    def __getitem__(self, key: slice | int) -> EnsembleVector | NDArray[Any]:
        """Apply integer-based indexing to all ensemble members.
        
        Parameters
        ----------
        key : slice or int
            Indexing key, e.g., slice(10, 40) or 25
        
        Returns
        -------
        EnsembleVector or ndarray
            If key is slice → EnsembleVector
            If key is int → ndarray with stacked results
        
        """
        import warnings
        
        # Get template result to determine return type
        template = self.ensemble._template
        template_result = template.iloc[key]
        
        # Case 1: Slice → returns Vector → return EnsembleVector
        if isinstance(template_result, Vector):
            # Apply the same indexing to all members
            sliced_data = self.ensemble._data.__getitem__((slice(None), key))
            return EnsembleVector(
                members=sliced_data,
                template=template_result,
                copy=False,
            )
        
        # Case 2: Returns scalar → return stacked numpy array
        else:
            warnings.warn(
                "Indexing that reduces dimensionality returns a raw numpy array. "
                "Result shape: (n_members,). "
                "Consider using slice indexing (e.g., iloc[10:11]) to maintain EnsembleVector structure.",
                UserWarning,
                stacklevel=2
            )
            # Apply indexing to each member and stack results
            return self.ensemble._data.__getitem__((slice(None), key))
    
    def __setitem__(self, key: slice | int, value: NDArray[Any] | float) -> None:
        """Set values in all ensemble members using integer-based indexing.
        
        Parameters
        ----------
        key : slice or int
            Indexing key for where to set values
        value : array or scalar
            Values to set.
        
        """
        value_arr = jnp.asarray(value)
        
        # Check if data is a JAX array (has .at attribute) or numpy array
        is_jax_array = hasattr(self.ensemble._data, 'at')
        
        # If value has a leading dimension matching ensemble size, apply different values per member
        if value_arr.ndim > 0 and value_arr.shape[0] == self.ensemble.size:
            # Use immutable update for JAX, in-place for numpy
            if is_jax_array:
                for i in range(self.ensemble.size):
                    self.ensemble._data = self.ensemble._data.at[i, key].set(value_arr[i])
            else:
                for i in range(self.ensemble.size):
                    self.ensemble._data[i, key] = value_arr[i]
        else:
            # Apply the same value to all members
            if is_jax_array:
                self.ensemble._data = self.ensemble._data.at[:, key].set(value)
            else:
                self.ensemble._data[:, key] = value


class EnsembleVectorValueLocator:
    """Value-based indexing for EnsembleVector (accessed via .vloc or .loc).
    
    Applies value-based indexing to all members of the ensemble.
    Uses the template's index structure to convert values to integer indices.
    
    Parameters
    ----------
    ensemble : EnsembleVector
        The ensemble to index
    strict : bool, default=True
        If True (vloc), requires exact value matches.
        If False (loc), allows approximate matches.
    
    """
    def __init__(self, ensemble: EnsembleVector, strict: bool = True):
        self.ensemble = ensemble
        self.strict = strict
    
    def __getitem__(
        self, 
        key: slice | int | float | str
    ) -> EnsembleVector | NDArray[Any]:
        """Apply value-based indexing to all ensemble members.
        
        Parameters
        ----------
        key : slice, int, float, or str
            Indexing key with values, e.g., slice(None, '5MeV') or '2.5MeV'
        
        Returns
        -------
        EnsembleVector or ndarray
            If key produces Vector → EnsembleVector
            If key produces scalar → ndarray with stacked results
        
        """
        import warnings
        
        # Get template result to determine return type
        template = self.ensemble._template
        template_result = template.vloc[key] if self.strict else template.loc[key]
        
        # Case 1: Produces a Vector → return EnsembleVector
        if isinstance(template_result, Vector):
            # Convert value-based key to integer slice
            if isinstance(key, slice):
                sx = template.X_index.index_slice(key, strict=self.strict)
            else:
                sx = template.X_index.index_expression(key, strict=self.strict)
            
            # Apply integer indexing to all members
            sliced_data = self.ensemble._data.__getitem__((slice(None), sx))
            
            return EnsembleVector(
                members=sliced_data,
                template=template_result,
                copy=False,
            )
        
        # Case 2: Returns scalar → return stacked numpy array
        else:
            warnings.warn(
                "Indexing that reduces dimensionality returns a raw numpy array. "
                "Result shape: (n_members,). "
                "Consider using slice indexing to maintain EnsembleVector structure.",
                UserWarning,
                stacklevel=2
            )
            # Convert value-based key to integer index
            if isinstance(key, slice):
                sx = template.X_index.index_slice(key, strict=self.strict)
            else:
                sx = template.X_index.index_expression(key, strict=self.strict)
            
            # Apply indexing to all members
            return self.ensemble._data.__getitem__((slice(None), sx))
    
    def __setitem__(
        self,
        key: slice | int | float | str,
        value: NDArray[Any] | float
    ) -> None:
        """Set values in all ensemble members using value-based indexing.
        
        Parameters
        ----------
        key : slice, int, float, or str
            Value-based indexing key
        value : array or scalar
            Values to set
        
        """
        # Convert value-based key to integer key using template
        template = self.ensemble._template
        
        if isinstance(key, slice):
            sx = template.X_index.index_slice(key, strict=self.strict)
        else:
            sx = template.X_index.index_expression(key, strict=self.strict)
        
        # Use the integer-based setter
        value_arr = jnp.asarray(value)
        
        # Check if data is a JAX array (has .at attribute) or numpy array
        is_jax_array = hasattr(self.ensemble._data, 'at')
        
        # If value has a leading dimension matching ensemble size, apply different values per member
        if value_arr.ndim > 0 and value_arr.shape[0] == self.ensemble.size:
            if is_jax_array:
                for i in range(self.ensemble.size):
                    self.ensemble._data = self.ensemble._data.at[i, sx].set(value_arr[i])
            else:
                for i in range(self.ensemble.size):
                    self.ensemble._data[i, sx] = value_arr[i]
        else:
            # Apply the same value to all members
            if is_jax_array:
                self.ensemble._data = self.ensemble._data.at[:, sx].set(value)
            else:
                self.ensemble._data[:, sx] = value


# Register type matcher for automatic dispatch
Ensemble.register_matcher(
    matcher=lambda x: isinstance(x, Vector),
    ensemble_class=EnsembleVector,
    priority=0
)

