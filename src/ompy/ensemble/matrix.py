"""Ensemble matrix for uncertainty quantification through Monte Carlo sampling."""
from __future__ import annotations

import warnings
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any, Self

import numpy as np
from numpy.typing import NDArray

from .._version import version as __version__
from ..accel import h5py_available, jax_working
from ..array.error_matrix import AsymmetricMatrix
from ..array.index import Index
from ..array.matrix import Matrix
from ..array.matrixmetadata import MatrixMetadata
from ..array.vector import Vector
from ..rendering.html import collapse, table
from ..stubs import Pathlike
from ..version import warn_version
from .array import EnsembleArray, ArrayType
from .base import Ensemble
from .meta import EnsembleMeta
from .vector import EnsembleVector

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






class EnsembleMatrix[Array: ArrayType](EnsembleArray[Matrix, Array]):
    """Container for a Monte Carlo ensemble of aligned matrices.
    
    The ``EnsembleMatrix`` stores multiple :class:`Matrix` instances that share
    the same shape and index structure. It enables efficient uncertainty quantification
    through Monte Carlo sampling and provides statistical summaries of the ensemble.
    
    All arithmetic operations (``+``, ``-``, ``*``, ``/``, ``**``) are supported and
    applied element-wise to each member. To apply :class:`Matrix` methods to all members,
    use the ``.each`` property, or the ``.map()`` and ``.apply()`` methods.
    
    Parameters
    ----------
    members : Sequence[Matrix] or ndarray
        Either a sequence of :class:`Matrix` instances or a 3D array of shape
        ``(n_members, n_x, n_y)`` containing the values of each ensemble member.
        If an array is provided, ``template`` must also be specified.
    template : Matrix, optional
        A :class:`Matrix` instance providing the index structure and metadata
        for the ensemble. Required when ``members`` is an ndarray.
    copy : bool, default: False
        If ``True``, copy the input data. Otherwise, store references where possible.
    metadata : MatrixMetadata, optional
        Metadata to override in the template matrix.
    
    Attributes
    ----------
    size : int
        Number of ensemble members.
    shape : tuple[int, int]
        Shape ``(n_x, n_y)`` of each member matrix.
    data : ndarray
        The underlying 3D array of shape ``(n_members, n_x, n_y)``.
    template : Matrix
        Template matrix containing shared index structure and metadata.
    
    Examples
    --------
    Create an ensemble from a list of matrices:
    
    >>> import numpy as np
    >>> from ompy import Matrix
    >>> from ompy.ensemble import EnsembleMatrix
    >>> 
    >>> # Create some example matrices
    >>> X = np.linspace(0, 10, 11)
    >>> Y = np.linspace(0, 15, 16)
    >>> matrices = []
    >>> for i in range(100):
    ...     values = np.random.rand(11, 16) * (i + 1)
    ...     matrices.append(Matrix(X=X, Y=Y, values=values))
    >>> 
    >>> # Create ensemble
    >>> ensemble = EnsembleMatrix(members=matrices)
    >>> print(f"Ensemble has {ensemble.size} members")
    >>> print(f"Each member has shape {ensemble.shape}")
    
    Perform arithmetic operations:
    
    >>> # Operations are applied to each member
    >>> scaled = ensemble * 2.0
    >>> normalized = ensemble / ensemble.sum()  # Note: sum() returns a list
    
    Apply Matrix methods to the entire ensemble:
    
    >>> # Rebin all members using .each
    >>> rebinned = ensemble.each.rebin(axis="X", factor=2.0)
    >>> print(f"Rebinned shape: {rebinned.shape}")
    >>> 
    >>> # Normalize all members
    >>> normalized = ensemble.each.normalize(axis=0)
    >>> 
    >>> # Use .apply() for custom transformations
    >>> result = ensemble.apply(lambda m: m.rebin(axis="X", factor=2.0))
    >>> 
    >>> # Use .map() to get list of results
    >>> sums = ensemble.map(lambda m: m.sum())
    
    Compute statistical summaries:
    
    >>> # Get median of ensemble
    >>> median_matrix = ensemble.median()
    >>> 
    >>> # Get mean with custom summary function
    >>> mean_matrix = ensemble.summary("mean")
    >>> 
    >>> # Get percentiles
    >>> p16, p84 = ensemble.percentile([16, 84])
    
    Convert to error matrix with confidence intervals:
    
    >>> # Create AsymmetricMatrix with 68% confidence interval
    >>> error_matrix = ensemble.summarize(alpha=0.68)
    >>> # error_matrix.values contains the median
    >>> # error_matrix.lerr contains lower error (median - 16th percentile)
    >>> # error_matrix.uerr contains upper error (84th percentile - median)
    
    Iterate over members:
    
    >>> # Access individual members
    >>> first_member = ensemble[0]  # Returns a Matrix
    >>> 
    >>> # Iterate over all members
    >>> for i, member in enumerate(ensemble):
    ...     print(f"Member {i} sum: {member.sum()}")
    
    Save and load ensembles:
    
    >>> # Save to HDF5
    >>> ensemble.save("my_ensemble.h5")
    >>> 
    >>> # Load from HDF5
    >>> loaded = EnsembleMatrix.from_path("my_ensemble.h5")
    
    Notes
    -----
    - All ensemble members must have identical shapes and compatible indices.
    - Arithmetic operations maintain the ensemble structure and return new
      :class:`EnsembleMatrix` instances.
    - When a :class:`Matrix` method is called on the ensemble, it is delegated
      to each member. Methods returning :class:`Matrix` objects return a new
      :class:`EnsembleMatrix`; methods returning scalars return a list of results.
    - The template matrix defines the shared structure (indices, metadata) but
      its values may differ from the actual member values.
    
    See Also
    --------
    AsymmetricMatrix : Matrix with asymmetric error bars.
    Matrix : Base matrix class for 2D data.
    
    """

    def __init__(
        self,
        members: Sequence[Matrix] | Array,
        *,
        template: Matrix | None = None,
        copy: bool = False,
        metadata: MatrixMetadata | None = None,
        _meta: EnsembleMeta | None = None,  # Internal: explicit meta
    ):
        if isinstance(members, (np.ndarray, jnp.ndarray)):
            if members.ndim != 3:
                raise ValueError(
                    f"Ensemble array must be 3D (n, x, y). Got shape {members.shape}."
                )
            if template is None:
                raise ValueError("template must be provided when constructing from ndarray.")
            self._data = members.copy() if copy else members
            self._template = template.clone(copy=copy)
            if self._data.shape[1:] != self._template.shape:
                raise ValueError(
                    "Template shape does not match ensemble members. "
                    f"Expected {self._template.shape}, got {self._data.shape[1:]}."
                )
        else:
            matrices = list(members)
            if not matrices:
                raise ValueError("EnsembleMatrix requires at least one member.")
            first = matrices[0]
            for i, mat in enumerate(matrices[1:], start=1):
                if mat.shape != first.shape:
                    raise ValueError(
                        f"Matrix {i} has shape {mat.shape}, expected {first.shape}."
                    )
                if not first.is_compatible_with_X(mat):
                    raise ValueError(f"Matrix {i} X index is incompatible with the ensemble.")
                if not first.is_compatible_with_Y(mat):
                    raise ValueError(f"Matrix {i} Y index is incompatible with the ensemble.")
            self._data = jnp.stack([jnp.array(mat.values, copy=copy) for mat in matrices], axis=0)
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
        self.iloc = EnsembleIndexLocator(self)
        self.vloc = EnsembleValueLocator(self, strict=True)
        self.loc = EnsembleValueLocator(self, strict=False)

    @property
    def shape(self) -> tuple[int, int]:
        """Shape of each member matrix."""
        return self._data.shape[1], self._data.shape[2]

    def _is_compatible_member(self, other: Any) -> bool:
        """Check if other is a Matrix."""
        return isinstance(other, Matrix)

    def _ensure_member_compat(self, matrix: Matrix) -> None:
        if not self._template.is_compatible_with_X(matrix):
            raise ValueError("X index is incompatible with the ensemble template.")
        if not self._template.is_compatible_with_Y(matrix):
            raise ValueError("Y index is incompatible with the ensemble template.")
        if matrix.values.shape != self.shape:
            raise ValueError(
                f"Matrix shape {matrix.values.shape} does not match ensemble members {self.shape}."
            )

    def _ensure_ensemble_compat(self, other: EnsembleMatrix) -> None:
        if self.shape != other.shape:
            raise ValueError(
                f"Ensemble member shapes do not match: {self.shape} vs {other.shape}."
            )
        if self.size != other.size:
            raise ValueError(
                f"Ensemble sizes do not match: {self.size} vs {other.size}."
            )
        if not self._template.is_compatible_with_X(other._template):
            raise ValueError("X indices between ensembles are incompatible.")
        if not self._template.is_compatible_with_Y(other._template):
            raise ValueError("Y indices between ensembles are incompatible.")

    def _validate_operand_shape(self, operand: Array) -> None:
        """Validate array operand is 2D with correct shape."""
        if operand.ndim == 2:
            if operand.shape != self.shape:
                raise ValueError(
                    f"Operand has shape {operand.shape}, expected {self.shape}."
                )
        elif operand.ndim != 0:
            raise ValueError("Operand must be scalar or 2D array matching member shape.")

    def __getitem__(self, index: int | slice) -> Matrix | Self:
        """Access ensemble member(s) by index.
        
        Parameters
        ----------
        index : int or slice
            Index of the member to retrieve (0-based), or slice for multiple members.
        
        Returns
        -------
        Matrix or EnsembleMatrix
            - If ``index`` is an int: Returns the member as a :class:`Matrix` instance.
            - If ``index`` is a slice: Returns a new :class:`EnsembleMatrix` with the
              selected members (no copy of data).
        
        Examples
        --------
        >>> ensemble = EnsembleMatrix(members=matrices)
        >>> 
        >>> # Get first member
        >>> first = ensemble[0]
        >>> isinstance(first, Matrix)
        True
        >>> 
        >>> # Get last member
        >>> last = ensemble[-1]
        >>> 
        >>> # Get subset of members (returns new ensemble)
        >>> subset = ensemble[5:20]
        >>> subset.size
        15
        >>> isinstance(subset, EnsembleMatrix)
        True
        
        See Also
        --------
        __iter__ : Iterate over all members.
        iter_matrices : Explicit iterator over members.
        
        """
        if isinstance(index, slice):
            # Return new ensemble with sliced members (no copy)
            return type(self)(members=self._data[index], template=self._template, copy=False)
        else:
            # Return individual member
            return self._template.clone(values=self._data[index], copy=False)

    def __iter__(self) -> Iterable[Matrix]:
        """Enable iteration over ensemble members.
        
        Yields
        ------
        Matrix
            Each ensemble member as a :class:`Matrix` instance.
        
        Examples
        --------
        >>> ensemble = EnsembleMatrix(members=matrices)
        >>> 
        >>> # Iterate using for loop
        >>> for member in ensemble:
        ...     print(member.shape)
        >>> 
        >>> # Convert to list
        >>> member_list = list(ensemble)
        >>> 
        >>> # Use in comprehension
        >>> sums = [mat.sum() for mat in ensemble]
        
        See Also
        --------
        __getitem__ : Access members by index.
        
        """
        for i in range(self.size):
            yield self[i]

    def __repr__(self) -> str:
        """String representation of the EnsembleMatrix.
        
        Returns
        -------
        str
            A concise string showing the ensemble size and member shape.
        
        Examples
        --------
        >>> ensemble = EnsembleMatrix(members=matrices)
        >>> print(ensemble)
        EnsembleMatrix(size=100, shape=(11, 16))
        
        """
        return f"EnsembleMatrix(size={self.size}, shape={self.shape})"

    def _repr_html_(self) -> str:
        """Generate HTML representation for Jupyter notebook display.
        
        Shows ensemble information and the template matrix structure.
        Since all members share the same indices and metadata, we display
        the template information along with the number of ensemble members.
        
        Returns
        -------
        str
            HTML string for rich display in Jupyter notebooks.
        
        """
        # Ensemble-specific information
        ensemble_info = [
            ("Ensemble size", str(self.size)),
            ("Member shape", f"{self.shape[0]} × {self.shape[1]}"),
            ("Array type", self._data.__class__.__name__),
        ]
        
        # Template information (similar to Matrix)
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
            <h3 style="margin: 0 0 10px 0; color: #2c3e50;">EnsembleMatrix</h3>
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
                <div style="margin-top: 10px;">
                    {collapse(self._template.Y_index._repr_html_(), "Y index")}
                </div>
            </div>
            
            {metadata_html}
        </div>
        """
        
        return html

    def summarize(
        self,
        *,
        summary: str | Callable[..., Array] = "median",
        alpha: float = 0.68,
        summary_kwargs: dict[str, Any] | None = None,
        percentile_kwargs: dict[str, Any] | None = None,
        clip: bool = False,
    ) -> AsymmetricMatrix:  # type: ignore[name-defined]
        """Convert ensemble to an asymmetric error matrix with confidence intervals.
        
        Computes a central value (e.g., median) and confidence intervals from the
        ensemble distribution, returning an :class:`AsymmetricMatrix` with lower
        and upper error bars.
        
        Parameters
        ----------
        summary : str or callable, default: "median"
            Function to compute the central value. See :meth:`summary` for options.
        alpha : float, default: 0.68
            Confidence level for the interval, between 0 and 1.
            
            - ``0.68`` gives approximately 1-sigma (68% confidence)
            - ``0.95`` gives approximately 2-sigma (95% confidence)
            
            The lower and upper percentiles are computed as:
            
            - Lower: ``(1 - alpha) / 2 * 100``
            - Upper: ``(1 + alpha) / 2 * 100``
        
        summary_kwargs : dict, optional
            Keyword arguments for the summary function.
        percentile_kwargs : dict, optional
            Keyword arguments for the percentile computation.
        clip : bool, default: False
            If ``True``, clip error bars to ensure they don't go below zero
            or above the maximum value in the ensemble.
        
        Returns
        -------
        AsymmetricMatrix
            Matrix with central values and asymmetric error bars:
            
            - ``values``: Central value (from summary function)
            - ``lerr``: Lower error (central - lower percentile)
            - ``uerr``: Upper error (upper percentile - central)
        
        Examples
        --------
        >>> ensemble = EnsembleMatrix(members=matrices)
        >>> 
        >>> # 68% confidence interval (1-sigma)
        >>> error_mat = ensemble.summarize(alpha=0.68)
        >>> 
        >>> # 95% confidence interval (2-sigma)
        >>> error_mat_95 = ensemble.summarize(alpha=0.95)
        >>> 
        >>> # Use mean as central value
        >>> error_mat_mean = ensemble.summarize(summary="mean", alpha=0.68)
        >>> 
        >>> # Access the components
        >>> central = error_mat.values
        >>> lower_error = error_mat.lerr
        >>> upper_error = error_mat.uerr
        >>> 
        >>> # Plot with error bars
        >>> fig, ax = error_mat.plot(n_sigma=1.0)
        
        Notes
        -----
        For a Gaussian distribution, ``alpha=0.68`` corresponds to the 16th and
        84th percentiles (mean ± 1σ), and ``alpha=0.95`` corresponds to the
        2.5th and 97.5th percentiles (mean ± 2σ).
        
        See Also
        --------
        summary : Compute statistical summary of ensemble.
        percentile : Compute arbitrary percentiles.
        AsymmetricMatrix : Matrix with asymmetric error bars.
        
        """
        # Import here to avoid circular imports
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0, 1].")
        
        # Compute central value
        summary_fn = _resolve_summary(summary, axis=0, kwargs=summary_kwargs)
        central_values = summary_fn(self._data)
        summary_matrix = self._template.clone(values=central_values, copy=True)
        
        # Compute confidence intervals
        lower_q = (1 - alpha) / 2 * 100
        upper_q = (1 + alpha) / 2 * 100
        percentile_kwargs = {} if percentile_kwargs is None else dict(percentile_kwargs)
        lower = jnp.percentile(self._data, lower_q, axis=0, **percentile_kwargs)
        upper = jnp.percentile(self._data, upper_q, axis=0, **percentile_kwargs)
        return AsymmetricMatrix.from_CI(summary_matrix, lower, upper, clip=clip)

    def to_hdf5(
        self,
        path: Pathlike,
        *,
        exist_ok: bool = False,
        compression: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Persist the full ensemble (members + template) to HDF5.
        
        Saves the ensemble to an HDF5 file, including all member matrices,
        the template structure, and metadata. The file can be loaded later
        using :meth:`from_hdf5` or :meth:`from_path`.
        
        Parameters
        ----------
        path : str or Path
            Path to the HDF5 file. If the extension is not ``.h5`` or ``.hdf5``,
            it will be changed to ``.h5``.
        exist_ok : bool, default: False
            If ``False``, raises an error if the file already exists.
            If ``True``, overwrites existing files.
        compression : str, optional
            Compression algorithm to use. Common options are:
            
            - ``"gzip"``: Good compression, moderate speed (default)
            - ``"lzf"``: Fast compression, less compression ratio
            - ``None``: No compression
        
        **kwargs
            Additional keyword arguments passed to ``h5py.create_dataset``.
        
        Examples
        --------
        >>> ensemble = EnsembleMatrix(members=matrices)
        >>> 
        >>> # Save with default compression
        >>> ensemble.to_hdf5("results.h5")
        >>> 
        >>> # Save with custom compression
        >>> ensemble.to_hdf5("results.h5", compression="lzf")
        >>> 
        >>> # Overwrite existing file
        >>> ensemble.to_hdf5("results.h5", exist_ok=True)
        >>> 
        >>> # Load it back
        >>> loaded = EnsembleMatrix.from_hdf5("results.h5")
        
        Notes
        -----
        The HDF5 file contains:
        
        - ``members``: 3D dataset with all member values
        - ``template/X_index``: X-axis index information
        - ``template/Y_index``: Y-axis information  
        - ``template/meta``: Metadata dictionary
        - ``template/values``: Template matrix values
        - File attributes: version, kind, n_members
        
        See Also
        --------
        save : Alias for this method.
        from_hdf5 : Load ensemble from HDF5.
        from_path : Load ensemble from file (auto-detects format).
        
        """
        if not h5py_available():
            raise ImportError("h5py is not installed; cannot save EnsembleMatrix.")
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
            "Y_index": self._template.Y_index.to_dict(),
            "meta": asdict(self._template.metadata),
            "values": np.asarray(self._template.values),
        }

        with h5py.File(path, "w") as f:
            f.attrs["version"] = __version__
            f.attrs["kind"] = "EnsembleMatrix"
            f.attrs["n_members"] = self.size
            f.attrs["meta_token"] = self._meta.token
            f.attrs["meta_stage"] = str(self._meta.stage) if self._meta.stage else ""
            f.create_dataset("members", data=np.asarray(self._data), **dataset_kwargs)
            f.create_group("template")
            dict_to_hdf5(f, template_payload, "template/")

    @classmethod
    def from_hdf5(cls, path: Pathlike) -> EnsembleMatrix:
        """Load an ensemble from an HDF5 file.
        
        Loads a complete :class:`EnsembleMatrix` that was previously saved
        using :meth:`to_hdf5` or :meth:`save`.
        
        Parameters
        ----------
        path : str or Path
            Path to the HDF5 file to load.
        
        Returns
        -------
        EnsembleMatrix
            The loaded ensemble with all members, indices, and metadata restored.
        
        Raises
        ------
        ImportError
            If h5py is not installed.
        FileNotFoundError
            If the file does not exist.
        
        Examples
        --------
        >>> # Save an ensemble
        >>> ensemble.to_hdf5("my_ensemble.h5")
        >>> 
        >>> # Load it back
        >>> loaded = EnsembleMatrix.from_hdf5("my_ensemble.h5")
        >>> 
        >>> # Verify they match
        >>> assert loaded.size == ensemble.size
        >>> assert loaded.shape == ensemble.shape
        
        See Also
        --------
        from_path : Load from file with automatic format detection.
        to_hdf5 : Save ensemble to HDF5.
        
        """
        if not h5py_available():
            raise ImportError("h5py is not installed; cannot load EnsembleMatrix.")
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
        Y_index = Index.from_dict(template_dict["Y_index"])
        metadata = template_dict.get("meta", {})
        template_values = template_dict.get("values")
        template = Matrix(
            X=X_index,
            Y=Y_index,
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
    def from_path(cls, path: Pathlike) -> EnsembleMatrix:
        """Load an ensemble from a file path with automatic format detection.
        
        Convenience method that automatically detects the file format based on
        the file extension and loads the ensemble appropriately.
        
        Parameters
        ----------
        path : str or Path
            Path to the file containing the ensemble. Currently supports
            HDF5 files with extensions ``.h5`` or ``.hdf5``.
        
        Returns
        -------
        EnsembleMatrix
            The loaded ensemble.
        
        Raises
        ------
        ValueError
            If the file extension is not recognized.
        
        Examples
        --------
        >>> # Save ensemble
        >>> ensemble.save("results.h5")
        >>> 
        >>> # Load with automatic format detection
        >>> loaded = EnsembleMatrix.from_path("results.h5")
        
        See Also
        --------
        from_hdf5 : Load specifically from HDF5 format.
        save : Save ensemble to file.
        
        """
        suffix = Path(path).suffix.lower()
        if suffix not in {".h5", ".hdf5"}:
            raise ValueError(
                f"Unsupported file extension {suffix!r}. Expected '.h5' or '.hdf5'."
            )
        return cls.from_hdf5(path)


class EnsembleIndexLocator:
    """Integer-based indexing for EnsembleMatrix (accessed via .iloc).
    
    Applies integer-based indexing to all members of the ensemble.
    When slicing results in a Matrix, returns a new EnsembleMatrix.
    When slicing results in a Vector or scalar, returns a stacked numpy array.
    
    Examples
    --------
    >>> ensemble = EnsembleMatrix(members=matrices)
    >>> 
    >>> # Get a submatrix from all members
    >>> sub_ensemble = ensemble.iloc[10:40, 20:50]  # Returns EnsembleMatrix
    >>> 
    >>> # Get a slice that produces vectors
    >>> vectors = ensemble.iloc[25, :]  # Returns array of shape (n_members, n_y)
    >>> 
    >>> # Get a single value from all members
    >>> values = ensemble.iloc[25, 30]  # Returns array of shape (n_members,)
    
    """
    def __init__(self, ensemble: EnsembleMatrix):
        self.ensemble = ensemble
    
    def __getitem__(self, key: tuple[slice | int, slice | int]) -> EnsembleMatrix | NDArray[Any]:
        """Apply integer-based indexing to all ensemble members.
        
        Parameters
        ----------
        key : tuple of slices/ints
            Indexing key, e.g., (slice(10, 40), slice(20, 50)) or (10, slice(None))
        
        Returns
        -------
        EnsembleMatrix or ndarray
            If key produces Matrix → EnsembleMatrix
            If key produces Vector or scalar → ndarray with stacked results
        
        """
        import warnings
        
        # Get template result to determine return type
        template = self.ensemble._template
        template_result = template.iloc[key]
        
        # Case 1: Both dimensions are slices → returns Matrix → return EnsembleMatrix
        if isinstance(template_result, Matrix):
            # Apply the same indexing to all members
            sliced_data = self.ensemble._data.__getitem__((slice(None),) + key)
            return EnsembleMatrix(
                members=sliced_data,
                template=template_result,
                _meta=self.ensemble._meta,  # Preserve lineage
                copy=False,
            )
        
        # Case 2: One dimension is int, other is slice → returns Vector → return EnsembleVector
        elif isinstance(template_result, Vector):
            # Apply indexing to each member and stack results
            sliced_data = self.ensemble._data.__getitem__((slice(None),) + key)
            return EnsembleVector(
                members=sliced_data,
                template=template_result,
                _meta=self.ensemble._meta,  # Preserve lineage
                copy=False,
            )
        
        # Case 3: Both dimensions are ints → returns scalar → return stacked array
        else:
            warnings.warn(
                "Indexing that returns scalars produces a raw numpy array of shape (n_members,). "
                "Consider using slicing (e.g., iloc[10:11, 20:21]) to maintain ensemble structure.",
                UserWarning,
                stacklevel=2
            )
            # Apply indexing to each member and stack results
            results = []
            for i in range(self.ensemble.size):
                member_result = self.ensemble._data[i].__getitem__(key)
                results.append(member_result)
            return jnp.stack(results, axis=0)
    
    def __setitem__(self, key: tuple[slice | int, slice | int], value: NDArray[Any] | float) -> None:
        """Set values in all ensemble members using integer-based indexing.
        
        Parameters
        ----------
        key : tuple of slices/ints
            Indexing key for where to set values
        value : array or scalar
            Values to set. Can be:
            - Scalar: same value for all members
            - Array matching the sliced shape: same array for all members
            - Array with leading dimension matching n_members: different values per member
        
        """
        value_arr = jnp.asarray(value)
        
        # Check if data is a JAX array (has .at attribute) or numpy array
        is_jax_array = hasattr(self.ensemble._data, 'at')
        
        # If value has a leading dimension matching ensemble size, apply different values per member
        if value_arr.ndim > 0 and value_arr.shape[0] == self.ensemble.size:
            # Use immutable update for JAX, in-place for numpy
            if is_jax_array:
                for i in range(self.ensemble.size):
                    self.ensemble._data = self.ensemble._data.at[i, key[0], key[1]].set(value_arr[i])
            else:
                for i in range(self.ensemble.size):
                    self.ensemble._data[i][key] = value_arr[i]
        else:
            # Apply the same value to all members
            full_key = (slice(None),) + key
            if is_jax_array:
                self.ensemble._data = self.ensemble._data.at[full_key].set(value)
            else:
                self.ensemble._data[full_key] = value


class EnsembleValueLocator:
    """Value-based indexing for EnsembleMatrix (accessed via .vloc or .loc).
    
    Applies value-based indexing to all members of the ensemble.
    Uses the template's index structure to convert values to integer indices.
    
    Parameters
    ----------
    ensemble : EnsembleMatrix
        The ensemble to index
    strict : bool, default=True
        If True (vloc), requires exact value matches.
        If False (loc), allows approximate matches.
    
    Examples
    --------
    >>> ensemble = EnsembleMatrix(members=matrices)
    >>> 
    >>> # Strict value-based slicing
    >>> sub_ensemble = ensemble.vloc[:'5MeV', '1MeV':'10MeV']
    >>> 
    >>> # Flexible value-based slicing (allows approximate matches)
    >>> sub_ensemble = ensemble.loc[:'5.1MeV', '1.05MeV':]
    
    """
    def __init__(self, ensemble: EnsembleMatrix, strict: bool = True):
        self.ensemble = ensemble
        self.strict = strict
    
    def __getitem__(
        self, 
        key: tuple[slice | int | float | str, slice | int | float | str]
    ) -> EnsembleMatrix | NDArray[Any]:
        """Apply value-based indexing to all ensemble members.
        
        Parameters
        ----------
        key : tuple
            Indexing key with values, e.g., (slice(None, '5MeV'), slice('1MeV', '10MeV'))
        
        Returns
        -------
        EnsembleMatrix or ndarray
            If key produces Matrix → EnsembleMatrix
            If key produces Vector or scalar → ndarray with stacked results

        """
        
        # Get template result to determine return type and the integer indices
        template = self.ensemble._template
        template_result = template.vloc[key] if self.strict else template.loc[key]
        
        # Case 1: Both dimensions produce a Matrix → return EnsembleMatrix
        if isinstance(template_result, Matrix):
            # Apply the same value-based indexing to all members
            # We need to extract the integer slices used by the template
            x_key, y_key = key if isinstance(key, tuple) else (key, slice(None))
            
            # Convert value-based keys to integer slices
            if isinstance(x_key, slice):
                sx = template.X_index.index_slice(x_key, strict=self.strict)
            else:
                sx = template.X_index.index_expression(x_key, strict=self.strict)
            
            if isinstance(y_key, slice):
                sy = template.Y_index.index_slice(y_key, strict=self.strict)
            else:
                sy = template.Y_index.index_expression(y_key, strict=self.strict)
            
            # Apply integer indexing to all members
            int_key = (sx, sy)
            sliced_data = self.ensemble._data.__getitem__((slice(None),) + int_key)
            
            return EnsembleMatrix(
                members=sliced_data,
                template=template_result,
                _meta=self.ensemble._meta,  # Preserve lineage
                copy=False,
            )
        
        # Case 2: One dimension is value, other is slice → returns Vector → return EnsembleVector
        elif isinstance(template_result, Vector):
            
            # Convert value-based key to integer key
            x_key, y_key = key if isinstance(key, tuple) else (key, slice(None))
            
            if isinstance(x_key, slice):
                sx = template.X_index.index_slice(x_key, strict=self.strict)
            else:
                sx = template.X_index.index_expression(x_key, strict=self.strict)
            
            if isinstance(y_key, slice):
                sy = template.Y_index.index_slice(y_key, strict=self.strict)
            else:
                sy = template.Y_index.index_expression(y_key, strict=self.strict)
            
            int_key = (sx, sy)
            sliced_data = self.ensemble._data.__getitem__((slice(None),) + int_key)
            
            return EnsembleVector(
                members=sliced_data,
                template=template_result,
                _meta=self.ensemble._meta,  # Preserve lineage
                copy=False,
            )
        
        # Case 3: Both dimensions are values (returns scalar) → return stacked array
        else:
            warnings.warn(
                "Indexing that returns scalars produces a raw numpy array of shape (n_members,). "
                "Consider using slicing to maintain ensemble structure.",
                UserWarning,
                stacklevel=2
            )
            # Convert value-based key to integer key
            x_key, y_key = key if isinstance(key, tuple) else (key, slice(None))
            
            if isinstance(x_key, slice):
                sx = template.X_index.index_slice(x_key, strict=self.strict)
            else:
                sx = template.X_index.index_expression(x_key, strict=self.strict)
            
            if isinstance(y_key, slice):
                sy = template.Y_index.index_slice(y_key, strict=self.strict)
            else:
                sy = template.Y_index.index_expression(y_key, strict=self.strict)
            
            int_key = (sx, sy)
            
            # Apply indexing to each member and stack results
            results = []
            for i in range(self.ensemble.size):
                member_result = self.ensemble._data[i].__getitem__(int_key)
                results.append(member_result)
            return jnp.stack(results, axis=0)
    
    def __setitem__(
        self,
        key: tuple[slice | int | float | str, slice | int | float | str],
        value: NDArray[Any] | float
    ) -> None:
        """Set values in all ensemble members using value-based indexing.
        
        Parameters
        ----------
        key : tuple
            Value-based indexing key
        value : array or scalar
            Values to set
        
        """
        # Convert value-based key to integer key using template
        template = self.ensemble._template
        x_key, y_key = key if isinstance(key, tuple) else (key, slice(None))
        
        if isinstance(x_key, slice):
            sx = template.X_index.index_slice(x_key, strict=self.strict)
        else:
            sx = template.X_index.index_expression(x_key, strict=self.strict)
        
        if isinstance(y_key, slice):
            sy = template.Y_index.index_slice(y_key, strict=self.strict)
        else:
            sy = template.Y_index.index_expression(y_key, strict=self.strict)
        
        int_key = (sx, sy)
        
        # Use the integer-based setter with JAX's immutable API
        value_arr = jnp.asarray(value)
        
        # If value has a leading dimension matching ensemble size, apply different values per member
        if value_arr.ndim > 0 and value_arr.shape[0] == self.ensemble.size:
            for i in range(self.ensemble.size):
                self.ensemble._data = self.ensemble._data.at[i].__getitem__(int_key).set(value_arr[i])
        else:
            # Apply the same value to all members using JAX's at[] API
            full_key = (slice(None),) + int_key
            self.ensemble._data = self.ensemble._data.at[full_key].set(value)
        
        # Also update the template
        if self.strict:
            self.ensemble._template.vloc[key] = value if not (
                isinstance(value_arr, jnp.ndarray) and value_arr.shape[0] == self.ensemble.size
            ) else jnp.mean(value_arr, axis=0)
        else:
            self.ensemble._template.loc[key] = value if not (
                isinstance(value_arr, jnp.ndarray) and value_arr.shape[0] == self.ensemble.size
            ) else jnp.mean(value_arr, axis=0)


# Register type matcher for automatic dispatch
Ensemble.register_matcher(
    matcher=lambda x: isinstance(x, Matrix),
    ensemble_class=EnsembleMatrix,
    priority=0
)

