"""Ensemble of aligned vector pairs with custom field names."""
from __future__ import annotations

from collections import namedtuple
from dataclasses import asdict, is_dataclass, fields as dataclass_fields
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..accel import h5py_available, jax_working
from ..rendering.html import collapse, table
from ..stubs import Pathlike
from ..version import warn_version
from .._version import version as __version__
from ..array.index import Index
from ..array.vector import Vector
from ..array.vectormetadata import VectorMetadata
from .base import Ensemble
from .meta import EnsembleMeta
from .struct import EnsembleStruct
from .vector import EnsembleVector
from typing import Sequence

if h5py_available():
    from ..array.filehandling import dict_to_hdf5, hdf5_to_dict
else:  # pragma: no cover
    dict_to_hdf5 = hdf5_to_dict = None  # type: ignore[assignment]

if jax_working():
    import jax.numpy as jnp
else:
    jnp = np


class EnsemblePairVector(EnsembleStruct):
    """Ensemble of aligned vector pairs with custom field names.
    
    Stores two aligned :class:`EnsembleVector` instances as a structure-of-arrays.
    Guarantees samplewise coupling via shared :class:`EnsembleMeta` (same n and token).
    
    All operations preserve alignment. Attempting to combine misaligned ensembles
    raises :class:`ValueError`.
    
    Parameters
    ----------
    _meta : EnsembleMeta, optional
        Internal parameter for explicit meta. If None, uses first field's meta.
    _field_names : tuple[str, str], optional
        Internal parameter for field names.
    **fields : EnsembleVector
        Exactly 2 ensemble vectors with custom field names.
        Must be aligned (same meta).
    
    Examples
    --------
    Create from existing ensembles:
    
    >>> pair = EnsemblePairVector(ex=ensemble_ex, gamma=ensemble_gamma)
    >>> pair = EnsemblePairVector(a=ensemble_a, b=ensemble_b)
    
    Access fields (zero-copy):
    
    >>> ex_ensemble = pair.ex
    >>> gamma_ensemble = pair.gamma
    >>> 
    >>> # Or by name
    >>> ex_ensemble = pair.get_field('ex')
    
    Map over fields:
    
    >>> # Joint mapping (function gets all fields)
    >>> result = pair.map(lambda f: (f['ex'] + f['gamma'], f['ex'] - f['gamma']))
    >>> 
    >>> # Per-field mapping (kwargs - cleanest!)
    >>> scaled = pair.map(ex=lambda ex: ex * 2.0, gamma=lambda g: g + 10)
    >>> 
    >>> # Partial update (zero-copy for unchanged fields!)
    >>> updated = pair.map(ex=lambda ex: ex.each.rebin(factor=2.0))
    
    Iterate over members:
    
    >>> for member in pair:
    ...     print(member.ex, member.gamma)  # namedtuple with field names
    
    Summarize to error representation:
    
    >>> errors = pair.summarize(alpha=0.68)
    >>> ex_error = errors['ex']  # AsymmetricVector
    >>> gamma_error = errors['gamma']  # AsymmetricVector
    
    """
    
    def __init__(
        self,
        *,
        _meta: EnsembleMeta | None = None,
        _field_names: tuple[str, str] | None = None,
        **fields: EnsembleVector
    ):
        """Create from named ensemble vector fields.
        
        Parameters
        ----------
        _meta : EnsembleMeta, optional
            Internal: explicit meta to use. If None, inferred from fields.
        _field_names : tuple[str, str], optional
            Internal: field names for reconstruction.
        **fields : EnsembleVector
            Exactly 2 ensemble vectors with custom names.
            Must be aligned (same meta).
        
        Raises
        ------
        ValueError
            If not exactly 2 fields provided
            If fields are not aligned (different meta)
        
        """
        if len(fields) != 2:
            raise ValueError(
                f"EnsemblePairVector requires exactly 2 fields, got {len(fields)}"
            )
        
        # Extract field names and values
        if _field_names is not None:
            # Internal use: field names explicitly provided
            self._field_names = _field_names
        else:
            # User construction: infer from kwargs
            self._field_names = tuple(fields.keys())
        
        field_values = list(fields.values())
        
        # Validate all fields are EnsembleVector
        for name, field in fields.items():
            if not isinstance(field, EnsembleVector):
                raise TypeError(
                    f"Field '{name}' must be EnsembleVector, got {type(field).__name__}"
                )
        
        # Store fields (zero-copy)
        self._fields = {name: value for name, value in zip(self._field_names, field_values)}
        
        # Set meta
        if _meta is None:
            # Use first field's meta
            _meta = field_values[0].meta
        else:
            # Validate provided meta matches fields
            _meta.validate_alignment(field_values[0].meta)
        
        self._meta = _meta
        
        # Create namedtuple class for iteration
        self._MemberTuple = namedtuple('Member', self._field_names)

    def first(self) -> EnsembleVector:
        return self._fields[self._field_names[0]]
    
    def second(self) -> EnsembleVector:
        return self._fields[self._field_names[1]]
    
    @classmethod
    def from_stacked(
        cls,
        stacked_data: dict[str, NDArray[Any]],
        templates: dict[str, Vector],
        *,
        meta: EnsembleMeta | None = None,
    ) -> EnsemblePairVector:
        """Create from stacked arrays with custom field names.
        
        Parameters
        ----------
        stacked_data : dict[str, ndarray]
            Dict mapping field names to stacked arrays.
            Each array should have shape (n_members, n_bins).
        templates : dict[str, Vector]
            Dict mapping field names to template vectors.
        meta : EnsembleMeta, optional
            Explicit meta to use. If None, creates new meta.
            If provided, validates n matches array sizes.
        
        Returns
        -------
        EnsemblePairVector
            New pair vector with aligned fields
        
        Raises
        ------
        ValueError
            If not exactly 2 fields
            If stacked arrays have different sizes
            If meta.n doesn't match array sizes
        
        Examples
        --------
        >>> pair = EnsemblePairVector.from_stacked(
        ...     stacked_data={'ex': ex_array, 'gamma': gamma_array},
        ...     templates={'ex': ex_template, 'gamma': gamma_template}
        ... )
        
        """
        if len(stacked_data) != 2:
            raise ValueError(
                f"EnsemblePairVector requires exactly 2 fields, got {len(stacked_data)}"
            )
        
        if set(stacked_data.keys()) != set(templates.keys()):
            raise ValueError("stacked_data and templates must have same keys")
        
        names = tuple(stacked_data.keys())
        
        # Validate all arrays have same size
        sizes = [arr.shape[0] for arr in stacked_data.values()]
        if len(set(sizes)) > 1:
            raise ValueError(
                f"Stacked arrays have different sizes: "
                f"{dict(zip(names, sizes))}"
            )
        
        n = sizes[0]
        
        # Create or validate meta
        if meta is None:
            meta = EnsembleMeta.create(n=n)
        else:
            if meta.n != n:
                raise ValueError(f"meta.n ({meta.n}) != array size ({n})")
        
        # Build ensemble vectors with shared meta (zero-copy)
        ensembles = {}
        for name in names:
            ensembles[name] = EnsembleVector(
                members=stacked_data[name],
                template=templates[name],
                _meta=meta,
                copy=False
            )
        
        return cls(**ensembles, _meta=meta, _field_names=names)
    
    @classmethod
    def from_pairs(
        cls,
        pairs: list[tuple[Vector, Vector]] | list[list[Vector]],
        *,
        field_names: tuple[str, str] = ("first", "second")
    ) -> EnsemblePairVector:
        """Create from a sequence of vector pairs.
        
        Convenience constructor for when you have a list of (Vector, Vector) tuples
        and want to create an ensemble of aligned pairs.
        
        Parameters
        ----------
        pairs : list of tuples or list of lists
            Sequence of (Vector, Vector) pairs. Each pair should have exactly 2 vectors.
        field_names : tuple[str, str], default: ("first", "second")
            Names for the two fields.
        
        Returns
        -------
        EnsemblePairVector
            New ensemble with aligned vector pairs
        
        Raises
        ------
        ValueError
            If pairs is empty or elements are not pairs
        
        Examples
        --------
        >>> pairs = [(vec1_a, vec1_b), (vec2_a, vec2_b), (vec3_a, vec3_b)]
        >>> ensemble = EnsemblePairVector.from_pairs(pairs)
        >>> ensemble.first  # EnsembleVector([vec1_a, vec2_a, vec3_a])
        >>> ensemble.second  # EnsembleVector([vec1_b, vec2_b, vec3_b])
        >>> 
        >>> # With custom field names
        >>> ensemble = EnsemblePairVector.from_pairs(
        ...     pairs, field_names=("ex", "gamma")
        ... )
        >>> ensemble.ex  # First components
        >>> ensemble.gamma  # Second components
        
        """
        if not pairs:
            raise ValueError("Cannot create EnsemblePairVector from empty sequence")
        
        # Validate all elements are pairs
        for i, pair in enumerate(pairs):
            if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                raise ValueError(
                    f"Element {i} is not a pair: {type(pair).__name__} with length {len(pair) if hasattr(pair, '__len__') else '?'}"
                )
            if not isinstance(pair[0], Vector) or not isinstance(pair[1], Vector):
                raise TypeError(
                    f"Element {i} pair contains non-Vector elements: "
                    f"({type(pair[0]).__name__}, {type(pair[1]).__name__})"
                )
        
        # Unzip pairs into two lists
        first_vectors = [pair[0] for pair in pairs]
        second_vectors = [pair[1] for pair in pairs]
        
        # Create EnsembleVector instances with shared meta
        meta = EnsembleMeta.create(n=len(pairs))
        first_ensemble = EnsembleVector(members=first_vectors, _meta=meta)
        second_ensemble = EnsembleVector(members=second_vectors, _meta=meta)
        
        # Create EnsemblePairVector with custom field names
        fields = {field_names[0]: first_ensemble, field_names[1]: second_ensemble}
        return cls(**fields, _meta=meta, _field_names=field_names)
    
    # ========== Field Access ==========
    
    def field_names(self) -> tuple[str, ...]:
        """Return field names."""
        return self._field_names
    
    def get_field(self, name: str) -> EnsembleVector:
        """Get field by name (zero-copy view).
        
        Parameters
        ----------
        name : str
            Field name
        
        Returns
        -------
        EnsembleVector
            The field ensemble (zero-copy reference)
        
        Raises
        ------
        KeyError
            If field not found
        
        """
        if name not in self._fields:
            raise KeyError(
                f"No field '{name}'. Available fields: {self._field_names}"
            )
        return self._fields[name]
    
    def __getattr__(self, name: str) -> EnsembleVector:
        """Access fields as attributes (e.g., pair.ex, pair.gamma).
        
        This enables convenient field access:
        >>> pair.ex  # Instead of pair.get_field('ex')
        
        """
        # Avoid recursion for private attributes
        if name.startswith('_'):
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")
        
        # Try to get as field
        if hasattr(self, '_fields') and name in self._fields:
            return self._fields[name]
        
        raise AttributeError(
            f"'{type(self).__name__}' has no attribute '{name}'. "
            f"Available fields: {getattr(self, '_field_names', [])}"
        )
    
    # ========== Iteration ==========
    
    def __iter__(self):
        """Iterate over members (returns namedtuples).
        
        Yields
        ------
        namedtuple
            Named tuple with fields matching field_names()
        
        Examples
        --------
        >>> pair = EnsemblePairVector(ex=ex_ensemble, gamma=gamma_ensemble)
        >>> for member in pair:
        ...     print(member.ex, member.gamma)  # namedtuple!
        
        """
        for i in range(self.size):
            field_values = [self._fields[name][i] for name in self._field_names]
            yield self._MemberTuple(*field_values)
    
    def __getitem__(self, index: int | slice):
        """Get member(s) by index.
        
        Parameters
        ----------
        index : int or slice
            - If int: returns member as namedtuple
            - If slice: returns new EnsemblePairVector with selected members
        
        Returns
        -------
        namedtuple or EnsemblePairVector
            - If int: Named tuple with field values for this member
            - If slice: New ensemble with sliced fields (no copy)
        
        Examples
        --------
        >>> # Get single member
        >>> member = ensemble[5]  # Returns namedtuple
        >>> 
        >>> # Get slice of ensemble
        >>> subset = ensemble[4:9]  # Returns new EnsemblePairVector with 5 members
        >>> subset = ensemble[::2]  # Every other member
        
        """
        if isinstance(index, slice):
            # Slice each field (EnsembleVector now supports slicing)
            sliced_fields = {name: self._fields[name][index] for name in self._field_names}
            
            # Determine new size from the sliced fields
            first_field = next(iter(sliced_fields.values()))
            new_size = len(first_field)
            
            # Create new meta with updated size (preserve token and stage)
            from .meta import EnsembleMeta
            new_meta = EnsembleMeta(
                n=new_size, 
                token=self._meta.token, 
                stage=self._meta.stage
            )
            
            return type(self)(**sliced_fields, _field_names=self._field_names, _meta=new_meta)
        else:
            # Return single member as namedtuple
            field_values = [self._fields[name][index] for name in self._field_names]
            return self._MemberTuple(*field_values)
    
    # ========== Construction Helper ==========
    
    def _from_fields(self, **fields) -> EnsemblePairVector:
        """Construct from fields preserving meta."""
        return type(self)(**fields, _meta=self._meta, _field_names=self._field_names)
    
    def _with_new_meta(self, meta: EnsembleMeta) -> EnsemblePairVector:
        """Create a new EnsemblePairVector with different metadata.
        
        Used by with_stage() to create copies with updated stage information.
        
        Parameters
        ----------
        meta : EnsembleMeta
            New metadata to use
        
        Returns
        -------
        EnsemblePairVector
            New instance with updated metadata
        
        """
        # Create new field ensembles with updated meta
        new_fields = {}
        for name in self._field_names:
            old_field = self._fields[name]
            # Each field is an EnsembleVector - call its _with_new_meta
            new_fields[name] = old_field._with_new_meta(meta)
        
        return type(self)(**new_fields, _meta=meta, _field_names=self._field_names)
    
    # ========== Persistence ==========
    
    def to_hdf5(
        self,
        path: Pathlike,
        *,
        exist_ok: bool = False,
        compression: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Save pair vector to HDF5.
        
        Parameters
        ----------
        path : str or Path
            Path to HDF5 file
        exist_ok : bool, default: False
            If False, raises if file exists
        compression : str, optional
            Compression algorithm (e.g., "gzip", "lzf")
        **kwargs
            Additional arguments for h5py.create_dataset
        
        """
        if not h5py_available():
            raise ImportError("h5py is not installed")
        if dict_to_hdf5 is None:
            raise ImportError("hdf5 helper utilities not available")
        
        import h5py
        
        path = Path(path)
        if path.suffix.lower() not in {".h5", ".hdf5"}:
            path = path.with_suffix(".h5")
        if not exist_ok and path.exists():
            raise FileExistsError(f"{path} already exists")
        
        with h5py.File(path, "w") as f:
            # Store metadata
            f.attrs["version"] = __version__
            f.attrs["kind"] = "EnsemblePairVector"
            f.attrs["n_members"] = self.size
            f.attrs["meta_token"] = self._meta.token
            f.attrs["meta_stage"] = str(self._meta.stage) if self._meta.stage else ""
            f.attrs["field_names"] = list(self._field_names)
            
            # Store each field
            for name in self._field_names:
                field_group = f.create_group(name)
                field = self._fields[name]
                
                # Store field data
                dataset_kwargs = {}
                if compression:
                    dataset_kwargs["compression"] = compression
                dataset_kwargs.update(kwargs)
                
                field_group.create_dataset("values", data=field.stacked, **dataset_kwargs)
                
                # Store template
                template_payload = {
                    "X_index": field.template.X_index.to_dict(),
                    "meta": asdict(field.template.metadata) if field.template.metadata else {},
                    "values": field.template.values,
                }
                dict_to_hdf5(field_group, template_payload, "template/")
    
    @classmethod
    def from_hdf5(cls, path: Pathlike) -> EnsemblePairVector:
        """Load pair vector from HDF5.
        
        Parameters
        ----------
        path : str or Path
            Path to HDF5 file
        
        Returns
        -------
        EnsemblePairVector
            Loaded pair vector
        
        """
        if not h5py_available():
            raise ImportError("h5py is not installed")
        if hdf5_to_dict is None:
            raise ImportError("hdf5 helper utilities not available")
        
        import h5py
        
        with h5py.File(path, "r") as f:
            # Load metadata
            version = f.attrs.get("version", "unknown")
            warn_version(version)
            
            kind = f.attrs["kind"]
            if kind != "EnsemblePairVector":
                raise ValueError(f"File contains {kind}, expected EnsemblePairVector")
            
            n_members = f.attrs["n_members"]
            meta_token = f.attrs["meta_token"]
            meta_stage_str = f.attrs.get("meta_stage", "")  # Backward compatible
            field_names = tuple(f.attrs["field_names"])
            
            # Reconstruct meta with stage
            from ..pipeline import Stage
            meta_stage = Stage.from_any(meta_stage_str) if meta_stage_str else None
            meta = EnsembleMeta(n=n_members, token=meta_token, stage=meta_stage)
            
            # Load each field
            templates = {}
            stacked_data = {}
            
            for name in field_names:
                field_group = f[name]
                
                # Load values
                stacked_data[name] = np.array(field_group["values"])
                
                # Load template
                template_payload = hdf5_to_dict(field_group, "template/")
                X_index = Index.from_dict(template_payload["X_index"])
                # VectorMetadata is a dataclass, construct from dict
                metadata = VectorMetadata(**template_payload["meta"]) if template_payload["meta"] else None
                templates[name] = Vector(
                    X=X_index,
                    values=template_payload["values"],
                    metadata=metadata
                )
        
        # Construct using from_stacked to ensure proper alignment
        return cls.from_stacked(stacked_data, templates, meta=meta)
    
    @classmethod
    def from_path(cls, path: Pathlike) -> EnsemblePairVector:
        """Load from file with automatic format detection.
        
        Currently only supports HDF5.
        
        Parameters
        ----------
        path : str or Path
            Path to file
        
        Returns
        -------
        EnsemblePairVector
            Loaded pair vector
        
        """
        path = Path(path)
        if path.suffix.lower() in {".h5", ".hdf5"}:
            return cls.from_hdf5(path)
        else:
            raise ValueError(
                f"Unknown file extension: {path.suffix}. "
                "Supported: .h5, .hdf5"
            )
    
    def save(
        self,
        path: Pathlike,
        *,
        exist_ok: bool = False,
        compression: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Save to HDF5 (alias for to_hdf5)."""
        self.to_hdf5(path, exist_ok=exist_ok, compression=compression, **kwargs)
    
    # ========== Representation ==========
    
    def __repr__(self) -> str:
        """String representation."""
        field_str = ", ".join(f"{name}=EnsembleVector" for name in self._field_names)
        return f"EnsemblePairVector(size={self.size}, fields=[{field_str}])"
    
    def _repr_html_(self) -> str:
        """HTML representation for Jupyter notebooks."""
        
        # Build summary table (main info card - always visible)
        rows = []
        rows.append(("Property", "Value"))
        rows.append(("Type", "EnsemblePairVector"))
        rows.append(("Size (members)", str(self.size)))
        rows.append(("Fields", ", ".join(self._field_names)))
        rows.append(("Meta Token", f"0x{self._meta.token:016x}"))
        
        # Add stage if set
        if self._meta.stage is not None:
            rows.append(("Stage", str(self._meta.stage)))
        
        summary_table = table(rows)
        
        # Build field shape table (collapsed)
        shape_rows = [("Field", "Shape")]
        for name in self._field_names:
            field = self._fields[name]
            shape_rows.append((name, str(field.shape)))
        
        shape_table = table(shape_rows)
        
        # Build individual field previews (each collapsed separately)
        field_previews = []
        for name in self._field_names:
            field = self._fields[name]
            # Each field gets its own collapsible section
            field_collapsed = collapse(field._repr_html_(), f"Field: {name}")
            field_previews.append(field_collapsed)
        
        field_section = "\n".join(field_previews)
        
        html = f"""
        <div>
        <h3>EnsemblePairVector</h3>
        {summary_table}
        {collapse(shape_table, "Field Shapes")}
        {collapse(field_section, "Field Details")}
        </div>
        """
        
        return html

    def plot(self):
        summary = self.summarize()
        (_,a), (_,b) = summary.items()
        return a.plot(), b.plot()


# ========== Type Matchers for Automatic Dispatch ==========

def _is_dataclass_with_2_vectors(x):
    """Check if x is a dataclass with exactly 2 Vector fields."""
    try:
        if not is_dataclass(x):
            return False
        fields = dataclass_fields(x)
        if len(fields) != 2:
            return False
        values = [getattr(x, f.name) for f in fields]
        return all(isinstance(v, Vector) for v in values)
    except (AttributeError, TypeError):
        return False


def _wrap_dataclass_pairs(sequence: Sequence[Any]) -> EnsemblePairVector:
    """Wrapper to convert sequence of dataclasses to EnsemblePairVector with field names."""
    if not sequence:
        raise ValueError("Cannot wrap empty sequence")
    
    # Get field names from first dataclass
    fields = dataclass_fields(sequence[0])
    field_names = tuple(f.name for f in fields)
    
    # Extract pairs
    pairs = []
    for item in sequence:
        values = [getattr(item, f.name) for f in fields]
        pairs.append(tuple(values))
    
    return EnsemblePairVector.from_pairs(pairs, field_names=field_names)


def _is_namedtuple_with_2_vectors(x):
    """Check if x is a namedtuple with exactly 2 Vector fields."""
    try:
        # Namedtuples have _fields attribute and are tuples
        if not (isinstance(x, tuple) and hasattr(x, '_fields')):
            return False
        if len(x._fields) != 2:
            return False
        return all(isinstance(v, Vector) for v in x)
    except (AttributeError, TypeError):
        return False


def _wrap_namedtuple_pairs(sequence):
    """Wrapper to convert sequence of namedtuples to EnsemblePairVector with field names."""
    if not sequence:
        raise ValueError("Cannot wrap empty sequence")
    
    # Get field names from first namedtuple
    field_names = sequence[0]._fields
    
    # Extract pairs (namedtuples are already tuples)
    pairs = [tuple(item) for item in sequence]
    
    return EnsemblePairVector.from_pairs(pairs, field_names=field_names)


def _is_dict_with_2_vectors(x):
    """Check if x is a dict with exactly 2 Vector values."""
    try:
        if not isinstance(x, dict):
            return False
        if len(x) != 2:
            return False
        return all(isinstance(v, Vector) for v in x.values())
    except (AttributeError, TypeError):
        return False


def _wrap_dict_pairs(sequence):
    """Wrapper to convert sequence of dicts to EnsemblePairVector with field names."""
    if not sequence:
        raise ValueError("Cannot wrap empty sequence")
    
    # Get field names from first dict (preserve order)
    field_names = tuple(sequence[0].keys())
    
    # Extract pairs in consistent order
    pairs = []
    for item in sequence:
        values = [item[name] for name in field_names]
        pairs.append(tuple(values))
    
    return EnsemblePairVector.from_pairs(pairs, field_names=field_names)


def _is_vector_pair(x):
    """Check if x is a tuple/list of exactly 2 Vector instances."""
    try:
        return (
            isinstance(x, (tuple, list))
            and len(x) == 2
            and not hasattr(x, '_fields')  # Exclude namedtuples
            and isinstance(x[0], Vector)
            and isinstance(x[1], Vector)
        )
    except (AttributeError, IndexError, TypeError):
        return False


def _wrap_vector_pairs(sequence):
    """Wrapper to convert sequence of plain pairs to EnsemblePairVector."""
    return EnsemblePairVector.from_pairs(sequence)


# Register matchers in priority order (highest first)

# Priority 15: Dataclasses (most specific structure)
Ensemble.register_matcher(
    matcher=_is_dataclass_with_2_vectors,
    ensemble_class=_wrap_dataclass_pairs,
    priority=15
)

# Priority 12: Namedtuples (structured with field names)
Ensemble.register_matcher(
    matcher=_is_namedtuple_with_2_vectors,
    ensemble_class=_wrap_namedtuple_pairs,
    priority=12
)

# Priority 11: Dicts (structured with keys as field names)
Ensemble.register_matcher(
    matcher=_is_dict_with_2_vectors,
    ensemble_class=_wrap_dict_pairs,
    priority=11
)

# Priority 10: Plain tuples/lists (no field name information)
Ensemble.register_matcher(
    matcher=_is_vector_pair,
    ensemble_class=_wrap_vector_pairs,
    priority=10
)

