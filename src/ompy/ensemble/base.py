"""Base class for all ensemble types."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, Iterable, TypeVar, Self, Callable, Sequence, Any


from .meta import EnsembleMeta
from ..pipeline import Stage

T = TypeVar('T')

# Type for matcher functions: callable that takes first element and returns bool
TypeMatcher = Callable[[Any], bool]


class Ensemble(ABC, Generic[T]):
    """Abstract base for all ensemble types.
    
    Provides core functionality:
    - Alignment metadata tracking (n, token)
    - Size and length properties
    - Iteration protocol
    - Alignment validation
    - Type-based dispatch via .wrap()
    
    Subclasses:
    - EnsembleArray: For Matrix/Vector ensembles
    - EnsembleStruct: For structured/compound ensembles
    
    """
    
    _meta: EnsembleMeta
    
    # Registry for type-based dispatch in wrap()
    # List of (priority, matcher_func, ensemble_class) tuples
    _type_matchers: list[tuple[int, TypeMatcher, type[Ensemble]]] = []
    
    @property
    def meta(self) -> EnsembleMeta:
        """Alignment metadata (n members, lineage token).
        
        Returns
        -------
        EnsembleMeta
            Immutable metadata containing ensemble size and lineage token
        
        """
        return self._meta
    
    @property
    def size(self) -> int:
        """Number of ensemble members."""
        return self._meta.n
    
    def __len__(self) -> int:
        """Return ensemble size."""
        return self.size
    
    @abstractmethod
    def __iter__(self) -> Iterable[T]:
        """Iterate over ensemble members."""
        ...
    
    @abstractmethod
    def __getitem__(self, index: int) -> T:
        """Access i-th member."""
        ...
    
    @property
    def stage(self) -> Stage | None:
        """The pipeline stage this ensemble represents.
        
        Can be set directly for in-place modification or via with_stage()
        for immutable copy.
        
        Returns None for generic or intermediate ensembles.
        
        Examples
        --------
        >>> from ompy.pipeline import Stage
        >>> ensemble = EnsembleVector(vectors)
        >>> ensemble.stage  # None
        >>> 
        >>> # Setter (in-place)
        >>> ensemble.stage = "raw"  # String works!
        >>> ensemble.stage = Stage.UNFOLDED
        >>> ensemble.stage = None
        >>> 
        >>> # Immutable copy
        >>> ensemble_raw = ensemble.with_stage(Stage.RAW)
        >>> ensemble_raw.stage  # Stage.RAW
        
        """
        return self._meta.stage
    
    @stage.setter
    def stage(self, value: Stage | str | int | None) -> None:
        """Set the pipeline stage (in-place).
        
        For immutable copy, use .with_stage() instead.
        
        Parameters
        ----------
        value : Stage | str | int | None
            Pipeline stage to set. Can be:
            - Stage enum value: Stage.RAW
            - String (case-insensitive): "raw", "UNFOLDED", "first_generation"
            - Integer: 1, 2, 3 (enum value)
            - None (no stage)
        
        Examples
        --------
        >>> ensemble.stage = "raw"
        >>> ensemble.stage = Stage.UNFOLDED
        >>> ensemble.stage = 1  # Stage.RAW
        >>> ensemble.stage = None
        """
        value = Stage.from_any(value)
        
        # Update meta in-place (create new frozen meta)
        self._meta = EnsembleMeta(
            n=self._meta.n,
            token=self._meta.token,
            stage=value
        )
    
    def __stage__(self) -> Stage | None:
        """Stage accessor for get_stage() helper from pipeline module."""
        return self._meta.stage
    
    def with_stage(self, stage: Stage | str | int | None, inplace: bool = False):
        """Set pipeline stage (in-place or return copy).
        
        Useful for marking ensembles with semantic context about what
        processing stage they represent in the analysis pipeline.
        
        Parameters
        ----------
        stage : Stage | str | int | None
            Pipeline stage to set. Can be:
            - Stage enum value: Stage.RAW
            - String (case-insensitive): "raw", "UNFOLDED", "first_generation"
            - Integer: 1, 2, 3 (enum value)
            - None (no stage)
        inplace : bool, default: False
            If True, modify in-place and return None.
            If False, return a new ensemble with updated stage.
        
        Returns
        -------
        Self | None
            New ensemble with updated stage (if inplace=False), or None (if inplace=True)
        
        Examples
        --------
        >>> from ompy.pipeline import Stage
        >>> raw = EnsembleVector(vectors)
        >>> 
        >>> # All of these work:
        >>> raw_marked = raw.with_stage(Stage.RAW)
        >>> raw_marked = raw.with_stage("raw")
        >>> raw_marked = raw.with_stage("UNFOLDED")
        >>> raw_marked = raw.with_stage(1)  # Stage.RAW value
        >>> 
        >>> # In-place modification
        >>> raw.with_stage("unfolded", inplace=True)
        >>> raw.stage  # Stage.UNFOLDED
        >>> 
        >>> # Or use property setter
        >>> raw.stage = "first generation"  # Even easier!
        
        """
        # Normalize stage using from_any
        stage = Stage.from_any(stage)
        
        if inplace:
            # Modify in-place via property setter
            self.stage = stage
            return None
        else:
            # Return new ensemble - use abstract method
            new_meta = EnsembleMeta(
                n=self._meta.n,
                token=self._meta.token,
                stage=stage
            )
            return self._with_new_meta(new_meta)
    
    @abstractmethod
    def _with_new_meta(self, meta: EnsembleMeta) -> Self:
        """Create a new ensemble with different metadata.
        
        Subclasses must implement this to support with_stage().
        
        Parameters
        ----------
        meta : EnsembleMeta
            New metadata to use
        
        Returns
        -------
        Self
            New ensemble instance with updated metadata
        """
        ...
    
    def _validate_alignment(self, other: Ensemble | EnsembleMeta) -> None:
        """Validate alignment with another ensemble or meta.
        
        Parameters
        ----------
        other : Ensemble or EnsembleMeta
            Ensemble or metadata to validate against
        
        Raises
        ------
        ValueError
            If not aligned (different n or token)
        
        """
        if isinstance(other, Ensemble):
            self._meta.validate_alignment(other._meta)
        else:
            self._meta.validate_alignment(other)
    
    @abstractmethod
    def map(self, *args, **kwargs):
        """Map function over ensemble.
        
        Signature varies by subclass:
        - EnsembleArray: map(func) -> list
        - EnsembleStruct: map(func | **field_funcs) -> EnsembleStruct
        """
        ...
    
    @abstractmethod
    def summarize(self, **kwargs):
        """Convert to summary/error representation.
        
        Return type varies by subclass:
        - EnsembleMatrix: AsymmetricMatrix
        - EnsembleVector: AsymmetricVector
        - EnsembleStruct: dict of error objects
        """
        ...

    def plot(self, *args, **kwargs):
        """Plot the ensemble summary.

        Alias for :meth:`summarize().plot()`.

        See :meth:`summarize().plot()` for more details.

        """
        return self.summarize().plot(*args, **kwargs)
    
    @classmethod
    def from_path(cls, path: str | Any) -> Ensemble:
        """Load ensemble from file, automatically detecting the subclass.
        
        Reads the 'kind' attribute from the HDF5 file to determine which
        Ensemble subclass saved it, then dispatches to that subclass's
        from_hdf5() method.
        
        Parameters
        ----------
        path : str or Path
            Path to the HDF5 file containing a saved ensemble.
        
        Returns
        -------
        Ensemble
            The loaded ensemble (EnsembleMatrix, EnsembleVector, or
            EnsemblePairVector, depending on what was saved).
        
        Examples
        --------
        >>> # Save any ensemble type
        >>> ensemble = EnsembleMatrix(matrices)
        >>> ensemble.save('file.h5')
        >>> 
        >>> # Load without knowing the type
        >>> loaded = Ensemble.from_path('file.h5')
        >>> type(loaded)
        <class 'ompy.ensemble.matrix.EnsembleMatrix'>
        >>> 
        >>> # Works with all ensemble types
        >>> loaded == ensemble  # True
        
        """
        from pathlib import Path
        
        # Import h5py
        try:
            import h5py
        except ImportError:
            raise ImportError("h5py is required to load ensembles from HDF5 files")
        
        # Read the 'kind' attribute
        path = Path(path)
        with h5py.File(path, "r") as f:
            kind = f.attrs["kind"]
        
        # Import subclasses and dispatch (importing here avoids circular imports)
        from .matrix import EnsembleMatrix
        from .vector import EnsembleVector
        from .pair_vector import EnsemblePairVector
        
        registry = {
            "EnsembleMatrix": EnsembleMatrix,
            "EnsembleVector": EnsembleVector,
            "EnsemblePairVector": EnsemblePairVector,
        }
        
        ensemble_class = registry[kind]
        return ensemble_class.from_hdf5(path)
    
    @classmethod
    def register_matcher(
        cls,
        matcher: TypeMatcher,
        ensemble_class: type[Ensemble] | Callable[[Sequence[Any]], Ensemble],
        priority: int = 0
    ) -> None:
        """Register a type matcher for automatic dispatch in wrap().
        
        Matchers are tried in order of priority (higher first), then
        registration order for equal priorities.
        
        Parameters
        ----------
        matcher : callable
            Function that takes a single element and returns True if this
            ensemble type should handle sequences of that element type.
        ensemble_class : type[Ensemble] or callable
            The Ensemble subclass to instantiate for matching elements, or
            a callable that takes a sequence and returns an Ensemble instance.
        priority : int, default: 0
            Higher priority matchers are tried first. Use this to resolve
            ambiguous cases (e.g., tuple matching should have higher priority
            than generic object matching).
        
        Examples
        --------
        >>> from ompy.array import Matrix
        >>> 
        >>> # Register EnsembleMatrix for Matrix elements
        >>> Ensemble.register_matcher(
        ...     matcher=lambda x: isinstance(x, Matrix),
        ...     ensemble_class=EnsembleMatrix,
        ...     priority=0
        ... )
        >>> 
        >>> # Register with higher priority for special cases
        >>> Ensemble.register_matcher(
        ...     matcher=lambda x: isinstance(x, tuple) and len(x) == 2,
        ...     ensemble_class=lambda seq: EnsemblePairVector.from_pairs(seq),
        ...     priority=10  # Check tuples before general types
        ... )
        
        """
        cls._type_matchers.append((priority, matcher, ensemble_class))
        # Sort by priority (descending)
        cls._type_matchers.sort(key=lambda x: x[0], reverse=True)
    
    @classmethod
    def wrap(cls, sequence: Sequence[Any]) -> Ensemble:
        """Dispatch to appropriate Ensemble subclass based on element type.
        
        Examines the first element of the sequence and uses registered type
        matchers to determine which Ensemble subclass to instantiate.
        
        Parameters
        ----------
        sequence : Sequence
            A sequence of elements to wrap in an ensemble. The type of the
            first element determines which Ensemble subclass is used.
        
        Returns
        -------
        Ensemble
            An instance of the appropriate Ensemble subclass:
            - EnsembleMatrix for sequences of Matrix
            - EnsembleVector for sequences of Vector
            - EnsemblePairVector for sequences of (Vector, Vector) tuples
        
        Raises
        ------
        ValueError
            If sequence is empty
        TypeError
            If no registered matcher handles the element type
        
        Examples
        --------
        >>> from ompy.array import Matrix, Vector
        >>> 
        >>> # Automatically create EnsembleMatrix
        >>> matrices = [Matrix(...), Matrix(...), Matrix(...)]
        >>> ensemble = Ensemble.wrap(matrices)
        >>> type(ensemble)
        <class 'ompy.ensemble.matrix.EnsembleMatrix'>
        >>> 
        >>> # Automatically create EnsembleVector
        >>> vectors = [Vector(...), Vector(...)]
        >>> ensemble = Ensemble.wrap(vectors)
        >>> type(ensemble)
        <class 'ompy.ensemble.vector.EnsembleVector'>
        >>> 
        >>> # Automatically create EnsemblePairVector
        >>> pairs = [(vec1_a, vec1_b), (vec2_a, vec2_b)]
        >>> ensemble = Ensemble.wrap(pairs)
        >>> type(ensemble)
        <class 'ompy.ensemble.pair_vector.EnsemblePairVector'>
        
        """
        if not sequence:
            raise ValueError("Cannot wrap empty sequence")
        
        first_element = sequence[0]
        
        # Try each registered matcher in priority order
        for priority, matcher, ensemble_class in cls._type_matchers:
            if matcher(first_element):
                return ensemble_class(sequence)
        
        # No matcher found
        raise TypeError(
            f"No registered Ensemble type can handle elements of type "
            f"{type(first_element).__name__}. Available matchers: "
            f"{len(cls._type_matchers)}"
        )

    @classmethod
    def can_wrap(cls, x: Any) -> bool:
        """Check if any registered matcher can handle the sequence."""
        for priority, matcher, ensemble_class in cls._type_matchers:
            if matcher(x):
                return True
        return False