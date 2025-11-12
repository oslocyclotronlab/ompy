"""Base class for structured ensemble types."""
from __future__ import annotations

from abc import abstractmethod
from typing import Any, Callable, TYPE_CHECKING

from .base import Ensemble
from .meta import EnsembleMeta

if TYPE_CHECKING:
    from ..pipeline import Stage


class EnsembleStruct(Ensemble):
    """Abstract base for structured/compound ensembles.
    
    Structured ensembles contain named fields, where each field is itself an Ensemble.
    All fields must be aligned (share the same EnsembleMeta).
    
    Provides:
    - Field access by name (via properties and .get_field())
    - Unified .map() for joint or per-field transformations
    - Iteration over members (returns namedtuples)
    - Alignment validation
    
    Does NOT provide:
    - Arithmetic operations (subclasses can add if needed)
    - Array-specific operations
    
    Subclasses must define:
    - Field names and storage
    - How to construct from fields
    - Member type (for iteration)
    
    """
    
    
    @abstractmethod
    def field_names(self) -> tuple[str, ...]:
        """Return tuple of field names in order.
        
        Returns
        -------
        tuple[str, ...]
            Field names
        
        """
        ...
    
    @abstractmethod
    def get_field(self, name: str) -> Ensemble:
        """Get field by name (zero-copy view).
        
        Parameters
        ----------
        name : str
            Field name
        
        Returns
        -------
        Ensemble
            The field ensemble (zero-copy reference)
        
        Raises
        ------
        KeyError
            If field name not found
        
        """
        ...
    
    @Ensemble.stage.setter  # type: ignore[attr-defined]
    def stage(self, value: Stage | str | int | None) -> None:
        """Set the pipeline stage (in-place) for struct and all fields.
        
        Overrides the base Ensemble stage setter to also update the stage
        on all field ensembles. This ensures that structured ensembles and
        their components have consistent stage information.
        
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
        >>> pair = EnsemblePairVector(ex=ensemble_ex, gamma=ensemble_gamma)
        >>> pair.stage = "raw"
        >>> pair.ex.stage      # Also Stage.RAW
        >>> pair.gamma.stage   # Also Stage.RAW
        >>> 
        >>> # All fields updated together
        >>> pair.stage = Stage.UNFOLDED
        >>> assert pair.ex.stage == Stage.UNFOLDED
        >>> assert pair.gamma.stage == Stage.UNFOLDED
        
        """
        from ..pipeline import Stage
        
        # Normalize the stage value
        value = Stage.from_any(value)
        
        # Update self meta (call parent setter)
        # Update meta in-place (create new frozen meta)
        self._meta = EnsembleMeta(
            n=self._meta.n,
            token=self._meta.token,
            stage=value
        )
        
        # Update all field ensembles
        for name in self.field_names():
            field = self.get_field(name)
            field.stage = value
    
    def fields(self) -> dict[str, Ensemble]:
        """Return dict of all fields.
        
        Returns
        -------
        dict[str, Ensemble]
            Mapping of field names to ensemble objects
        
        """
        return {name: self.get_field(name) for name in self.field_names()}
    
    def map(
        self,
        func: Callable | None = None,
        **field_funcs: Callable
    ):
        """Map over fields - joint or per-field.
        
        Provides two calling styles:
        
        1. **Joint mapping** (single function gets all fields):
           Function receives dict of fields, returns dict or tuple of new fields.
           
        2. **Per-field mapping** (kwargs):
           Provide functions for specific fields. Unspecified fields are unchanged.
        
        Parameters
        ----------
        func : callable, optional
            Joint function with signature: f(fields: dict) -> dict | tuple
            If returns tuple, maps to field_names() in order.
        **field_funcs : callable
            Per-field functions: field_name=function
        
        Returns
        -------
        EnsembleStruct
            New struct with transformed fields (preserves meta)
        
        Raises
        ------
        ValueError
            If result fields are not aligned with original meta
        
        Examples
        --------
        >>> # Joint mapping (receives all fields)
        >>> result = pair.map(lambda f: (f['a'] + f['b'], f['a'] - f['b']))
        >>> 
        >>> # Per-field mapping (kwargs)
        >>> scaled = pair.map(a=lambda a: a * 2.0, b=lambda b: b + 10)
        >>> 
        >>> # Partial update (zero-copy for unchanged fields!)
        >>> updated = pair.map(a=lambda a: a.each.normalize())
        
        """
        if func is not None and field_funcs:
            raise ValueError("Provide either func or field_funcs, not both")
        
        if func is not None:
            # Joint mapping: function receives all fields
            result = func(self.fields())
            
            if isinstance(result, dict):
                new_fields = result
            elif isinstance(result, (tuple, list)):
                if len(result) != len(self.field_names()):
                    raise ValueError(
                        f"Function returned {len(result)} values, "
                        f"expected {len(self.field_names())}"
                    )
                new_fields = dict(zip(self.field_names(), result))
            else:
                raise ValueError("Map function must return dict or tuple")
        else:
            # Per-field mapping (or no mapping)
            new_fields = {}
            for name in self.field_names():
                if name in field_funcs:
                    new_fields[name] = field_funcs[name](self.get_field(name))
                else:
                    new_fields[name] = self.get_field(name)  # Zero-copy reuse!
        
        # Validate all results are aligned
        for name, field in new_fields.items():
            if isinstance(field, Ensemble):
                self._validate_alignment(field)
        
        return self._from_fields(**new_fields)
    
    @abstractmethod
    def _from_fields(self, **fields) -> EnsembleStruct:
        """Construct from dict of fields (preserving meta).
        
        Internal method for creating new instances from transformed fields.
        
        Parameters
        ----------
        **fields : Ensemble
            Field values
        
        Returns
        -------
        EnsembleStruct
            New instance with same meta
        
        """
        ...
    
    def unzip(self) -> tuple[Ensemble, ...]:
        """Return fields as tuple of aligned ensembles.
        
        Returns fields in the same order as field_names().
        
        Returns
        -------
        tuple[Ensemble, ...]
            Tuple of field ensembles (zero-copy views)
        
        Examples
        --------
        >>> ex_ensemble, gamma_ensemble = pair.unzip()
        
        """
        return tuple(self.get_field(name) for name in self.field_names())
    
    as_tuple = unzip  # Alias
    
    def summarize(self, **kwargs) -> dict[str, Any]:
        """Summarize each field to error representation.
        
        Calls .summarize() on each field ensemble and returns results as dict.
        
        Parameters
        ----------
        **kwargs
            Passed to each field's .summarize() method
            (typically: summary="median", alpha=0.68, etc.)
        
        Returns
        -------
        dict[str, AsymmetricVector/Matrix]
            Dict mapping field names to error objects
        
        Examples
        --------
        >>> errors = pair.summarize(alpha=0.68)
        >>> ex_error = errors['ex']  # AsymmetricVector
        >>> gamma_error = errors['gamma']  # AsymmetricVector
        
        """
        return {
            name: self.get_field(name).summarize(**kwargs)
            for name in self.field_names()
        }

