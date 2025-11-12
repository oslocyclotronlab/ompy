"""
Base Result class for pipeline results.

All pipeline functions that return Results should subclass from Result
and implement the __unwrap__() protocol.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Any, Self

from .stage import Stage

@dataclass(kw_only=True)
class Settings:
    @classmethod
    def from_dict(cls, kwargs: dict[str, Any], *, error_on_leftover: bool = True) -> Self:
        """Create Settings from a dict of kwargs.
        
        Args:
            kwargs: Dictionary of keyword arguments
            error_on_leftover: If True, raise ValueError if kwargs contains unused keys
            
        Raises:
            ValueError: If a required parameter is missing or if error_on_leftover=True
                       and there are unused kwargs
        """
        # Get required fields (those without defaults)
        required = {f.name for f in cls.__dataclass_fields__.values() 
                   if f.default is f.default_factory is None}
        
        # Check all required fields are present
        missing = required - kwargs.keys()
        if missing:
            raise ValueError(f"Missing required parameters: {missing}")
            
        # Get valid field names
        valid_fields = set(cls.__dataclass_fields__.keys())
        
        # Extract only valid kwargs
        settings_kwargs = {k: v for k, v in kwargs.items() if k in valid_fields}
        
        # Check for leftover kwargs
        leftover = set(kwargs) - valid_fields
        if leftover and error_on_leftover:
            raise ValueError(f"Unexpected parameters: {leftover}")
            
        return cls(**settings_kwargs)

    def consume(self, kwargs: dict[str, Any], *, error_on_leftover: bool = False) -> tuple[Self, dict[str, Any]]:
        """Update Settings from a dict of kwargs and return leftover kwargs.
        
        Args:
            kwargs: Dictionary of keyword arguments
            error_on_leftover: If True, raise ValueError if kwargs contains unused keys
            
        Returns:
            Dictionary of unused kwargs
            
        Raises:
            ValueError: If error_on_leftover=True and there are unused kwargs
        """
        # Get valid field names
        valid_fields = set(self.__class__.__dataclass_fields__.keys())
        
        # Extract valid and leftover kwargs
        settings_kwargs = {k: v for k, v in kwargs.items() if k in valid_fields}
        leftover_kwargs = {k: v for k, v in kwargs.items() if k not in valid_fields}
        
        # Check for leftover kwargs if requested
        if leftover_kwargs and error_on_leftover:
            raise ValueError(f"Unexpected parameters: {set(leftover_kwargs)}")

        return self.__class__(**(asdict(self) | settings_kwargs)),  leftover_kwargs
    
    def update(self, **kwargs) -> Self:
        """Create a new Settings instance with updated fields.
        
        Args:
            **kwargs: Fields to update
            
        Returns:
            New Settings instance with updated values
            
        Example:
            >>> settings = DecompositionSettings(iterations=500)
            >>> new_settings = settings.update(iterations=1000, lam_rho2=1e-3)
        """
        return self.__class__(**(asdict(self) | kwargs))


@dataclass
class ResultMeta:
    """Metadata for Result objects.
    
    Attributes:
        stage: The pipeline stage this result represents
        method: The method/algorithm used to produce this result
        parameters: Dictionary of parameters used
    """
    stage: Stage = Stage.from_any(None)
    method: str | None = None

    def __post_init__(self):
        self.stage = Stage.from_any(self.stage)

@dataclass(kw_only=True)
class Result[T](ABC):
    """Abstract base class for pipeline results.
    
    All Result subclasses must implement:
    - __unwrap__() method for lifting protocol
    
    The Result provides semantic context about how data was created,
    tracked via ResultMeta which includes the pipeline stage.
    
    Subclasses may optionally implement .best() or other methods
    for accessing the primary result, but only __unwrap__() is 
    required for the lifting decorator to work.
    
    Examples:
        >>> class UnfoldedResult(Result[Matrix]):
        ...     def __init__(self, matrix: Matrix, meta: ResultMeta):
        ...         self._matrix = matrix
        ...         self.meta = meta
        ...     
        ...     def __unwrap__(self) -> Matrix:
        ...         return self._matrix
        ...     
        ...     def best(self) -> Matrix:  # Optional!
        ...         return self._matrix
    """
    
    meta: ResultMeta = field(default_factory=ResultMeta)
    
    @abstractmethod
    def __unwrap__(self) -> T:
        """Unwrap protocol for lifting.
        
        This is the ONLY required method for Result subclasses.
        The lifting decorator calls this to extract the wrapped value.
        
        Returns:
            The unwrapped value (Matrix, Vector, tuple, etc.)
        """
        ...
    
    @property
    def stage(self) -> Stage:
        """The pipeline stage this result represents."""
        return self.meta.stage

    @stage.setter
    def stage(self, value: Stage):
        self.meta.stage = Stage.from_any(value)
    
    def __stage__(self) -> Stage:
        """Alternative access to stage (for get_stage helper)."""
        return self.meta.stage

