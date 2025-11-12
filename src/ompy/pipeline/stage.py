"""
Pipeline stage enumeration.

Defines semantic stages in the OMpy analysis pipeline.
"""
from __future__ import annotations
from enum import Enum, auto
from typing import Any


class Stage(Enum):
    """Semantic stages in the OMpy analysis pipeline.
    
    Each stage represents a distinct processing step with specific
    semantic meaning about how the data was created.
    
    Examples
    --------
    >>> # Create from string (case-insensitive)
    >>> Stage.from_str("raw")
    Stage.RAW
    >>> Stage.from_str("UNFOLDED")
    Stage.UNFOLDED
    
    >>> # Create from any representation
    >>> Stage.from_any("first_generation")
    Stage.FIRST_GENERATION
    >>> Stage.from_any(Stage.RAW)
    Stage.RAW
    >>> Stage.from_any(None)
    None
    """
    RAW = auto()
    UNFOLDED = auto()
    FIRST_GENERATION = auto()
    DECOMPOSED = auto()
    NORMALIZED = auto()
    
    def __str__(self) -> str:
        return self.name.lower()
    
    def __repr__(self) -> str:
        return f"Stage.{self.name}"
    
    @classmethod
    def from_str(cls, s: str) -> Stage:
        """Create Stage from string (case-insensitive).
        
        Parameters
        ----------
        s : str
            Stage name (case-insensitive). Can be:
            - "raw", "RAW", "Raw"
            - "unfolded", "UNFOLDED"
            - "first_generation", "FIRST_GENERATION"
            - "decomposed", "DECOMPOSED"
            - "normalized", "NORMALIZED"
        
        Returns
        -------
        Stage
            The corresponding Stage enum value
        
        Raises
        ------
        ValueError
            If string doesn't match any stage name
        
        Examples
        --------
        >>> Stage.from_str("raw")
        Stage.RAW
        >>> Stage.from_str("UNFOLDED")
        Stage.UNFOLDED
        >>> Stage.from_str("First_Generation")
        Stage.FIRST_GENERATION
        """
        # Normalize to uppercase
        s_upper = s.upper()
        
        try:
            return cls[s_upper]
        except KeyError:
            # Try with underscores/spaces normalized
            # "first generation" -> "FIRST_GENERATION"
            s_normalized = s.replace(' ', '_').replace('-', '_').upper()
            try:
                return cls[s_normalized]
            except KeyError:
                valid_names = [stage.name for stage in cls]
                raise ValueError(
                    f"Invalid stage name: '{s}'. "
                    f"Valid names: {', '.join(valid_names)}"
                )
    
    @classmethod
    def from_any(cls, value: Any) -> "Stage | None":
        """Create Stage from various representations.
        
        Handles:
        - None → None
        - Stage → Stage (passthrough)
        - str → Stage.from_str()
        - int → Stage(value)
        
        Dispatches on input type for ergonomic Stage creation.
        
        Parameters
        ----------
        value : Any
            Input value to convert to Stage
        
        Returns
        -------
        Stage | None
            Stage enum value, or None if input is None
        
        Raises
        ------
        ValueError
            If value cannot be converted to Stage
        TypeError
            If value type is not supported
        
        Examples
        --------
        >>> Stage.from_any(None)
        None
        >>> Stage.from_any("raw")
        Stage.RAW
        >>> Stage.from_any(Stage.UNFOLDED)
        Stage.UNFOLDED
        >>> Stage.from_any(1)  # By value
        Stage.RAW
        """
        # Dispatch on type
        if value is None:
            return None
        elif isinstance(value, cls):
            # Already a Stage - passthrough
            return value
        elif isinstance(value, str):
            # String - convert case-insensitively
            return cls.from_str(value)
        elif isinstance(value, int):
            # Integer - convert by enum value
            try:
                return cls(value)
            except ValueError:
                valid_values = [stage.value for stage in cls]
                raise ValueError(
                    f"Invalid stage value: {value}. "
                    f"Valid values: {valid_values}"
                )
        else:
            # Unsupported type
            raise TypeError(
                f"Cannot convert {type(value).__name__} to Stage. "
                f"Supported types: None, Stage, str, int"
            )

