"""Alignment metadata for ensemble tracking."""
from __future__ import annotations

import secrets
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..pipeline import Stage


@dataclass(frozen=True)
class EnsembleMeta:
    """Immutable metadata for ensemble alignment tracking.
    
    Every ensemble value carries metadata to track:
    - Number of members (n)
    - Lineage token for alignment validation
    - Pipeline stage (optional semantic context)
    
    Token is a random 64-bit ID that's shared by all ensembles derived from
    the same original ensemble. It prevents accidental mixing of unrelated
    ensembles with the same size.
    
    The stage attribute tracks where in the analysis pipeline this ensemble
    sits (e.g., RAW, UNFOLDED, FIRST_GENERATION). It's optional and can be
    None for generic or intermediate ensembles.
    
    Attributes
    ----------
    n : int
        Number of ensemble members (leading axis size)
    token : int
        Random 64-bit ID for tracking ensemble lineage
    stage : Stage | None
        Pipeline stage this ensemble represents (optional)
    
    Examples
    --------
    >>> # Create new meta (fresh token)
    >>> meta = EnsembleMeta.create(n=100)
    >>> 
    >>> # With stage
    >>> from ompy.pipeline import Stage
    >>> meta = EnsembleMeta.create(n=100, stage=Stage.RAW)
    >>> 
    >>> # Check alignment
    >>> meta1 = EnsembleMeta.create(n=100)
    >>> meta2 = EnsembleMeta.create(n=100)
    >>> meta1.validate_alignment(meta2)  # Raises! Different tokens
    >>> 
    >>> # Same lineage
    >>> meta3 = meta1  # Same object, same token
    >>> meta1.validate_alignment(meta3)  # OK!
    
    """
    
    n: int
    token: int
    stage: Stage | None = None
    
    @classmethod
    def create(cls, n: int, stage: "Stage | str | int | None" = None) -> EnsembleMeta:
        """Create new metadata with fresh random token.
        
        Parameters
        ----------
        n : int
            Number of ensemble members
        stage : Stage | str | int | None, optional
            Pipeline stage for this ensemble. Can be:
            - Stage enum value: Stage.RAW
            - String (case-insensitive): "raw", "UNFOLDED", "first_generation"
            - Integer: 1, 2, 3 (enum value)
            - None (no stage)
        
        Returns
        -------
        EnsembleMeta
            New metadata with unique token
        
        Examples
        --------
        >>> meta = EnsembleMeta.create(n=100)
        >>> meta.n
        100
        >>> isinstance(meta.token, int)
        True
        
        >>> # With stage (various formats)
        >>> from ompy.pipeline import Stage
        >>> meta = EnsembleMeta.create(n=100, stage=Stage.RAW)
        >>> meta = EnsembleMeta.create(n=100, stage="raw")
        >>> meta = EnsembleMeta.create(n=100, stage="UNFOLDED")
        >>> meta.stage
        <Stage.RAW: 1>
        
        """
        if n <= 0:
            raise ValueError(f"n must be positive, got {n}")
        
        # Normalize stage using from_any
        if stage is not None:
            try:
                from ..pipeline import Stage as StageEnum
                stage = StageEnum.from_any(stage)
            except ImportError:
                # Pipeline not available, keep as-is
                pass
        
        token = secrets.randbits(64)
        return cls(n=n, token=token, stage=stage)
    
    def validate_alignment(self, other: EnsembleMeta) -> None:
        """Validate alignment with another ensemble metadata.
        
        Parameters
        ----------
        other : EnsembleMeta
            Metadata to check alignment against
        
        Raises
        ------
        ValueError
            If n differs or tokens don't match
        
        Examples
        --------
        >>> meta1 = EnsembleMeta.create(n=100)
        >>> meta2 = EnsembleMeta.create(n=100)
        >>> meta1.validate_alignment(meta2)  # Raises! Different tokens
        
        """
        if self.n != other.n:
            raise ValueError(
                f"Unaligned ensembles: n {self.n} vs {other.n}"
            )
        if self.token != other.token:
            raise ValueError(
                f"Unaligned ensembles: token mismatch "
                f"(0x{self.token:016x} vs 0x{other.token:016x})"
            )
    
    def __eq__(self, other: object) -> bool:
        """Check if metas are identical (same n and token).
        
        Note: Stage is NOT compared for equality - only n and token matter
        for alignment validation.
        """
        if not isinstance(other, EnsembleMeta):
            return False
        return self.n == other.n and self.token == other.token
    
    def _repr_html_(self) -> str:
        """HTML representation for Jupyter notebooks.
        
        Returns a formatted table showing ensemble metadata.
        """
        stage_str = f"<code>{self.stage}</code>" if self.stage else "<em>None</em>"
        
        html = f"""
        <div style="padding: 10px; border: 1px solid #ddd; border-radius: 4px; background: #f9f9f9; font-family: monospace;">
            <strong>EnsembleMeta</strong>
            <table style="margin-top: 8px; border-collapse: collapse;">
                <tr>
                    <td style="padding: 4px 12px 4px 0; color: #666;">Members:</td>
                    <td style="padding: 4px;"><strong>{self.n}</strong></td>
                </tr>
                <tr>
                    <td style="padding: 4px 12px 4px 0; color: #666;">Token:</td>
                    <td style="padding: 4px;"><code>0x{self.token:016x}</code></td>
                </tr>
                <tr>
                    <td style="padding: 4px 12px 4px 0; color: #666;">Stage:</td>
                    <td style="padding: 4px;">{stage_str}</td>
                </tr>
            </table>
        </div>
        """
        return html
    
