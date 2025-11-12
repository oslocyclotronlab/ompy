from __future__ import annotations
from abc import abstractmethod
from typing import Any, Generic, Sequence
from numpy.typing import NDArray
from ..array.abstractarray import AbstractArray
from typing import TypeVar, Iterable, Callable, Self
from ..accel import jax_working
from ..stubs import Pathlike
from .base import Ensemble
from .meta import EnsembleMeta
import operator
import numpy as np
from typing import TYPE_CHECKING

if jax_working():
    import jax.numpy as jnp
else:
    jnp = np

if TYPE_CHECKING:
    import jax
    type JaxArray = jax.Array

type ArrayType = NDArray[Any] | JaxArray

MemberType = TypeVar('MemberType', bound=AbstractArray)

class MemberView(Generic[MemberType]):
    """View object for applying operations to all ensemble members.
    
    Accessed via the `.each` property on ensemble objects. Provides a clear
    and explicit interface for delegating method calls to each member.
    
    Examples
    --------
    >>> ensemble = EnsembleMatrix(members=matrices)
    >>> 
    >>> # Call methods on all members
    >>> rebinned = ensemble.each.rebin(axis='X', factor=2.0)
    >>> normalized = ensemble.each.normalize(axis=0)
    >>> 
    >>> # Get list of scalar results
    >>> sums = ensemble.each.sum()  # Returns [sum1, sum2, ...]
    >>> 
    >>> # Iterate over members
    >>> for member in ensemble.each:
    ...     print(member.shape)
    
    """
    
    def __init__(self, ensemble: EnsembleArray[MemberType]):
        self._ensemble = ensemble
    
    def __iter__(self) -> Iterable[MemberType]:
        """Iterate over ensemble members."""
        return iter(self._ensemble)
    
    def __getattr__(self, name: str) -> Any:
        """Delegate method calls to each member.
        
        Returns
        -------
        EnsembleArray or list
            If method returns same type as members → returns new ensemble
            Otherwise → returns list of results
        """
        # Prevent delegation of magic methods
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(
                f"MemberView has no attribute '{name}'"
            )
        
        # Check if attribute exists on template
        try:
            template_attr = getattr(self._ensemble._get_template(), name)
        except (AttributeError, Exception):
            raise AttributeError(
                f"Member type has no attribute '{name}'"
            )
        
        # If not callable, return template's attribute value
        if not callable(template_attr):
            return template_attr
        
        # Create wrapper that applies method to all members
        def member_method(*args: Any, **kwargs: Any) -> Any:
            return self._ensemble._apply_to_members(name, *args, **kwargs)
        
        return member_method


class EnsembleArray[MemberType: AbstractArray, Array: ArrayType](Ensemble[MemberType]):
    """Abstract base class for array-like ensemble containers.
    
    Provides common functionality for ensembles of matrices, vectors, or other
    array-like objects. Subclasses must implement member-specific behavior.
    
    Inherits from Ensemble[T] and adds:
    - Stacked data array (_data)
    - Template for structure (_template)
    - Arithmetic operations
    - Statistical methods
    - .each delegation
    
    """
    
    _data: Array
    _template: MemberType
    
    @abstractmethod
    def __init__(self, members: Sequence[MemberType] | Array, **kwargs):
        """Initialize ensemble from members or array."""
        ...
    
    @abstractmethod
    def __getitem__(self, index: int | slice) -> MemberType | Self:
        """Access member(s) by index.
        
        Parameters
        ----------
        index : int or slice
            - If int: returns individual member
            - If slice: returns new ensemble with selected members
        """
        ...
    
    @abstractmethod
    def __iter__(self) -> Iterable[MemberType]:
        """Iterate over members."""
        ...

    @property
    def members(self) -> list[MemberType]:
        return list(self)
    
    @property
    @abstractmethod
    def shape(self) -> tuple[int, ...]:
        """Shape of each member (subclass-specific)."""
        ...
    
    def _get_template(self) -> MemberType:
        """Get the template member."""
        return self._template
    
    @property
    def each(self) -> MemberView[MemberType]:
        """Access member operations through this view.
        
        Returns a view object that delegates method calls to all members.
        Methods that return the same type as members will return a new ensemble.
        Other methods return a list of results.
        
        Examples
        --------
        >>> # Call methods on all members
        >>> rebinned = ensemble.each.rebin(factor=2.0)
        >>> 
        >>> # Get list of results
        >>> sums = ensemble.each.sum()
        >>> 
        >>> # Iterate over members
        >>> for member in ensemble.each:
        ...     print(member.sum())
        
        """
        return MemberView(self)
    
    def _with_new_meta(self, meta: EnsembleMeta) -> Self:
        """Create a new ensemble with different metadata.
        
        Implementation of abstract method from Ensemble base class.
        """
        return type(self)(
            members=self._data,
            template=self._template,
            _meta=meta,
            copy=False
        )

    def map(self, func: Callable[[MemberType], Any]) -> list[Any]:
        """Apply a function to each member and return list of results.
        
        This method does NOT modify the ensemble or create a new one.
        It simply applies the function and returns results as a list.
        
        Parameters
        ----------
        func : callable
            Function to apply to each member. Signature: `func(member) -> result`
        
        Returns
        -------
        list
            List containing the result of applying func to each member.
        
        Examples
        --------
        >>> # Get sum of each member
        >>> sums = ensemble.map(lambda m: m.sum())
        >>> 
        >>> # Get custom metric from each member
        >>> maxvals = ensemble.map(lambda m: m.values.max())
        >>> 
        >>> # Apply complex operation
        >>> results = ensemble.map(lambda m: m.rebin(factor=2.0).sum())
        
        See Also
        --------
        apply : Apply function and update ensemble.
        each : Property-based delegation.
        
        """
        return [func(member) for member in self]
    
    def apply(
        self, 
        func: Callable[[MemberType], MemberType | None], 
        inplace: bool = False
    ) -> Self | None:
        """Apply a function to each member, updating the ensemble.
        
        The function should either:
        1. Return a modified member (when inplace=False)
        2. Modify the member in-place and return None (when inplace=True)
        
        Parameters
        ----------
        func : callable
            Function to apply. Signature: `func(member) -> member_or_none`
            - If inplace=False: Must return a new member object
            - If inplace=True: Should modify member in-place, return None
        inplace : bool, default=False
            If True, modifies this ensemble in-place and returns None.
            If False, returns a new ensemble with modified members.
        
        Returns
        -------
        EnsembleArray or None
            New ensemble if inplace=False, None if inplace=True.
        
        Examples
        --------
        >>> # Create new ensemble with transformed members
        >>> rebinned = ensemble.apply(lambda m: m.rebin(factor=2.0))
        >>> 
        >>> # Modify in place
        >>> ensemble.apply(lambda m: m.rebin(factor=2.0, inplace=True), inplace=True)
        >>> 
        >>> # Chain operations
        >>> result = ensemble.apply(lambda m: m.rebin(factor=2.0).normalize())
        
        See Also
        --------
        map : Apply function and return list of results.
        each : Property-based delegation.
        
        """
        if inplace:
            # Apply function in-place, updating self._data
            updated_members = []
            for member in self:
                result = func(member)
                if result is not None:
                    raise ValueError(
                        "With inplace=True, function must modify member in-place and return None. "
                        f"Got return value of type {type(result).__name__}"
                    )
                updated_members.append(member.values)
            self._data = jnp.stack(updated_members, axis=0)
            
            # Also apply to template if possible
            try:
                template_copy = self._template.clone(copy=True)
                func(template_copy)
                self._template = template_copy
            except Exception:
                # If template update fails, keep old template
                pass
            
            return None
        else:
            # Create new ensemble with results
            results = []
            for member in self:
                result = func(member)
                if result is None:
                    raise ValueError(
                        "With inplace=False, function must return a new member. "
                        "Got None. Did you mean inplace=True?"
                    )
                results.append(result)
            
            # Return new ensemble with updated members (preserve lineage)
            return type(self)(members=results, _meta=self._meta)
    
    def _apply_to_members(self, method_name: str, *args, **kwargs) -> Any:
        """Internal: Apply a method by name to all members.
        
        Used by MemberView for delegation.
        """
        # Clone template to test return type
        test_template = self._template.clone(copy=True)
        test_attr = getattr(test_template, method_name)
        template_result = test_attr(*args, **kwargs)
        
        # Handle inplace operations (return None)
        if template_result is None:
            updated_members = []
            for member in self:
                member_copy = member.clone(copy=True)
                getattr(member_copy, method_name)(*args, **kwargs)
                updated_members.append(member_copy.values)
            self._data = jnp.stack(updated_members, axis=0)
            
            # Apply to actual template
            getattr(self._template, method_name)(*args, **kwargs)
            return None
        
        # Handle methods that return same type as member
        elif isinstance(template_result, type(self._template)):
            results = []
            for member in self:
                result = getattr(member, method_name)(*args, **kwargs)
                results.append(result)
            
            # Create new ensemble with results (preserve lineage)
            return type(self)(members=results, _meta=self._meta)
        
        # For other return types, return list
        else:
            results = []
            for member in self:
                result = getattr(member, method_name)(*args, **kwargs)
                results.append(result)
            return results
    
    # ========== Properties (array-specific) ==========
    
    @property
    def stacked(self) -> Array:
        """Zero-copy view of stacked data (ensemble axis is leading).
        
        Returns
        -------
        ndarray
            The underlying data array with shape (n_members, ...).
            This is a zero-copy view of the internal data.
        
        """
        return self._data
    
    @property
    def data(self) -> Array:
        """The underlying data array of shape (n_members, ...).
        
        Alias for .stacked property.
        """
        return self._data
    
    @property
    def template(self) -> MemberType:
        """Template member containing shared index structure and metadata."""
        return self._template
    
    @template.setter
    def template(self, template: MemberType) -> Self:
        self._template = template
        return self
    
    # ========== Statistical methods (common to all ensembles) ==========
    
    def mean(self) -> MemberType:
        """Compute the mean of the ensemble.
        
        Returns
        -------
        MemberType
            Member containing the mean value of each bin across all members.
        
        Examples
        --------
        >>> mean_result = ensemble.mean()
        
        """
        values = jnp.mean(self._data, axis=0)
        return self._template.clone(values=values, copy=True)
    
    def median(self) -> MemberType:
        """Compute the median of the ensemble.
        
        Returns
        -------
        MemberType
            Member containing the median value of each bin across all members.
        
        Examples
        --------
        >>> median_result = ensemble.median()
        
        """
        values = jnp.median(self._data, axis=0)
        return self._template.clone(values=values, copy=True)
    
    def percentile(
        self,
        q: float | Sequence[float],
        *,
        percentile_kwargs: dict[str, Any] | None = None,
    ) -> Array:
        """Compute percentiles of the ensemble distribution.
        
        Parameters
        ----------
        q : float or sequence of floats
            Percentile(s) to compute, in range [0, 100].
        percentile_kwargs : dict, optional
            Additional keyword arguments passed to the percentile function.
        
        Returns
        -------
        ndarray
            Array of percentile values.
        
        Examples
        --------
        >>> p50 = ensemble.percentile(50)
        >>> p16, p84 = ensemble.percentile([16, 84])
        
        """
        percentile_kwargs = {} if percentile_kwargs is None else dict(percentile_kwargs)
        # JAX requires q to be an array, not a list
        q_arr = jnp.asarray(q) if isinstance(q, (list, tuple)) else q
        return jnp.percentile(self._data, q_arr, axis=0, **percentile_kwargs)
    
    @abstractmethod
    def summarize(
        self,
        *,
        summary: str | Callable[..., Array] = "median",
        alpha: float = 0.68,
        summary_kwargs: dict[str, Any] | None = None,
        percentile_kwargs: dict[str, Any] | None = None,
        clip: bool = False,
    ):
        """Convert ensemble to error object with confidence intervals.
        
        Must be implemented by subclasses to return appropriate error type:
        - EnsembleMatrix returns AsymmetricMatrix
        - EnsembleVector returns AsymmetricVector
        
        Parameters
        ----------
        summary : str or callable, default: "median"
            Function to compute central value ("mean", "median", etc.)
        alpha : float, default: 0.68
            Confidence level (0.68 ≈ 1σ, 0.95 ≈ 2σ)
        summary_kwargs : dict, optional
            Keyword arguments for summary function
        percentile_kwargs : dict, optional
            Keyword arguments for percentile computation
        clip : bool, default: False
            Whether to clip error bars to non-negative values
        
        Returns
        -------
        AsymmetricMatrix or AsymmetricVector
            Object with central values and error bars
        
        """
        ...
    
    # ========== Arithmetic operations (common to all ensembles) ==========
    
    def _unary_operation(self, op: Callable[[Any], Any]) -> Self:
        """Apply unary operation to all members."""
        result_members = op(self._data)
        result_template_values = op(self._template.values)
        template = self._template.clone(
            values=jnp.asarray(result_template_values),
            copy=False,
        )
        return type(self)(
            members=jnp.asarray(result_members),
            template=template,
            _meta=self._meta,  # Preserve lineage token
            copy=False,
        )
    
    def _binary_operation(
        self,
        other: MemberType | Self | Array | float,
        op: Callable[[Any, Any], Any],
        *,
        reverse: bool = False,
    ) -> Self:
        """Apply binary operation with another object."""
        # Case 1: EnsembleArray + EnsembleArray
        if isinstance(other, type(self)):
            self._validate_alignment(other)  # Validate meta alignment
            self._ensure_ensemble_compat(other)  # Validate shape compatibility
            lhs_members = other._data if reverse else self._data
            rhs_members = self._data if reverse else other._data
            lhs_template_vals = other._template.values if reverse else self._template.values
            rhs_template_vals = self._template.values if reverse else other._template.values
        
        # Case 2: EnsembleArray + Member (Matrix/Vector)
        elif self._is_compatible_member(other):
            self._ensure_member_compat(other)
            lhs_members = other.values if reverse else self._data
            rhs_members = self._data if reverse else other.values
            lhs_template_vals = other.values if reverse else self._template.values
            rhs_template_vals = self._template.values if reverse else other.values
        
        # Case 3: EnsembleArray + scalar or array
        else:
            if np.isscalar(other):
                operand = other
            else:
                operand = jnp.asarray(other)
                self._validate_operand_shape(operand)
            lhs_members = operand if reverse else self._data
            rhs_members = self._data if reverse else operand
            lhs_template_vals = operand if reverse else self._template.values
            rhs_template_vals = self._template.values if reverse else operand
        
        # Apply operation
        result_members = op(lhs_members, rhs_members)
        result_template_values = op(lhs_template_vals, rhs_template_vals)
        template = self._template.clone(
            values=jnp.asarray(result_template_values),
            copy=False,
        )
        return type(self)(
            members=jnp.asarray(result_members),
            template=template,
            _meta=self._meta,  # Preserve lineage token
            copy=False,
        )
    
    # Arithmetic operators
    def __add__(self, other: MemberType | Self | Array | float) -> Self:
        return self._binary_operation(other, operator.add)
    
    def __radd__(self, other: MemberType | Self | Array | float) -> Self:
        return self._binary_operation(other, operator.add, reverse=True)
    
    def __sub__(self, other: MemberType | Self | Array | float) -> Self:
        return self._binary_operation(other, operator.sub)
    
    def __rsub__(self, other: MemberType | Self | Array | float) -> Self:
        return self._binary_operation(other, operator.sub, reverse=True)
    
    def __mul__(self, other: MemberType | Self | Array | float) -> Self:
        return self._binary_operation(other, operator.mul)
    
    def __rmul__(self, other: MemberType | Self | Array | float) -> Self:
        return self._binary_operation(other, operator.mul, reverse=True)
    
    def __truediv__(self, other: MemberType | Self | Array | float) -> Self:
        return self._binary_operation(other, operator.truediv)
    
    def __rtruediv__(self, other: MemberType | Self | Array | float) -> Self:
        return self._binary_operation(other, operator.truediv, reverse=True)
    
    def __pow__(self, power: float) -> Self:
        return self._unary_operation(lambda data: data**power)
    
    def __neg__(self) -> Self:
        return self._unary_operation(operator.neg)
    
    def __pos__(self) -> Self:
        return self._unary_operation(operator.pos)
    
    # ========== Persistence (common wrapper) ==========
    
    def save(
        self,
        path: Pathlike,
        *,
        exist_ok: bool = False,
        compression: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Save ensemble to file (alias for to_hdf5)."""
        self.to_hdf5(path, exist_ok=exist_ok, compression=compression, **kwargs)
    
    # ========== Abstract methods for subclasses ==========
    
    @abstractmethod
    def _is_compatible_member(self, other: Any) -> bool:
        """Check if other is a compatible member type (Matrix/Vector)."""
        ...
    
    @abstractmethod
    def _ensure_member_compat(self, member: MemberType) -> None:
        """Validate member is compatible with ensemble."""
        ...
    
    @abstractmethod
    def _ensure_ensemble_compat(self, other: Self) -> None:
        """Validate other ensemble is compatible."""
        ...
    
    @abstractmethod
    def _validate_operand_shape(self, operand: Array) -> None:
        """Validate array operand has compatible shape."""
        ...
    
    @abstractmethod
    def to_hdf5(
        self,
        path: Pathlike,
        *,
        exist_ok: bool = False,
        compression: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Save ensemble to HDF5 file."""
        ...
    
    @classmethod
    @abstractmethod
    def from_hdf5(cls, path: Pathlike) -> Self:
        """Load ensemble from HDF5 file."""
        ...
    
    @classmethod
    @abstractmethod
    def from_path(cls, path: Pathlike) -> Self:
        """Load ensemble from file with automatic format detection."""
        ...
    
    @abstractmethod
    def __repr__(self) -> str:
        """String representation."""
        ...
    
    @abstractmethod
    def _repr_html_(self) -> str:
        """HTML representation for Jupyter."""
        ...