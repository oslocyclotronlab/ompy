"""
Lifting decorator for analysis pipeline functions.

This module provides a @lift decorator that automatically handles dispatch
for functions that work on single Matrix/Vector objects, making them work
seamlessly with:
- Iterable[T] (lists, tuples, generators)
- Result objects (passed through as-is)
- EnsembleMatrix/EnsembleVector (automatically unwraps members if they're Results)
- Resampling2D

Key Semantic:
    Ensemble types have special meaning - they represent "production pipeline"
    data where you only care about clean results. Therefore, when processing
    an Ensemble, members are automatically unwrapped if they're Result objects,
    processed, and the results are wrapped back into the Ensemble.

The decorator preserves type hints and provides clean error messages.
"""
from __future__ import annotations

from functools import wraps
from typing import (
    Any, Callable, Generic, TypeVar,
    Protocol, ParamSpec
)
from collections.abc import Sequence
from dataclasses import dataclass, field
import inspect
import types

from .stage import Stage

# Type variables
T = TypeVar('T')  # Generic type
R = TypeVar('R')  # Result type
P = ParamSpec('P')  # Parameters


class Unwrappable(Protocol):
    """Protocol for Result objects that support unwrapping via __unwrap__().
    
    Any object that implements this protocol can be automatically unwrapped
    when it appears as a member of an Ensemble.
    """
    def __unwrap__(self) -> Any: ...


class HasMembers(Protocol):
    """Protocol for Ensemble types that have a .members attribute."""
    @property
    def members(self) -> list[Any]: ...


class Resamplable(Protocol):
    """Protocol for Resampling2D that has get_eta method."""
    def get_eta(self, i: int) -> Any: ...
    def __len__(self) -> int: ...


@dataclass
class LiftConfig:
    """Configuration for the lift decorator.
    
    Attributes:
        unwrap_param: Name of runtime parameter for unwrap control (default: '_lift_unwrap')
        elide_param: Name of runtime parameter for validation control (default: '_lift_elide')
        unwrap_method: Method name to call for unwrapping (default: '__unwrap__')
        return_tuple_for_iterables: If True, return tuples instead of lists for iterables
        propagate_kwargs: If True, pass through all kwargs to the lifted function
        parallel: If True (future), use parallel processing for iterables
        preserve_container: If True, return same container type (list->list, tuple->tuple)
        expects: Expected Stage for validation (None = no validation)
        progress: If True, show progress bar when processing sequences (opt-out via _lift_disable_tqdm)
        progress_desc: Description for progress bar (defaults to function name)
        progress_disable_param: Runtime parameter to disable progress (default: '_lift_disable_tqdm')
        progress_leave_param: Runtime parameter to control bar persistence (default: '_lift_leave_tqdm')
        override_kwargs: Dict of kwargs to inject when processing containers (e.g., {'disable_tqdm': True})
        ensemble_strategy: Optional callable that controls ensemble processing flow.
            Signature: (ensemble, func, args, kwargs) -> Iterator[Result]
            If None, uses default strategy (process each member independently).
    """
    unwrap_param: str = '_lift_unwrap'
    elide_param: str = '_lift_elide'
    unwrap_method: str = '__unwrap__'
    return_tuple_for_iterables: bool = False
    propagate_kwargs: bool = True
    parallel: bool = False
    preserve_container: bool = True
    expects: Stage | None = None
    produces: Stage | None = None
    
    # Progress bar configuration
    progress: bool = True  # Default: show progress for sequences (opt-out)
    progress_desc: str | None = None
    progress_disable_param: str = '_lift_disable_tqdm'
    progress_leave_param: str = '_lift_leave_tqdm'
    override_kwargs: dict[str, Any] = field(default_factory=dict)
    
    # Ensemble processing strategy
    ensemble_strategy: Callable[[Any, Callable, tuple, dict], Any] | None = None
        
    def __post_init__(self):
        self.expects = Stage.from_any(self.expects)
        self.produces = Stage.from_any(self.produces)
        if self.override_kwargs is None or len(self.override_kwargs) == 0:
            self.override_kwargs = {'leave_tqdm': False}


# ============================================================================
# Ensemble Processing Strategies
# ============================================================================

def _default_ensemble_strategy(ensemble: Any, func: Callable, args: tuple, kwargs: dict) -> Any:
    """Default strategy: process each member independently with same kwargs.
    
    This is the identity strategy - each member is processed with identical
    kwargs, yielding results one by one.
    
    Args:
        ensemble: The ensemble to process
        func: The lifted function to call on each member
        args: Positional arguments for func
        kwargs: Keyword arguments for func
        
    Yields:
        Results from processing each member
    """
    for member in ensemble:
        yield func(member, *args, **kwargs)


def template_strategy(ensemble: Any, func: Callable, args: tuple, kwargs: dict) -> Any:
    """Strategy that extracts template from first result for consistent grids.
    
    Processes the first ensemble member normally, then extracts template
    parameters from the result. Remaining members are processed with these
    template parameters merged into kwargs, ensuring consistent output grids.
    
    This is useful for functions like decomposition where the output grid
    depends on data (via heuristic cutting), but you want all ensemble
    members to produce compatible grids for wrapping into EnsembleVector.
    
    Args:
        ensemble: The ensemble to process
        func: The lifted function to call on each member
        args: Positional arguments for func
        kwargs: Keyword arguments for func
        
    Yields:
        Results from processing each member (first without template, rest with)
        
    Example:
        >>> @lift(ensemble_strategy=template_strategy)
        ... def decompose(FG: Matrix, **kwargs) -> DecompositionResult:
        ...     return optimize(FG, **kwargs)
        >>> 
        >>> results = decompose(ensemble)  # Automatic template extraction!
    """
    # Process first member normally
    first_result = func(ensemble[0], *args, **kwargs)
    yield first_result
    
    # Try to extract template kwargs from first result
    template_kwargs = {}
    
    # Method 1: Use protocol method if available
    if hasattr(first_result, 'to_template_kwargs') and callable(first_result.to_template_kwargs):
        template_kwargs = first_result.to_template_kwargs()
        kwargs = {**kwargs, **template_kwargs}
    
    # Process remaining members with template
    for member in ensemble[1:]:
        yield func(member, *args, **kwargs)


# ============================================================================
# Public Helper Functions
# ============================================================================

def unwrap(wrapper: Any) -> Any:
    """Public helper to unwrap an object via __unwrap__() protocol.
    
    This avoids the visual noise of .__unwrap__() everywhere.
    
    Args:
        wrapper: Object implementing __unwrap__() protocol
    
    Returns:
        The unwrapped value
    
    Example:
        >>> result = UnfoldedResult(...)
        >>> matrix = unwrap(result)  # Cleaner than result.__unwrap__()
    """
    if hasattr(wrapper, '__unwrap__') and callable(wrapper.__unwrap__):
        return wrapper.__unwrap__()
    return wrapper


def get_stage(obj: Any) -> Stage | None:
    """Get the pipeline stage of an object.
    
    Works with Result objects (via .stage or .__stage__()) and
    Ensemble objects (via .stage attribute).
    
    Args:
        obj: Object to get stage from
    
    Returns:
        Stage enum value, or None if object has no stage
    
    Example:
        >>> result = UnfoldedResult(...)
        >>> get_stage(result)  # Stage.UNFOLDED
        >>> 
        >>> ensemble = EnsembleVector(...).with_stage(Stage.RAW)
        >>> get_stage(ensemble)  # Stage.RAW
    """
    # Try __stage__() method first
    if hasattr(obj, '__stage__') and callable(obj.__stage__):
        return obj.__stage__()
    
    # Try .stage property
    if hasattr(obj, 'stage'):
        stage = obj.stage
        if isinstance(stage, Stage):
            return stage
    
    return None

    
def set_stage(obj: Any, stage: Stage) -> None:
    """Set the stage of an object."""
    if hasattr(obj, 'stage'):
        obj.stage = stage
    elif hasattr(obj, '__stage__'):
        obj.__stage__ = stage


def with_stage(obj: Any, stage: Stage) -> Any:
    """Set the stage of an object and return it."""
    set_stage(obj, stage)
    return obj


# ============================================================================
# Internal Helper Functions
# ============================================================================

def _is_result_type(obj: Any) -> bool:
    """Check if object is a Result type (implements __unwrap__)."""
    return hasattr(obj, '__unwrap__') and callable(obj.__unwrap__)


def _is_ensemble_type(obj: Any) -> bool:
    """Check if object is an Ensemble type (EnsembleMatrix/EnsembleVector)."""
    # Use duck typing to avoid import issues
    return (hasattr(obj, 'members') and 
            hasattr(obj, '__len__') and 
            hasattr(type(obj).__name__, '__class__') and
            'Ensemble' in type(obj).__name__)


def _is_resampling2d(obj: Any) -> bool:
    """Check if object is a Resampling2D type."""
    # Use duck typing to avoid import issues
    return hasattr(obj, 'get_eta') and hasattr(obj, '__len__') and callable(obj.get_eta)


def _unwrap_if_result(obj: Any, method: str = '__unwrap__') -> Any:
    """Unwrap a Result object using the specified method.
    
    Args:
        obj: Object to potentially unwrap
        method: Method name to call for unwrapping
    
    Returns:
        Unwrapped value if obj is a Result, otherwise obj unchanged
    """
    if not _is_result_type(obj):
        return obj
    
    # Try configured unwrap method
    if hasattr(obj, method) and callable(getattr(obj, method)):
        return getattr(obj, method)()
    
    # Fallback to common names for backwards compatibility
    for fallback in ['__unwrap__', 'best', 'unwrap', 'get', 'value']:
        if hasattr(obj, fallback) and callable(getattr(obj, fallback)):
            return getattr(obj, fallback)()
    
    # No unwrap method found - return as-is
    return obj


def _is_generator(obj: Any) -> bool:
    """Check if object is a generator or iterator."""
    return isinstance(obj, types.GeneratorType) or hasattr(obj, '__next__')


def _is_iterable_but_not_string(obj: Any) -> bool:
    """Check if object is iterable but not a string, bytes, or generator."""
    # Exclude generators - they get special handling
    if _is_generator(obj):
        return False
    
    return (
        isinstance(obj, (list, tuple, Sequence)) 
        and not isinstance(obj, (str, bytes))
    )



class Lifter(Generic[P, R]):
    """Core lifting implementation.
    
    This class handles the actual dispatch logic for lifted functions.
    
    Key Semantics:
        - Result objects are transparent wrappers: unwrap input, return new Result
        - Ensemble objects are preserving wrappers: unwrap members, process, 
          unwrap outputs, re-wrap into Ensemble
        - Lists/tuples/generators are generic containers: process each element
    """
    
    def __init__(
        self, 
        func: Callable[P, R],
        config: LiftConfig,
    ):
        """Initialize the lifter.
        
        Args:
            func: The base function that operates on single objects
            config: Configuration for lifting behavior
        """
        self.func = func
        self.config = config
        self.signature = inspect.signature(func)
        
        # Check for parameter collision
        reserved_params = [
            (config.unwrap_param, 'unwrap_param'),
            (config.elide_param, 'elide_param'),
            (config.progress_disable_param, 'progress_disable_param'),
            (config.progress_leave_param, 'progress_leave_param'),
        ]
        
        for param_name, config_name in reserved_params:
            if param_name in self.signature.parameters:
                raise ValueError(
                    f"Function '{func.__name__}' already has parameter '{param_name}'. "
                    f"Use a different name: @lift({config_name}='_other_name')"
                )
        
        # Store function metadata
        wraps(func)(self)
        
    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> Any:
        """Call the lifted function with automatic dispatch."""
        # Extract control parameters from kwargs
        elide = kwargs.pop(self.config.elide_param, False)
        
        # Extract progress control parameters
        self._progress_disabled = kwargs.pop(self.config.progress_disable_param, False)
        self._progress_leave = kwargs.pop(self.config.progress_leave_param, True)
        
        # Get the first positional argument (the data to process)
        if not args:
            raise TypeError(f"{self.func.__name__}() missing required positional argument")
        
        data = args[0]
        rest_args = args[1:]
        
        # Validate input if validation is configured and not elided
        if not elide and self.config.expects is not None:
            self._validate_input(data)
        
        # Dispatch based on input type
        result = self._dispatch(data, rest_args, kwargs)
        
        return result
    
    def _validate_input(self, data: Any) -> None:
        """Validate that input has expected stage.
        
        The expects parameter must be a Stage enum value.
        Validates the stage of:
        - Result objects (via get_stage)
        - Ensemble objects (via .stage attribute)
        
        Args:
            data: Input data to validate
        
        Raises:
            ValueError: If stage doesn't match expected
        """
        expects = self.config.expects
        
        if expects is None:
            return
        
        # Get the stage of the input data
        data_stage = get_stage(data)
        
        # If data has no stage, we can't validate (pass through)
        if data_stage is None:
            return
        
        # Validate stage matches
        if data_stage != expects:
            raise ValueError(
                f"{self.func.__name__} expects data with stage {expects}, "
                f"got {data_stage}. "
                f"To bypass this check, use {self.config.elide_param}=True"
            )
    
    def _dispatch(
        self, 
        data: Any, 
        rest_args: tuple,
        kwargs: dict[str, Any],
    ) -> Any:
        """Dispatch to appropriate handler based on data type.
        
        Dispatch order:
        1. Result - unwrap, process, return new Result
        2. Ensemble - unwrap members, process, unwrap outputs, re-wrap
        3. Resampling2D - extract, process, return list
        4. Generator - process lazily, return generator
        5. Iterable (list/tuple) - process each, preserve container
        6. Single object - process directly
        """
        
        # Case 1: Result object - unwrap before passing to function
        if _is_result_type(data):
            data = _unwrap_if_result(data, self.config.unwrap_method)
            return with_stage(self.func(data, *rest_args, **kwargs), self.config.produces)
        
        # Case 2: EnsembleMatrix/EnsembleVector - special handling
        if _is_ensemble_type(data):
            return self._handle_ensemble(data, rest_args, kwargs)
        
        # Case 3: Resampling2D - convert to members and process
        if _is_resampling2d(data):
            return self._handle_resampling(data, rest_args, kwargs)
        
        # Case 4: Generator - preserve lazy evaluation
        if _is_generator(data):
            return self._handle_generator(data, rest_args, kwargs)
        
        # Case 5: Iterable (list/tuple) - process each element
        if _is_iterable_but_not_string(data):
            return self._handle_iterable(data, rest_args, kwargs)
        
        # Case 6: Single object - just call the function
        return with_stage(self.func(data, *rest_args, **kwargs), self.config.produces)
    
    def _handle_ensemble(
        self,
        ensemble: Any,
        rest_args: tuple,
        kwargs: dict[str, Any],
    ) -> Any:
        """Handle Ensemble using strategy pattern for flexible processing.
        
        Uses ensemble_strategy (if provided) or default strategy to control
        processing flow. Results are unwrapped immediately to minimize memory.
        
        Key semantic: Ensemble can only hold Matrix/Vector, not Result objects.
        Therefore, after processing (which returns Results), we must unwrap
        the Results back to Matrix/Vector before reconstructing the Ensemble.
        
        Progress bar shown by default for Ensembles (opt-out with _lift_disable_tqdm).
        Inner function progress bars are disabled via override_kwargs.
        
        Flow:
            Ensemble[Matrix] 
            → strategy yields Results
            → unwrap each Result immediately (memory efficient!)
            → Ensemble[Matrix] (reconstructed)
        """
        # Get strategy (use default if not provided)
        strategy = self.config.ensemble_strategy or _default_ensemble_strategy
        
        # Apply kwargs overrides for container processing
        call_kwargs = self.config.override_kwargs | kwargs
        
        # Get iterator from strategy
        result_iterator = strategy(ensemble, self.func, rest_args, call_kwargs)
        
        # Setup progress bar (if enabled)
        show_progress = self.config.progress and not self._progress_disabled
        
        if show_progress:
            try:
                from tqdm.autonotebook import tqdm
                desc = self.config.progress_desc or f"{self.func.__name__} (ensemble)"
                result_iterator = tqdm(
                    result_iterator,
                    desc=desc,
                    leave=self._progress_leave,
                    total=len(ensemble)
                )
            except ImportError:
                # tqdm not available, proceed without progress
                pass
        
        # Process results and unwrap immediately (memory efficient!)
        processed = []
        for result in result_iterator:
            # Unwrap immediately to free memory
            unwrapped_result = _unwrap_if_result(result, self.config.unwrap_method)
            
            # Validate before continuing
            if not ensemble.can_wrap(unwrapped_result):
                raise TypeError(
                    f"Cannot wrap result of type {type(unwrapped_result).__name__} into an Ensemble"
                )
            
            processed.append(unwrapped_result)
        
        # Reconstruct ensemble with processed members
        ensemble_out = ensemble.wrap(processed)
        
        return with_stage(ensemble_out, self.config.produces)
    
    def _handle_resampling(
        self,
        resampling: Any,
        rest_args: tuple,
        kwargs: dict[str, Any],
    ) -> Any:
        """Handle Resampling2D by extracting members via get_eta.
        
        Progress bar shown by default (opt-out with _lift_disable_tqdm).
        """
        members = [resampling.get_eta(i) for i in range(len(resampling))]
        
        # Apply kwargs overrides for container processing
        if self.config.override_kwargs:
            call_kwargs = {**kwargs, **self.config.override_kwargs}
        else:
            call_kwargs = kwargs
        
        # Setup progress bar for Resampling (if enabled)
        iterator = range(len(members))
        show_progress = self.config.progress and not self._progress_disabled
        
        if show_progress:
            try:
                from tqdm.autonotebook import tqdm
                desc = self.config.progress_desc or f"{self.func.__name__} (resampling)"
                iterator = tqdm(
                    iterator,
                    desc=desc,
                    leave=self._progress_leave
                )
            except ImportError:
                pass
        
        # Process each member (recursively dispatches)
        results = []
        for i in iterator:
            result = self._dispatch(members[i], rest_args, call_kwargs)
            results.append(result)
        
        if self.config.return_tuple_for_iterables:
            return tuple(results)
        return results
    
    def _handle_generator(
        self,
        generator: Any,
        rest_args: tuple,
        kwargs: dict[str, Any],
    ) -> Any:
        """Handle generator without consuming it - preserve lazy evaluation.
        
        Returns a new generator that processes items one at a time.
        No progress bar by default for generators (lazy evaluation).
        """
        # Apply kwargs overrides for generator items
        if self.config.override_kwargs:
            call_kwargs = {**kwargs, **self.config.override_kwargs}
        else:
            call_kwargs = kwargs
        
        def process_lazy():
            for item in generator:
                # Recursively dispatch with overridden kwargs
                result = self._dispatch(item, rest_args, call_kwargs)
                yield result
        
        return process_lazy()
    
    def _handle_iterable(
        self,
        iterable: Sequence,
        rest_args: tuple,
        kwargs: dict[str, Any],
    ) -> Any:
        """Handle list/tuple by processing each element.
        
        Recursively dispatches on each element, which handles:
        - Unwrapping Results in the list
        - Nested structures like [[Matrix, Matrix]]
        - Mixed types like [Matrix, Result, Matrix]
        
        Progress bar shown by default for sequences (opt-out with _lift_disable_tqdm).
        Inner function progress bars are disabled via override_kwargs.
        """
        # Determine if we should preserve the container type
        is_tuple = isinstance(iterable, tuple)
        
        # Apply kwargs overrides for container processing
        if self.config.override_kwargs:
            call_kwargs = {**kwargs, **self.config.override_kwargs}
        else:
            call_kwargs = kwargs
        
        # Setup progress bar for sequences (if enabled)
        iterator = iterable
        show_progress = self.config.progress and not self._progress_disabled
        
        if show_progress:
            try:
                from tqdm.autonotebook import tqdm
                desc = self.config.progress_desc or self.func.__name__
                iterator = tqdm(
                    iterable, 
                    desc=desc, 
                    leave=self._progress_leave
                )
            except ImportError:
                # tqdm not available, proceed without progress
                pass
        
        results = []
        for item in iterator:
            # Recursively dispatch each item with overridden kwargs
            result = self._dispatch(item, rest_args, call_kwargs)
            results.append(result)
        
        # Preserve container type if configured
        if self.config.preserve_container and is_tuple:
            return tuple(results)
        elif self.config.return_tuple_for_iterables:
            return tuple(results)
        return results


def lift(
    *,
    expects: Stage | None = None,
    produces: Stage | None = None,
    return_tuple: bool = False,
    preserve_container: bool = True,
    parallel: bool = False,
    unwrap_param: str = '_lift_unwrap',
    elide_param: str = '_lift_elide',
    unwrap_method: str = '__unwrap__',
    progress: bool = True,
    progress_desc: str | None = None,
    progress_disable_param: str = '_lift_disable_tqdm',
    progress_leave_param: str = '_lift_leave_tqdm',
    override_kwargs: dict[str, Any] | None = None,
    ensemble_strategy: Callable[[Any, Callable, tuple, dict], Any] | None = None,
) -> Callable[[Callable[P, R]], Lifter[P, R]]:
    """Decorator to lift a function to work with collections and ensemble types.
    
    The @lift decorator makes a function that works on single objects (Matrix, Vector)
    automatically work with:
    - Result objects: Automatically unwrapped via __unwrap__()
    - list/tuple: Process each element, preserve container type
    - Generator: Preserve lazy evaluation
    - EnsembleMatrix/EnsembleVector: Unwrap members, process, unwrap outputs, re-wrap
    - Resampling2D: Extract members and process
    
    Key Semantics:
        - Result is transparent: unwrap input → process → return new Result
        - Ensemble is preserving: unwrap members → process → unwrap outputs → re-wrap
        - List/tuple/generator: process each element
    
    Args:
        expects: Expected Stage for validation. Must be a Stage enum value.
            Validates both Result.stage and Ensemble.stage attributes.
            None: no validation (default)
        return_tuple: If True, return tuples instead of lists for iterables
        preserve_container: If True, preserve input container type (list→list, tuple→tuple)
        parallel: If True, use parallel processing (not yet implemented)
        unwrap_param: Name of runtime parameter (default: '_lift_unwrap')
        elide_param: Name of runtime validation bypass parameter (default: '_lift_elide')
        unwrap_method: Method name for unwrapping (default: '__unwrap__')
        progress: If True, show progress bar for sequences/ensembles (default: True, opt-out)
        progress_desc: Description for progress bar (default: function name)
        progress_disable_param: Runtime param to disable progress (default: '_lift_disable_tqdm')
        progress_leave_param: Runtime param to keep bar after completion (default: '_lift_leave_tqdm')
        override_kwargs: Dict of kwargs to inject when processing containers.
            Useful for disabling inner progress bars: {'disable_tqdm': True}
        ensemble_strategy: Optional callable to control ensemble processing flow.
            Signature: (ensemble, func, args, kwargs) -> Iterator[Result]
            If None, uses default strategy (process each member independently).
            Use template_strategy for automatic template extraction from first member.
    
    Examples:
        Basic usage:
        >>> @lift()
        ... def unfold(matrix: Matrix) -> UnfoldedResult:
        ...     return compute_unfold(matrix)
        
        >>> unfold(matrix)                  # Matrix → UnfoldedResult
        >>> unfold([m1, m2, m3])            # [Matrix, ...] → [UnfoldedResult, ...]
        >>> unfold(result)                  # Result → unwraps → UnfoldedResult
        >>> unfold(ensemble)                # Ensemble → processes → Ensemble
        
        With validation:
        >>> from ompy.stage import Stage
        >>> @lift(expects=Stage.UNFOLDED)
        ... def firstgen(matrix: Matrix) -> FirstGenerationResult:
        ...     return compute_fg(matrix)
        
        >>> ensemble = EnsembleVector(...).with_stage(Stage.UNFOLDED)
        >>> firstgen(ensemble)              # Validates stage, then processes
        
        Bypass validation:
        >>> firstgen(wrong_ensemble, _lift_elide=True)  # Skips validation
        
        Generator support (lazy evaluation):
        >>> def load_many():
        ...     for i in range(1000):
        ...         yield load_matrix(i)
        >>> results = unfold(load_many())   # Returns generator, not list!
        >>> for r in results:               # Processes one at a time
        ...     save(r)
        
        Progress bars and keyword override:
        >>> @lift(
        ...     progress=True,  # Default: enabled
        ...     progress_desc="Processing matrices",
        ...     override_kwargs={'disable_tqdm': True}  # Disable inner bars
        ... )
        ... def expensive(matrix, disable_tqdm=False):
        ...     # Has internal tqdm
        ...     for step in tqdm(..., disable=disable_tqdm):
        ...         compute(step)
        ...     return result
        
        >>> expensive(matrix)              # Single: shows internal tqdm
        >>> expensive([m1, ..., m100])     # List: outer bar only, inner disabled
        >>> expensive(ensemble)             # Ensemble: outer bar only
        >>> expensive(matrices, _lift_disable_tqdm=True)  # No outer bar
        >>> expensive(matrices, _lift_leave_tqdm=True)     # Keep bar after done
        
        Ensemble strategy for template extraction:
        >>> from ompy.pipeline.lifting import lift, template_strategy
        >>> 
        >>> @lift(
        ...     expects='first generation',
        ...     produces='decomposed',
        ...     ensemble_strategy=template_strategy  # Automatic template!
        ... )
        ... def decompose(FG: Matrix, **kwargs) -> DecompositionResult:
        ...     return optimize(FG, **kwargs)
        >>> 
        >>> # Process ensemble - template extracted automatically from first member
        >>> results = decompose(ensemble)  # All members get consistent grids!
    
    Returns:
        A decorator that lifts the function
    """
    config = LiftConfig(
        expects=expects,
        produces=produces,
        unwrap_param=unwrap_param,
        elide_param=elide_param,
        unwrap_method=unwrap_method,
        return_tuple_for_iterables=return_tuple,
        preserve_container=preserve_container,
        parallel=parallel,
        progress=progress,
        progress_desc=progress_desc,
        progress_disable_param=progress_disable_param,
        progress_leave_param=progress_leave_param,
        override_kwargs=override_kwargs,
        ensemble_strategy=ensemble_strategy,
    )
    
    def decorator(func: Callable[P, R]) -> Lifter[P, R]:
        return Lifter(func, config)
    
    return decorator


# Type-aware lifting helpers for better type hints
def make_lifted_signature(
    base_func: Callable,
    config: LiftConfig,
) -> str:
    """Generate type signature for lifted function.
    
    This is mainly for documentation purposes to show what types
    the lifted function accepts and returns.
    """
    sig = inspect.signature(base_func)
    
    # Get return type
    return_type = sig.return_annotation
    
    # Build lifted signature
    lines = [
        f"Lifted signature for {base_func.__name__}:",
        f"  Single: {sig}",
        f"  Iterable: (Iterable[T], ...) -> Iterable[{return_type}]",
        f"  Ensemble: (Ensemble[T], ...) -> Ensemble[{return_type}]",
    ]
    
    if config.strip_results:
        lines.append(f"  Result: (Result[T], ...) -> {return_type} [auto-strips .best()]")
    
    return "\n".join(lines)

