from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Literal, Self

__all__ = ['PlotSettings', 'PlotKind', 'PlotScale']

PlotKind = Literal["step", "plot", "line", "bar", "dot", "scatter", "poisson"]
PlotScale = Literal["linear", "log", "symlog"]


@dataclass(frozen=True, slots=True)
class PlotSettings:
    """Visual presentation settings for plotting.
    
    Contains ONLY visual/styling information, not semantic info like labels.
    
    Attributes:
        yscale: Y-axis scale ('linear', 'log', 'symlog')
        xscale: X-axis scale ('linear', 'log', 'symlog')
        kind: Plot kind ('step', 'line', 'bar', 'dot', 'scatter', 'poisson')
        plot_kwargs: Additional matplotlib kwargs (colors, linewidth, etc.)
    
    Examples:
        >>> # Create settings
        >>> settings = PlotSettings(yscale='log', kind='line')
        
        >>> # Use presets
        >>> settings = PlotSettings.log_line()
        
        >>> # Chain modifications
        >>> settings = PlotSettings.log_line().with_kwargs(color='red', linewidth=2)
    """
    yscale: PlotScale = 'linear'
    xscale: PlotScale = 'linear'
    kind: PlotKind = 'step'
    plot_kwargs: dict[str, Any] = field(default_factory=dict)
    
    def clone(self, **kwargs: Any) -> Self:
        """Create a modified copy.
        
        Args:
            **kwargs: PlotSettings attributes to override
        
        Returns:
            New PlotSettings with specified changes
        """
        updates = {}
        for key in self.__dataclass_fields__:
            if key == 'plot_kwargs':
                # Special handling: merge plot_kwargs
                if 'plot_kwargs' in kwargs:
                    updates[key] = self.plot_kwargs | kwargs.pop('plot_kwargs')
                else:
                    updates[key] = self.plot_kwargs
            else:
                updates[key] = kwargs.pop(key, getattr(self, key))
        
        # Any remaining kwargs go into plot_kwargs
        if kwargs:
            updates['plot_kwargs'] = updates['plot_kwargs'] | kwargs
        
        return self.__class__(**updates)
    
    def with_yscale(self, scale: PlotScale) -> Self:
        """Set y-axis scale."""
        return self.clone(yscale=scale)
    
    def with_xscale(self, scale: PlotScale) -> Self:
        """Set x-axis scale."""
        return self.clone(xscale=scale)
    
    def with_log_scale(self) -> Self:
        """Set y-axis to log scale."""
        return self.clone(yscale='log')
    
    def with_linear_scale(self) -> Self:
        """Set y-axis to linear scale."""
        return self.clone(yscale='linear')
    
    def with_kind(self, kind: PlotKind) -> Self:
        """Set plot kind."""
        return self.clone(kind=kind)
    
    def with_kwargs(self, **kwargs: Any) -> Self:
        """Add or update matplotlib kwargs.
        
        Args:
            **kwargs: Matplotlib keyword arguments (color, linewidth, etc.)
        
        Examples:
            >>> settings = PlotSettings.log_line().with_kwargs(
            ...     color='red', linewidth=2, alpha=0.8
            ... )
        """
        return self.clone(plot_kwargs=self.plot_kwargs | kwargs)
    
    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Self:
        """Create PlotSettings from a dictionary.
        
        This is a convenience method to allow passing dicts directly.
        
        Args:
            d: Dictionary with keys: yscale, xscale, kind, plot_kwargs
        
        Returns:
            New PlotSettings instance
        
        Examples:
            >>> settings = PlotSettings.from_dict({
            ...     'yscale': 'log',
            ...     'kind': 'line',
            ...     'plot_kwargs': {'color': 'red'}
            ... })
            
            >>> # Minimal
            >>> settings = PlotSettings.from_dict({'yscale': 'log'})
        """
        return cls(
            yscale=d.get('yscale', 'linear'),
            xscale=d.get('xscale', 'linear'),
            kind=d.get('kind', 'step'),
            plot_kwargs=d.get('plot_kwargs', {})
        )
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation.
        
        Returns:
            Dictionary with yscale, xscale, kind, plot_kwargs
        
        Examples:
            >>> settings = PlotSettings.log_line()
            >>> d = settings.to_dict()
            >>> # d == {'yscale': 'log', 'xscale': None, 'kind': 'line', 'plot_kwargs': {}}
        """
        return {
            'yscale': self.yscale,
            'xscale': self.xscale,
            'kind': self.kind,
            'plot_kwargs': self.plot_kwargs,
        }
    
    # ============ PRESET FACTORY METHODS (Visual Only) ============
    
    @classmethod
    def log_step(cls, **kwargs: Any) -> Self:
        """Preset: Log scale with step plot.
        
        Suitable for histograms, spectra, count data.
        """
        return cls(yscale='log', kind='step', plot_kwargs=kwargs)
    
    @classmethod
    def log_line(cls, **kwargs: Any) -> Self:
        """Preset: Log scale with line plot.
        
        Suitable for training curves, convergence plots.
        """
        return cls(yscale='log', kind='line', plot_kwargs=kwargs)
    
    @classmethod
    def linear_line(cls, **kwargs: Any) -> Self:
        """Preset: Linear scale with line plot.
        
        Suitable for residuals, smooth functions.
        """
        return cls(yscale='linear', kind='line', plot_kwargs=kwargs)
    
    @classmethod
    def linear_step(cls, **kwargs: Any) -> Self:
        """Preset: Linear scale with step plot."""
        return cls(yscale='linear', kind='step', plot_kwargs=kwargs)
    
    @classmethod
    def poisson_log(cls, **kwargs: Any) -> Self:
        """Preset: Poisson error bars on log scale.
        
        Suitable for counting data with statistical errors.
        """
        return cls(yscale='log', kind='poisson', plot_kwargs=kwargs)
    
    @classmethod
    def poisson_linear(cls, **kwargs: Any) -> Self:
        """Preset: Poisson error bars on linear scale."""
        return cls(yscale='linear', kind='poisson', plot_kwargs=kwargs)
    
    @classmethod
    def scatter_log(cls, **kwargs: Any) -> Self:
        """Preset: Scatter plot on log scale."""
        return cls(yscale='log', kind='scatter', plot_kwargs=kwargs)
    
    @classmethod
    def scatter_linear(cls, **kwargs: Any) -> Self:
        """Preset: Scatter plot on linear scale."""
        return cls(yscale='linear', kind='scatter', plot_kwargs=kwargs)
    
    @classmethod
    def bar(cls, yscale: PlotScale = 'linear', **kwargs: Any) -> Self:
        """Preset: Bar plot."""
        return cls(yscale=yscale, kind='bar', plot_kwargs=kwargs)
    
    @classmethod
    def styled(
        cls,
        color: str | None = None,
        linewidth: float | None = None,
        linestyle: str | None = None,
        marker: str | None = None,
        alpha: float | None = None,
        **kwargs: Any
    ) -> Self:
        """Create settings with common style parameters.
        
        Args:
            color: Line/marker color
            linewidth: Line width
            linestyle: Line style ('-', '--', '-.', ':')
            marker: Marker style
            alpha: Transparency (0-1)
            **kwargs: Additional matplotlib kwargs
        
        Example:
            >>> settings = PlotSettings.styled(
            ...     color='red', linewidth=2, linestyle='--', alpha=0.7
            ... )
        """
        plot_kwargs = {}
        if color is not None:
            plot_kwargs['color'] = color
        if linewidth is not None:
            plot_kwargs['linewidth'] = linewidth
        if linestyle is not None:
            plot_kwargs['linestyle'] = linestyle
        if marker is not None:
            plot_kwargs['marker'] = marker
        if alpha is not None:
            plot_kwargs['alpha'] = alpha
        plot_kwargs.update(kwargs)
        return cls(plot_kwargs=plot_kwargs)

