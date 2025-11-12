from __future__ import annotations
from dataclasses import dataclass
from typing import Any

from .plotsettings import PlotSettings

__all__ = ['VectorPreset', 'register_preset', 'get_preset', 'list_presets']


@dataclass(frozen=True, slots=True)
class VectorPreset:
    """Complete preset combining semantic info and visual settings.
    
    A VectorPreset defines:
    - Semantic info: what the vector represents (xalias, xlabel, ylabel)
    - Visual info: how to plot it by default (PlotSettings)
    
    Attributes:
        xalias: Short alias for x-axis (e.g., 'iteration', 'E', 't')
        xlabel: Full x-axis label (e.g., 'Iterations', 'Energy', 'Time [s]')
        ylabel: Y-axis label (e.g., 'Loss', 'Counts', 'Amplitude')
        plot_settings: Default visual settings for plotting
    
    Examples:
        >>> # Create custom preset
        >>> preset = VectorPreset(
        ...     xalias='time',
        ...     xlabel='Time [ms]',
        ...     ylabel='Voltage [V]',
        ...     plot_settings=PlotSettings.linear_line()
        ... )
        >>> 
        >>> # Use with Vector
        >>> vec = Vector.from_preset(preset, times, voltages)
    """
    xalias: str
    xlabel: str
    ylabel: str
    plot_settings: PlotSettings
    
    def apply_to_vector_kwargs(self) -> dict[str, Any]:
        """Convert to Vector constructor kwargs.
        
        Returns:
            Dictionary with xalias, xlabel, vlabel, plot_settings
        """
        return {
            'xalias': self.xalias,
            'xlabel': self.xlabel,
            'vlabel': self.ylabel,
            'plot_settings': self.plot_settings
        }
    
    def with_ylabel(self, ylabel: str) -> VectorPreset:
        """Create new preset with different ylabel."""
        return VectorPreset(
            xalias=self.xalias,
            xlabel=self.xlabel,
            ylabel=ylabel,
            plot_settings=self.plot_settings
        )
    
    def with_plot_settings(self, plot_settings: PlotSettings) -> VectorPreset:
        """Create new preset with different plot settings."""
        return VectorPreset(
            xalias=self.xalias,
            xlabel=self.xlabel,
            ylabel=self.ylabel,
            plot_settings=plot_settings
        )
    
    def with_xlabel(self, xlabel: str) -> VectorPreset:
        """Create new preset with different xlabel."""
        return VectorPreset(
            xalias=self.xalias,
            xlabel=xlabel,
            ylabel=self.ylabel,
            plot_settings=self.plot_settings
        )


# ============ GLOBAL PRESET REGISTRY ============

_BUILTIN_PRESETS: dict[str, VectorPreset] = {
    'iterations': VectorPreset(
        xalias='iteration',
        xlabel='Iterations',
        ylabel='Loss',
        plot_settings=PlotSettings.log_line()
    ),
    'iterations_linear': VectorPreset(
        xalias='iteration',
        xlabel='Iterations',
        ylabel='Value',
        plot_settings=PlotSettings.linear_line()
    ),
    'chi_squared': VectorPreset(
        xalias='iteration',
        xlabel='Iterations',
        ylabel=r'$\chi^2$',
        plot_settings=PlotSettings.log_line()
    ),
    'likelihood': VectorPreset(
        xalias='iteration',
        xlabel='Iterations',
        ylabel='Log-likelihood',
        plot_settings=PlotSettings.linear_line()
    ),
    'spectrum': VectorPreset(
        xalias='E',
        xlabel='Energy',
        ylabel='Counts',
        plot_settings=PlotSettings.log_step()
    ),
    'spectrum_linear': VectorPreset(
        xalias='E',
        xlabel='Energy',
        ylabel='Counts',
        plot_settings=PlotSettings.linear_step()
    ),
    'residuals': VectorPreset(
        xalias='',
        xlabel='Energy',
        ylabel='Residual',
        plot_settings=PlotSettings.linear_line()
    ),
    'ratio': VectorPreset(
        xalias='',
        xlabel='Energy',
        ylabel='Ratio',
        plot_settings=PlotSettings.linear_line()
    ),
    'cross_section': VectorPreset(
        xalias='E',
        xlabel='Energy',
        ylabel=r'$\sigma$ [mb]',
        plot_settings=PlotSettings.log_line()
    ),
    'nld': VectorPreset(
        xalias='E',
        xlabel='Excitation Energy',
        ylabel=r'$\rho$ [MeV$^{-1}$]',
        plot_settings=PlotSettings.log_step()
    ),
    'gsf': VectorPreset(
        xalias='E',
        xlabel=r'$E_\gamma$',
        ylabel=r'$f_{E1}$ [MeV$^{-3}$]',
        plot_settings=PlotSettings.log_step()
    ),
    'transmission': VectorPreset(
        xalias='E',
        xlabel=r'$E_\gamma$',
        ylabel='Transmission Coefficient',
        plot_settings=PlotSettings.log_line()
    ),
}

# Custom user presets
_CUSTOM_PRESETS: dict[str, VectorPreset] = {}


def register_preset(name: str, preset: VectorPreset) -> None:
    """Register a custom preset.
    
    Args:
        name: Preset name (will be used in Vector.from_preset(name, ...))
        preset: VectorPreset object
    
    Examples:
        >>> # Define custom preset
        >>> my_preset = VectorPreset(
        ...     xalias='t',
        ...     xlabel='Time [ms]',
        ...     ylabel='Amplitude',
        ...     plot_settings=PlotSettings.linear_line()
        ... )
        >>> 
        >>> # Register it
        >>> register_preset('my_timeseries', my_preset)
        >>> 
        >>> # Use it
        >>> vec = Vector.from_preset('my_timeseries', times, amplitudes)
    """
    if name in _BUILTIN_PRESETS:
        raise ValueError(f"Cannot override built-in preset '{name}'")
    _CUSTOM_PRESETS[name] = preset


def get_preset(name: str) -> VectorPreset:
    """Get a preset by name.
    
    Args:
        name: Preset name
    
    Returns:
        VectorPreset object
    
    Raises:
        ValueError: If preset not found
    
    Examples:
        >>> preset = get_preset('iterations')
        >>> preset = get_preset('my_custom_preset')
    """
    # Check custom first (allows shadowing built-ins if needed)
    if name in _CUSTOM_PRESETS:
        return _CUSTOM_PRESETS[name]
    if name in _BUILTIN_PRESETS:
        return _BUILTIN_PRESETS[name]
    
    available = list_presets()
    raise ValueError(
        f"Unknown preset '{name}'. Available presets: {', '.join(available)}"
    )


def list_presets() -> list[str]:
    """List all available presets (built-in and custom).
    
    Returns:
        List of preset names
    
    Example:
        >>> presets = list_presets()
        >>> print(presets)
        ['iterations', 'spectrum', 'residuals', ...]
    """
    return sorted(list(_BUILTIN_PRESETS.keys()) + list(_CUSTOM_PRESETS.keys()))


def unregister_preset(name: str) -> None:
    """Remove a custom preset.
    
    Args:
        name: Preset name to remove
    
    Raises:
        ValueError: If trying to remove a built-in preset
        KeyError: If preset doesn't exist
    """
    if name in _BUILTIN_PRESETS:
        raise ValueError(f"Cannot unregister built-in preset '{name}'")
    del _CUSTOM_PRESETS[name]



