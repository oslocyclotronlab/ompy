from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Self

from .plotsettings import PlotSettings
from ..units import Unit
from ..stubs import Unitlike


@dataclass(frozen=True, slots=True)
class VectorMetadata:
    valias: str = ''
    vlabel: str = 'Counts'
    vunit: Unit = ''  # Unit for values (y-axis), defaults to dimensionless
    name: str = ''
    misc: dict[str, Any] = field(default_factory=dict)
    plot_settings: PlotSettings = field(default_factory=PlotSettings)

    def __post_init__(self):
        """Convert dict to PlotSettings if needed, and ensure vunit is a Unit."""
        if self.plot_settings is not None and isinstance(self.plot_settings, dict):
            object.__setattr__(self, 'plot_settings', PlotSettings.from_dict(self.plot_settings))
        # Convert vunit to Unit if it's not already
        object.__setattr__(self, 'vunit', Unit(self.vunit))

    def clone(self, valias: str | None = None, vlabel: str | None = None,
              vunit: Unitlike | None = None,
              name: str | None = None, misc: dict[str, Any] | None = None,
              plot_settings: PlotSettings | dict[str, Any] | None | object = None) -> Self:
        valias = valias if valias is not None else self.valias
        vlabel = vlabel if vlabel is not None else self.vlabel
        vunit = vunit if vunit is not None else self.vunit
        name = name if name is not None else self.name
        misc = misc if misc is not None else self.misc
        # Special handling for plot_settings: None is a valid value to set
        if plot_settings is None:
            plot_settings = self.plot_settings
        return self.__class__(valias, vlabel, vunit, name, misc, plot_settings)

    def update(self, **kwargs: Any) -> Self:
        return self.clone(**kwargs)

    def add_comment(self, key: str, value: Any) -> Self:
        return self.update(misc=self.misc | {key: value})
