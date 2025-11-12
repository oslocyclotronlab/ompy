from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class MatrixMetadata:
    """Stores metadata for a Matrix.

    """
    valias: str = ''
    vlabel: str = 'Counts'
    vunit: Any = ''  # Unit for values (z-axis), defaults to dimensionless
    name: str = ''
    misc: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Ensure vunit is a Unit."""
        if self.vunit is not None and not hasattr(self.vunit, 'dimensionless'):
            from ..units import Unit
            object.__setattr__(self, 'vunit', Unit(self.vunit))

    def clone(self, valias: str | None = None, vlabel: str | None = None,
              vunit: Any | None = None,
              name: str | None = None, misc: dict[str, Any] | None = None) -> MatrixMetadata:
        valias = valias if valias is not None else self.valias
        vlabel = vlabel if vlabel is not None else self.vlabel
        vunit = vunit if vunit is not None else self.vunit
        name = name if name is not None else self.name
        misc = misc if misc is not None else self.misc
        return MatrixMetadata(valias, vlabel, vunit, name, misc)

    def update(self, **kwargs) -> MatrixMetadata:
        return self.clone(**kwargs)

    def add_comment(self, key: str, value: Any) -> MatrixMetadata:
        return self.update(misc=self.misc | {key: value})
