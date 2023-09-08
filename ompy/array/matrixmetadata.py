from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class MatrixMetadata:
    """Stores metadata for a Matrix.

    """
    valias: str = ''
    vlabel: str = 'Counts'
    name: str = ''
    misc: dict[str, Any] = field(default_factory=dict)

    def clone(self, valias: str | None = None, vlabel: str | None = None,
              name: str | None = None, misc: dict[str, Any] | None = None) -> MatrixMetadata:
        valias = valias if valias is not None else self.valias
        vlabel = vlabel if vlabel is not None else self.vlabel
        name = name if name is not None else self.name
        misc = misc if misc is not None else self.misc
        return MatrixMetadata(valias, vlabel, name, misc)

    def update(self, **kwargs) -> MatrixMetadata:
        return self.clone(**kwargs)

    def add_comment(self, key: str, value: Any) -> MatrixMetadata:
        return self.update(misc=self.misc | {key: value})
