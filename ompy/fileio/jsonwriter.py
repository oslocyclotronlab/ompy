from __future__ import annotations
from typing import TYPE_CHECKING, Any, Self
from .writer import PrimitiveWriter
from pathlib import Path
from .. import __full_version__
import json

if TYPE_CHECKING:
    from .tree import Primitive, DataNode


class JSONWriter(PrimitiveWriter):
    def __init__(self):
        super().__init__()
        self.data: dict[str, Any] = {}
        self.path: Path | None = None

    def open(self, path: Path | str) -> Self:
        self.path = Path(path).with_suffix('.json')
        return self

    def write(self, data: Primitive):
        self.data[data.label] = dict(value=data.data,
                                     link=False,
                                     type='primitive')

    def write_link(self, data: DataNode, path: Path):
        self.data[data.label] = dict(value=str(path),
                                     link=True,
                                     type='data.dtype')

    def write_version(self):
        self.data['__version'] = __full_version__

    def write_self(self):
        self.data['__writer'] = 'JSONWriter'

    def close(self):
        if self.path is None:
            raise Exception("No path set for JSONWriter")
        with self.path.open("w") as f:
            json.dump(self.data, f, indent=4)

        # Reset writer on close
        self.path = None
        self.data = {}
