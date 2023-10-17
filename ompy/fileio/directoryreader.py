
from __future__ import annotations
from typing import TYPE_CHECKING, Literal, Generator
from .visitor import Visitor
from pathlib import Path
from contextlib import contextmanager
from .reader import Reader
from .numpyreader import NumpyReader
from .jsonreader import JSONReader

if TYPE_CHECKING:
    from .tree import (Node, DataNode, Root, Primitive,
        PrimitiveComposite, TreeConvertableNode,
        TreeComposite, Array, Unsupported)


class DirectoryReader(Visitor):
    """
    The reader must select the correct reader from
    the saved data.

    Make it possible to walk over a saved tree without loading, and query it.
        "show all Bootstraps"
    """
    def __init__(self, path: Path | str):
        self.path: Path = Path(path)
        self.cwd: Path = self.path

    def read(self) -> Root:
        root = Root()
        self.cwd = self.path
        return root

    def find_data_file(self, path: Path) -> Path:
        matches = list(path.glob('data.*'))
        if len(matches) == 0:
            raise FileNotFoundError(f"No data file found in {path}")
        elif len(matches) > 1:
            raise FileNotFoundError(f"Multiple data files found in {path}")
        return matches[0]
