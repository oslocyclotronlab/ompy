from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Self, TYPE_CHECKING
from pathlib import Path

if TYPE_CHECKING:
    from .tree import DataNode

WRITERS: dict[str, type[Writer]] = {}

class Writer(ABC):
    def __init__(self):
        self.file = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not cls.__abstractmethods__:
            WRITERS[cls.__name__] = cls

    """
    Abstract base class for all writers.
    """
    def open(self, path: Path | str, **kwargs) -> Self:
        self.file = Path(path).open(**kwargs)
        return self

    @abstractmethod
    def write(self, data: DataNode):
        """
        Write data to the writer.

        :param data: The data to write.
        """
        pass

    def close(self) -> None:
        """
        Close the writer.
        """
        if self.file is not None:
            self.file.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class PrimitiveWriter(Writer):
    @abstractmethod
    def write_version(self):
        pass

    @abstractmethod
    def write_self(self):
        pass

    def write_meta(self):
        self.write_version()
        self.write_self()

    @abstractmethod
    def write_link(self, data: DataNode, path: Path):
        pass
