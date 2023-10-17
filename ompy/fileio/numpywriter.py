from __future__ import annotations
from typing import TYPE_CHECKING, Self
from .writer import Writer
import numpy as np
from pathlib import Path

if TYPE_CHECKING:
    from .tree import Array

class NumpyWriter(Writer):
    def __init__(self, dtype='auto', format='npz'):
        super().__init__()
        self.dtype = dtype
        self.format = format
        self.path: Path | None = None

    def open(self, path: Path | str) -> Self:
        self.path = Path(path).with_suffix(f'.{self.format}')
        return self

    def write(self, data: Array):
        assert self.path is not None

        array: np.ndarray = data.data
        if self.dtype == 'auto':
            dtype = array.dtype
        else:
            dtype = np.dtype(self.dtype)

        if self.format == 'npy':
            np.save(self.path, array.astype(dtype))
        elif self.format == 'npz':
            np.savez(self.path, array.astype(dtype))
        else:
            raise ValueError(f'Unknown format {self.format}')
