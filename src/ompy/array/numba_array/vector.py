import numba as nb
from numba.experimental import jitclass
import numpy as np


@jitclass([('values', nb.types.float32[:])])
class Vector:
    def __init__(self, bins: np.ndarray, values: np.ndarray):
        self.values = values
        self.bins = bins
        self.overflow = 0.0
        self.underflow = 0.0

    def at(self, i: int) -> float:
        if i < 0 or i >= len(self.values):
            raise ValueError("Index is out of bounds")
        return self.values[i]

    def index(self, i: int) -> int:
        if i < self.bins[0]:
            raise ValueError("Value is less than the minimum bin")
        elif i > self.bins[-1]:
            raise ValueError("Value is greater than the maximum bin")
        else:
            return self._index(i)

    def _index(self, i: float) -> int:
        return np.searchsorted(self.bins, i, side='right')

    def increment(self, where: float, value: float):
        if where < self.bins[0]:
            self.underflow += value
        elif where > self.bins[-1]:
            self.overflow += value
        else:
            i = self._index(where)
            self.values[i] += value

    def __len__(self):
        return len(self.values)
