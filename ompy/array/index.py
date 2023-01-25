from __future__ import annotations
import numpy as np
import warnings
from abc import ABC, abstractmethod
from collections import OrderedDict
from .index_fn import _index_left, _index_mid, _index_mid_uniform, is_monotone, is_uniform, is_length_congruent
from .. import float64
from .rebin import _rebin_uniform_left_left, Preserve
from typing import TypeAlias, Literal, Type

"""
TODO Mixin architecture is not quite right. Read up.
-[ ] Implement binary search
-[ ] Implement conversion methods
How is rebinning handled?
"""



class Index(ABC):
    """ Abstract base class for indexers """

    def __init__(self, bins: np.ndarray):
        assert len(bins) > 0
        assert is_monotone(bins), "Bins must be monotone."
        self.X = bins


    def rebin(self, other: Index | np.ndarray, values: np.ndarray, preserve: Preserve = 'counts'):
        match other:
            case Index():
                pass
            case _:
                other = to_index(values, 'left')
        return self._rebin(other, values, preserve=preserve)

    @abstractmethod
    def _rebin(self, other: Index, values: np.ndarray, preserve: Preserve = 'counts') -> tuple[Index, np.ndarray]:
        pass

    @abstractmethod
    def index(self, x: float64) -> int:
        """ Index of the bin containing x """
        pass

    def __getitem__(self, i: int) -> float:
        return self.X[i]


class Edge(ABC):
    def __init__(self, boundary: float):
        self.boundary = boundary

    @abstractmethod
    def left_edge(self, i: int) -> float64:
        """ Left edge of the bin """
        pass

    @abstractmethod
    def right_edge(self, i: int) -> float64:
        """ Right edge of the bin """
        pass

    @abstractmethod
    def mid(self, i: int) -> float64:
        """ Mid point of the bin """
        pass

    @property
    def leftmost(self) -> float64:
        """ Left edge of the leftmost bin """
        return self.left_edge(0)

    @property
    def rightmost(self) -> float64:
        """ Right edge of the rightmost bin """
        return self.right_edge(-1)

    def is_inbounds(self, x: float64) -> bool:
        """ Check if index is in bounds """
        return self.leftmost <= x <= self.rightmost

    @abstractmethod
    def is_left(self) -> bool: ...

    @abstractmethod
    def to_left(self) -> Edge: ...

    @abstractmethod
    def to_mid(self) -> Edge: ...

    @abstractmethod
    def other_edge_cls(self) -> Type[Index]: ...


    def assert_inbounds(self, x: float64):
        """ Assert that x is in bounds """
        if x < self.leftmost:
            raise ValueError(f"{x} is smaller than leftmost edge {self.leftmost}")
        if x >= self.rightmost:
            raise ValueError(f"{x} is larger than rightmost edge {self.rightmost}")


class Left(Edge):
    def left_edge(self, i: int) -> float64:
        return self.X[i]

    def right_edge(self, i: int) -> float64:
        return self.X[i] + self.step(i)

    def mid(self, i: int) -> float64:
        return self.X[i] + self.step(i) / 2

    def to_left(self) -> Index:
        return self

    def to_mid(self) -> Index:
        bins = self.X + self.steps() / 2
        return self.other_edge_cls()(bins)

    def index(self, x: float64) -> int:
        self.assert_inbounds(x)
        return _index_left(self.X, x)

    def is_left(self) -> bool:
        return True


class Mid(Edge):
    def left_edge(self, i: int) -> float64:
        return self.X[i] - self.step(i) / 2

    def right_edge(self, i: int) -> float64:
        return self.X[i] + self.step(i) / 2

    def mid(self, i: int) -> float64:
        return self.X[i]

    def to_mid(self) -> Index:
        return self

    def to_left(self) -> Index:
        bins = self.X - self.steps() / 2
        return self.other_edge_cls()(bins)

    def index(self, x: float64) -> int:
        self.assert_inbounds(x)
        return _index_mid_uniform(self.X, x)

    def is_left(self) -> bool:
        return False


class Layout(ABC):
    def calibration(self) -> tuple[float, float]:
        """ Energy calibration relative to channel number"""
        raise NotImplementedError()

    @abstractmethod
    def steps(self) -> float64: ...

    @abstractmethod
    def step(self, i: int) -> float64: ...

    @abstractmethod
    def is_uniform(self) -> bool: ...


class Uniform(Layout):
    """ Index for equidistant binning """
    def __init__(self, bins: np.ndarray):
        super().__init__(bins)
        assert is_uniform(bins)
        self.dX = bins[1] - bins[0]

    def step(self, i: int) -> float64:
        return self.dX

    def steps(self) -> np.ndarray:
        return np.repeat(self.dX, len(self))

    def is_uniform(self) -> bool:
        return True

    def is_congruent(self, other) -> bool:
        if not is_length_congruent(self.X, other.X):
            return False
        if not other.uniform():
            return False
        dx = self.dX
        dy = other.dX
        if np.isclose(dx, dy) and np.isclose(self[0] % dx, other[0] % dy):
            return True
        return False


class LeftUniformIndex(Left, Uniform, Index):
    """ Index for left-binning """
    def __init__(self, bins: np.ndarray):
        super().__init__(bins)

    def _rebin(self, other: Index, values: np.ndarray, preserve: Preserve = 'counts') -> tuple[Index, np.ndarray]:
        if not other.is_uniform():
            raise NotImplementedError()
        if not other.is_left():
            other_ = other.to_left()
            new = _rebin_uniform_left_left(self.X, other_.X, values, preserve=preserve)
        else:
            new = _rebin_uniform_left_left(self.X, other.X, values, preserve=preserve)
        return other, new

    def other_edge_cls(self) -> Type[Index]:
        return MidUniformIndex


class MidUniformIndex(Mid, Uniform, Index):
    """ Index for uniform mid-binning """
    def __init__(self, bins: np.ndarray):
        super().__init__(bins)

    def _rebin(self, other: Index, values: np.ndarray, preserve: Preserve = 'counts') -> tuple[Index, np.ndarray]:
        if not other.is_uniform():
            raise NotImplementedError()
        this = self.to_left()
        if not other.is_left():
            other_ = other.to_left()
            new = _rebin_uniform_left_left(this.X, other_.X, values, preserve=preserve)
        else:
            new = _rebin_uniform_left_left(this.X, other.X, values, preserve=preserve)
        return other, new

    def other_edge_cls(self) -> Type[Index]:
        return LeftUniformIndex


class NonUniform(Layout):
    """ Index for non-equidistant binning """
    def __init__(self, bins: np.ndarray):
        super().__init__(bins)
        self.dX = np.diff(self.X)

    def is_uniform(self) -> bool:
        return False

    def step(self, i: int) -> np.float64:
        return self.dX[i]

    def steps(self) -> np.ndarray:
        return self.dX

    def _rebin(self, other: Index, values: np.ndarray, preserve: Preserve = 'counts') -> tuple[Index, np.ndarray]:
        raise NotImplementedError()


class LeftNonUniformIndex(Left, NonUniform, Index):
    def __init__(self, bins: np.ndarray):
        super().__init__(bins)

    def index(self, x: float64) -> int:
        self.assert_inbounds(x)
        return _index_left(self.X, x)

    def other_edge_cls(self) -> Type[Index]:
        return MidNonUniformIndex


class MidNonUniformIndex(Mid, NonUniform, Index):
    def __init__(self, bins: np.ndarray):
        super().__init__(bins)

    def index(self, x: float64) -> int:
        self.assert_inbounds(x)
        return _index_mid(self.X, x)

    def other_edge_cls(self) -> Type[Index]:
        return LeftNonUniformIndex



Edges: TypeAlias = Literal['left', 'mid']
def to_index(X: np.ndarray, edge: Edges = 'left') -> Index:
    X = np.asarray(X)
    if edge not in {'left', 'mid'}:
        raise ValueError(f"`edge` must be on of {Edges} not {edge}")
    if not is_monotone(X):
        raise ValueError("Indices must be monotone")
    if is_uniform(X):
        if edge == 'left':
            return LeftUniformIndex(X)
        else:
            return MidUniformIndex(X)
    else:
        if edge == 'left':
            return LeftNonUniformIndex(X)
        else:
            return MidNonUniformIndex(X)
