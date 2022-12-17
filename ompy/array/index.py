import numpy as np
import warnings
from abc import ABC, abstractmethod
from collections import OrderedDict

try:
    from numba import njit, int32, float32, float64
    from numba.experimental import jitclass
except ImportError:
    warnings.warn("Numba could not be imported. Falling back to non-jiting which will be much slower")
    int32 = np.int32
    float32 = np.float32
    float64 = np.float64

    def nop_decorator(func, *aargs, **kkwargs):
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    njit = nop_decorator
    jitclass = nop_decorator


class Index(ABC):
    """ Abstract base class for indexers """

    def __init__(self, bins: np.ndarray):
        assert len(bins) > 0
        assert is_monotone(bins), "Bins must be monotone."
        self.X = bins

    @abstractmethod
    def index(self, x: float64) -> int:
        """ Index of the bin containing x """
        pass

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

    def assert_inbounds(self, x: float64):
        """ Assert that x is in bounds """
        if x < self.leftmost:
            raise ValueError(f"{x} is smaller than leftmost edge {self.leftmost}")
        if x > self.rightmost:
            raise ValueError(f"{x} is larger than rightmost edge {self.rightmost}")


class LeftEdge:
    def left_edge(self, i: int) -> float64:
        return self.X[i]

    def right_edge(self, i: int) -> float64:
        return self.X[i] + self.dX

    def mid(self, i: int) -> float64:
        return self.X[i] + self.dX / 2


class MidEdge:
    def left_edge(self, i: int) -> float64:
        return self.X[i] - self.dX / 2

    def right_edge(self, i: int) -> float64:
        return self.X[i] + self.dX / 2

    def mid(self, i: int) -> float64:
        return self.X[i]


class UniformIndex(ABC, Index):
    """ Index for equidistant binning """
    def __init__(self, bins: np.ndarray):
        super().__init__(bins)
        assert is_equidistant(bins)
        self.dX = bins[1] - bins[0]

    @property
    def step(self) -> float64:
        return self.dX


class LeftUniformIndex(UniformIndex, LeftEdge):
    """ Index for left-binning """
    def __init__(self, bins: np.ndarray):
        super().__init__(bins)

    def index(self, x: float64) -> int:
        self.assert_inbounds(x)
        i = 0
        while i < len(self.X):
            if x > self.X[i]:
                return i
            i += 1
        return i - 1


class MidUniformIndex(UniformIndex, MidEdge):
    """ Index for uniform mid-binning """
    def __init__(self, bins: np.ndarray):
        super().__init__(bins)

    def index(self, x: float64) -> int:
        raise NotImplementedError("Not implemented yet")
        self.assert_inbounds(x)
        i = 0
        while i < len(self.X):
            if self.X[i] > x:
                return i - 1
            i += 1
        return i - 1


class NonUniformIndex(ABC, Index):
    """ Index for non-equidistant binning """
    def __init__(self, bins: np.ndarray):
        super().__init__(bins)


class LeftNonUniformIndex(NonUniformIndex, LeftEdge):
    def __init__(self, bins: np.ndarray):
        super().__init__(bins)

    def index(self, x: float64) -> int:
        raise NotImplementedError("Not implemented yet")


class MidNonUniformIndex(NonUniformIndex, MidEdge):
    def __init__(self, bins: np.ndarray):
        super().__init__(bins)

    def index(self, x: float64) -> int:
        raise NotImplementedError()


@njit
def is_monotone(x: np.ndarray) -> bool:
    """ Check if x is strictly monotone increasing """
    for i in range(len(x) - 1):
        if x[i] >= x[i + 1]:
            return False
    return True


@njit
def is_equidistant(X: np.ndarray) -> bool:
    """ Check if X is equidistant """
    dX = X[1] - X[0]
    return np.allclose(X[1:] - X[:-1], dX)