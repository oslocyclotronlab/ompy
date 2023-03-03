from __future__ import annotations
from typing import TYPE_CHECKING, overload, Literal
from .stubs import Axes, Point, PointUnit, Points, Unitlike, ArrayBool, PointI
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from abc import ABC, abstractmethod
from .library import from_unit

if TYPE_CHECKING:
    from .array import Matrix


class Geometry(ABC):

    # @abstractmethod
    @overload
    def apply(self, matrix: Matrix, inplace: Literal[False] = ...) -> None: ...

    # @abstractmethod
    @overload
    def apply(self, matrix: Matrix, inplace: Literal[True] = ...) -> Matrix: ...

    # @abstractmethod
    def apply(self, matrix: Matrix, inplace: bool = False) -> Matrix | None:
        ...

    def plot(self, matrix: Matrix, ax: Axes | None = None, **kwargs) -> Axes:
        if ax is None:
            ax: Axes = plt.subplots()[1]
        return self.draw(matrix, ax, **kwargs)

    @abstractmethod
    def draw(self, matrix: Matrix, ax: Axes, **kwargs) -> Axes:
        ...

    @staticmethod
    def resolve(point: PointUnit, matrix: Matrix) -> PointI:
        ix = matrix.index_X(point[0])
        iy = matrix.index_Y(point[1])
        return ix, iy

    @staticmethod
    @overload
    def maybe_resolve(point: PointUnit, matrix: Matrix) -> PointI: ...
    @staticmethod
    @overload
    def maybe_resolve(point: None, matrix: Matrix) -> None: ...

    @staticmethod
    def maybe_resolve(point: PointUnit | None, matrix: Matrix) -> PointI | None:
        if point is not None:
            return Geometry.resolve(point, matrix)
        return None



class Line(Geometry):
    def __init__(self, *, p1: PointUnit | None = None,
                 p2: PointUnit | None = None,
                 slope: float | None = None):
        super().__init__()
        self.p1 = p1
        self.p2 = p2
        self.slope = slope

    def resolve_self(self, matrix: Matrix) -> (PointI, PointI):
        p1: Point | None = Geometry.maybe_resolve(self.p1, matrix)
        p2: Point | None = Geometry.maybe_resolve(self.p2, matrix)
        slope = self.slope
        # Correct for binning
        if slope is not None:
            slope *= matrix.dX[0] / matrix.dY[0]
        p: Points = refine_points(p1, p2, slope)
        return p

    def above(self, matrix: Matrix) -> ArrayBool:
        p1, p2 = self.resolve_self(matrix)
        mask = line_mask(matrix, p1, p2)
        return mask

    def at(self, matrix: Matrix) -> ArrayBool:
        p1, p2 = self.resolve_self(matrix)
        mask = line_mask(matrix, p1, p2, 'at')
        return mask

    def below(self, matrix: Matrix) -> ArrayBool:
        p1, p2 = self.resolve_self(matrix)
        mask = line_mask(matrix, p1, p2, 'below')
        return mask

    def draw(self, matrix: Matrix, ax: Axes, color='r') -> Axes:
        mask = self.at(matrix)
        X, Y = matrix._plot_mesh()
        # Ugly hack to plot mask
        masked = np.ma.array(mask, mask=~mask)
        palette = plt.cm.gray.with_extremes(over=color)
        ax.pcolormesh(Y, X, masked, cmap=palette,
                      norm=colors.Normalize(vmin=-1.0, vmax=0.5))
        return ax

    def parallel_shift(self, delta: Unitlike) -> Line:
        p1_ = from_unit(self.p1, 'keV')
        p2_ = from_unit(self.p2, 'keV')
        d = from_unit(delta, 'keV')
        p1, p2 = refine_points(p1_, p2_, self.slope)
        p1 = np.array(p1)
        p2 = np.array(p2)
        # Normal to the line
        n = (p2 - p1) / np.linalg.norm(p2 - p1)
        # 90 degrees to the line
        v = np.array([-n[1], n[0]])
        # Shift the line by thickness
        p3 = p1 + v * d
        p4 = p2 + v * d
        return Line(p1=p3, p2=p4)


class ThickLine(Geometry):
    def __init__(self, line: Line, thickness: Unitlike):
        super().__init__()
        self.line = line
        self.thickness = thickness
        self.upper = line.parallel_shift(thickness)
        self.lower = line.parallel_shift(-self.thickness)

    def within(self, matrix: Matrix) -> ArrayBool:
        mask = self.upper.above(matrix) & self.lower.below(matrix)
        return mask

    def draw(self, matrix: Matrix, ax: Axes, color='r') -> Axes:
        self.line.draw(matrix, ax, color)
        self.upper.draw(matrix, ax, color='y')
        return self.lower.draw(matrix, ax, color='y')



def line_mask(matrix: Matrix, p1: PointI, p2: PointI,
              where: str = 'above') -> ArrayBool:
    """Create a mask for above (True) and below (False) a line

    Args:

    Returns:
        The boolean array with counts below the line set to False

    NOTE:
        This method and Jørgen's original method give 2 pixels difference
        Probably because of how the interpolated line is drawn
    """
    Ix = p1[0], p2[0]
    Iy = p1[1], p2[1]

    # Interpolate between the two points
    assert(Ix[1] != Ix[0])
    a = (Iy[1]-Iy[0])/(Ix[1]-Ix[0])
    b = Iy[0] - a*Ix[0]
    line = lambda x: a*x + b  # NOQA E731

    # Mask all indices below this line to 0
    X = np.arange(matrix.shape[1])
    Y = np.arange(matrix.shape[0])
    i_mesh, j_mesh = np.meshgrid(X, Y)
    match where:
        case 'above':
            mask = j_mesh > line(i_mesh)
        case 'below':
            mask = j_mesh < line(i_mesh)
        case 'at':
            mask = (j_mesh >= line(i_mesh)) & (j_mesh <= line(i_mesh)+1)
        case _:
            raise ValueError("'where' must be 'above', 'below' or 'at'")
    return mask


def refine_points(p1: Point | None = None, p2: Point | None = None,
                    slope: float | None = None) -> Points:
    if p1 is None and p2 is None:
        raise ValueError("Must provide at least 1 point")
    if p1 is not None and p2 is not None:
        return sort_points(p1, p2)
    if slope is None:
        raise ValueError("Provide both 1 point and a slope")

    p = p1 if p2 is None else p2
    assert p is not None

    return points_from_partial(slope, p)

def sort_points(p1: Point, p2: Point) -> Points:
    if p1[0] < p2[0]:
        return p1, p2
    if p1[0] > p2[0]:
        return p2, p1
    if p1[1] < p2[1]:
        return p1, p2
    return p2, p1

def points_from_partial(slope: float, point: Point) -> Points:
    X, Y = point
    intercept = Y - slope*X
    return points_from_ab(intercept, slope)

def points_from_ab(intercept: float, slope: float) -> Points:
    X0, X1 = 0.0, 5000.0
    Y0, Y1 = intercept, slope*X1 + intercept
    return ((X0, Y0), (X1, Y1))
