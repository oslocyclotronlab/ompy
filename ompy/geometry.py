from __future__ import annotations
from typing import TYPE_CHECKING, overload, Literal
from .stubs import Axes, Point, PointUnit, Points, Unitlike, ArrayBool, PointI
from .helpers import make_axes
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from abc import ABC, abstractmethod
from .library import from_unit

if TYPE_CHECKING:
    from .array import Matrix

def extrapolate_index(index):
    dx = index[1] - index[0]
    return lambda x: (x - index[0])/dx


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

    @make_axes
    def plot(self, matrix: Matrix, ax: Axes, **kwargs) -> Axes:
        return self.draw(matrix, ax, **kwargs)

    @abstractmethod
    def draw(self, matrix: Matrix, ax: Axes, **kwargs) -> Axes:
        ...

    @staticmethod
    def resolve(point: PointUnit, matrix: Matrix) -> PointI:
        ix = matrix.index_X(point[0])
        iy = matrix.index_Y(point[1])
        #ix = extrapolate_index(matrix.X)(point[0])
        #iy = extrapolate_index(matrix.Y)(point[1])
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
                 slope: float | None = None,
                 intercept: float | None = None):
        super().__init__()
        self.p1 = p1
        self.p2 = p2
        self.slope = slope
        self.intercept = intercept

    def above(self, matrix: Matrix) -> ArrayBool:
        return self.line_mask(matrix, 'above')

    def at(self, matrix: Matrix, tol=1e-2) -> ArrayBool:
        return self.line_mask(matrix, 'at', tol)

    def below(self, matrix: Matrix) -> ArrayBool:
        return self.line_mask(matrix, 'below')

    def draw_mask(self, matrix: Matrix, ax: Axes, **kwargs) -> Axes:
        mask = self.at(matrix, tol=kwargs.pop('tol', 40))
        X, Y = matrix._plot_mesh()
        # Ugly hack to plot mask
        masked = np.ma.array(mask, mask=~mask)
        palette = plt.cm.gray.with_extremes(over=kwargs.pop('color', 'r'))
        ax.pcolormesh(Y, X, masked, cmap=palette,
                      norm=colors.Normalize(vmin=-1.0, vmax=0.5))
        return ax

    def draw(self, matrix: Matrix, ax: Axes, **kwargs) -> Axes:
        slope, intercept = get_line_parameters(p0=self.p1, p1=self.p2,
                                               slope=self.slope, intercept=self.intercept)
        line = slope * matrix.Y + intercept
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax.plot(matrix.Y, line, **kwargs)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        return ax

    def parallel_shift(self, delta: Unitlike) -> Line:
        p1, p2 = parallel_shift(delta, self.p1, self.p2, self.slope, self.intercept)
        return Line(p1=p1, p2=p2)

    def line_mask(self, matrix: Matrix,
                  where: str = 'above',
                  tol: float = 1e-2) -> ArrayBool:
        """Create a mask for above (True) and below (False) a line

        Args:

        Returns:
            The boolean array with counts below the line set to False

        NOTE:
            This method and Jørgen's original method give 2 pixels difference
            Probably because of how the interpolated line is drawn
        """    
        # Create meshgrid for X and Y
        X_grid, Y_grid = np.meshgrid(matrix.Y, matrix.X)

        # Calculate the slope and intercept of the line
        
        slope, intercept = get_line_parameters(p0=self.p1, p1=self.p2,
                                               slope=self.slope, intercept=self.intercept)

        # Calculate the line passing through the points
        line = X_grid * slope + intercept

        match where:
            case 'above':
                mask = Y_grid > line
            case 'below':
                mask = Y_grid < line
            case 'at':
                mask = np.abs(Y_grid - line) < tol
            case _:
                raise ValueError("'where' must be 'above', 'below' or 'at'")
        return mask

def get_line_parameters(p0: None | Point = None,
                        p1: None | Point = None, 
                        slope: float | Point = None, 
                        intercept: float | None = None) -> tuple[float, float]:
    if (p0 is not None) and (p1 is not None):
        # Calculate the slope and intercept using the two points
        slope = (p1[1] - p0[1]) / (p1[0] - p0[0])
        intercept = p0[1] - slope * p0[0]
    elif (slope is not None) and (intercept is not None):
        # Both slope and intercept are given, no need to calculate anything
        pass
    elif (p0 is not None) and (slope is not None):
        # Calculate the intercept using the point and slope
        intercept = p0[1] - slope * p0[0]
    elif (p0 is not None) and (intercept is not None):
        # Calculate the slope using the point and intercept
        slope = (p0[1] - intercept) / p0[0]
    else:
        raise ValueError("Insufficient information to define a line.")
    assert slope is not None
    assert intercept is not None

    return slope, intercept

def parallel_shift(delta: float,
                   p0: Point | None = None,
                   p1: Point | None = None,
                   slope: float | None = None, intercept: float | None = None) -> Points:
    """
    Shifts a line defined by two points or slope and intercept, in parallel by a distance delta.

    :param p0: The first point on the line, as a tuple of (x, y) coordinates.
    :param p1: The second point on the line, as a tuple of (x, y) coordinates.
    :param slope: The slope of the line.
    :param intercept: The intercept of the line.
    :param delta: The distance to parallel shift the line.
    :return: Two new points (p3, p4) that define the shifted line.
    """
    # Get the slope and intercept of the line
    slope, intercept = get_line_parameters(p0, p1, slope, intercept)

    if p0 is None:
        # Use an arbitrary point on the line if p0 is not provided
        p0 = (0, intercept)

    if p1 is None:
        # Use another arbitrary point on the line if p1 is not provided
        p1 = (1, slope + intercept)

    p0, p1 = np.array(p0), np.array(p1)

    # Normal to the line
    n = (p1 - p0) / np.linalg.norm(p1 - p0)

    # 90 degrees to the line
    v = np.array([-n[1], n[0]])

    # Shift the line by delta
    p3 = p0 + v * delta
    p4 = p1 + v * delta

    return p3, p4

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

    def draw(self, matrix: Matrix, ax: Axes, **kwargs) -> Axes:
        kw = {'color': 'r'} | kwargs
        self.line.draw(matrix, ax, **kw)
        self.upper.draw(matrix, ax, ls='--', **kw)
        return self.lower.draw(matrix, ax, ls='--', **kw)

    def within(self, matrix: Matrix) -> ArrayBool:
        mask = self.upper.above(matrix) & self.lower.below(matrix)
        return mask
