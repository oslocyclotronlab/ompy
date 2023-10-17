from typing import Union
from . import ureg, Q_
from pathlib import Path
from pint._typing import UnitLike
from pint import Quantity
import numpy as np
from numpy.typing import NDArray as Array
from numpy.typing import ArrayLike
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.colorbar import Colorbar
from matplotlib.collections import QuadMesh
from matplotlib.lines import Line2D
from matplotlib.container import ErrorbarContainer, BarContainer
from typing import TypeAlias, Any, Literal, TypeGuard
from nptyping import NDArray, Shape, Floating
from matplotlib.collections import PathCollection

Unit = type(ureg('keV'))
keV = Unit
QuantityLike: TypeAlias = Quantity | int | float | str
Unitlike: TypeAlias = UnitLike
Unitful: TypeAlias = QuantityLike | Unitlike
Pathlike: TypeAlias = Path | str
ArrayKeV = type(Q_(np.asarray([3.0, 4.0]), 'keV'))
ArrayFloat: TypeAlias = Array[np.float_]
ArrayBool: TypeAlias = Array[np.bool_]
ArrayInt: TypeAlias = Array[np.int_]
array: TypeAlias = np.ndarray
arraylike: TypeAlias = ArrayLike | array # | ArrayKeV
PointUnit: TypeAlias = tuple[Unitlike, Unitlike]
Point: TypeAlias = tuple[Any, Any]
PointF: TypeAlias = tuple[float, float]
PointI: TypeAlias = tuple[int, int]
Points: TypeAlias = tuple[Point, Point]
numeric: TypeAlias = int | float | np.number
LineKwargs: TypeAlias = dict[str, Any]
ErrorBarKwargs: TypeAlias = dict[str, Any]
ErrorPlotKind: TypeAlias = Literal['line', 'fill']
Lines: TypeAlias = tuple[Line2D, ...] | Line2D | list[Line2D]
Plot1D: TypeAlias = tuple[Axes, Line2D]
Plots1D: TypeAlias = tuple[Axes, Lines | list[Lines]]
Plot2D: TypeAlias = tuple[Axes, Any]
Plots2D: TypeAlias = tuple[list[Axes], list[Any]]
PlotError1D: TypeAlias = tuple[Axes, ErrorbarContainer]
PlotScatter1D: TypeAlias = tuple[Axes, PathCollection]
PlotBar1D: TypeAlias = tuple[Axes, BarContainer]
VectorPlot: TypeAlias = Plot1D | Plots1D | PlotError1D | PlotScatter1D | PlotBar1D

array1D: TypeAlias = NDArray[Shape["*"], Floating]
array2D: TypeAlias = NDArray[Shape["*, *"], Floating]
array3D: TypeAlias = NDArray[Shape["*, *, *"], Floating]

def is_lines(x) -> TypeGuard[Lines]:
    for line in x:
        if not isinstance(line, Line2D):
            return False
    return True
