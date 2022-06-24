from typing import Union
from . import ureg, Q_
from pathlib import Path
#from pint._typing import UnitLike
import numpy as np
from numpy.typing import NDArray as Array
from numpy.typing import ArrayLike
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.colorbar import Colorbar
from matplotlib.collections import QuadMesh
from typing import TypeAlias, Any
#Unitlike = UnitLike | int | float
Unit = type(ureg('keV'))
Unitlike: TypeAlias = float | int | str | Unit
Pathlike: TypeAlias = Path | str
ArrayKeV = type(Q_(np.asarray([3.0, 4.0]), 'keV'))
ArrayFloat: TypeAlias = Array[np.float_]
ArrayBool: TypeAlias = Array[np.bool_]
ArrayInt: TypeAlias = Array[np.int_]
array: TypeAlias = np.ndarray
arraylike: TypeAlias = ArrayLike | array | ArrayKeV
PointUnit: TypeAlias = tuple[Unitlike, Unitlike]
Point: TypeAlias = tuple[Any, Any]
PointF: TypeAlias = tuple[float, float]
PointI: TypeAlias = tuple[int, int]
Points: TypeAlias = tuple[Point, Point]
