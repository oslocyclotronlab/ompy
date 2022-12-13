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
keV = Unit
Unitlike: TypeAlias = float | int | str | Unit
Pathlike: TypeAlias = Path | str
ArrayKeV = type(Q_(np.asarray([3.0, 4.0]), 'keV'))
ArrayFloat: TypeAlias = Array[np.float_]
ArrayBool: TypeAlias = Array[np.bool_]
ArrayInt: TypeAlias = Array[np.int_]
array: TypeAlias = np.ndarray
arraylike: TypeAlias = ArrayLike | array | ArrayKeV
PointUnit: TypeAlias = (Unitlike, Unitlike)
Point: TypeAlias = (Any, Any)
PointF: TypeAlias = (float, float)
PointI: TypeAlias = (int, int)
Points: TypeAlias = (Point, Point)
numeric: TypeAlias = int | float | np.number
