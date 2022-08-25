from . import Matrix, Vector
from .abstractarray import AbstractArray, to_plot_axis
from typing import Union, Tuple, Optional, overload, Literal
import numpy as np
from .stubs import array


def zeros_like(array: AbstractArray,
               **kwargs) -> AbstractArray:
    if isinstance(array, Matrix):
        return Matrix(Ex=array.Ex, Eg=array.Eg,
                      values=np.zeros_like(array.values, **kwargs))
    elif isinstance(array, Vector):
        return Vector(E=array.E, values=np.zeros_like(array.values, **kwargs))
    else:
        raise ValueError(f"Expected Array, not {type(array)}.")

@overload
def empty_like(array: Matrix, **kwargs) -> Matrix: ...


@overload
def empty_like(array: Vector, **kwargs) -> Vector: ...


def empty_like(array: AbstractArray,
               **kwargs) -> AbstractArray:
    if isinstance(array, Matrix):
        return Matrix(Ex=array.Ex, Eg=array.Eg,
                      values=np.empty_like(array.values, **kwargs))
    elif isinstance(array, Vector):
        return Vector(E=array.E, values=np.empty_like(array.values, **kwargs))
    else:
        raise ValueError(f"Expected Array, not {type(array)}.")


def zeros(array: array | int | Tuple[int, int],
          **kwargs) -> AbstractArray:
    raise NotImplementedError()
    if isinstance(array, np.ndarray):
        if array.ndim == 1:
            return Vector(values=np.zeros_like(array, **kwargs))
        elif array.ndim == 2:
            return Matrix(values=np.zeros_like(array, **kwargs))
        else:
            raise ValueError("Array must have dimension < 3.")
    elif isinstance(array, (tuple, list)):
        if len(array) == 1:
            return Vector(values=np.zeros(array, **kwargs))
        elif len(array) == 2:
            return Matrix(values=np.zeros(array, **kwargs))
        else:
            raise ValueError("Array must have dimension < 3.")
    else:
        raise ValueError(f"Expected numpy array or iterable, not {type(array)}.")


def sum(array: Union[Matrix, Vector, np.ndarray],
        axis: Union[int, str] | None = None,
        **kwargs) -> Union[Vector, np.ndarray]:
    raise NotImplementedError()
    if isinstance(array, Vector):
        return array.sum(**kwargs)
    elif isinstance(array, Matrix):
        axis = to_plot_axis(axis)
        val = array.sum(axis=axis, **kwargs)
        return Vector()
