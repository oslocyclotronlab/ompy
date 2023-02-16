from . import Matrix, Vector
from .abstractarray import AbstractArray, to_plot_axis
from typing import Union, Tuple, Optional, overload, Literal
import numpy as np
from ..stubs import array

@overload
def zeros_like(array: Vector, **kwargs) -> Vector: ...
@overload
def zeros_like(array: Matrix, **kwargs) -> Matrix: ...


def zeros_like(array: AbstractArray,
               **kwargs) -> AbstractArray:
    match array:
        case Vector():
            return array.clone(values=np.zeros_like(array.values), **kwargs)
        case Matrix():
            return array.clone(values=np.zeros_like(array.values), **kwargs)
        case np.ndarray():
            if array.ndim != 1:
                raise ValueError("Numpy array must be vector.")
            return Vector(X=array, values=np.zeros_like(array), **kwargs)
        case _:
            raise ValueError(f"Expected Array, not {type(array)}.")


@overload
def empty_like(array: Matrix, **kwargs) -> Matrix: ...


@overload
def empty_like(array: Vector, **kwargs) -> Vector: ...


def empty_like(array: AbstractArray,
               **kwargs) -> AbstractArray:
    match array:
        case Vector():
            return array.clone(values=np.empty_like(array.values), **kwargs)
        case Matrix():
            return array.clone(values=np.empty_like(array.values), **kwargs)
        case np.ndarray():
            if array.ndim != 1:
                raise ValueError("Numpy array must be vector.")
            return Vector(X=array, values=np.empty_like(array), **kwargs)
        case _:
            raise ValueError(f"Expected Array, not {type(array)}.")

@overload
def empty(ex: ..., eg: None, **kwargs) -> Vector: ...
@overload
def empty(ex: ..., eg: array, **kwargs) -> Matrix: ...

def empty(**kwargs):
    if eg is None:
        values = np.empty(len(ex), **kwargs)
        return Vector(E=ex, values=values)
    values = np.empty((len(ex), len(eg)), **kwargs)
    return Matrix(values=values, Ex=ex, Eg=eg)


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