from .matrix import Matrix
from .vector import Vector
from .abstractarray import AbstractArray
from typing import Union, Tuple
import numpy as np


def zeros_like(array: AbstractArray,
               **kwargs) -> AbstractArray:
    if isinstance(array, Matrix):
        return Matrix(Ex=array.Ex, Eg=array.Eg,
                      values=np.zeros_like(array.values, **kwargs))
    elif isinstance(array, Vector):
        return Vector(E=array.E, values=np.zeros_like(array.values, **kwargs))
    else:
        raise ValueError(f"Expected Array, not {type(array)}.")


def zeros(array: Union[np.ndarray,
                       Tuple[int],
                       Tuple[int, int],
                       int],
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

