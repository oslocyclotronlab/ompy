from . import Matrix, Vector
from .abstractarray import AbstractArray, to_plot_axis
from typing import Union, Tuple, Optional, overload, Literal, Callable
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


def linspace(start, stop, num, *, edge = 'left', **kwargs) -> Vector:
    bins = np.linspace(start, stop, num, **kwargs)
    return Vector(X=bins, values=np.zeros(len(bins), dtype=float))

@overload
def fmap(array: Vector, func: Callable[[np.ndarray], np.ndarray], *args, **kwargs) -> Vector: ...
@overload
def fmap(array: Matrix, func: Callable[[np.ndarray], np.ndarray], *args, **kwargs) -> Matrix: ...

def fmap(array: AbstractArray, func: Callable[[np.ndarray], np.ndarray], *args, **kwargs) -> AbstractArray:
    """ `functor_map`. Applies a function to the values of an array. Equal to a Haskell fmap <&>"""
    match array:
        case Vector():
            return array.clone(values=func(*args, **kwargs))
        case Matrix():
            return array.clone(values=func(*args, **kwargs))
        case _:
            raise ValueError(f"Expected Array, not {type(array)}.")

@overload
def umap(array: Vector, func: Callable[[np.ndarray], np.ndarray], *args, **kwargs) -> Vector: ...
@overload
def umap(array: Matrix, func: Callable[[np.ndarray], np.ndarray], *args, **kwargs) -> Matrix: ...

def umap(array: AbstractArray, func: Callable[[np.ndarray], np.ndarray], *args, **kwargs) -> AbstractArray:
    """ `unwrap_map`. Applies a function to the values of an array. Equal to a Haskell fmap <&>.

    This is the same as fmap, but uses the array as the first argument of the function and
    unwraps other arguments.
    """
    args = [a.values if isinstance(a, AbstractArray) else a for a in args]
    kwargs = {k: v.values if isinstance(v, AbstractArray) else v for k, v in kwargs.items()}
    match array:
        case Vector():
            return array.clone(values=func(array.values, *args, **kwargs))
        case Matrix():
            return array.clone(values=func(array.values, *args, **kwargs))
        case _:
            raise ValueError(f"Expected Array, not {type(array)}.")


def omap(func: Callable[[np.ndarray], np.ndarray], *args, **kwargs) -> any:
    """ `out_of_map`. Applies `func` while unwrapping the arguments. """
    args = [a.values if isinstance(a, AbstractArray) else a for a in args]
    kwargs = {k: v.values if isinstance(v, AbstractArray) else v for k, v in kwargs.items()}
    return func(*args, **kwargs)


@overload
def xmap(array: Vector, func: Callable[[np.ndarray], np.ndarray], *args, **kwargs) -> Vector: ...
@overload
def xmap(array: Matrix, func: Callable[[np.ndarray], np.ndarray], *args, **kwargs) -> Matrix: ...

def xmap(array: AbstractArray, func: Callable[[np.ndarray], np.ndarray], *args, **kwargs) -> AbstractArray:
    """ `x unwrap_map`. Applies a function to the index of an array. Equal to a Haskell fmap <&>.

    This is the same as fmap, but uses the index of the array as the first argument of the function and
    unwraps other arguments.
    """
    args = [a.values if isinstance(a, AbstractArray) else a for a in args]
    kwargs = {k: v.values if isinstance(v, AbstractArray) else v for k, v in kwargs.items()}
    match array:
        case Vector():
            return array.clone(values=func(array.X, *args, **kwargs))
        case Matrix():
            return array.clone(values=func(array.X, *args, **kwargs))
        case _:
            raise ValueError(f"Expected Array, not {type(array)}.")
