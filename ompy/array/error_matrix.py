from __future__ import annotations
from .matrix import Matrix
from abc import abstractmethod
import numpy as np
from typing import Literal, Iterable
from ..stubs import arraylike, Unitlike
from .index import Index, Edge


class ErrorMatrix(Matrix):
    @classmethod
    @abstractmethod
    def add_error(cls, matrix: Matrix, *args, **kwargs) -> ErrorMatrix: ...

    @classmethod
    def from_matrix(cls, matrix: Matrix, *args, **kwargs) -> ErrorMatrix:
        return cls.add_error(matrix, *args, **kwargs)

    def save(self, path: str, *args, **kwargs) -> None:
        raise NotImplementedError()

    @classmethod
    def from_path(cls, path: str, *args, **kwargs) -> ErrorMatrix:
        raise NotImplementedError()


class AsymmetricMatrix(ErrorMatrix):
    def __init__(self, *, lerr: Iterable[float],
                 uerr: Iterable[float],
                 order: Literal['C', 'F'] = 'C',
                 dtype: type = np.float64,
                 copy: bool = False,
                 **kwargs):
        kwargs['copy'] = copy
        kwargs['dtype'] = dtype
        kwargs['order'] = order
        super().__init__(**kwargs)
        if copy:
            def fetch(x):
                return np.asarray(x, dtype=dtype, order=order).copy()
        else:
            def fetch(x):
                return np.asarray(x, dtype=dtype, order=order)
        self.lerr: np.ndarray = fetch(lerr)
        self.uerr: np.ndarray = fetch(uerr)
        if self.lerr.shape != self.values.shape:
            raise ValueError("lerr must have the same shape as values. Got"
                             f" {self.lerr.shape} and {self.values.shape}")
        if self.uerr.shape != self.values.shape:
            raise ValueError("uerr must have the same shape as values. Got"
                             f" {self.uerr.shape} and {self.values.shape}")

    @classmethod
    def add_error(cls, matrix: Matrix, lerr: np.ndarray, uerr: np.ndarray,
                  **kwargs) -> AsymmetricMatrix:
        return cls(values=matrix.values, lerr=lerr, uerr=uerr, **kwargs)


    @classmethod
    def from_CI(cls, matrix: Matrix, lower: np.ndarray,
                upper: np.ndarray, clip: bool = False, **kwargs) -> AsymmetricMatrix:
        lerr = matrix.values - lower
        uerr = upper - matrix.values
        if np.any(lerr < 0) or np.any(uerr < 0):
            if clip:
                lerr = np.maximum(lerr, 0)
                uerr = np.maximum(uerr, 0)
            else:
                raise ValueError(("CI must be greater than or equal to the "
                                  "values. Might be due to numerical precision. "
                                  "Consider setting `clip=True`"))
        return cls(X=matrix.X_index, values=matrix.values, lerr=lerr, uerr=uerr, **kwargs)


    def clone(self, X: Index | None = None, Y: Index | None = None,
              values: np.ndarray | None = None,
              lerr: np.ndarray | None = None, uerr: np.ndarray | None = None,
              metadata = None, copy: bool = False,
              **kwargs) -> Matrix:
        """ Copies the object.

        Any keyword argument will override the equivalent
        attribute in the copy. For example, matrix.clone(Eg=[1,2,3])
        tries to set the gamma energy to [1,2,3].

        kwargs: Any keyword argument is overwritten
            in the copy.
        Returns:
            The copy
        """
        X = X if X is not None else self.X_index
        Y = Y if Y is not None else self.Y_index
        values = values if values is not None else self.values
        lerr = lerr if lerr is not None else self.lerr
        uerr = uerr if uerr is not None else self.uerr
        metadata = metadata if metadata is not None else self.metadata
        metadata = metadata.update(**kwargs)
        return Matrix(X=X, Y=Y, values=values, lerr=lerr, uerr=uerr, metadata=metadata, copy=copy)
