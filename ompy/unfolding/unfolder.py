from __future__ import annotations
from collections import Counter
import numpy as np
from .. import Vector, Matrix, Response, zeros_like
from ..array import pack_into_matrix, to_index
from ..helpers import maybe_set
from ..response import ResponseName
from abc import ABC, abstractmethod
from typing import Literal, TypeAlias, overload, Self
from tqdm.autonotebook import tqdm
from .result import Parameters2D, ResultMeta2D, Parameters1D, ResultMeta1D
from .result1d import UnfoldedResult1D
from .result2d import UnfoldedResult2D, UnfoldedResult2DSimple
from .stubs import Space
from ..stubs import array1D


UNFOLDER_CLASSES: dict[str, type[Unfolder]] = {}


class Unfolder(ABC):
    """ Abstract base class for unfolding algorithms

    Parameters
    ----------
    R: Matrix
        The unsmoothed response matrix
    G: Matrix
        The gaussian smoothing matrix
    """

    def __init__(self, R: Matrix, G: Matrix):
        if R.shape != G.shape:
            raise ValueError(
                f"R and G must have the same shape, got {R.shape} and {G.shape}")
        if not R.X_index.is_compatible_with(R.Y_index):
            raise ValueError("R must be square")
        if not R.X_index.is_compatible_with(G.X_index):
            raise ValueError(
                f"R and G must have the same axes.\n{R.X_index.summary()}\n{G.X_index.summary()}")
        self.R: Matrix = R
        self.G: Matrix = G


    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        UNFOLDER_CLASSES[cls.__name__] = cls

    @staticmethod
    def resolve_name(name: str) -> type[Unfolder]:
        return UNFOLDER_CLASSES[name]

    @staticmethod
    def resolve_method(unfolder: type[Unfolder] | str) -> type[Unfolder]:
        if isinstance(unfolder, str):
            return Unfolder.resolve_name(unfolder)
        else:
            return unfolder

    @staticmethod
    @abstractmethod
    def supports_background() -> bool: ...

    @classmethod
    def from_response(cls, response: Response, data: Matrix | Vector, **kwargs) -> Self:
        R, G = response.specialize_like(data)
        return cls(R, G, **kwargs)

    @classmethod
    def from_db(cls, db: ResponseName, data: Matrix | Vector, **kwargs) -> Self:
        response = Response.from_db(db)
        return cls.from_response(response, data, **kwargs)

    @overload
    def unfold(self, data: Matrix, background: Matrix | None = None, **kwargs) -> UnfoldedResult2D:
        ...

    @overload
    def unfold(self, data: Vector, background: Vector | None = None, **kwargs) -> UnfoldedResult1D:
        ...

    @overload
    def unfold(self, data: list[Vector], background: list[Vector] | None = None,
               **kwargs) -> list[UnfoldedResult1D]:
        ...

    def unfold(self, data: Matrix | Vector | list[Vector],
               background: Matrix | Vector | list[Vector] | None = None,
               **kwargs) -> UnfoldedResult2D | UnfoldedResult1D | list[UnfoldedResult1D]:
        match data, background:
            case Matrix(), Matrix() | None:
                return self.unfold_matrix(data, background, **kwargs)
            case Vector(), Vector() | None:
                return self.unfold_vector(data, background, **kwargs)
            case list(), list() | None:
                return self.unfold_vectors(data, background, **kwargs)
            case _:
                raise ValueError(
                    f"Expected both Matrix, Vector or list of Vectors, got {type(data), type(background)}")

    def unfold_vector(self, data: Vector, background: Vector | None = None, initial: InitialVector = 'raw', R: str | Matrix = 'R', G: str | Matrix = 'G', **kwargs) -> UnfoldedResult1D:
        space, R = self._resolve_response(R)
        if not R.is_compatible_with(data.X_index):
            raise ValueError("R and data must have the same axes")
        if background is not None:
            if not self.supports_background():
                raise ValueError("This unfolding algorithm does not support background subtraction.")
            if not R.is_compatible_with(background.X_index):
                raise ValueError(
                    "The background has different index from the data.")
        R = R.T
        if G == 'G':
            G = self.G.T
        elif isinstance(G, str):
            raise ValueError(f"Unknown G: {G}")
        else:
            G = G

        initial_: Vector = initial_vector(data, initial)
        return self._unfold_vector(R=R, data=data, background=background,
                                   initial=initial_, G=G, space=space, **kwargs)


    def unfold_vectors(self, data: list[Vector],
                       background: list[Vector] | None = None,
                       initial: InitialVector | list[InitialVector] = 'raw',
                       R: str | Matrix = 'R',
                       G: str | Matrix = 'G', **kwargs) -> list[UnfoldedResult1D]:
        space, R = self._resolve_response(R)
        # All vectors must be the same shape
        if len(data) <= 1:
            raise ValueError("At least two vectors are required. Use unfold_vector for single vector.")
        c = Counter([len(v) for v in data])
        if len(c) != 1:
            raise ValueError("All vectors must have the same length."
                             f"Got lengths: {c}")

        for i, vec in enumerate(data):
            if not R.is_compatible_with(vec.X_index):
                raise ValueError(f"`R` and vector {i} must have compatible axes")
        if background is not None:
            #if not self.supports_background():
            #    raise ValueError("This unfolding algorithm does not support background subtraction.")
            if len(background) != len(data):
                raise ValueError("`background` must have the same length as `data`.")
            for i, vec in enumerate(background):
                if not R.is_compatible_with(vec.X_index):
                    raise ValueError(f"`R` and background vector {i} must have compatible axes.")
        R = R.T
        if G == 'G':
            G = self.G.T
        elif isinstance(G, str):
            raise ValueError(f"Unknown G: {G}")
        else:
            G = G

        if isinstance(initial, list):
            initials: list[Vector] = [initial_vector(data[i], initial[i])
                                      for i in range(len(data))]
        else:
            initials = [initial_vector(data[i], initial) for i in range(len(data))]
        return self._unfold_vectors(R=R, data=data, background=background,
                                   initial=initials, G=G, space=space, **kwargs)

    def unfold_matrix(self, data: Matrix, background: Matrix | None = None,
                      initial: InitialMatrix = 'raw',
                      R: str | Matrix | tuple[str, Matrix] = 'R', G: str | Matrix = 'G',
                      G_ex: Matrix | None = None,
                      **kwargs) -> UnfoldedResult2D:
        space, R = self._resolve_response(R)
        if not self.R.X_index.is_compatible_with(data.Y_index):
            raise ValueError("R and data must have the same axes."
                             f"\n\nThe index of R:\n{R.X_index.summary()}"
                             f"\n\nThe index of data:\n{data.Y_index.summary()}")
        if background is not None:
            if not self.supports_background():
                raise ValueError("This unfolding algorithm does not support background subtraction.")
            if not background.is_compatible_with(data):
                raise ValueError(
                    "The background has different indices from the data.")
        R = R.T

        if G == 'G':
            G = self.G.T
        elif isinstance(G, str):
            raise ValueError(f"Unknown G: {G}")
        else:
            G = G

        if G_ex is None:
            G_ex = Matrix(X=data.X_index, Y=data.X_index, values = np.eye(data.shape[0]))

        use_previous, initial = initial_matrix(data, initial)
        return self._unfold_matrix(R, data, background, initial, use_previous,
                                   space, G, G_ex=G_ex, **kwargs)

    @abstractmethod
    def _unfold_vector(self, R: Matrix, data: Vector, background: Vector | None,
                       initial: Vector, space: Space, G: Matrix | None = None,
                       **kwargs) -> UnfoldedResult1D: ...

    def _unfold_vectors(self, R: Matrix, data: list[Vector], background: list[Vector] | None,
                       initial: list[Vector], space: Space, G: Matrix,
                       **kwargs) -> list[UnfoldedResult1D]:
        """
        A default implementation of unfolding a list of vectors 

        Packs the vectors into a matrix and calls unfold_matrix()
        """
        mat = pack_into_matrix(data)
        bg = None if background is None else pack_into_matrix(background)
        init = pack_into_matrix(initial)
        result = self.unfold_matrix(mat, bg, init, R=(space, R.T), G=G, **kwargs)
        return result


    def _unfold_matrix(self, R: Matrix, data: Matrix,
                       background: Matrix | None, initial: Matrix,
                       use_previous: bool, space: Space,
                       G: Matrix, G_ex: Matrix, **kwargs) -> UnfoldedResult2DSimple:
        """ A default, simple implementation of unfolding a matrix

         """
        best = np.zeros((data.shape[0], R.shape[1]))
        N = data.shape[0]
        time = np.zeros(N)
        bins = np.zeros(N)
        #masks = np.zeros_like(data)
        pbar = tqdm(range(N))
        for i in pbar:
            vec: Vector = data.iloc[i, :]
            # We only want to unfold up to the diagonal + resolution
            j = vec.last_nonzero()
            pbar.set_description(f"Ex = {data.X_index[i] * data.X_index.unit:~} ({j})")
            vec: Vector = vec.iloc[:j]
            if background is not None:
                bvec: Vector | None = background.iloc[i, :j]
            else:
                bvec = None
            if use_previous and i > 0:
                init = best[i-1, :j]
            else:
                init = initial.iloc[i, :j]
            R_: Matrix = R.iloc[:j, :j]
            if G is not None:
                G_ = G.iloc[:j, :j]
            else:
                G_ = None
            res = self._unfold_vector(R_, vec, bvec, init, space=space, G=G_, **kwargs)
            best[i, :j] = res.best()
            time[i] = res.meta.time
            bins[i] = j
        parameters = Parameters2D(raw=data, background=background, R=R, G=G,
                                  initial=initial)
        meta = ResultMeta2D(time=time, space=space, parameters=parameters,
                            method=res.meta.method)
        best = data.clone(values=best)
        return UnfoldedResult2DSimple(meta=meta, u=best)

    def _resolve_response(self, R: str | Matrix | tuple[Space, Matrix]) -> tuple[Space, Matrix]:
        match R:
            case Matrix():
                return 'unknown', R
            case 'R':
                return R, self.R
            case 'G':
                return R, self.G
            case 'GR':
                return R, self.G@self.R
            case 'RG':
                return R, self.R@self.G
            case (Space as space, Matrix() as mat):
                return space, mat
            case _:
                raise ValueError(f"Invalid R {R}")


InitialVector: TypeAlias = Literal['raw', 'random'] | float | np.ndarray | Vector
InitialMatrix: TypeAlias = Literal['raw', 'random'] | float | np.ndarray | Matrix


def initial_vector(data: Vector, initial: InitialVector) -> Vector:
    match initial:
        case str():
            match initial:
                case 'raw':
                    return data.copy()
                case 'random':
                    return data.copy(values=np.random.poisson(np.median(data.values), len(data)))
        case float():
            return data.clone(values=float(initial) + zeros_like(data))
        case np.ndarray():
            return data.clone(values=initial.copy())
        case Vector():
            return initial.copy()
        case _:
            raise ValueError(f"Invalid initial value {initial}")


def initial_matrix(data: Matrix, initial: InitialMatrix) -> tuple[bool, Matrix]:
    match initial:
        case 'raw':
            return False, data.copy()
        case float():
            return False, zeros_like(data) + initial
        case np.ndarray():
            return False, data.copy(values=initial)
        case Matrix():
            return False, initial.copy()
        case 'random':
            return False, data.copy(values=np.random.poisson(np.median(data.values), data.shape))
        case 'previous':
            return True, data.copy()
        case _:
            raise ValueError(f"Invalid initial value {initial}")


def mask_511(data: Vector) -> np.ndarray:
    mask = np.ones_like(data.values, dtype=bool)
    eps = 50
    start = 510-eps
    stop = 510+eps
    if stop < data.X_index.leftmost:
        return mask
    if start > data.X_index.rightmost:
        return mask
    stop = min(stop, data.X_index[-1])
    start = max(start, data.X_index[0])
    start = data.X_index.index(start)
    stop = data.X_index.index(stop)
    mask[start:stop] = False
    return mask


def make_mask(data: Vector, mask) -> Vector:
    match mask:
        case np.ndarray():
            return data.clone(values=mask, dtype=bool)
        case Vector():
            return mask
        case None:
            return data.clone(values=np.ones_like(data.values, dtype=bool),
                              dtype=bool)
        case _:
            return data.clone(values=mask(data), dtype=bool)
