from __future__ import annotations
from collections import Counter
import numpy as np
from .. import Vector, Matrix, zeros_like, JAX_AVAILABLE
from ..array import pack_into_matrix, Array
from abc import ABC, abstractmethod
from typing import Literal, TypeAlias, overload, Self, TYPE_CHECKING, Iterable
from tqdm.autonotebook import tqdm
from .result import Parameters2D, ResultMeta2D, Result
from .result1d import UnfoldedResult1D
from .result2d import UnfoldedResult2D, UnfoldedResult2DSimple
from .stubs import Space, Mask, Mask1D, Mask2D
from ..rendering.html import collapse, table
import warnings

if TYPE_CHECKING:
    from ..detector import Detector


UNFOLDER_CLASSES: dict[str, type[Unfolder]] = {}


class Unfolder(ABC):
    """Abstract base class for unfolding algorithms

    Attributes
    ----------
    R: Matrix
        The unsmoothed response matrix
    G: Matrix
        The gaussian smoothing matrix
    """

    def __init__(
        self,
        D: Matrix | None = None,
        G_eg: Matrix | None = None,
        G_ex: Matrix | None = None,
        detector: Detector | None = None,
        space: Space = "mu",
        warn_int_data: bool = True,
    ):
        # We must transpose D and G_eg because of convention
        # Better to it here than expecting the user to remember it.
        self._D: Matrix | None = D
        self._G_eg: Matrix | None = G_eg
        self._G_ex: Matrix | None = G_ex
        self._detector: Detector | None = detector
        self.cached_array_hash: int | None = None
        self.space: Space = space
        self.warn_int_data: bool = warn_int_data
        if space != "mu":
            raise NotImplementedError(f"Space {space} is not implemented")

    @property
    def D(self) -> Matrix:
        if self._D is None:
            raise ValueError("D is not set")
        return self._D

    @property
    def G_eg(self) -> Matrix:
        if self._G_eg is None:
            raise ValueError("G_eg is not set")
        return self._G_eg

    @property
    def G_ex(self) -> Matrix:
        if self._G_ex is None:
            raise ValueError("G_ex is not set")
        return self._G_ex

    def check_matrices(self) -> None:
        # The matrices must satisfy y = G_ex @ mat @ D @ G_eg
        # yeah, you dingus, you need the matrix
        if self._D is not None and self._G_eg is not None:
            try:
                self._D.X_index.is_compatible_with(self._G_eg.Y_index, do_raise=True)
            except Exception as e:
                raise ValueError(
                    "D and G_eg must have compatible axes.\n"
                    f"D.shape: {self._D.shape} ?= {self._G_eg.shape} = G_eg.shape"
                ) from e

    def set_matrices(self, array: Matrix | Vector, reset: bool = False) -> None:
        if self.warn_int_data:
            if np.issubdtype(array.values.dtype, np.integer):
                warnings.warn(
                    "You are providing integer data to an unfolding algorithm. "
                    "This is not recommended because it may lead to unexpected "
                    "behavior. Recommended to use float data."
                )

        # Either a detector is set, or all matrices are set
        if self._detector is None:
            need_Gex = isinstance(array, Matrix)
            if (
                self._D is None
                or self._G_eg is None
                or (need_Gex and self._G_ex is None)
            ):
                raise ValueError(
                    "You must either provide all matrices at initialization, or "
                    "use `from_detector(detector)` to set them from a detector."
                )
            # Detector is not set but matrices are set. Ensure they are compatible
            self.check_array(array)
            self.cached_array = array
            return

        # Detector is set. Need to check if the matrices can be reused
        if not reset and self.cached_array_hash is not None:
            if self.cached_array_hash == self.hash_array(array):
                return

        # If we get here, we need to specialize the matrices
        try:
            G_ex, (D, G_eg) = self._detector.specialize_like(array)
        except ValueError as e:
            raise ValueError(f"Detector {self._detector} does not implement specialize_like() as expected.\n"
                             "It probably doesn't have a discrete component.\n"
                             "Check that you provided the correct detector.") from e
        self._D = D
        self._G_eg = G_eg
        self._G_ex = G_ex
        self.cached_array_hash = self.hash_array(array)

    @staticmethod
    def hash_array(array: Matrix | Vector) -> int:
        # We don't care about the values, only the shape and index
        match array:
            case Matrix():
                return hash((array.shape, array.X_index, array.Y_index))
            case Vector():
                return hash((array.shape, array.X_index))
            case _:
                raise ValueError(f"Invalid array type: {type(array)}")

    def check_array(self, array: Matrix | Vector) -> None:
        self.check_matrices()
        # If array is a vector, G_ex is 1, so we can ignore it
        if isinstance(array, Matrix):
            if self._G_ex is None:
                raise ValueError("When unfolding a matrix, G_ex must be provided.")
            try: 
                self.G_ex.X_index.is_compatible_with(array.X_index, do_raise=True)
            except Exception as e:
                raise ValueError(
                    "G_ex must be compatible with the array. "
                    f"Got {self.G_ex.shape} and {array.shape}"
                ) from e
        try:
            self.D.X_index.is_compatible_with(array.Y_index, do_raise=True)
        except Exception as e:
            raise ValueError(
                "D must be compatible with the array. "
                f"Got {self.D.shape} and {array.shape}"
            ) from e
        # We don't need to check G_eg since check_matrices() already did that
        # we only need to ensure it exists
        if self._G_eg is None:
            raise ValueError("G_eg must be provided.")

    def check_background(
        self, data: Matrix | Vector, background: tuple[Matrix, ...] | tuple[Vector, ...] = ()
    ) -> None:
        return
        if background:
            if not self.supports_background():
                raise ValueError(
                    "This unfolding algorithm does not support background subtraction."
                )
            for i, bg in enumerate(background):
                if not bg.is_compatible_with(data):
                    raise ValueError(f"The background #{i} has different indices from the data.")

    @classmethod
    def from_detector(cls, detector: Detector) -> Self:
        # A bit verbose, but it is vestigal and fits in the
        # pattern established by other classes
        return cls(detector=detector)

    @classmethod
    def from_result(cls, result: Result) -> Self:
        return cls(D=result.D, G_eg=result.G_eg, G_ex=result.G_ex)

    @classmethod
    def from_result_constructor(cls, result: Result) -> Self:
        return cls.resolve_method(result.meta.method).from_result(result)

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

    @overload
    def unfold(
        self, data: Matrix, background: Matrix | None = None, **kwargs
    ) -> UnfoldedResult2D: ...

    @overload
    def unfold(
        self, data: Vector, background: Vector | None = None, **kwargs
    ) -> UnfoldedResult1D: ...

    @overload
    def unfold(
        self,
        data: list[Vector],
        background: list[Vector] | Vector | None = None,
        **kwargs,
    ) -> list[UnfoldedResult1D]: ...

    def unfold(
        self,
        data: Matrix | Vector | list[Vector],
        background: tuple[Matrix, ...] | tuple[Vector, ...] | list[tuple[Vector, ...]] = (),
        mask: Mask = "last nonzero",
        **kwargs,
    ) -> UnfoldedResult2D | UnfoldedResult1D | list[UnfoldedResult1D]:
        match data:
            case Matrix():
                return self.unfold_matrix(data, background, mask=mask, **kwargs)
            case Vector():
                return self.unfold_vector(data, background, mask=mask, **kwargs)
            case list():
                return self.unfold_vectors(data, background, mask=mask, **kwargs)
            case _:
                raise ValueError(
                    f"Expected both Matrix, Vector or list of Vectors, got {type(data), type(background)}"
                )

    def unfold_vector(
        self,
        data: Vector,
        background: tuple[Vector, ...] = (),
        initial: InitialVector = "raw",
        mask: Mask1D = "last nonzero",
        **kwargs,
    ) -> UnfoldedResult1D:
        self.set_matrices(data)
        #self.check_background(data, background)

        initial_: Vector = initial_vector(data, initial)
        mask: np.ndarray = make_mask(data, mask)
        return self._unfold_vector(
            data=data,
            background=background,
            initial=initial_,
            D=self.D,
            G_eg=self.G_eg,
            mask=mask,
            **kwargs,
        )

    def unfold_vectors(
        self,
        data: list[Vector],
        background: list[tuple[Vector, ...]] = (),
        initial: InitialVector | list[InitialVector] = "raw",
        mask: Mask1D | list[Mask1D] = "last nonzero",
        **kwargs,
    ) -> list[UnfoldedResult1D]:
        # All vectors must be the same shape
        if len(data) <= 1:
            raise ValueError(
                "At least two vectors are required. Use unfold_vector for single vector."
            )
        c = Counter([len(v) for v in data])
        if len(c) != 1:
            raise ValueError(
                "All vectors must have the same length." f"Got lengths: {c}"
            )
        self.set_matrices(data[0])
        match background:
            case None:
                pass
            case Vector():
                self.check_background(data[0], background)
                background = [background for i in range(len(data))]
            case Iterable():
                if len(background) != len(data):
                    raise ValueError(
                        "`background` must have the same length as `data`."
                    )
                for raw, bg in zip(data, background):
                    self.check_background(raw, bg)

        match mask:
            case Vector() | np.ndarray() | str():
                mask1d = make_mask(data[0], mask)
                # The memory is shared
                masks = [mask1d for i in range(len(data))]
            case Iterable():
                if len(mask) != len(data):
                    raise ValueError("`mask` must have the same length as `data`.")
                # could share memory here, but i'm lazy
                masks = [make_mask(data[i], mask[i]) for i in range(len(data))]

        if isinstance(initial, list):
            initials: list[Vector] = [
                initial_vector(data[i], initial[i]) for i in range(len(data))
            ]
        else:
            initials = [initial_vector(data[i], initial) for i in range(len(data))]

        return self._unfold_vectors(
            D=self.D,
            G_eg=self.G_eg,
            data=data,
            background=background,
            initial=initials,
            mask=masks,
            **kwargs,
        )

    def unfold_matrix(
        self,
        data: Matrix,
        background: tuple[Matrix, ...] = (),
        initial: InitialMatrix = "raw",
        mask: Mask2D = "last nonzero",
        **kwargs,
    ) -> UnfoldedResult2D:
        self.set_matrices(data)
        self.check_background(data, background)
        use_previous, initial = initial_matrix(data, initial)
        specialized_mask: np.ndarray = make_mask(data, mask)
        return self._unfold_matrix(
            data=data,
            background=background,
            initial=initial,
            D=self.D,
            G_eg=self.G_eg,
            G_ex=self.G_ex,
            mask=specialized_mask,
            **kwargs,
        )

    @abstractmethod
    def _unfold_vector(
        self,
        data: Vector,
        background: Vector | None,
        initial: Vector,
        D: Matrix,
        G_eg: Matrix,
        **kwargs,
    ) -> UnfoldedResult1D: ...

    def _unfold_vectors(
        self,
        data: list[Vector],
        background: list[Vector] | None,
        initial: list[Vector],
        D: Matrix,
        G_eg: Matrix,
        mask: list[np.ndarray],
        **kwargs,
    ) -> list[UnfoldedResult1D]:
        """
        A default implementation of unfolding a list of vectors

        Packs the vectors into a matrix and calls unfold_matrix()
        """
        warnings.warn(
            "This is the fallback method for `unfold_vectors`."
            " Depending on the implementation of `unfold_matrix`, "
            " there might be scaling or row correlation effects, "
            " giving different results compared to `unfold_vector`.\n"
            "USE WITH CARE!"
        )
        mat = pack_into_matrix(data)
        bg = None if background is None else pack_into_matrix(background)
        init = pack_into_matrix(initial)
        result = self.unfold_matrix(mat, bg, init, R=(space, R.T), G=G, **kwargs)
        return result

    def _unfold_matrix(
        self,
        data: Matrix,
        G_eg: Matrix,
        G_ex: Matrix,
        background: Matrix | None,
        initial: Matrix,
        use_previous: bool,
        **kwargs,
    ) -> UnfoldedResult2DSimple:
        """A default, simple implementation of unfolding a matrix"""
        # This used to be implemented, but as i've learned more about unfolding,
        # I've realized that it is not a good idea to have a default implementation.
        # The user *must* provide a mask to mask out the region of interest.
        raise NotImplementedError("This is not implemented")
        best = np.zeros((data.shape[0], R.shape[1]))
        N = data.shape[0]
        time = np.zeros(N)
        bins = np.zeros(N)
        # masks = np.zeros_like(data)
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
                init = best[i - 1, :j]
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
        parameters = Parameters2D(
            raw=data, background=background, R=R, G=G, initial=initial
        )
        meta = ResultMeta2D(
            time=time, space=space, parameters=parameters, method=res.meta.method
        )
        best = data.clone(values=best)
        return UnfoldedResult2DSimple(meta=meta, u=best)

    def _repr_html_(self) -> str:
        """
        Generate HTML representation for Jupyter notebook display.
        Provides information about matrices, detector, and space.
        Uses existing table() and collapse() functions.
        """

        # Helper function for matrix info
        def matrix_info(matrix, name):
            if matrix is None:
                return f"{name}: None"
            return f"{name}: {matrix.__class__.__name__} of shape {matrix.shape}"

        # Create the main info table data
        info_data = [
            ("Space", self.space),
            (
                "Cached Array Hash",
                (
                    self.cached_array_hash
                    if self.cached_array_hash is not None
                    else "None"
                ),
            ),
        ]

        # Create matrices info
        matrices_info = []

        if self._D is not None:
            matrices_info.append(("D Matrix", matrix_info(self._D, "D")))
            if hasattr(self._D, "_repr_html_"):
                d_html = collapse(self._D._repr_html_(), "D Matrix Details")
            else:
                d_html = ""
        else:
            matrices_info.append(("D Matrix", "None"))
            d_html = ""

        if self._G_eg is not None:
            matrices_info.append(("G_eg Matrix", matrix_info(self._G_eg, "G_eg")))
            if hasattr(self._G_eg, "_repr_html_"):
                g_eg_html = collapse(self._G_eg._repr_html_(), "G_eg Matrix Details")
            else:
                g_eg_html = ""
        else:
            matrices_info.append(("G_eg Matrix", "None"))
            g_eg_html = ""

        if self._G_ex is not None:
            matrices_info.append(("G_ex Matrix", matrix_info(self._G_ex, "G_ex")))
            if hasattr(self._G_ex, "_repr_html_"):
                g_ex_html = collapse(self._G_ex._repr_html_(), "G_ex Matrix Details")
            else:
                g_ex_html = ""
        else:
            matrices_info.append(("G_ex Matrix", "None"))
            g_ex_html = ""

        # Create detector info
        detector_info = "None"
        detector_html = ""
        if self._detector is not None:
            detector_info = f"{self._detector.__class__.__name__}"
            matrices_info.append(("Detector", detector_info))
            if hasattr(self._detector, "_repr_html_"):
                detector_html = collapse(
                    self._detector._repr_html_(), "Detector Details"
                )
        else:
            matrices_info.append(("Detector", "None"))

        # Build the full HTML output
        html = f"""
        <div class="matrix-container" style="margin: 10px 0; font-family: sans-serif;">
            <div class="main-info">
                <h3>{self.__class__.__name__} Information</h3>
                {table(info_data, color="#e6f7ff")}
            </div>
            
            <div class="matrices-info" style="margin-top: 15px;">
                <h3>Matrices and Detector</h3>
                {table(matrices_info, color="#e6fffa")}
            </div>
            
            <div class="details-section" style="margin-top: 15px;">
                {d_html}
                {g_eg_html}
                {g_ex_html}
                {detector_html}
            </div>
        </div>
        """

        return html


InitialVector: TypeAlias = Literal["raw", "random"] | float | np.ndarray | Vector
InitialMatrix: TypeAlias = Literal["raw", "random"] | float | np.ndarray | Matrix


def initial_vector(data: Vector, initial: InitialVector) -> Vector:
    match initial:
        case str():
            match initial:
                case "raw":
                    return data.copy()
                case "random":
                    return data.copy(
                        values=np.random.poisson(np.median(data.values), len(data))
                    )
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
        case "raw":
            return False, data.copy()
        case float():
            return False, zeros_like(data) + initial
        case np.ndarray():
            return False, data.copy(values=initial)
        case Matrix():
            return False, initial.copy()
        case "random":
            return False, data.copy(
                values=np.random.poisson(np.median(data.values), data.shape)
            )
        case "previous":
            return True, data.copy()
        case _:
            raise ValueError(f"Invalid initial value {initial}")


def mask_511(data: Vector) -> np.ndarray:
    mask = np.ones_like(data.values, dtype=bool)
    eps = 50
    start = 510 - eps
    stop = 510 + eps
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


def make_mask(data: Matrix | Vector, mask: Mask) -> np.ndarray:
    if isinstance(data, Matrix):
        return make_mask_matrix(data, mask)
    else:
        return make_mask_vector(data, mask)


def make_mask_matrix(data: Matrix, mask: Mask2D) -> np.ndarray:
    match mask:
        case np.ndarray():
            return mask
        case Matrix():
            if not data.is_compatible_with(mask):
                raise ValueError("Mask must be compatible with data")
            return mask.values
        case "tril":
            if data.shape[0] != data.shape[1]:
                raise ValueError("tril mask only works for square matrices")
            return np.tril(np.ones_like(data.values, dtype=bool))
        case "last nonzero":
            return last_nonzero_matrix(data)
        case _:
            raise ValueError(f"Invalid mask {mask}")

def last_nonzero_matrix(data: Matrix) -> np.ndarray:
    # Number of columns
    n_cols = data.shape[1]
    # For each element, give its column index if non-zero, else –1
    idx = np.where(data != 0, np.arange(n_cols), -1)
    # Find the last non-zero index in each row
    last = idx.max(axis=1)              # shape (n_rows,)
    # Build a row of column indices 0,1,…,n_cols-1
    cols = np.arange(n_cols)            # shape (n_cols,)
    # Broadcast compare: for each row i, cols <= last[i]
    mask = cols <= last[:, None]        # shape (n_rows, n_cols)
    return mask

if JAX_AVAILABLE:
    import jax
    import jax.numpy as jnp
    @jax.jit
    def last_nonzero_matrix(data: jnp.ndarray) -> jnp.ndarray:
        """
        For each row in `data`, returns a boolean mask where columns up to
        (and including) the last non-zero element are True.
        """
        n_cols = data.shape[1]
        # Replace non-zero entries with their column index, zeros → -1
        idx = jnp.where(data != 0, jnp.arange(n_cols), -1)
        # Find last non-zero index per row
        last = jnp.max(idx, axis=1)          # shape (n_rows,)
        # Compare every column index against each row’s last index
        cols = jnp.arange(n_cols)            # shape (n_cols,)
        mask = cols <= last[:, None]         # broadcasts to (n_rows, n_cols)
        return mask

def make_mask_vector(data: Vector, mask: Mask1D) -> np.ndarray:
    match mask:
        case np.ndarray():
            return mask
        case Vector():
            if not data.is_compatible_with(mask):
                raise ValueError("Mask must be compatible with data")
            return mask.values
        case "last nonzero":
            mask = np.zeros_like(data, dtype=bool)
            mask[: data.last_nonzero()] = True
            return mask
        case _:
            raise ValueError(f"Invalid mask {mask}")
