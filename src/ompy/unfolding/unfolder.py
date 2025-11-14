from __future__ import annotations
from collections import Counter
import numpy as np
from ..array import Vector, Matrix, Array
from ..array.ops import zeros_like, pack_into_matrix
from ..accel import jax_available
from abc import ABC, abstractmethod
from typing import Literal, TypeAlias, overload, Self, TYPE_CHECKING, Iterable
from tqdm.autonotebook import tqdm
from ..pipeline.stage import Stage
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

    Unlike most of OMpy's design, this is stateful. It
    was necessary to make it userfriendly.

    Attributes
    ----------
    R: Matrix
        The unsmoothed response matrix
    G: Matrix
        The gaussian smoothing matrix
    """

    def __init__(
        self,
        D_eg: Matrix | None = None,
        G_eg: Matrix | None = None,
        D_ex: Matrix | None = None,
        G_ex: Matrix | None = None,
        efficiency: Vector | None = None,
        detector: Detector | None = None,
        space: Space = "mu",
        warn_int_data: bool = True,
    ):
        self._D_eg: Matrix | None = D_eg
        self._D_ex: Matrix | None = D_ex
        self._G_eg: Matrix | None = G_eg
        self._G_ex: Matrix | None = G_ex
        self._detector: Detector | None = detector
        self.cached_array_hash: int | None = None
        self.space: Space = space
        self.warn_int_data: bool = warn_int_data
        self.efficiency: Vector | None = efficiency
        if space != "mu":
            raise NotImplementedError(f"Space {space} is not implemented")

    @property
    def D_eg(self) -> Matrix:
        if self._D_eg is None:
            raise ValueError("D_eg is not set")
        return self._D_eg

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

    @property
    def D_ex(self) -> Matrix:
        if self._D_ex is None:
            raise ValueError("D is not set")
        return self._D_ex

    def check_matrices(self) -> None:
        # The matrices must satisfy y = G_ex @ mat @ D @ G_eg
        # yeah, you dingus, you need the matrix
        if self._D_eg is not None and self._G_eg is not None:
            try:
                self._D_eg.X_index.is_compatible_with(self._G_eg.Y_index, do_raise=True)
            except Exception as e:
                raise ValueError(
                    "D and G_eg must have compatible axes.\n"
                    f"D.shape: {self._D_eg.shape} ?= {self._G_eg.shape} = G_eg.shape"
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
                self._D_eg is None
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
            matrices_ex, matrices_eg = self._detector.specialize_like(array)
        except ValueError as e:
            raise ValueError(f"Detector {self._detector} does not implement specialize_like() as expected.\n"
                             "It probably doesn't have a discrete component.\n"
                             "Check that you provided the correct detector.") from e
        if not isinstance(matrices_eg, Matrix):
            self._D_eg = matrices_eg.D
            self._G_eg = matrices_eg.G
        else:
            # Identity matrix was dropped
            self._G_eg = matrices_eg
        if not isinstance(matrices_ex, Matrix):
            self._D_ex = matrices_ex.D
            self._G_ex = matrices_ex.G
        else:
            # Identity matrix was dropped
            self._G_ex = matrices_ex
        self.cached_array_hash = self.hash_array(array)

        self.efficiency = self._detector.efficiency_like(array)

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
                self.D_eg.X_index.is_compatible_with(array.Y_index, do_raise=True)
            except Exception as e:
                raise ValueError(
                    "D_eg must be compatible with the array. "
                    f"Got {self.D_eg.shape} and {array.shape}"
                ) from e
        else:
            try:
                self.D_eg.X_index.is_compatible_with(array.X_index, do_raise=True)
            except Exception as e:
                raise ValueError(
                    "D_eg must be compatible with the array. "
                    f"Got {self.D_eg.shape} and {array.shape}"
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
    def from_detector(cls, detector: Detector, **kwargs) -> Self:
        # A bit verbose, but it is vestigal and fits in the
        # pattern established by other classes
        return cls(detector=detector, **kwargs)

    @classmethod
    def from_result(cls, result: Result) -> Self:
        return cls(D_eg=result.D_eg, G_eg=result.G_eg, G_ex=result.G_ex)

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
        result = self._unfold_vector(
            data=data,
            background=background,
            initial=initial_,
            D=self.D_eg,
            G_eg=self.G_eg,
            mask=mask,
            efficiency=self.efficiency,
            **kwargs,
        )
        # Set stage for pipeline compatibility
        result.meta.stage = Stage.UNFOLDED
        return result

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
            case [*rest]:  # Iterable
                if len(background) != len(data):
                    raise ValueError(
                        "`background` must have the same length as `data`."
                    )
                for raw, bg in zip(data, background):
                    self.check_background(raw, bg)

        match mask:
            case None:
                masks = None
            case Vector() | np.ndarray() | str():
                mask1d = make_mask(data[0], mask)
                # The memory is shared
                masks = [mask1d for i in range(len(data))]
            case [*rest]: # Iterable
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

        results = self._unfold_vectors(
            D=self.D_eg,
            G_eg=self.G_eg,
            data=data,
            background=background,
            initial=initials,
            mask=masks,
            efficiency=self.efficiency,
            **kwargs,
        )
        # Set stage for pipeline compatibility
        for result in results:
            result.meta.stage = Stage.UNFOLDED
        return results

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
        result = self._unfold_matrix(
            data=data,
            background=background,
            initial=initial,
            D=self.D_eg,
            G_eg=self.G_eg,
            G_ex=self.G_ex,
            mask=specialized_mask,
            efficiency=self.efficiency,
            **kwargs,
        )
        # Set stage for pipeline compatibility
        result.meta.stage = Stage.UNFOLDED
        return result

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

    def correct_efficiency[T: Matrix | Vector](self, data: T) -> T:
        eff: Vector | None = self.efficiency
        if eff is None:
            raise ValueError("Efficiency is not set.")
        return data / eff

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

        if self._D_eg is not None:
            matrices_info.append(("D_eg Matrix", matrix_info(self._D_eg, "D_eg")))
            if hasattr(self._D_eg, "_repr_html_"):
                d_eg_html = collapse(self._D_eg._repr_html_(), "D_eg Matrix Details")
            else:
                d_eg_html = ""
        else:
            matrices_info.append(("D Matrix", "None"))
            d_eg_html = ""

        if self._G_eg is not None:
            matrices_info.append(("G_eg Matrix", matrix_info(self._G_eg, "G_eg")))
            if hasattr(self._G_eg, "_repr_html_"):
                g_eg_html = collapse(self._G_eg._repr_html_(), "G_eg Matrix Details")
            else:
                g_eg_html = ""
        else:
            matrices_info.append(("G_eg Matrix", "None"))
            g_eg_html = ""

        if self._D_ex is not None:
            matrices_info.append(("D_ex Matrix", matrix_info(self._D_ex, "D_ex")))
            if hasattr(self._D_ex, "_repr_html_"):
                d_ex_html = collapse(self._D_ex._repr_html_(), "D_ex Matrix Details")
            else:
                d_ex_html = ""
        else:
            matrices_info.append(("D_ex Matrix", "None"))
            d_ex_html = ""

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
                {d_eg_html}
                {g_eg_html}
                {d_ex_html}
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
        case "diagonal":
            return diagonal_mask(data)
        case "last nonzero":
            return last_nonzero_matrix(data.values)
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

if jax_available():
    import jax
    import jax.numpy as jnp

    def last_nonzero_matrix(matrix: jax.Array) -> jax.Array:
        """
        Creates a mask for a JAX matrix where, for each row, scanning from the last
        column downwards:
        - Elements are 0 while the original matrix elements are 0.
        - Switches to 1 at and after the first non-zero element.

        Args:
            matrix: A JAX array (matrix).

        Returns:
            A JAX array representing the mask.
        """

        # Reverse each row to process from right to left
        reversed_matrix = jnp.flip(matrix, axis=-1)

        # Calculate a cumulative sum of non-zero indicators.
        # This will be 0 until the first non-zero element (from the right),
        # then become 1 and stay 1.
        non_zero_indicator = (reversed_matrix != 0).astype(matrix.dtype)
        cumulative_non_zero = jnp.cumsum(non_zero_indicator, axis=-1)

        # The mask is 1 where cumulative_non_zero is greater than 0, and 0 otherwise.
        # This effectively handles the "switches to 1 at and after the first non-zero"
        # condition from the right.
        mask = (cumulative_non_zero > 0).astype(matrix.dtype)

        # Flip the mask back to the original order
        final_mask = jnp.flip(mask, axis=-1)

        return ~final_mask.astype(bool)

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
        case float() | np.floating() | str():
            i = data.index(mask)
            mask = np.zeros_like(data, dtype=bool)
            mask[:i] = True
            return mask
        case _:
            raise ValueError(f"Invalid mask {mask}")


def diagonal_mask(data: Matrix) -> np.ndarray:
    # Create meshgrid of x and y coordinates
    x = data.X
    y = data.Y
    X, Y = np.meshgrid(x, y)
    # Mask where y >= x
    return (Y >= X).T