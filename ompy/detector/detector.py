from __future__ import annotations
from abc import ABC, abstractmethod, ABCMeta
from ..stubs import Unitlike, Axes
from ..stubs import array as Array
from ..library import from_unit, into_unit
from .. import u, Vector, Matrix, empty, Index
from ..response import DiscreteInterpolation, Response, ResponseMatrices, Components
import numpy as np
import matplotlib.pyplot as plt
from typing import overload, Literal, Callable, TypeAlias, Self, Type
import warnings
from functools import partial
from ..numbalib import njit
from .. import JAX_WORKING, JAX_AVAILABLE
from ..array.ufunc import eye

FWHM_TO_SIGMA = 1 / (2 * np.sqrt(2 * np.log(2)))
Backend: TypeAlias = Literal["numpy", "jax"]
BACKEND: Backend = "jax" if JAX_WORKING else "numpy"

"""
TODO:
    - [ ] Make point-and-click calibrator
    - [ ] Fix the _, __ naming
"""

# We need to map the integer types to the float types
# because integer array inputs give zero as output
DTYPE_MAP = {
    np.int8: np.float32,
    np.int16: np.float32,
    np.int32: np.float32,
    np.int64: np.float64,
    np.uint8: np.float32,
    np.uint16: np.float32,
    np.uint32: np.float32,
    np.uint64: np.float64,
}
DTYPES_MAP = {np.dtype(k): np.dtype(v) for k, v in DTYPE_MAP.items()}


def dtype_map(dtype: np.dtype) -> np.dtype:
    try:
        # .get() doesn't work
        return DTYPES_MAP[dtype]
    except KeyError:
        return dtype


class DetectorMeta(type):
    """Check that the subclass implements the methods correctly.

    When `cls.implements_discrete_response()` is `True`, the subclass must
    override `_discrete_response()`.

    """

    def __new__(cls, name, bases, class_dict):
        new_class = super().__new__(cls, name, bases, class_dict)

        # Check if the class implements the method and requires overriding
        if (
            name != "Detector"
            and hasattr(new_class, "implements_discrete_response")
            and new_class.implements_discrete_response()
            and new_class._discrete_response is Detector._discrete_response
        ):
            raise TypeError(
                f"{name} must override '_discrete_response' since 'implements_discrete_response' returns True"
            )

        return new_class


class CombinedMeta(ABCMeta, DetectorMeta):
    pass


class Detector(ABC, metaclass=CombinedMeta):
    def __init__(self, title: str = ""):
        self.title = title

    def resolution_sigma(self, energy: Unitlike) -> Unitlike:
        fwhm = self.FWHM(energy)
        return fwhm * FWHM_TO_SIGMA

    def _resolution_sigma(self, energy: float) -> float:
        fwhm = self._FWHM(energy)
        return fwhm * FWHM_TO_SIGMA

    def __resolution_sigma(self, energy: float) -> float:
        fwhm = self.__FWHM(energy)
        return fwhm * FWHM_TO_SIGMA

    def sigma(self, energy: Unitlike) -> Unitlike:
        return self.resolution_sigma(energy)

    def _sigma(self, energy: float) -> float:
        return self._resolution_sigma(energy)

    def FWHM(self, energy: Unitlike) -> Unitlike:
        e = from_unit(energy, "keV")
        e = np.atleast_1d(e)
        # We have to run a check to ensure _FWHM preserves the shape of the input
        fwhm = np.atleast_1d(self._FWHM(e))
        if len(fwhm) < len(e):
            # If the shape didn't match, so _FWHM probably returned a scalar.
            # We need to vectorize it to ensure it matches the shape of the input
            fwhm = np.vectorize(self._FWHM)(e)
        return fwhm * u("keV")

    def __FWHM(self, energy: np.ndarray) -> np.ndarray:
        return np.asarray([self._FWHM(e) for e in energy])

    @abstractmethod
    def _FWHM(self, energy: float) -> float: ...

    def resolution_e(self, e: Unitlike, sigma: float = 2) -> Unitlike:
        """Find the index of the diagonal + resolution of sigma"""
        res = self.resolution_sigma(e)
        Ex = into_unit(e, "keV") + sigma * res
        return Ex

    @overload
    def resolution_gauss(
        self,
        E: Index | Array | Vector,
        mu: Unitlike,
        as_array: Literal[False] = ...,
        normalize: bool = ...,
    ) -> Vector: ...

    @overload
    def resolution_gauss(
        self,
        E: Index | Array | Vector,
        mu: Unitlike,
        as_array: Literal[True] = ...,
        normalize: bool = ...,
    ) -> Array: ...

    def resolution_gauss(
        self,
        E: Index | Array | Vector,
        mu: Unitlike,
        as_array: bool = False,
        normalize: bool = True,
    ) -> Array | Vector:
        _E = E
        if isinstance(E, Vector):
            E = E.to("keV").E_true
        elif isinstance(E, Index):
            E = E.to_unit("keV").bins
        elif isinstance(E, Matrix):
            raise ValueError(
                "Matrix input not supported as it is ambiguous whether the energy is along the rows or columns.\n"
                "Specify an axis to resolve the ambiguity, e.g. `matrix.X` or `matrix.Y`."
            )
        Eg = from_unit(mu, "keV")
        sigma = self._resolution_sigma(Eg)
        gauss = ngaussian(E, Eg, sigma)
        if normalize:
            gauss = gauss / gauss.sum()
        if as_array:
            return gauss
        return Vector(E=_E, values=gauss)

    @overload
    @abstractmethod
    def resolution_matrix(
        self,
        array: Vector | Matrix,
        *,
        as_array: Literal[False],
        backend: Backend | None = None,
    ) -> Matrix: ...

    @overload
    def resolution_matrix(
        self,
        array: Vector | Matrix,
        *,
        as_array: Literal[True],
        backend: Backend | None = None,
    ) -> np.ndarray | jnp.ndarray: ...

    def resolution_matrix(
        self,
        array: Vector | Matrix,
        *,
        as_array: bool = False,
        backend: Backend | None = None,
    ) -> Matrix | np.ndarray | jnp.ndarray:
        if backend is None:
            backend = BACKEND

        E = self._get_energy_axis(array)
        E_ = E.to_unit("keV").bins

        if backend == "numpy" or backend == "jax":
            R = self._resolution_matrix_numpy(E, E_)
        elif False:  # backend == 'jax':
            R = self._resolution_matrix_jax(E, E_)
        else:
            raise ValueError(f"Unsupported backend: {backend}")

        if as_array:
            return R

        matrix = self._matrix_from_array(R, E, dtype=dtype_map(array.dtype))
        # Set title if the subclass didn't set it
        if not matrix.title:
            matrix.title = f"{self.title} resolution"
        return matrix

    @abstractmethod
    def _get_energy_axis(self, array: Vector | Matrix) -> Vector: ...

    @abstractmethod
    def _matrix_from_array(self, R: np.ndarray | jnp.ndarray, E: Vector) -> Matrix: ...

    def _resolution_matrix_numpy(self, E, E_):
        R = np.zeros((len(E_), len(E)))
        for i, e in enumerate(E_):
            R[i, :] = self.resolution_gauss(E, e, as_array=True)
        return R

    def _resolution_matrix_jax(self, E, E_):
        if not JAX_WORKING:
            raise RuntimeError("JAX is not available")
        raise NotImplementedError("JAX resolution matrix not implemented")
        # The convolved matrix works, but it has numerical artifacts when
        # the convolution is too coarse. Since the correction is so
        # small, we can just use the non-convolved matrix.

        sigma = jnp.array([self._resolution_sigma(e) for e in E_]).squeeze()
        # return create_convolved_gaussian_matrix(jnp.array(E_), sigma)
        return create_gaussian_matrix(jnp.array(E_), sigma)

    def resolution_like(self, array: Vector | Matrix) -> Vector | Matrix:
        return self.resolution_matrix(array)

    def plot_FWHM(
        self, ax: Axes | None = None, start=0, stop="10MeV", n=100, **kwargs
    ) -> Axes:
        if ax is None:
            ax = plt.subplots()[1]
        assert ax is not None
        start = from_unit(start, "keV")
        stop = from_unit(stop, "keV")
        e = np.linspace(start, stop, n)
        ax.plot(e, self.__FWHM(e), **kwargs)
        ax.set_xlabel(r"Gamma energy $E_\gamma$[keV]")
        ax.set_ylabel("FWHM [keV]")
        ax.set_title("FWHM of {}".format(self.__class__.__name__))
        ax2 = ax.twinx()
        ax2.set_ylabel(r"$\sigma$ [keV]")
        ax2.plot(e, self.__resolution_sigma(e), **kwargs)

        return ax

    def discrete_like(self, E: Index | Vector | Matrix | Array, **kwargs) -> Matrix:
        """Create a discrete response matrix for the given energy axis.


        This is a wrapper around _matrix_from_array() that subclasses implement.

        Args:
            E: The energy axis to create the response matrix for. Can be:
                - Index: An energy calibration
                - Vector: A 1D array with energy calibration
                - Matrix: A 2D array with energy calibration. Picks the energy axis
                that the detector operates on.
                - Array: A numpy array of energies

        Returns:
            Matrix: The discrete response matrix
        """
        dtype = None
        match E:
            case Vector() | Matrix():
                dtype = E.dtype
                E = self._get_energy_axis(E)
            case Index():
                # Index passes through the dtype
                pass
            case np.ndarray():
                dtype = E.dtype
            case _:
                raise ValueError(f"Unsupported type: {type(E)}")
        D = self._discrete_response(E, **kwargs)
        matrix = self._matrix_from_array(D, E, dtype=dtype_map(dtype))
        # Set title if the subclass didn't set it
        if not matrix.title:
            matrix.title = f"{self.title} discrete response"
        return matrix

    def _discrete_response(self, E: Index | Array, **kwargs) -> Matrix:
        """
        Returns a discrete response matrix for the given energy axis.

        By default, returns an identity matrix that represents a perfect detector
        response (no resolution effects). Subclasses may override this behavior by
        implementing their own discrete response matrix.

        Args:
            E: The energy axis to create the response matrix for. Can be:
                - Vector: A 1D array with energy calibration
                - Index: An energy calibration

        Returns:
            Matrix: The discrete response matrix
        """
        return eye(E)

    @staticmethod
    def implements_discrete_response() -> bool:
        return False

    def copy(self, title: str | None = None) -> Self:
        return self.clone(title=title, copy=True)

    def clone(self, title: str | None = None, copy: bool = False) -> Self:
        title = self.title if title is None else title
        return self.__class__(title=title)

    @abstractmethod
    def _get_energy_dim(self, array: Vector | Matrix) -> Literal[0, 1]: ...

    @overload
    def specialize_like(
        self, array: Vector | Matrix, *, drop_eye: Literal[True]
    ) -> Matrix | ResponseMatrices: ...

    @overload
    def specialize_like(
        self, array: Vector | Matrix, *, drop_eye: Literal[False]
    ) -> ResponseMatrices: ...

    def specialize_like(
        self, array: Vector | Matrix, *, drop_eye: bool = True,
        components: Components | None = None
    ) -> Matrix | ResponseMatrices:
        if drop_eye and not self.implements_discrete_response():
            return self.resolution_like(array)
        else:
            return ResponseMatrices(
                D=self.discrete_like(array, components=components), G=self.resolution_like(array)
            )


class EgDetector(Detector):
    def _get_energy_dim(self, array: Vector | Matrix) -> Literal[1]:
        return 1

    def _get_energy_axis(self, array: Vector | Matrix) -> Vector:
        return array.Y_index if array.ndim == 2 else array.X_index

    def _matrix_from_array(
        self, R: np.ndarray | jnp.ndarray, E: Vector, dtype
    ) -> Matrix:
        matrix = Matrix(X=E, Y=E, values=R, dtype=dtype)
        matrix.ylabel = r"Measured $E_\gamma$"
        matrix.xlabel = r"True $E_\gamma$"
        return matrix


class ExDetector(Detector):
    def _get_energy_dim(self, array: Vector | Matrix) -> Literal[0]:
        return 0

    def _get_energy_axis(self, array: Vector | Matrix) -> Vector:
        return array.X_index

    def _matrix_from_array(
        self, R: np.ndarray | jnp.ndarray, E: Vector, dtype
    ) -> Matrix:
        if JAX_WORKING and isinstance(R, jnp.ndarray):
            R = jnp.transpose(R)
        else:
            R = np.transpose(R)
        matrix = Matrix(X=E, Y=E, values=R, dtype=dtype)
        matrix.ylabel = r"Measured $E_{\mathrm{in}}$"
        matrix.xlabel = r"True $E_{\mathrm{in}}$"
        return matrix


class LambdaEgDetector(EgDetector):
    def __init__(self, func: Callable[[float], float], title=""):
        super().__init__(title)
        self.func = func

    def _FWHM(self, e: float) -> float:
        return self.func(e)

    def __str__(self) -> str:
        return f"LambdaEgDetector with lambda={self.func}"


class LambdaExDetector(ExDetector):
    def __init__(self, func: Callable[[float], float], title=""):
        super().__init__(title)
        self.func = func

    def _FWHM(self, e: float) -> float:
        return self.func(e)

    def __str__(self) -> str:
        return f"LambdaExDetector with lambda={self.func}"


class CompoundDetector:
    def __init__(self, eg_detector: EgDetector, ex_detector: ExDetector):
        self.eg_detector = eg_detector
        self.ex_detector = ex_detector

    @overload
    def cut_at_resolution_sharp(
        self, mat: Matrix, *, eg_sigma: ..., ex_sigma: ..., inplace: Literal[False]
    ) -> Matrix: ...

    @overload
    def cut_at_resolution_sharp(
        self, mat: Matrix, *, eg_sigma: ..., ex_sigma: ..., inplace: Literal[True]
    ) -> None: ...

    def cut_at_resolution_sharp(
        self,
        mat: Matrix,
        *,
        eg_sigma: float = 3,
        ex_sigma: float = 3,
        inplace: bool = False,
    ) -> Matrix | None:
        """Return a matrix with the resolution of sigma"""
        mask = np.zeros_like(mat.values, dtype=bool)
        for i in range(mat.shape[0]):
            e_diagonal = mat.Ex[i] * mat.Ex_index.unit
            ex = self.ex_detector.resolution_e(e_diagonal, sigma=-ex_sigma)
            eg = self.eg_detector.resolution_e(e_diagonal, sigma=eg_sigma)
            if not mat.Ex_index.is_inbounds(ex) or not mat.Eg_index.is_inbounds(eg):
                continue
            ex_i = mat.index_Ex(ex)
            eg_i = mat.index_Eg(eg)
            mask[ex_i, eg_i:] = True

        if inplace:
            mat.values[mask] = 0.0
        else:
            matrix = mat.clone()
            matrix.values[mask] = 0.0
            return matrix

    @overload
    def cut_at_resolution(
        self, mat: Matrix, *, eg_sigma: ..., ex_sigma: ..., inplace: Literal[False]
    ) -> Matrix: ...

    @overload
    def cut_at_resolution(
        self, mat: Matrix, *, eg_sigma: ..., ex_sigma: ..., inplace: Literal[True]
    ) -> None: ...

    def cut_at_resolution(
        self,
        mat: Matrix,
        *,
        eg_sigma: float = 3,
        ex_sigma: float = 3,
        inplace: bool = False,
    ) -> Matrix | None:
        """Return a matrix with the resolution of sigma"""
        if inplace:
            values = mat.values
        else:
            values = mat.values.copy()

        sigma_eg = self.eg_detector.resolution_sigma(mat.Eg)
        sigma_ex = self.ex_detector.resolution_sigma(mat.Ex)
        sigma_ex = np.atleast_1d(sigma_ex.to("keV").magnitude)
        sigma_eg = np.atleast_1d(sigma_eg.to("keV").magnitude)
        if len(sigma_ex) == 1:
            sigma_ex = np.full_like(mat.Ex, sigma_ex[0])
        if len(sigma_eg) == 1:
            sigma_eg = np.full_like(mat.Eg, sigma_eg[0])
        # set the dtype to that of the matrix
        sigma_ex = sigma_ex.astype(dtype_map(mat.values.dtype))
        sigma_eg = sigma_eg.astype(dtype_map(mat.values.dtype))
        Ex = mat.Ex.astype(dtype_map(mat.values.dtype))
        Eg = mat.Eg.astype(dtype_map(mat.values.dtype))
        values = cut_at_resolution(values, Ex, Eg, sigma_ex, sigma_eg, ex_sigma, eg_sigma)
        if not inplace:
            return mat.clone(values=values)

    def specialize_like(
        self, array: Vector | Matrix, *, drop_eye: bool = True
    ) -> (
        tuple[ResponseMatrices, ResponseMatrices]
        | tuple[Matrix, Matrix]
        | tuple[Matrix, ResponseMatrices]
        | tuple[ResponseMatrices, Matrix]
    ):
        return self.ex_detector.specialize_like(
            array, drop_eye=drop_eye
        ), self.eg_detector.specialize_like(array, drop_eye=drop_eye)

    def copy(self, **kwargs) -> Self:
        return self.clone(*kwargs, copy=True)

    def clone(
        self,
        eg_detector: EgDetector | None = None,
        ex_detector: ExDetector | None = None,
    ) -> Self:
        eg_detector = eg_detector or self.eg_detector
        ex_detector = ex_detector or self.ex_detector
        return self.__class__(eg_detector=eg_detector, ex_detector=ex_detector)


# @njit
def cut_at_resolution(mat, Ex, Eg, sigma_ex, sigma_eg, ex_sigma, eg_sigma):
    for i in range(mat.shape[0]):
        e_x = Ex[i]
        s_ex = sigma_ex[i]
        for j in range(mat.shape[1]):
            e_g = Eg[j]
            if e_g < e_x:
                continue
            s_eg = sigma_eg[j]
            distance_eg = e_g - e_x
            distance_ex = e_x - e_g  # ?
            factor_ex = ngaussian(distance_ex, 0, ex_sigma * s_ex)
            factor_eg = ngaussian(distance_eg, 0, eg_sigma * s_eg)
            factor = np.sqrt(factor_ex * factor_eg)
            mat[i, j] *= factor


@njit
def ngaussian(x: np.ndarray, mu: float, sigma: float):
    return np.exp(-np.power(x - mu, 2.0) / (2 * np.power(sigma, 2.0)))


if JAX_WORKING:
    import jax
    import jax.numpy as jnp
    from functools import partial

    def gaussian(x, mu, sigma):
        """
        Compute the Gaussian (normal) distribution.

        This function calculates the probability density of a Gaussian distribution
        for given x values, mean (mu), and standard deviation (sigma).

        Args:
            x (array): The input values.
            mu (float): The mean of the distribution.
            sigma (float): The standard deviation of the distribution.

        Returns:
            array: The probability density values for the input x.

        Reasoning:
            The Gaussian distribution is a fundamental probability distribution
            used to model many natural phenomena. In the context of detector
            resolution, it represents the spread of measured values around the
            true energy value.
        """
        return jnp.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * jnp.sqrt(2 * jnp.pi))

    def create_gaussian_matrix(E, E_, sigma):
        R = np.zeros((len(E_), len(E)))
        for i, e in enumerate(E_):
            R[i, :] = self.resolution_gauss(E, e, as_array=True)
        return R

    def uniform(x, mu, dmu):
        """
        Compute a uniform distribution.

        This function calculates the probability density of a uniform distribution
        centered at mu with a half-width of dmu.

        Args:
            x (array): The input values.
            mu (float): The center of the uniform distribution.
            dmu (float): The half-width of the uniform distribution.

        Returns:
            array: The probability density values for the input x.

        Reasoning:
            The uniform distribution represents an equal probability within a
            certain range. In detector modeling, it can represent the bin width
            of the energy channels, assuming equal probability within each bin.
        """
        return jnp.where((x >= mu - dmu) & (x <= mu + dmu), 1 / (2 * dmu), 0)

    @partial(jax.jit, static_argnums=(4,))
    def convolve_gaussian_uniform(x, mu_g, sigma, dmu, num_points):
        """
        Convolve a Gaussian distribution with a uniform distribution.

        This function performs the convolution of a Gaussian (representing detector
        resolution) with a uniform distribution (representing bin width).

        Args:
            x (array): The energy values.
            mu_g (float): The mean of the Gaussian distribution.
            sigma (float): The standard deviation of the Gaussian.
            dmu (float): The half-width of the uniform distribution.
            num_points (int): The number of points to use in the convolution.

        Returns:
            array: The convolved distribution.

        Reasoning:
            Convolution of the Gaussian (detector resolution) with a uniform
            distribution (bin width) more accurately models the detector response.
            This accounts for both the spread due to detector resolution and the
            discretization of energy channels.
        """
        t = jnp.linspace(mu_g - 4 * sigma - dmu, mu_g + 4 * sigma + dmu, num_points)
        dt = t[1] - t[0]

        g = gaussian(t, mu_g, sigma)
        u = uniform(x[:, jnp.newaxis] - t, 0, dmu)

        return jnp.sum(g * u, axis=1) * dt

    def optimal_num_points(sigma, dx):
        """
        Calculate the optimal number of points for convolution.

        This function determines the number of points to use in the convolution
        to ensure accurate results while balancing computational efficiency.

        Args:
            sigma (float or array): The standard deviation(s) of the Gaussian.
            dx (float): The bin width.

        Returns:
            int: The optimal number of points for convolution.

        Reasoning:
            The number of points needs to be large enough to capture the full
            width of the Gaussian and uniform distributions, but not so large
            as to unnecessarily increase computation time. The calculation
            ensures at least 10 points per sigma and a minimum of 2000 points.
        """
        if isinstance(sigma, (np.ndarray, jnp.ndarray)):
            min_sigma = jnp.min(sigma).item()
            max_sigma = jnp.max(sigma).item()
        else:
            min_sigma = sigma
            max_sigma = sigma
        num_points = max(int(10 * (8 * max_sigma + dx) / min(min_sigma, dx)), 5000)
        return num_points

    def create_convolved_gaussian_matrix(x, sigma, num_points=None):
        """
        Create a matrix of convolved Gaussian distributions.

        This function generates a matrix where each row represents the convolved
        Gaussian distribution for a specific energy, accounting for both detector
        resolution and bin width effects.

        Args:
            x (array): The energy values.
            sigma (array): The standard deviations for each energy value.
            num_points (int, optional): The number of points to use in convolution.

        Returns:
            array: A matrix where each row is a convolved distribution.

        Reasoning:
            This matrix represents the detector response function, showing how
            the detector would respond to monoenergetic inputs across its energy
            range. It's crucial for accurate modeling of detector behavior in
            spectroscopy applications.
        """
        n = len(x)
        dx = x[1] - x[0]  # Assume constant bin width
        dmu = dx / 2

        if num_points is None:
            num_points = optimal_num_points(sigma, dx)

        # Calculate convolved Gaussian distributions for each row
        convolved = jax.vmap(
            lambda mu, s: convolve_gaussian_uniform(x, mu, s, dmu, num_points)
        )(x, sigma)

        # Normalize each row
        convolved /= jnp.sum(convolved, axis=1, keepdims=True)

        return np.asarray(convolved)

    @jax.jit
    def cut_at_resolution(mat: jnp.ndarray,
                        Ex: jnp.ndarray,
                        Eg: jnp.ndarray,
                        sigma_ex: jnp.ndarray,
                        sigma_eg: jnp.ndarray,
                        ex_sigma: float,
                        eg_sigma: float) -> jnp.ndarray:
        """
        For each row i and column j of `mat`, if Eg[j] >= Ex[i], scale mat[i,j] by
        sqrt(ngaussian(Ex[i]-Eg[j], 0, ex_sigma*sigma_ex[i]) *
            ngaussian(Eg[j]-Ex[i], 0, eg_sigma*sigma_eg[j]))
        Otherwise leave mat[i,j] unchanged.
        """
        # compute Eg[j] - Ex[i] for all (i,j)
        d_eg = Eg[None, :] - Ex[:, None]         # shape (n_i, n_j)
        mask = d_eg >= 0                         # only these get modified

        # distances for the two gaussians
        d_ex = -d_eg                             # Ex[i] - Eg[j] = -(Eg-Ex)

        # build the σ matrices
        std_ex = (ex_sigma * sigma_ex)[:, None]  # shape (n_i, 1)
        std_eg = (eg_sigma * sigma_eg)[None, :]  # shape (1, n_j)

        # gaussian factors
        gauss_ex = jnp.exp(-d_ex**2 / (2 * std_ex**2))
        gauss_eg = jnp.exp(-d_eg**2 / (2 * std_eg**2))

        # combined factor, defaulting to 1.0 when mask is False
        factor = jnp.where(~mask,
                        jnp.sqrt(gauss_ex * gauss_eg),
                        1.0)

        return mat * factor.T