import numpy as np
import matplotlib.pyplot as plt
import warnings
from copy import deepcopy
from uncertainties import unumpy
from pathlib import Path
from typing import Optional, Union, Any, Tuple, List
from scipy.optimize import minimize
from .ensemble import Ensemble
from .matrix import Matrix, Vector
from .decomposition import chisquare_diagonal, nld_gsf_product
from .library import div0
from .action import Action


class Extractor:
    """ Extracts nld and γSF from an Ensemble or a Matrix

    Basically a wrapper around a minimization routine with bookeeping.
    By giving an Ensemble instance and an Action cutting a Matrix into
    the desired shape, nuclear level density (nld) and gamma strength function
    (gsf/γSF) are extracted. The results are exposed in the attributes
    self.nld and self.gsf, as well as saved to disk. The saved results are
    used if filenames match, or can be loaded manually with load().

    The method decompose(matrix, [std]) extracts the nld and gsf from a single
    Matrix. If no std is given, the error function will be the square
    error.

    Attributes:
        ensemble: The Ensemble instance to extract nld and gsf from.
        size: The number of (nld, gsf) pairs to extract. Must be equal to
           or smaller than ensemble.size.
        regenerate: Whether to force extraction from matrices even if
            previous results are found on disk.
        method: The scipy.minimization method to use. Defaults to Powell.
        options: The scipy.minimization options to use.
        nld: The nuclear level densities extracted.
        gsf: The gamma strength functions extracted.
        trapezoid: The Action cutting the matrices of the Ensemble
           into the desired shape where from the nld and gsf will
           be extracted from.
    """
    def __init__(self, ensemble: Optional[Ensemble] = None,
                 trapezoid: Optional[Action] = None):
        self.ensemble = ensemble
        self._path = Path('extraction_ensemble')
        self._path.mkdir()
        self.size = 10 if ensemble is None else ensemble.size
        self.regenerate = False
        self.method = 'Powell'
        self.options = {'disp': True, 'ftol': 1e-3, 'maxfev': None}
        self.nld: List[Vector] = []
        self.gsf: List[Vector] = []
        self.trapezoid = trapezoid

    def __call__(self, ensemble: Optional[Ensemble] = None,
                 trapezoid: Optional[Action] = None):
        return self.extract_from(ensemble, trapezoid)

    def extract_from(self, ensemble: Optional[Ensemble] = None,
                     trapezoid: Optional[Action] = None):
        """ Decompose each first generation matrix in an Ensemble

        Saves the extracted nld and gsf to file, or loads them if
        already generated. Exposes the vectors in the attributes
        self.nld and self.gsf.

        Args:
            ensemble: The ensemble to extract nld and gsf from.
                Can be provided in __init__ instead.
            trapezoid: An Action describing the cut to apply
                to the matrices to obtain the desired region for
                extracting nld and gsf.
        Raises:
            ValueError if no Ensemble instance is provided here
                or earlier.
        """
        if ensemble is not None:
            self.ensemble = ensemble
        elif self.ensemble is None:
            raise ValueError("ensemble must be given")
        if trapezoid is not None:
            self.trapezoid = trapezoid
        elif self.trapezoid is None:
            raise ValueError("A 'trapezoid' cut must be given'")

        assert self.ensemble.size >= self.size, "Ensemble is too small"

        rhos = []
        gsfs = []
        for i in range(self.size):
            rho_path = self.save_path / f'rho_{i}.npy'
            gsf_path = self.save_path / f'gsf_{i}.npy'
            if rho_path.exists() and gsf_path.exists() and not self.regenerate:
                rhos.append(Vector(path=rho_path))
                gsfs.append(Vector(path=gsf_path))
            else:
                rho, gsf = self.step(i)
                rho.save(rho_path)
                gsf.save(gsf_path)
                rhos.append(rho)
                gsfs.append(gsf)

        self.nld = rhos
        self.gsf = gsfs

    def step(self, num: int) -> Tuple[Vector, Vector]:
        """ Wrapper around _extract in order to be consistent with other classes
        """
        return self._extract(num)

    def _extract(self, num: int) -> Tuple[Vector, Vector]:
        """ Extract nld and gsf from matrix number i from Ensemble

        Returns:
            The nld and gsf as Vectors
        """
        assert self.ensemble is not None
        assert self.trapezoid is not None
        matrix = self.ensemble.get_firstgen(num)
        std = deepcopy(self.ensemble.std_firstgen)
        self.trapezoid.act_on(matrix)
        self.trapezoid.act_on(std)
        nld, gsf = self.decompose(matrix, std)
        return nld, gsf
        # fit = FitRhoT(matrix, std, self.bin_width,
        # Ex_min, Ex_max, Eg_min, self.method, self.options)
        # fit.fit()
        # gsf = fit.T.values / (2*np.pi*(fit.T.E)**3)
        # return fit.rho, Vector(gsf, fit.T.E)

    def decompose(self, matrix: Matrix,
                  std: Optional[Matrix] = None,
                  x0: Optional[np.ndarray] = None,
                  product: bool = False) -> Tuple[Vector, Vector]:
        """ Decomposes a matrix into nld and γSF

        Algorithm:
            Creates the energy range for nld based on the diagonal
            energy resolution. Tries to minimize the product
                firstgen = nld·gsf
            using square error or chi square as error function.
        Args:
            matrix: The matrix to decompose. Should already
                be cut into appropiate size
            std: The standard deviation for the matrix. Must
                be the same size
            x0: The initial guess for nld and gsf.
            product: Whether to return the first generation matrix
               resulting from the product of nld and gsf.
        Returns:
            The nuclear level density, the gamma strength function,
            the energy range for nld and the energy range for γSF.

        TODO: Fix proportionality factor
        """
        # Why normalize?
        # matrix.values, std = normalize(matrix.values, std)
        if std is not None:
            assert matrix.shape == std.shape
            std.values = std.values.copy(order='C')

        resolution = matrix.diagonal_resolution()
        E_nld = np.linspace(-resolution.max(),
                            matrix.Ex.max()-matrix.Eg.min(),
                            max(matrix.Ex.size, matrix.Eg.size))

        if x0 is None:
            nld0 = np.ones(E_nld.size)
            T0, _ = matrix.projection('Eg')
            x0 = np.append(T0, nld0)
            assert T0.size == matrix.Eg.size

        def errfun(x: np.ndarray) -> float:
            # Add a non-negative constraint
            if np.any(x < 0):
                return 1e20

            gsf = x[:matrix.Eg.size]
            nld = x[matrix.Eg.size:]
            fit = nld_gsf_product(nld, gsf, resolution,
                                  E_nld, matrix.Eg, matrix.Ex)
            # chi = chisquare(matrix.values, fit, std, resolution,
            #                 matrix.Eg, matrix.Ex)
            if std is None:
                return np.sum((matrix.values - fit)**2)
            else:
                chi = chisquare_diagonal(matrix.values, fit, std.values,
                                         resolution, matrix.Eg, matrix.Ex)
                return chi

        res = minimize(errfun, x0=x0, method=self.method, options=self.options)
        gsf = res.x[:matrix.Eg.size]
        nld = res.x[matrix.Eg.size:]

        if product:
            values = nld_gsf_product(nld, gsf, resolution,
                                     E_nld, matrix.Eg, matrix.Ex)
            mat = Matrix(values=values, Ex=matrix.Ex, Eg=matrix.Eg)
            return Vector(nld, E_nld), Vector(gsf, matrix.Eg), mat
        else:
            return Vector(nld, E_nld), Vector(gsf, matrix.Eg)

    def load(self, path: Optional[Union[str, Path]] = None) -> None:
        """ Load already extracted nld and gsf from file

        Args:
            path: The path to the directory containing the
                files.
        """
        if path is not None:
            path = Path(path)
        else:
            path = self.save_path

        if self.nld or self.gsf:
            warnings.warn("Loading nld and gsf into non-empty instance")

        for fname in path.glob("nld[0-9]*.*"):
            self.nld.append(Vector(path=fname))

        for fname in path.glob("gsf[0-9]*.*"):
            self.gsf.append(Vector(path=fname))

        assert len(self.nld) == len(self.gsf), "Corrupt files"
        self.size = len(self.nld)

    def plot(self, ax: Optional[Any] = None, scale: str = 'log',
             **kwargs) -> None:
        if ax is None:
            fig, ax = plt.subplots(1, 2, constrained_layout=True)
        for nld, gsf in zip(self.nld, self.gsf):
            ax[0].plot(nld.E, nld.values, color='k',
                       alpha=1/self.size, **kwargs)
            ax[1].plot(gsf.E, gsf.values, color='k',
                       alpha=1/self.size, **kwargs)

        ax[0].errorbar(nld.E, self.nld_mean(), yerr=self.nld_std(),
                       fmt='o', ms=1)
        ax[1].errorbar(gsf.E, self.gsf_mean(), yerr=self.gsf_std(),
                       fmt='o', ms=1)

        ax[0].set_title("Level density")
        ax[1].set_title("γSF")
        if scale == 'log':
            ax[0].set_yscale("log")
            ax[1].set_yscale("log")
        return fig, ax

    def nld_mean(self) -> np.ndarray:
        return np.mean([nld.values for nld in self.nld], axis=0)

    def gsf_mean(self) -> np.ndarray:
        return np.mean([gsf.values for gsf in self.gsf], axis=0)

    def nld_std(self) -> np.ndarray:
        return np.std([nld.values for nld in self.nld], axis=0)

    def gsf_std(self) -> np.ndarray:
        return np.std([gsf.values for gsf in self.gsf], axis=0)

    @property
    def save_path(self) -> Path:
        return self._path

    @save_path.setter
    def path(self, path: Union[str, Path]) -> None:
        if isinstance(path, str):
            self._path = Path(path)
        elif isinstance(path, Path):
            self._path = path
        else:
            raise TypeError(f"path must be str or Path, got {type(path)}")


def normalize(values, std=0):
    matrix = unumpy.uarray(values, std)

    # normalize each Ex row to 1 (-> get decay probability)
    for i, total in enumerate(matrix.sum(axis=1)):
        matrix[i, :] = div0(matrix[i, :], total)
    values = unumpy.nominal_values(matrix)
    std = unumpy.std_devs(matrix)

    return values, std

