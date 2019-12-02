import numpy as np
import matplotlib.pyplot as plt
import warnings
import logging
from contextlib import redirect_stdout
from uncertainties import unumpy
import os

from pathlib import Path
from typing import Optional, Union, Any, Tuple, List
from scipy.optimize import minimize
from .ensemble import Ensemble
from .matrix import Matrix
from .vector import Vector
from .decomposition import chisquare_diagonal, nld_T_product
from .action import Action

if 'JPY_PARENT_PID' in os.environ:
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm

LOG = logging.getLogger(__name__)


class Extractor:
    """Extracts nld and γSF from an Ensemble or a Matrix

    Basically a wrapper around a minimization routine with bookeeping.
    By giving an `Ensemble` instance and an `Action` cutting a `Matrix` into
    the desired shape, nuclear level density (nld) and gamma strength function
    (gsf/γSF) are extracted. The results are exposed in the attributes
    self.nld and self.gsf, as well as saved to disk. The saved results are
    used if filenames match, or can be loaded manually with `load()`.

    The method `decompose(matrix, [std])` extracts the nld and gsf from a
    single Matrix.

    Attributes:
        ensemble (Ensemble): The Ensemble instance to extract nld and gsf from.
        regenerate (bool): Whether to force extraction from matrices even if
            previous results are found on disk. Defaults to True
        method (str): The scipy.minimization method to use. Defaults to Powell.
        options (dict): The scipy.minimization options to use.
        nld (list[Vector]): The nuclear level densities extracted.
        gsf (list[Vector]): The gamma strength functions extracted.
        trapezoid (Action[Matrix]): The Action cutting the matrices of the
            Ensemble into the desired shape where from the nld and gsf will be
            extracted from.
        path (path): The path to save and/or load nld and gsf to/from.
        extend_diagonal_by_resolution (bool, optional): If `True` (default),
            the fit will be extended beyond Ex=Eg by the (FWHM) of the
            resolution. Remember to set the resolution according to your
            experiment
        resolution_Ex (float or np.ndarray, optional): Resolution (FWHM) along
            Ex axis (particle detector resolution). Defaults to 150 keV


    TODO:
        - If path is given, it tries to load. If path is later set,
          it is not created. This is a very common pattern. Consider
          superclassing the disk book-keeping.
    """
    def __init__(self, ensemble: Optional[Ensemble] = None,
                 trapezoid: Optional[Action] = None,
                 path: Optional[Union[str, Path]] = None):
        """
        ensemble (Ensemble, optional): see above
        trapezoid (Action[Matrix], optional): see above
        path (Path or str, optional): see above
        """
        self.ensemble = ensemble
        self.regenerate = False
        self.method = 'Powell'
        self.options = {'disp': True, 'ftol': 1e-3, 'maxfev': None}
        self.nld: List[Vector] = []
        self.gsf: List[Vector] = []
        self.trapezoid = trapezoid

        if path is not None:
            self.path = Path(path)
            self.path.mkdir(exist_ok=True)
            self.load(self.path)
        else:
            self.path = Path('extraction_ensemble')
            self.path.mkdir(exist_ok=True)

        self.extend_fit_by_resolution: bool = False
        self.resolution_Ex = 150  # keV

    def __call__(self, ensemble: Optional[Ensemble] = None,
                 trapezoid: Optional[Action] = None):
        return self.extract_from(ensemble, trapezoid)

    def extract_from(self, ensemble: Optional[Ensemble] = None,
                     trapezoid: Optional[Action] = None,
                     regenerate: Optional[bool] = None):
        """Decompose each first generation matrix in an Ensemble

        If `regenerate` is `True` it saves the extracted nld and gsf to file,
        or loads them if already generated. Exposes the vectors in the
        attributes self.nld and self.gsf.

        Args:
            ensemble (Ensemble, optional): The ensemble to extract nld and gsf
                from. Can be provided in when initializing instead.
            trapezoid (Action, optional): An Action describing the cut to apply
                to the matrices to obtain the desired region for extracting nld
                and gsf.
            regenerate (bool, optional): Whether to regenerate all nld and gsf
                even if they are found on disk.

        Raises:
            ValueError: If no Ensemble instance is provided here or earlier.
        """
        if ensemble is not None:
            self.ensemble = ensemble
        elif self.ensemble is None:
            raise ValueError("ensemble must be given")
        if trapezoid is not None:
            self.trapezoid = trapezoid
        elif self.trapezoid is None:
            raise ValueError("A 'trapezoid' cut must be given'")
        if regenerate is None:
            regenerate = self.regenerate
        self.path = Path(self.path)  # TODO: Fix

        nlds = []
        gsfs = []
        for i in tqdm(range(self.ensemble.size)):
            nld_path = self.path / f'nld_{i}.npy'
            gsf_path = self.path / f'gsf_{i}.npy'
            if nld_path.exists() and gsf_path.exists() and not regenerate:
                nlds.append(Vector(path=nld_path))
                gsfs.append(Vector(path=gsf_path))
            else:
                nld, gsf = self.step(i)
                nld.save(nld_path)
                gsf.save(gsf_path)
                nlds.append(nld)
                gsfs.append(gsf)

        self.nld = nlds
        self.gsf = gsfs

    def step(self, num: int) -> Tuple[Vector, Vector]:
        """ Wrapper around _extract in order to be consistent with other classes

        Args:
            num: Number of the fg matrix to extract
        """
        nld, gsf = self._extract(num)
        return nld, gsf

    def _extract(self, num: int) -> Tuple[Vector, Vector]:
        """ Extract nld and gsf from matrix number i from Ensemble

        Args:
            num: Number of the fg matrix to extract

        Returns:
            The nld and gsf as Vectors
        """
        assert self.ensemble is not None
        assert self.trapezoid is not None
        matrix = self.ensemble.get_firstgen(num).copy()
        std = self.ensemble.std_firstgen.copy()
        # ensure same vuts for all ensemble members if Eg_max is not given
        # (thus auto-determined) in the trapezoid.
        if num == 0:
            self.trapezoid.act_on(matrix)
            self.trapezoid.act_on(std)
            self.trapezoid.curry(Eg_max=matrix.Eg[-1])
        else:
            self.trapezoid.act_on(matrix)
            self.trapezoid.act_on(std)
        nld, gsf = self.decompose(matrix, std)
        return nld, gsf

    def decompose(self, matrix: Matrix,
                  std: Optional[Matrix] = None,
                  x0: Optional[np.ndarray] = None,
                  product: bool = False) -> Tuple[Vector, Vector]:
        """ Decomposes a matrix into nld and γSF

        Algorithm:
            Creates the energy range for nld based on the diagonal
            energy resolution. Tries to minimize the product::

                firstgen = nld·gsf

            using (weighted) chi square as error function.

            If first nld / last gsf elements cannot be constrained, as there
            are no entries for them in the matrix, they will be set to `np.nan`

        Args:
            matrix: The matrix to decompose. Should already
                be cut into appropiate size
            std: The standard deviation for the matrix. Must
                be the same size as the matrix. If no std is provided,
                square error will be used instead of chi square.
            x0: The initial guess for nld and gsf.
            product: Whether to return the first generation matrix
               resulting from the product of nld and gsf.

        Returns:
            The nuclear level density and the gamma strength function
            as Vectors.
            Optionally returns `nld*γSF` if `product` is `True`

        """
        if np.any(matrix.values < 0):
            raise ValueError("input matrix has to have positive entries only.")
        if std is not None:
            std = std.copy()
            if np.any(std.values < 0):
                raise ValueError("std has to have positive entries only.")
            assert matrix.shape == std.shape, \
                f"matrix.shape: {matrix.shape} != std.shape : {std.shape}"
            std.values = std.values.copy(order='C')
            matrix.values, std.values = normalize(matrix, std)
            matrix.Ex = matrix.Ex.copy(order='C')
            matrix.Eg = matrix.Eg.copy(order='C')
        else:
            matrix.values, _ = normalize(matrix)

        # Eg and Ex *must* have the same step size for the
        # decomposition to make sense.
        dEx = matrix.Ex[1] - matrix.Ex[0]
        dEg = matrix.Eg[1] - matrix.Eg[0]
        assert dEx == dEg, \
            "Ex and Eg must have the same bin width. Currently they have"\
            f"dEx: {dEx:.1f} and dEg: {dEg:.1f}. You have to rebin.\n"\
            "The `ensemble` class has a `rebin` method."
        bin_width = dEx

        # create nld energy array
        Emin = matrix.Ex.max()-matrix.Eg.max()
        Emax = matrix.Ex.max()-matrix.Eg.min()
        E_nld = np.linspace(Emin, Emax, np.ceil((Emax-Emin)/bin_width)+1)

        if self.extend_fit_by_resolution:
            resolution = self.diagonal_resolution(matrix)
        else:
            resolution = np.zeros_like(matrix.Ex)

        if x0 is None:  # default initial guess
            nld0 = np.ones(E_nld.size)
            T0, _ = matrix.projection('Eg')
            x0 = np.append(T0, nld0)
            assert T0.size == matrix.Eg.size
        assert len(x0) == matrix.Eg.size + E_nld.size

        def errfun(x: np.ndarray) -> float:
            # Add a non-negative constraint
            if np.any(x < 0):
                return 1e20

            T = x[:matrix.Eg.size]
            nld = x[matrix.Eg.size:]
            fit = nld_T_product(nld, T, resolution,
                                E_nld, matrix.Eg, matrix.Ex)
            if std is None:
                return np.sum((matrix.values - fit)**2)
            else:
                chi = chisquare_diagonal(matrix.values, fit, std.values,
                                         resolution, matrix.Eg, matrix.Ex)
                return chi

        LOG.info("Minimizing")
        LOG.write = lambda msg: LOG.info(msg) if msg != '\n' else None
        with redirect_stdout(LOG):
            res = minimize(errfun, x0=x0, method=self.method, options=self.options)
        T = res.x[:matrix.Eg.size]
        nld = res.x[matrix.Eg.size:]

        # Set elements that couldn't be constrained (no entries) to np.na
        n_nan_nld, n_nan_gsf = self.unconstrained_elements(matrix, E_nld,
                                                           resolution)
        if n_nan_gsf > 0:
            T[-n_nan_gsf:] = np.nan
        nld[:n_nan_nld] = np.nan
        # Convert transmission coefficient to the more useful
        # gamma strength function
        gsf = T/(2*np.pi*matrix.Eg**3)

        if product:
            nld_0 = np.where(np.isnan(nld), np.zeros_like(nld), nld)
            T_0 = np.where(np.isnan(T), np.zeros_like(T), T)
            values = nld_T_product(nld_0, T_0, resolution,
                                   E_nld, matrix.Eg, matrix.Ex)
            mat = Matrix(values=values, Ex=matrix.Ex, Eg=matrix.Eg)
            return Vector(nld, E_nld), Vector(gsf, matrix.Eg), mat
        else:
            return Vector(nld, E_nld), Vector(gsf, matrix.Eg)

    @staticmethod
    def unconstrained_elements(matrix: Matrix,
                               E_nld: np.ndarray,
                               resolution: np.ndarray) -> Tuple[int, int]:
        """
        Indices of elements close to the diagonal in gsf and nld that
        cannot be constrained, as the bins don't have counts

        Args:
            matrix (Matrix): Input matrix
            E_nld (np.ndarray): Energy array of the nld
            resolution (np.ndarray): Resolution at `Ex=Ex`

        Returns:
            Number of unconstrained elements of nld and gsf
        """
        dEx = matrix.Ex[1] - matrix.Ex[0]
        dEg = matrix.Eg[1] - matrix.Eg[0]
        assert dEx == dEg

        lastEx = matrix[-1, :].copy()

        # account for resolution
        iEgmax = matrix.index_Eg(matrix.Ex[-1] + resolution[-1])
        lastEx[iEgmax+1:] = 0

        n_nan_gsf = (matrix.shape[1]-1) - np.nonzero(lastEx)[0][-1]
        Efirst_nld = matrix.Ex[-1] - matrix.Eg[-(1+n_nan_gsf)]
        n_nan_nld = np.abs(E_nld-Efirst_nld).argmin()
        return n_nan_nld, n_nan_gsf

    def diagonal_resolution(self, matrix: Matrix) -> np.ndarray:
        """Detector resolution at the Ex=Eg diagonal

        Uses gaussian error propagations which assumes independence of
        resolutions along Ex and Eg axis.

        Args:
            matrix (Matrix): Matrix for which the sesoluton shall be calculated

        Returns:
            resolution at Ex = Eg.
        """
        dEx = matrix.Ex[1] - matrix.Ex[0]
        dEg = matrix.Eg[1] - matrix.Eg[0]
        assert dEx == dEg

        dE_resolution = np.sqrt(self.resolution_Ex**2
                                + self.resolution_Eg(matrix)**2)
        return dE_resolution

    @staticmethod
    def resolution_Eg(matrix: Matrix) -> np.ndarray:
        """Resolution along Eg axis for each Ex. Defaults in this class are for OSCAR.

        Args:
            matrix (Matrix): Matrix for which the sesoluton shall be calculated

        Returns:
            resolution
        """
        def fFWHM(E, p):
            return np.sqrt(p[0] + p[1] * E + p[2] * E**2)
        fwhm_pars = np.array([73.2087, 0.50824, 9.62481e-05])
        return fFWHM(matrix.Ex, fwhm_pars)

    def load(self, path: Optional[Union[str, Path]] = None) -> None:
        """ Load already extracted nld and gsf from file

        Args:
            path: The path to the directory containing the
                files.
        """
        if path is not None:
            path = Path(path)
        else:
            path = Path(self.path)  # TODO: Fix pathing
        LOG.debug("Loading from %s", path)

        if not path.exists():
            raise IOError(f"The path {path} does not exist.")
        if self.nld or self.gsf:
            warnings.warn("Loading nld and gsf into non-empty instance")

        for fname in path.glob("nld_[0-9]*.npy"):
            self.nld.append(Vector(path=fname))

        for fname in path.glob("gsf_[0-9]*.npy"):
            self.gsf.append(Vector(path=fname))

        assert len(self.nld) == len(self.gsf), "Corrupt files"

        if not self.nld:
            warnings.warn("Found no files")

        self.size = len(self.nld)

    def plot(self, ax: Optional[Any] = None, scale: str = 'log',
             plot_mean: bool = False,
             color='k', **kwargs) -> None:
        """ Basic visualization of nld and gsf

        Args:
            ax: An axis to plot onto
            scale: Scale to use
            plot_mean: Whether to plot individual samples or mean & std. dev
        """
        if ax is None:
            fig, ax = plt.subplots(1, 2, constrained_layout=True)
        else:
            fig = ax.figure

        if plot_mean:
            ax[0].errorbar(self.nld[0].E, self.nld_mean(), yerr=self.nld_std(),
                           fmt='o', ms=2, lw=1, color=color, **kwargs)
            ax[1].errorbar(self.gsf[0].E, self.gsf_mean(), yerr=self.gsf_std(),
                           fmt='o', ms=2, lw=1, color=color, **kwargs)
        else:
            for nld, gsf in zip(self.nld, self.gsf):
                ax[0].plot(nld.E, nld.values, color=color,
                           alpha=1/len(self.nld), **kwargs)
                ax[1].plot(gsf.E, gsf.values, color=color,
                           alpha=1/len(self.gsf), **kwargs)

        ax[0].set_title("Level density")
        ax[1].set_title("γSF")
        if scale == 'log':
            ax[0].set_yscale("log")
            ax[1].set_yscale("log")

        return fig, ax

    def nld_mean(self) -> np.ndarray:
        return np.nanmean([nld.values for nld in self.nld], axis=0)

    def gsf_mean(self) -> np.ndarray:
        return np.nanmean([gsf.values for gsf in self.gsf], axis=0)

    def nld_std(self) -> np.ndarray:
        return np.nanstd([nld.values for nld in self.nld], axis=0)

    def gsf_std(self) -> np.ndarray:
        return np.nanstd([gsf.values for gsf in self.gsf], axis=0)

    def ensemble_nld(self) -> Vector:
        energy = self.nld[0].E
        values = self.nld_mean()
        std = self.nld_std()
        return Vector(values=values, E=energy, std=std)

    def ensemble_gsf(self) -> Vector:
        energy = self.gsf[0].E
        values = self.gsf_mean()
        std = self.gsf_std()
        return Vector(values=values, E=energy, std=std)


def normalize(mat: Matrix,
              std: Optional[Matrix]) -> Tuple[np.ndarray, np.ndarray]:
    """Matrix normalization per row taking into account the std. dev

    Error propagation assuming gaussian error propagation.

    Args:
        mat (Matrix): input matrix
        std (Matrix, optional): Standard deviation at each bin

    Returns:
        Values of normalized matrix and normalized standard deviation


    """
    matrix = unumpy.uarray(mat.values, std.values if std is not None else None)

    # normalize each Ex row to 1 (-> get decay probability)
    for i, total in enumerate(matrix.sum(axis=1)):
        if total == 0:
            continue
        matrix[i, :] = np.true_divide(matrix[i, :], total)
    values = unumpy.nominal_values(matrix)
    std = unumpy.std_devs(matrix)

    return values, std
