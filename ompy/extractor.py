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
            previous results are found on disk. Defaults to False
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
        x0 (np.ndarray or str): Initial values. See `decompose`.
        randomize_initial_values (bool): Randomize initial values for
            decomposition. Defaults to True.
        seed (int): Random seed for reproducibility of results.
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

        self.x0 = None
        self.randomize_initial_values: bool = True
        self.seed: int = 98743215
        np.random.seed(self.seed)  # seed also in `extract_from`

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
        np.random.seed(self.seed)  # seed also in `__init__`
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
        # following lines might be superfluous now:
        # ensure same cuts for all ensemble members if Eg_max is not given
        # (thus auto-determined) in the trapezoid.
        if num == 0:
            self.trapezoid.act_on(matrix)
            self.trapezoid.curry(Eg_max=matrix.Eg[-1])
            self.trapezoid.act_on(std)
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
            x0: The initial guess for nld and gsf. If `np.ndarray`, ordered as
                (T0, nld0). Otherwise, if `str`, used as the method, see
                `guess_initial_values` (where also defaults are given).
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
        Emin = matrix.Ex.min()-matrix.Eg.max()
        Emax = matrix.Ex.max()-matrix.Eg.min()
        E_nld = np.linspace(Emin, Emax, int(np.ceil((Emax-Emin)/bin_width))+1)

        if self.extend_fit_by_resolution:
            resolution = self.diagonal_resolution(matrix)
        else:
            resolution = np.zeros_like(matrix.Ex)

        x0 = self.x0 if x0 is None else x0
        if x0 is None or isinstance(x0, str):  # default initials or method
            x0 = self.guess_initial_values(E_nld, matrix, x0)
        assert len(x0) == E_nld.size + matrix.Eg.size
        if self.randomize_initial_values:
            x0 = np.random.uniform(x0/5, x0*5)  # arb. choice for bounds

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
            res = minimize(errfun, x0=x0, method=self.method,
                           options=self.options)
        T = res.x[:matrix.Eg.size]
        nld = res.x[matrix.Eg.size:]

        # Set elements that couldn't be constrained (no entries) to np.na
        nld_counts0, T_counts0 = self.constraining_counts(matrix, resolution)
        T[T_counts0 == 0] = np.nan
        nld[nld_counts0 == 0] = np.nan

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

    def guess_initial_values(self, E_nld: np.ndarray, matrix: Matrix,
                             method: Optional[str] = None) -> np.ndarray:
        """Guess initial values `x0` for minimization rountine

        Note:
            Different initial guesses will affect the normalization constants
            needed later. In order to minimize the effort when changing the
            initial guess (`method`), one can try to provide initial guesses
            where the nld is approx(!) 1 in all bins.

        Args:
            E_nld: Energy array of nld
            matrix: Matrix to be sent to minimization
            method (optional): Method for initial guesses. Defaults to
                a backshifted Fermi-gas like initial value. This default is
                choosen, because we often find a CT-like result -- so we
                want something dissimilar as a start.

        Returns:
            x0: Initial guesses as a stacked array of (T0, nld0)
        """

        if E_nld[-1] > 100:
            LOG.info("Infering calibration that calibration is in keV.")
            E_nld = E_nld.copy()
            E_nld /= 1000

        if method is None or method == "BSFG-like":
            nld0 = self.x0_BSFG(E_nld)
        elif method == "CT-like":
            nld0 = self.x0_CT(E_nld)
        elif method == "parabola":
            nld0 = self.x0_parabola(E_nld)
        else:
            raise NotImplementedError(f"Method {method} not in "
                                      "['BSFG-like','CT-like'.")

        T0, _ = matrix.projection('Eg')
        assert T0.size == matrix.Eg.size
        x0 = np.append(T0, nld0)
        assert np.isfinite(x0).all
        return x0

    @staticmethod
    def x0_BSFG(E_nld: np.ndarray, E0: float = -.2, a: float = 15):
        """ Initial guess that resembles Backshifted Fermi-gas solution

        Note that this is like a fermi-gas after transformation with
        `Aexp(αE)`. Default parameters are chosen somewhat arbitrary, but
        resembling fitted values reported in EB2005.

        Args:
            E_nld: Energy array of nld in [MeV]
            E0 (optional): Back-shift in [MeV]. Defaults to -0.2.
            a (optional): Level density parameter in [MeV⁻¹]. Defaults to 15.

        Returns:
            nld0: Initial guess
        """

        U = E_nld - E0
        U[U <= 0] = 0.5  # workaround for negative energies (or E=0)
        fg = np.exp(2*np.sqrt(a*U)) / (a**(1/4)*U**(5/4))

        # transform for convenience, such that the result is close to 1
        # for many bins -- here: same value at 1/4 of the length as at end
        N = len(fg)
        alpha = np.log(fg[N//4]/fg[-1]) / (E_nld[-1] - E_nld[N//4])
        fg *= np.exp(alpha*E_nld)
        fg /= fg[N//2] * 0.7  # at half of the array, values = 1/0.7 = 1.4

        return fg

    @staticmethod
    def x0_CT(E_nld: np.ndarray) -> np.ndarray:
        """Initial guess that resembles CT solution

        This is the proposal of Schiller2000 -- however, note that
        a constant nld of 1 after the transformation with `Aexp(αE)` is just
        like the CT formula.

        Args:
            E_nld: Energy array of nld

        Returns:
            nld0: Initial guess
        """
        return np.ones(E_nld.size)

    @staticmethod
    def x0_parabola(E_nld: np.ndarray, E0: float = 4,
                    y0: float = 0.01, a: float = 1) -> np.ndarray:
        """Initial guess as parabola a(E_nld - E0)² + y0

        This is quite crazy; For E0 > 0: the NLD is not expected to
        reduce for higher Ex

        Args:
            E_nld: Energy array of nld [in MeV]
            E0: shift constant in x direction [in MeV]
            y0: shift constant in y direction.
            a: multiplier


        Returns:
            nld0: Initial guess
        """
        vals = a*(E_nld - E0)**2 + y0
        assert (vals >= 0).all(), "Negative nld is meaningless"
        return vals

    @staticmethod
    def constraining_counts(matrix: Matrix,
                            resolution: np.ndarray) -> Tuple[np.ndarray,
                                                             np.ndarray]:
        """ Number of counts constraining each nld bin and gsf bin

        Args:
            matrix (Matrix): Input matrix
            resolution (np.ndarray): Resolution at `Ex=Ex`

        Returns:
            Tuple[nld_counts, T_counts]: Number of counts constraining each
                nld and gsf bin
        """
        matrix = matrix.copy()

        # Mask elements beyond Ex + resolution* to 0 (not used in chi2)
        # * + halfbin to get closest bin (when calib uses midbins)
        Egs = np.tile(matrix.Eg, (matrix.shape[0], 1))  # reapeat Egs
        halfbin = (matrix.Eg[1]-matrix.Eg[0])/2
        Emax = (matrix.Ex + resolution)[:, np.newaxis] + halfbin
        matrix.values = np.ma.masked_array(matrix.values, Egs > Emax)

        # sum counts along diagonals
        start = - (matrix.shape[0] - 1)
        stop = matrix.shape[1]
        nld_counts = np.array([matrix.values.trace(offset=d)
                              for d in range(start, stop)])
        nld_counts = nld_counts[::-1]

        T_counts, _ = matrix.projection('Eg')

        return nld_counts, T_counts.data

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
        ax[1].set_title(r"$\gamma$SF")
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

    def __getstate__(self):
        """ `__getstate__` excluding `ensemble` attribute to save space """
        state = self.__dict__.copy()
        del state['ensemble']
        return state


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
