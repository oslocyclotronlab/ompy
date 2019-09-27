import numpy as np
import matplotlib.pyplot as plt
import warnings
from uncertainties import unumpy
from tqdm import tqdm
from pathlib import Path
from typing import Optional, Union, Any, Tuple, List
from scipy.optimize import minimize
from .ensemble import Ensemble
from .matrix import Matrix
from .vector import Vector
from .decomposition import chisquare_diagonal, nld_T_product
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
        ensemble (Ensemble): The Ensemble instance to extract nld and gsf from.
        size (int): The number of (nld, gsf) pairs to extract. Must be equal to
           or smaller than ensemble.size.
        regenerate (bool): Whether to force extraction from matrices even if
            previous results are found on disk. Defaults to True
        path (path): The path to save and/or load nld and gsf to/from.
        method (str): The scipy.minimization method to use. Defaults to Powell.
        options (dict): The scipy.minimization options to use.
        nld (list[Vector]): The nuclear level densities extracted.
        gsf (list[Vector]): The gamma strength functions extracted.
        trapezoid (Action[Matrix]): The Action cutting the matrices of the
            Ensemble
           into the desired shape where from the nld and gsf will
           be extracted from.

    TODO:
        - If path is given, it tries to load. If path is later set,
          it is not created. This is a very common pattern. Consider
          superclassing the disk book-keeping.
        - Make bin_width setable
    """
    def __init__(self, ensemble: Optional[Ensemble] = None,
                 trapezoid: Optional[Action] = None,
                 path: Optional[Union[str, Path]] = None):
        self.ensemble = ensemble
        self.size = 10 if ensemble is None else ensemble.size
        self.regenerate = False
        self.method = 'Powell'
        self.options = {'disp': True, 'ftol': 1e-3, 'maxfev': None}
        self.nld: List[Vector] = []
        self.gsf: List[Vector] = []
        self.trapezoid = trapezoid
        self.bin_width = None  # TODO: Make Setable

        if path is not None:
            self.path = Path(path)
            self.path.mkdir(exist_ok=True)
            try:
                self.load(self.path)
            except AssertionError:
                pass
        else:
            self.path = Path('extraction_ensemble')
            self.path.mkdir(exist_ok=True)

    def __call__(self, ensemble: Optional[Ensemble] = None,
                 trapezoid: Optional[Action] = None):
        return self.extract_from(ensemble, trapezoid)

    def extract_from(self, ensemble: Optional[Ensemble] = None,
                     trapezoid: Optional[Action] = None,
                     regenerate: Optional[bool] = None):
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
            regenerate: Whether to regenerate all nld and gsf even if
                they are found on disk.
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
        if regenerate is None:
            regenerate = self.regenerate
        self.path = Path(self.path)  # TODO: Fix

        assert self.ensemble.size >= self.size, "Ensemble is too small"

        nlds = []
        gsfs = []
        for i in tqdm(range(self.size)):
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
        """
        nld, gsf = self._extract(num)
        return nld, gsf

    def _extract(self, num: int) -> Tuple[Vector, Vector]:
        """ Extract nld and gsf from matrix number i from Ensemble

        Returns:
            The nld and gsf as Vectors
        """
        assert self.ensemble is not None
        assert self.trapezoid is not None
        matrix = self.ensemble.get_firstgen(num).copy()
        std = self.ensemble.std_firstgen.copy()
        self.trapezoid.act_on(matrix)
        self.trapezoid.act_on(std)
        nld, gsf = self.decompose(matrix, std)
        return nld, gsf

    def decompose(self, matrix: Matrix,
                  std: Optional[Matrix] = None,
                  x0: Optional[np.ndarray] = None,
                  bin_width: Optional[int] = None,
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
                be the same size as the matrix. If no std is provided,
                square error will be used instead of chi square.
            x0: The initial guess for nld and gsf.
            bin_width: NOT YET IMPLEMENTED!
                The bin width of the energy array of nld.
                Defaults to self.bin_width if None.
            product: Whether to return the first generation matrix
               resulting from the product of nld and gsf.
        Returns:
            The nuclear level density and the gamma strength function
            as Vectors.
            Optionally returns nld*γSF if product is True

        Todo:
            Implement automatic rebinning if bin_width is given(?)

        """
        if std is not None:
            assert matrix.shape == std.shape
            std.values = std.values.copy(order='C')
            matrix.values, std.values = normalize(matrix, std)
            matrix.Ex = matrix.Ex.copy(order='C')
            matrix.Eg = matrix.Eg.copy(order='C')
        else:
            matrix.values, _ = normalize(matrix)

        if bin_width is not None:
            bin_width = bin_width
        else:
            bin_width = self.bin_width
        if bin_width is not None:
            raise NotImplementedError("Bin-width cannot be set yet."
                                      "Rebin upfront.")

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
        resolution = matrix.diagonal_resolution()
        Emin = -resolution.max()
        Emax = matrix.Ex.max()-matrix.Eg.min()
        E_nld = np.linspace(Emin, Emax, np.ceil((Emax-Emin)/bin_width))

        if x0 is None:
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

        res = minimize(errfun, x0=x0, method=self.method, options=self.options)
        T = res.x[:matrix.Eg.size]
        nld = res.x[matrix.Eg.size:]

        # Convert transmission coefficient to the more useful
        # gamma strength function
        gsf = T/(2*np.pi*matrix.Eg**3)

        if product:
            values = nld_T_product(nld, T, resolution,
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
            path = Path(self.path)  # TODO: Fix pathing

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
             **kwargs) -> None:
        """ Basic visualization of nld and gsf

        Args:
            ax: An axis to plot onto
            scale: Scale to use
        TODO: Fix
        """
        if ax is None:
            fig, ax = plt.subplots(1, 2, constrained_layout=True)
        # for nld, gsf in zip(self.nld, self.gsf):
        #     ax[0].plot(nld.E, nld.values, color='k',
        #                alpha=1/self.size, **kwargs)
        #     ax[1].plot(gsf.E, gsf.values, color='k',
        #                alpha=1/self.size, **kwargs)
        else:
            fig = None

        ax[0].errorbar(self.nld[0].E, self.nld_mean(), yerr=self.nld_std(),
                       fmt='o', ms=1, lw=1)
        ax[1].errorbar(self.gsf[0].E, self.gsf_mean(), yerr=self.gsf_std(),
                       fmt='o', ms=1, lw=1)

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


def normalize(mat: Matrix, std: Optional[Matrix]):
    matrix = unumpy.uarray(mat.values, std.values if std is not None else None)

    # normalize each Ex row to 1 (-> get decay probability)
    for i, total in enumerate(matrix.sum(axis=1)):
        if total == 0:
            continue
        matrix[i, :] = np.true_divide(matrix[i, :], total)
    values = unumpy.nominal_values(matrix)
    std = unumpy.std_devs(matrix)

    return values, std
