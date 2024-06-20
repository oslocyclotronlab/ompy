import logging
import numpy as np
from .. import Matrix, Vector
from .. import zeros_like, NUMBA_AVAILABLE
from typing import TypeAlias, Literal
from tqdm.auto import tqdm
from ..numbalib import njit, prange, jitclass, float32
from dataclasses import dataclass, asdict
from ..unfolding import BootstrapMatrix

"""
TODO:
- [ ] Improve population normalization estimation
- [ ] Implement all generation method
- [ ] Handle Ei, Ef limits of N
      Compute in (Ex, Eg) and map to (Ei, Ef)?
- [ ] Make a backup implementation in numpy
- [x] Write a wrapper for bootstrap lists
- [ ] If I can construct AG from FG,
      I can use GD to find FG.
      Need to known population factor.
- [ ] How to handle bootstraped AG? Share N from median eta?
- Uncertainties accumulate along Ef, as Ef is the acumulation
  of all lower Ef.
- Bootstrap does not account for systematic error, in particular
  oversubstraction due to wrong N.
- I can see the shadow of poorly unfolded contaminants in
  (upper - lower).
  And diagonal lines with a steeper slope in continuum??
    - Need higher order calibration
- How does FG(median) compare to median(FG)?
- Make some nice graphics of the FG method
- Population is best constructed from singles spectra,
  which much be found during sorting. Didn't quite understand
  how. Ask more later.
- Can Independent component analysis work to decompose the AG?
  Or NMF.
- Unfolding - wavelet?
- FG check rhosigchi for Ex=Eg cutoff Februar 2016 ndim xdim smooth 300keV
- The method fails for RAINIER mu data. I suspect it is the Ex Eg bin resolution
  that makes small errors that accumulate, yielding negative bins.
  Should there be any lines at all in FG?
  I forgot to copy over my FG AG. Unable to check.
- Can smoothness conditions be enforced to regularize the solution?
  All vertical lines "of" the AG should be in the G, not FG. Right?
- Could there be flaws with the RAINIER simulation?
  When folded with G, there are gradients along Ex. I suspect there are
  some binning issues with the RAINIER discretization.
- I was mistaken. The FG method is not significantly worse when done on
  G folded data than on mu space.
  Maybe? Maybe a trick of the color scale.
  The discrete levels are completely identical. 
  BUG: For the example I used it suddenly deviates at ex=4.000MeV. Very strange
  This is so strange that I can't help but suspect a bug.
  Oh, maybe RAINIER! If the discrete levels are used up to 4MeV, maybe the problem
  is with the simulation not faithfully reproducing the quasi continuum.
  Nevertheless my intuition has recieved a blow.

"""

LOG = logging.getLogger(__name__)

if NUMBA_AVAILABLE:
    spec = [
        ('values', float32[:, :]),
        ('Ex', float32[:]),
        ('Eg', float32[:]),
    ]
else:
    spec = None

Backend: TypeAlias = Literal['numpy', 'numba']
DEFAULT_BACKEND: Backend = 'numba' if NUMBA_AVAILABLE else 'numpy'


@jitclass(spec=spec)
class MatrixNumba:
    def __init__(self, Ex, Eg, values):
        self.Ex = Ex
        self.Eg = Eg
        self.values = values

    def __getitem__(self, key):
        return self.values[key]

    def __setitem__(self, key, value):
        self.values[key] = value

    def index_Ex(self, e):
        k = int((e - self.Ex[0]) // (self.Ex[1] - self.Ex[0]))
        return k

if NUMBA_AVAILABLE:
    spec = [
        ('values', float32[:]),
        ('E', float32[:]),
    ]
else:
    spec = None

@jitclass(spec=spec)
class VectorNumba:
    def __init__(self, E, values):
        self.E = E
        self.values = values

    def __getitem__(self, key):
        return self.values[key]

def mat_to_numba(mat: Matrix) -> MatrixNumba:
    mat = mat.clone(dtype=np.float32)
    return MatrixNumba(mat.X, mat.Y, mat.values)

def vec_to_numba(vec: Vector) -> VectorNumba:
    vec = vec.clone(dtype='float32')
    return VectorNumba(vec.X, vec.values)


@dataclass
class FirstGenerationParameters:
    iterations: int = 10

FGP: TypeAlias = FirstGenerationParameters


@dataclass
class FirstGenerationResult:
    AG: Matrix
    FG: Matrix
    alpha: Matrix
    parameters: FirstGenerationParameters


def first_generation(AG: Matrix | list[Matrix] | BootstrapMatrix,
                     params: FGP = FGP(),
                     multiplicity: Vector | None = None,
                     population_norm: Matrix | None = None,
                     disable_tqdm: bool = False,
                     **kwargs) -> FirstGenerationResult | list[FirstGenerationResult]:
    match AG:
        case Matrix():
            return first_generation_matrix(AG, params, multiplicity, population_norm, disable_tqdm, **kwargs)
        case list():
            return first_generation_list(AG, params, multiplicity, population_norm, disable_tqdm, **kwargs)
        case BootstrapMatrix():
            AGs = [AG.get_eta(i) for i in range(len(AG))]
            return first_generation(AGs, params, multiplicity, population_norm, disable_tqdm, **kwargs)
        case x:
            raise TypeError(f"AG must be Matrix or list of Matrix, not {type(x)}")


def first_generation_list(AG: list[Matrix],
                     params: FGP = FGP(),
                     multiplicity: Vector | None = None,
                     population_norm: Matrix | None = None,
                     disable_tqdm: bool = False,
                     **kwargs) -> list[FirstGenerationResult]:
    res: list[FirstGenerationResult] = []
    tqdm_ = tqdm if not disable_tqdm else lambda x: x
    for i in tqdm_(range(len(AG))):
        LOG.info("First generation for AG[%d]", i)
        res.append(first_generation_matrix(AG[i], params, multiplicity, population_norm, disable_tqdm=True, **kwargs))
    return res


def first_generation_matrix(AG: Matrix,
                     params: FGP = FGP(),
                     multiplicity: Vector | None = None,
                     population_norm: Matrix | None = None,
                     disable_tqdm: bool = False,
                     backend: Backend = DEFAULT_BACKEND,
                     **kwargs) -> FirstGenerationResult:
    params = FGP(**(asdict(params) | kwargs))
    FG = zeros_like(AG)
    alphas = np.zeros((params.iterations, len(AG.Ex)))
    M = multiplicity_estimation(AG) if multiplicity is None else multiplicity
    if population_norm is None:
        N = population_normalization(AG, multiplicity = M, backend=backend)
    else:
        N = population_norm

    tqdm_ = tqdm if not disable_tqdm else lambda x: x

    FG = np.zeros_like(AG)
    FG[AG > 0] = 1
    AG_ = AG.values
    FG_prev = FG
    for i in tqdm_(range(params.iterations)):
        W = FG / FG.sum(axis=1)[:, np.newaxis]  # Normalize each Ex row
        G = G_step(AG, W, N)
        FG = AG_ - G
        alpha = (1 - 1/M) * (AG.sum(axis=1) / G.sum(axis=1))
        alpha[~np.isfinite(alpha)] = np.nan
        alphas[i, :] = alpha
        alpha_mean = np.nanmedian(alpha)
        alpha_low = np.nanpercentile(alpha, 25)
        alpha_high = np.nanpercentile(alpha, 75)
        

        diff = FG - FG_prev
        abs_diff = np.abs(diff).sum()
        rel_diff = np.nansum(abs(diff / FG))
        max_diff = np.max(np.abs(diff))
        LOG.info("Iteration %d:\n\tabs = %g, rel = %g, max = %g, \u03B1 = %g±(%g,%g)",
                 i+1, abs_diff, rel_diff, max_diff, alpha_mean, alpha_low, alpha_high)
        FG_prev = FG
    FG = AG.clone(values=FG, name='first generation')
    alpha = Matrix(Ex=AG.Ex,
                   i=np.arange(params.iterations),
                   values=alphas.T,
                   ylabel='iteration', Y_unit='',
                   xlabel='Ex',
                   name='alpha')
    res = FirstGenerationResult(AG=AG, FG=FG, alpha=alpha,
                                parameters=params)
    return res


def multiplicity_estimation(AG: Matrix) -> Vector:
    """ Estimate the multiplicity from all generations matrix

    See DOI: 10.1016/0168-9002(87)91221-6

    Args:
        AG: All generations matrix, most often from the unfolding step.
    
    Returns:
        Estimated multiplicity vector
    """
    Eg_sum = AG.sum(axis='Eg')
    Eg_expectation = (AG.Eg * AG).sum(axis='Eg') / Eg_sum
    multiplicity = AG.Ex / Eg_expectation
    multiplicity[multiplicity < 0] = 0
    multiplicity.xlabel = 'multiplicity'
    multiplicity.title = 'multiplicity estimation'
    return multiplicity
    

def population_normalization(AG: Matrix, multiplicity: Vector | None = None,
                             backend: Backend = DEFAULT_BACKEND) -> Matrix:
    if multiplicity is None:
        multiplicity = multiplicity_estimation(AG)
    Eg_sum = AG.sum(axis='Eg')
    match backend:
        case 'numba':
            N = population_normalization_njit(vec_to_numba(multiplicity), 
                                              vec_to_numba(Eg_sum)).values
        case 'numpy':
            N = population_normalization_np(AG.Ex, multiplicity.values, Eg_sum.values)
        case _:
            raise ValueError(f"Invalid backend: {backend}. Must be 'numpy' or 'numba'")
    N = Matrix(Ei=AG.Ex, Ef=AG.Ex, values=N,
               xlabel='Ei', ylabel='Ef', name='population normalization')
    return N

    
@njit
def population_normalization_njit(multiplicity: VectorNumba,
                                  Eg_sum: VectorNumba) -> MatrixNumba:
    Ex = multiplicity.E
    N = np.zeros((len(Ex), len(Ex)), dtype=multiplicity.values.dtype)
    for ei in prange(len(Ex)):
        for ef in prange(len(Ex)):
            if multiplicity[ef] == 0 or Eg_sum[ef] == 0:
                continue
            N[ei, ef] = multiplicity[ef] / multiplicity[ei] * Eg_sum[ei] / Eg_sum[ef]
    return MatrixNumba(Ex, Ex, N)


def population_normalization_np(Ex, M, N):
    ex, ef = np.meshgrid(np.arange(len(Ex)), np.arange(len(Ex)), indexing='ij')
    n = M[ef] / M[ex] * N[ex] / N[ef]
    return n


def G_step(AG: Matrix, W: np.ndarray, N: Matrix, backend: Backend = DEFAULT_BACKEND) -> np.ndarray:
    match backend:
        case 'numba':
            # This makes a copy :(
            return G_step_njit(mat_to_numba(AG), W, mat_to_numba(N))
        case 'numpy':
            raise NotImplementedError("Numpy backend not implemented")
            return G_step_np(AG.values, W, N)
        case _:
            raise ValueError(f"Invalid backend: {backend}. Must be 'numpy' or 'numba'")


@njit(parallel=True)
def G_step_njit(AG: MatrixNumba, W: np.ndarray, N: MatrixNumba) -> np.ndarray:
    G = np.zeros_like(AG.values)
    Ex = AG.Ex
    NEx = len(Ex)
    for i_ei in prange(NEx):
        for i_ef in range(i_ei):
            eg = Ex[i_ei] - Ex[i_ef]
            k = AG.index_Ex(eg)  # W.index_Eg
            if k < 0:
                continue  # break?
            factor = N[i_ei, i_ef] * W[i_ei, k]
            G[i_ei, :] += factor * AG.values[i_ef, :]
    return G
