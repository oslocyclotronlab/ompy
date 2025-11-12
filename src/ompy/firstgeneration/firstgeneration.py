import logging
from dataclasses import dataclass, field
from typing import Literal, TypeAlias

import numpy as np
from tqdm.auto import tqdm

from ..accel import numba_available
from ..array import Matrix, Vector
from ..array.ops import zeros_like
from ..numbalib import float32, jitclass, njit, prange
from ..pipeline.lifting import lift
from ..pipeline.result import Result, ResultMeta, Settings
from ..pipeline.stage import Stage

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
_HAS_NUMBA = numba_available()

if _HAS_NUMBA:
    spec = [
        ("values", float32[:, :]),
        ("Ex", float32[:]),
        ("Eg", float32[:]),
    ]
else:
    spec = None

Backend: TypeAlias = Literal["numpy", "numba_array"]
DEFAULT_BACKEND: Backend = "numba_array" if _HAS_NUMBA else "numpy"


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

    def index_Eg(self, e):
        k = int((e - self.Eg[0]) // (self.Eg[1] - self.Eg[0]))
        return k


if _HAS_NUMBA:
    spec = [
        ("values", float32[:]),
        ("E", float32[:]),
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
    vec = vec.clone(dtype="float32")
    return VectorNumba(vec.X, vec.values)


@dataclass
class FirstGenerationSettings(Settings):
    """Settings for first generation extraction.
    
    Inherits from pipeline.Settings which provides:
    - from_dict() - create from kwargs dict
    - consume() - extract matching kwargs and return leftovers
    - update() - create new instance with updated fields
    """
    # Common parameters
    iterations: int = 10
    backend: Backend = DEFAULT_BACKEND
    enforce_nonnegativity: bool = False
    relaxation: float = 0.5
    rel_tol: float | None = None
    abs_tol: float | None = None
    leave_tqdm: bool = True
    disable_tqdm: bool = False
    multiplicity: Vector | None = None
    population_norm: Matrix | None = None
    
    # Parameters for latent/gamma methods
    lambda_U: float = 1e-2  # Tikhonov regularization for U
    nnls_iters: int = 200   # Inner NNLS iterations for U
    lambda_gamma: float = 1e-2  # Ridge for gamma method
    pg_iters: int = 200     # Projected gradient iters for gamma
    gamma_min: float = 1.0
    gamma_max: float = 5.0


FGS: TypeAlias = FirstGenerationSettings


@dataclass(kw_only=True)
class FirstGenerationResult(Result[Matrix]):
    """Result from first generation extraction.
    
    Inherits from pipeline.Result[Matrix] which provides:
    - .stage property for pipeline tracking
    - __stage__() method
    - Consistent interface with other pipeline results
    """
    AG: Matrix
    FG: Matrix
    U: Matrix
    alpha: Matrix
    settings: FirstGenerationSettings
    reldiffs: Vector
    maxdiffs: Vector
    absdiffs: Vector
    meta: ResultMeta = field(default_factory=lambda: ResultMeta(
        stage=Stage.FIRST_GENERATION,
        method='first_generation'
    ))
    
    def __unwrap__(self) -> Matrix:
        """Unwrap protocol for pipeline compatibility.
        
        Returns the first-generation (FG) matrix, which is the primary
        result of the first-generation extraction.
        """
        return self.FG


@lift(expects=Stage.UNFOLDED, produces=Stage.FIRST_GENERATION)
def first_generation(
    AG: Matrix,
    settings: FGS | None = None,
    method: Literal['standard', 'latent', 'gamma'] = 'standard',
    **kwargs,
) -> FirstGenerationResult:
    """Extract first-generation gamma-ray spectrum from all-generations spectrum.
    
    The @lift decorator automatically handles:
    - list[Matrix] -> list[FirstGenerationResult]
    - EnsembleMatrix -> EnsembleMatrix (applies to all members)
    - Resampling2D -> convert to EnsembleMatrix first with .ensemble_eta()
    
    Parameters
    ----------
    AG : Matrix
        All-generations matrix (typically from unfolding, in eta space).
    params : FirstGenerationParameters
        Algorithm parameters (iterations, etc.).
    multiplicity : Vector, optional
        Gamma multiplicity vector. If None, estimated from AG.
    population_norm : Matrix, optional
        Population normalization matrix. If None, computed from AG.
    disable_tqdm : bool
        Disable progress bar.
    method : {'standard', 'latent', 'gamma'}
        Which algorithm variant to use:
        - 'standard': Basic iterative method
        - 'latent': Includes latent component U
        - 'gamma': Includes row gain factors
    **kwargs
        Additional parameters passed to the selected method.
        
    Returns
    -------
    FirstGenerationResult
        Result containing FG, AG, alpha, convergence metrics, etc.
        
    Examples
    --------
    Single matrix:
    >>> result = first_generation(AG_matrix, method='standard')
    
    List of matrices:
    >>> results = first_generation([AG1, AG2, AG3], method='latent')
    
    Ensemble from bootstrap:
    >>> resampling = unfold_result.resample(N=100)
    >>> ensemble_eta = resampling.ensemble_eta()
    >>> fg_ensemble = first_generation(ensemble_eta)
    >>> fg_mean = fg_ensemble.mean()
    
    Notes
    -----
    For Resampling2D input, convert to EnsembleMatrix first:
    >>> ensemble_eta = resampling.ensemble_eta()  # Use eta space for FG
    >>> results = first_generation(ensemble_eta)
    """
    # Create settings from kwargs if not provided
    if settings is None:
        settings = FGS()
    
    # Merge any additional kwargs into settings
    settings = settings.update(**kwargs)
    
    # Select the appropriate implementation
    match method:
        case 'standard':
            fn = first_generation_matrix
        case 'latent':
            fn = first_generation_latent
        case 'gamma':
            fn = first_generation_gamma
        case _:
            raise ValueError(f"Invalid method: {method}. Must be 'standard', 'latent', or 'gamma'")
    
    # Call the selected implementation (single Matrix case)
    return fn(AG, settings)


def first_generation_matrix(
    AG: Matrix,
    settings: FGS,
) -> FirstGenerationResult:
    AG = AG.as_numpy()
    FG = zeros_like(AG)
    alphas = np.zeros((settings.iterations, len(AG.Ex)))
    M = multiplicity_estimation(AG) if settings.multiplicity is None else settings.multiplicity
    if settings.population_norm is None:
        N = population_normalization(AG, multiplicity=M, backend=settings.backend)
    else:
        N = settings.population_norm

    tqdm_ = lambda x: tqdm(x, leave=settings.leave_tqdm) if not settings.disable_tqdm else lambda x: x

    FG = np.zeros_like(AG.values)
    FG[AG > 0] = 1
    AG_ = AG.values
    FG_prev = FG
    alpha = np.ones(AG.Ex.size)
    reldiffs = np.zeros(settings.iterations)
    maxdiffs = np.zeros(settings.iterations)
    absdiffs = np.zeros(settings.iterations)
    eta = settings.relaxation
    for i in tqdm_(range(settings.iterations)):
        W = FG / FG.sum(axis=1)[:, np.newaxis]  # Normalize each Ex row
        G = G_step(AG, W, N)
        #FG = AG_ - (1 + 0.5*alpha[:, np.newaxis]) * G
        FG = (1-eta)*FG + eta*(AG_ - G)
        if settings.enforce_nonnegativity:
            FG = np.clip(FG, 0, None)

        alpha = (1 - 1 / M) * (AG_.sum(axis=1) / G.sum(axis=1))
        alpha[~np.isfinite(alpha)] = np.nan
        alphas[i, :] = alpha
        alpha_mean = np.nanmedian(alpha)
        alpha_low = np.nanpercentile(alpha, 25)
        alpha_high = np.nanpercentile(alpha, 75)

        diff = FG - FG_prev
        abs_diff = np.abs(diff).sum()
        rel_diff = np.nansum(abs(diff / FG))
        max_diff = np.max(np.abs(diff))
        reldiffs[i] = rel_diff
        maxdiffs[i] = max_diff
        absdiffs[i] = abs_diff
        LOG.info(
            "Iteration %d:\n\tabs = %g, rel = %g, max = %g, \u03b1 = %g±(%g,%g)",
            i + 1,
            abs_diff,
            rel_diff,
            max_diff,
            alpha_mean,
            alpha_low,
            alpha_high,
        )

        # Check convergence
        if (settings.rel_tol is not None and rel_diff < settings.rel_tol) or \
           (settings.abs_tol is not None and abs_diff < settings.abs_tol):
            # Trim arrays to actual iterations performed
            alphas = alphas[:i+1]
            reldiffs = reldiffs[:i+1]
            maxdiffs = maxdiffs[:i+1]
            absdiffs = absdiffs[:i+1]
            break

        FG_prev = FG
    FG = AG.clone(values=FG, name="first generation")
    alpha = Matrix(
        Ex=AG.Ex,
        i=np.arange(len(alphas)),
        values=alphas.T,
        ylabel="iteration",
        Y_unit="",
        xlabel="Ex",
        name="alpha",
    )
    reldiffs = Vector(i=np.arange(len(reldiffs)), values=reldiffs, name="relative differences", vlabel='rel. diff.', xlabel='iteration')
    maxdiffs = Vector(i=np.arange(len(maxdiffs)), values=maxdiffs, name="maximum differences", vlabel='max. diff.', xlabel='iteration')
    absdiffs = Vector(i=np.arange(len(absdiffs)), values=absdiffs, name="absolute differences", vlabel='abs. diff.', xlabel='iteration')
    U = Matrix(Ex=AG.Ex, Ef=AG.Ex, values=np.zeros_like(AG.values), name="latent component")
    res = FirstGenerationResult(
        AG=AG, FG=FG, U=U, alpha=alpha, settings=settings,
        reldiffs=reldiffs, maxdiffs=maxdiffs, absdiffs=absdiffs
    )
    return res

def multiplicity_estimation(AG: Matrix) -> Vector:
    """Estimate the multiplicity from all generations matrix

    See DOI: 10.1016/0168-9002(87)91221-6

    Args:
        AG: All generations matrix, most often from the unfolding step.

    Returns:
        Estimated multiplicity vector
    """
    Eg_sum = AG.sum(axis="Eg")
    Eg_expectation = (AG.Eg * AG).sum(axis="Eg") / Eg_sum
    multiplicity = AG.Ex / Eg_expectation
    multiplicity[multiplicity < 0] = 0
    multiplicity.ylabel = "multiplicity"
    multiplicity.title = "multiplicity estimation"
    return multiplicity


def population_normalization(
    AG: Matrix, multiplicity: Vector | None = None, backend: Backend = DEFAULT_BACKEND
) -> Matrix:
    if multiplicity is None:
        multiplicity = multiplicity_estimation(AG)
    Eg_sum = AG.sum(axis="Eg")
    match backend:
        case "numba_array":
            N = population_normalization_njit(
                vec_to_numba(multiplicity), vec_to_numba(Eg_sum)
            ).values
        case "numpy":
            N = population_normalization_np(AG.Ex, multiplicity.values, Eg_sum.values)
        case _:
            raise ValueError(
                f"Invalid backend: {backend}. Must be 'numpy' or 'numba_array'"
            )
    N = Matrix(
        Ei=AG.Ex,
        Ef=AG.Ex,
        values=N,
        xlabel="Ei",
        ylabel="Ef",
        name="population normalization",
    )
    return N


@njit
def population_normalization_njit(
    multiplicity: VectorNumba, Eg_sum: VectorNumba
) -> MatrixNumba:
    Ex = multiplicity.E
    N = np.zeros((len(Ex), len(Ex)), dtype=multiplicity.values.dtype)
    for ei in prange(len(Ex)):
        if multiplicity[ei] == 0 or Eg_sum[ei] == 0:
            continue
        for ef in prange(len(Ex)):
            if multiplicity[ef] == 0 or Eg_sum[ef] == 0:
                continue
            N[ei, ef] = multiplicity[ef] / multiplicity[ei] * Eg_sum[ei] / Eg_sum[ef]
    return MatrixNumba(Ex, Ex, N)


def population_normalization_np(Ex, M, N):
    ex, ef = np.meshgrid(np.arange(len(Ex)), np.arange(len(Ex)), indexing="ij")
    n = M[ef] / M[ex] * N[ex] / N[ef]
    return n


def G_step(
    AG: Matrix, W: np.ndarray, N: Matrix, backend: Backend = DEFAULT_BACKEND
) -> np.ndarray:
    match backend:
        case "numba_array":
            # This makes a copy :(
            return G_step_njit(mat_to_numba(AG), W, mat_to_numba(N))
        case "numpy":
            raise NotImplementedError("Numpy backend not implemented")
            return G_step_np(AG.values, W, N)
        case _:
            raise ValueError(
                f"Invalid backend: {backend}. Must be 'numpy' or 'numba_array'"
            )


@njit(parallel=True)
def G_step_njit(AG: MatrixNumba, W: np.ndarray, N: MatrixNumba) -> np.ndarray:
    G = np.zeros_like(AG.values)
    Ex = AG.Ex
    NEx = len(Ex)
    for i_ei in prange(NEx):
        for i_ef in range(i_ei):
            eg = Ex[i_ei] - Ex[i_ef]
            #k = AG.index_Eg(eg)  # W.index_Eg
            k = AG.index_Ex(eg)
            # should it have been AG.index_Ex(eg)?
            if k < 0 or k >= len(AG.Eg):
                continue  # break?
            factor = N[i_ei, i_ef] * W[i_ei, k]
            G[i_ei, :] += factor * AG.values[i_ef, :]
    return G


@njit(parallel=True)
def build_C_matrix(AG: MatrixNumba, W: np.ndarray, N: MatrixNumba) -> np.ndarray:
    """
    Build the JxJ lower-triangular 'row-recycle' matrix C with entries:
        C[i, j] = N[i, j] * W[i, k_drop(i,j)]
    where k_drop(i,j) maps Eg = Ex[i] - Ex[j] onto the Eg grid.
    This C is independent of gamma column; it is used as (C @ X) per Eg column.
    """
    Ex, Eg = AG.Ex, AG.Eg
    J = len(Ex)
    C = np.zeros((J, J), dtype=np.float32)
    # Map drop energies to Eg bins once
    for i in prange(J):
        for j in range(i):  # only j < i contributes
            eg = Ex[i] - Ex[j]
            # Use Eg indexer (NOT Ex)
            k = AG.index_Eg(eg)
            if k < 0 or k >= len(Eg):
                continue
            C[i, j] = N[i, j] * W[i, k]
    return C


#@njit(parallel=True)
def apply_C_to_matrix(C: np.ndarray, X: np.ndarray) -> np.ndarray:
    """
    Column-wise multiply: for each gamma column g, (C @ X[:, g]).
    X is (J, K). Returns (J, K).
    """
    return C @ X

#@njit(parallel=True)
def nnls_tikhonov_columnwise(C: np.ndarray,
                             R: np.ndarray,
                             lambda_U: float = 1e-2,
                             iters: int = 200,
                             step: float | None = None) -> np.ndarray:
    """
    Solve, independently for each gamma column g:
        min_{u >= 0} 0.5 || C u - r ||^2 + 0.5 * lambda_U ||u||^2
    with projected gradient steps.  C is (J,J), R is (J,K), returns U (J,K).
    """
    J, K = R.shape
    U = np.zeros((J, K), dtype=np.float32)
    # Precompute
    Ct = C.T
    # We operate with the normal equations operator: A = CtC + lambda_U I
    # Lipschitz constant ~ ||A||2. Use a safe upper bound if not provided.
    v = np.empty(size=(J,))
    if step is None:
        A = Ct @ C
        # power iteration for a cheap spectral norm estimate (few steps)
        #v = np.random.normal(size=(J,)).astype(np.float32)

        for _ in range(10):
            v = A @ v
            n = np.linalg.norm(v) + 1e-12
            v /= n
        L = float(v @ (A @ v)) + lambda_U
        step = 1.0 / (L + 1e-8)

    I_lambda = lambda_U  # scalar
    for g in prange(K):
        u = U[:, g]
        r = R[:, g]
        # Precompute b = C^T r
        b = Ct @ r
        # Projected gradient on: (CtC + λI) u = C^T r  (in least-squares sense)
        for _ in range(iters):
            grad = (Ct @ (C @ u)) + I_lambda * u - b
            u = u - step * grad
            # projection to nonnegative orthant
            np.maximum(u, 0.0, out=u)
        U[:, g] = u
    return U


def first_generation_latent(
    AG: Matrix,
    settings: FGS,
) -> FirstGenerationResult:
    AG = AG.as_numpy()
    FG = zeros_like(AG)
    alphas = np.zeros((settings.iterations, len(AG.Ex)))
    M = multiplicity_estimation(AG) if settings.multiplicity is None else settings.multiplicity
    if settings.population_norm is None:
        N = population_normalization(AG, multiplicity=M, backend=settings.backend)
    else:
        N = settings.population_norm

    tqdm_ = tqdm if not settings.disable_tqdm else lambda x: x

    FG = np.zeros_like(AG)
    FG[AG > 0] = 1
    AG_ = AG.values
    FG_prev = FG
    alpha = np.ones(AG.Ex.size)
    reldiffs = np.zeros(settings.iterations)
    maxdiffs = np.zeros(settings.iterations)
    absdiffs = np.zeros(settings.iterations)
    eta = settings.relaxation

    # --- latent component alternating loop (replaces your current W/G/FG update) ---
    for i in tqdm_(range(settings.iterations)):
        # Build W from current FG estimate (mask/positivity left to user preference)
        # Normalize rows; guard zero rows
        row_sum = FG.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1.0
        W = FG / row_sum

        # Build C once per iteration from W,N
        C = build_C_matrix(mat_to_numba(AG), W, mat_to_numba(N))

        # Precompute R0 = A - C A  (all as ndarrays)
        CA = apply_C_to_matrix(C, AG_)
        R0 = AG_ - CA

        # --- U-update: NNLS per gamma column on  C U ≈ (R0 - P) ---
        U_new = nnls_tikhonov_columnwise_masked_njit(
        C.astype(np.float32),
        (R0 - FG).astype(np.float32),
        AG.Ex.astype(np.float32),
        AG.Eg.astype(np.float32),
        lambda_U=np.float32(settings.lambda_U),
        iters=settings.nnls_iters,
        step=-1.0,  # let it auto-pick a safe step per column
    )
 

        # --- P-update: closed form then projection/relaxation ---
        CU = apply_C_to_matrix(C, U_new)
        P_raw = R0 - CU                 # unconstrained P*
        P_proj = P_raw
        if settings.enforce_nonnegativity:
            P_proj = np.maximum(P_proj, 0.0)

        # Optional: per-row sum projection to mean multiplicity target (commented)
        # T = (AG.sum(axis=1).values / np.maximum(M.values, 1e-12)).astype(P_proj.dtype)
        # s = T / np.maximum(P_proj.sum(axis=1), 1e-12)
        # P_proj *= s[:, None]

        # Relaxed update (your 'lr' / relaxation)
        FG = (1 - eta) * FG + eta * P_proj

        # Diagnostics (unchanged)
        diff = FG - FG_prev
        abs_diff = np.abs(diff).sum()
        rel_diff = np.nansum(np.abs(diff) / np.maximum(np.abs(FG), 1e-12))
        max_diff = np.max(np.abs(diff))
        reldiffs[i] = rel_diff
        maxdiffs[i] = max_diff
        absdiffs[i] = abs_diff
        LOG.info(
            "Iteration %d:\n\tabs = %g, rel = %g, max = %g",
            i + 1, abs_diff, rel_diff, max_diff,
        )

        if (settings.rel_tol is not None and rel_diff < settings.rel_tol) or \
           (settings.abs_tol is not None and abs_diff < settings.abs_tol):
            alphas = alphas[:i+1]  # alpha no longer used; keep shape intact
            reldiffs = reldiffs[:i+1]
            maxdiffs = maxdiffs[:i+1]
            absdiffs = absdiffs[:i+1]
            break

        FG_prev = FG
    # --- end latent component alternating loop ---

    
    
    FG = AG.clone(values=FG, name="first generation")
    alpha = Matrix(
        Ex=AG.Ex,
        i=np.arange(len(alphas)),
        values=alphas.T,
        ylabel="iteration",
        Y_unit="",
        xlabel="Ex",
        name="alpha",
    )
    reldiffs = Vector(i=np.arange(len(reldiffs)), values=reldiffs, name="relative differences", vlabel='rel. diff.')
    maxdiffs = Vector(i=np.arange(len(maxdiffs)), values=maxdiffs, name="maximum differences", vlabel='max. diff.')
    absdiffs = Vector(i=np.arange(len(absdiffs)), values=absdiffs, name="absolute differences", vlabel='abs. diff.')
    U = Matrix(Ex=AG.Ex, Ef=AG.Ex, values=U_new, name="latent component")
    res = FirstGenerationResult(
        AG=AG, FG=FG, U=U, alpha=alpha, settings=settings,
        reldiffs=reldiffs, maxdiffs=maxdiffs, absdiffs=absdiffs
    )
    return res

    
@njit
def _build_mask_indices(Ex, Eg_g):
    """Return indices j where Ex[j] >= Eg[g]."""
    J = Ex.shape[0]
    cnt = 0
    for j in range(J):
        if Ex[j] >= Eg_g:
            cnt += 1
    idx = np.empty(cnt, dtype=np.int32)
    k = 0
    for j in range(J):
        if Ex[j] >= Eg_g:
            idx[k] = j
            k += 1
    return idx  # length Jg

@njit#(parallel=True)
def _build_CtC_and_b(C, idx, r):
    """
    Build CtC (Jg x Jg) and b = C[:,idx]^T r (Jg) using loops.
    C: (J,J), idx: (Jg,), r: (J,)
    """
    J = C.shape[0]
    Jg = idx.shape[0]
    CtC = np.zeros((Jg, Jg), dtype=np.float32)
    b   = np.zeros(Jg, dtype=np.float32)

    # b = (C[:,idx])^T r
    for a in prange(Jg):
        ja = idx[a]
        s = 0.0
        for i in range(J):
            s += C[i, ja] * r[i]
        b[a] = s

    # CtC = (C[:,idx])^T (C[:,idx])
    for a in range(Jg):
        ja = idx[a]
        for bcol in range(a, Jg):
            jb = idx[bcol]
            s = 0.0
            for i in range(J):
                s += C[i, ja] * C[i, jb]
            CtC[a, bcol] = s
            CtC[bcol, a] = s  # symmetric
    return CtC, b

@njit#(parallel=True)
def _power_iter_max_eig_sym(A, n_iter=8):
    """
    Rough spectral norm (largest eigenvalue) for symmetric PSD A via power iteration.
    """
    n = A.shape[0]
    if n == 0:
        return 0.0
    v = np.ones(n, dtype=np.float32) / np.sqrt(n)
    for _ in range(n_iter):
        # w = A @ v
        w = np.zeros(n, dtype=np.float32)
        for i in prange(n):
            s = 0.0
            Ai = A[i]
            for j in range(n):
                s += Ai[j] * v[j]
            w[i] = s
        # normalize
        normw = 0.0
        for i in range(n):
            normw += w[i] * w[i]
        normw = np.sqrt(normw)
        if normw == 0.0:
            return 0.0
        for i in prange(n):
            v[i] = w[i] / normw
    # Rayleigh quotient v^T A v
    tmp = np.zeros(n, dtype=np.float32)
    for i in prange(n):
        s = 0.0
        Ai = A[i]
        for j in range(n):
            s += Ai[j] * v[j]
        tmp[i] = s
    rq = 0.0
    for i in range(n):
        rq += v[i] * tmp[i]
    return rq

@njit#(parallel=True)
def _proj_grad_nnls_tikhonov(CtC, b, lambda_U, iters, step):
    """
    Solve min_{u>=0} 0.5||C u - r||^2 + 0.5*lambda_U||u||^2 in normal-eqs form:
    min 0.5|| (CtC + λI) u - b ||_A with PGD. CtC is (Jg,Jg), b is (Jg,)
    """
    Jg = b.shape[0]
    u = np.zeros(Jg, dtype=np.float32)
    if Jg == 0:
        return u

    # If step<=0, estimate from L ≈ λ_max(CtC) + lambda_U
    if step <= 0.0:
        L = _power_iter_max_eig_sym(CtC, 6) + lambda_U
        if L <= 0.0:
            L = 1.0
        step = 1.0 / L

    # PGD iterations
    for _ in range(iters):
        # grad = (CtC + λI) u - b
        # tmp = CtC @ u
        tmp = np.zeros(Jg, dtype=np.float32)
        for i in prange(Jg):
            s = 0.0
            Ai = CtC[i]
            for j in range(Jg):
                s += Ai[j] * u[j]
            tmp[i] = s
        for i in prange(Jg):
            g = tmp[i] + lambda_U * u[i] - b[i]
            u[i] = u[i] - step * g
            if u[i] < 0.0:
                u[i] = 0.0
    return u

@njit(parallel=True)
def nnls_tikhonov_columnwise_masked_njit(
    C,         # (J,J) float32
    R,         # (J,K) float32   (R = R0 - P in your outer loop)
    Ex, Eg,    # (J,), (K,) float32
    lambda_U=1e-2,
    iters=200,
    step=-1.0  # <=0 means auto-compute per column
):
    """
    For each gamma column g, solve with support mask Ex>=Eg[g]:
        min_{u[mask]>=0} 0.5|| C[:,mask] u_mask - R[:,g] ||^2 + 0.5*λ||u_mask||^2
    Returns U (J,K), with zeros outside mask.
    """
    J, K = R.shape
    U = np.zeros((J, K), dtype=np.float32)

    for g in prange(K):
        idx = _build_mask_indices(Ex, Eg[g])       # allowed rows (length Jg)
        Jg = idx.shape[0]
        if Jg == 0:
            continue

        CtC, b = _build_CtC_and_b(C, idx, R[:, g])
        # per-column step if requested
        st = step
        if st <= 0.0:
            # estimate L for this masked CtC
            L = _power_iter_max_eig_sym(CtC, 6) + lambda_U
            if L <= 0.0:
                L = 1.0
            st = 1.0 / L

        u = _proj_grad_nnls_tikhonov(CtC, b, lambda_U, iters, st)
        # scatter back into full-size column
        for k in range(Jg):
            U[idx[k], g] = u[k]
        # others remain 0
    return U



# === GAMMA

import numpy as np
from numba import njit

@njit
def apply_row_gains_to_A(A, gamma):
    """
    (Γ A): multiply each row j of A by gamma[j].
    A: (J,K) float32, gamma: (J,) float32  -> (J,K) float32
    """
    J, K = A.shape
    GA = np.empty((J, K), dtype=np.float32)
    for j in range(J):
        gj = gamma[j]
        for g in range(K):
            GA[j, g] = gj * A[j, g]
    return GA

@njit
def CA_matmul(C, X):
    """
    Column-wise multiply: (C @ X) with C(J,J), X(J,K) -> (J,K).
    """
    J, K = X.shape
    Y = np.zeros((J, K), dtype=np.float32)
    for g in range(K):
        for i in range(J):
            s = 0.0
            for j in range(J):
                s += C[i, j] * X[j, g]
            Y[i, g] = s
    return Y

@njit
def build_H_b_for_gamma(C, A, R):
    """
    Build normal-equation pieces for z = gamma-1 >= 0:
      min 0.5 z^T H z - b^T z   (H SPD, b)
    where v_j = C[:,j] A[j,:] (as a (J,K) 'column-image'), and
      H[i,j] = <v_i, v_j>_F = sum_{x,g} C[x,i]*C[x,j]*A[i,g]*A[j,g]
      b[i]   = <v_i, R>_F   = sum_{x,g} C[x,i]*A[i,g]*R[x,g]
    Shapes: C(J,J), A(J,K), R(J,K) -> H(J,J), b(J)
    """
    J, K = A.shape
    H = np.zeros((J, J), dtype=np.float32)
    b = np.zeros(J, dtype=np.float32)

    # Precompute A_row_norms*g products for speed? we do direct loops for njit clarity
    for i in range(J):
        for j in range(J):
            s = 0.0
            for x in range(J):
                cix_cxj = C[x, i] * C[x, j]
                if cix_cxj != 0.0:
                    for g in range(K):
                        s += cix_cxj * A[i, g] * A[j, g]
            H[i, j] = s

    for i in range(J):
        sb = 0.0
        for x in range(J):
            cxi = C[x, i]
            if cxi != 0.0:
                for g in range(K):
                    sb += cxi * A[i, g] * R[x, g]
        b[i] = sb

    return H, b

@njit
def power_iter_max_eig_sym(A, n_iter=12):
    """
    Largest eigenvalue (spectral norm) estimate for symmetric PSD matrix A (J,J).
    """
    J = A.shape[0]
    if J == 0:
        return 0.0
    v = np.full(J, 1.0 / np.sqrt(J), dtype=np.float32)
    for _ in range(n_iter):
        # w = A @ v
        w = np.zeros(J, dtype=np.float32)
        for i in range(J):
            s = 0.0
            Ai = A[i]
            for j in range(J):
                s += Ai[j] * v[j]
            w[i] = s
        # norm
        nrm = 0.0
        for i in range(J):
            nrm += w[i] * w[i]
        nrm = np.sqrt(nrm)
        if nrm == 0.0:
            return 0.0
        for i in range(J):
            v[i] = w[i] / nrm
    # Rayleigh quotient
    tmp = np.zeros(J, dtype=np.float32)
    for i in range(J):
        s = 0.0
        Ai = A[i]
        for j in range(J):
            s += Ai[j] * v[j]
        tmp[i] = s
    rq = 0.0
    for i in range(J):
        rq += v[i] * tmp[i]
    return rq

@njit
def solve_gamma_pg_njit(C, A, P, lambda_gamma, iters, gamma_min, gamma_max):
    """
    Projected-gradient on z = gamma - 1 >= 0 for:
      min 0.5 || R - Σ_j z_j v_j ||_F^2 + 0.5 λ ||z||^2,
    where R = A - P - C @ A  and v_j contribution is C[:,j] * A[j,:].
    Returns gamma (J,), with gamma in [gamma_min, gamma_max], gamma_min >= 1.
    """
    J, K = A.shape

    # R = A - P - C A
    CA = CA_matmul(C, A)
    R = np.empty((J, K), dtype=np.float32)
    for i in range(J):
        for g in range(K):
            R[i, g] = A[i, g] - P[i, g] - CA[i, g]

    # Build H and b for quadratic 0.5 z^T (H + λ I) z - b^T z
    H, b = build_H_b_for_gamma(C, A, R)

    # Add λI to H in the iterations via gradient (we include λ term in grad)
    # Step size <= 1 / (λ_max(H) + λ)
    L = power_iter_max_eig_sym(H, 10) + lambda_gamma
    if L <= 0.0:
        L = 1.0
    step = 1.0 / L

    # PGD on z >= 0
    z = np.zeros(J, dtype=np.float32)
    for _ in range(iters):
        # grad = (H + λI) z - b
        # Hz
        Hz = np.zeros(J, dtype=np.float32)
        for i in range(J):
            s = 0.0
            Hi = H[i]
            for j in range(J):
                s += Hi[j] * z[j]
            Hz[i] = s
        for i in range(J):
            grad = Hz[i] + lambda_gamma * z[i] - b[i]
            zi = z[i] - step * grad
            if zi < 0.0:
                zi = 0.0
            z[i] = zi

    # gamma = 1 + z, then clamp to [gamma_min, gamma_max]
    gamma = np.empty(J, dtype=np.float32)
    for i in range(J):
        gi = 1.0 + z[i]
        if gi < gamma_min:
            gi = gamma_min
        elif gi > gamma_max:
            gi = gamma_max
        gamma[i] = gi
    return gamma



def first_generation_gamma(
    AG: Matrix,
    settings: FGS,
) -> FirstGenerationResult:
    AG = AG.as_numpy()
    FG = zeros_like(AG)
    alphas = np.zeros((settings.iterations, len(AG.Ex)))
    M = multiplicity_estimation(AG) if settings.multiplicity is None else settings.multiplicity
    if settings.population_norm is None:
        N = population_normalization(AG, multiplicity=M, backend=settings.backend)
    else:
        N = settings.population_norm

    tqdm_ = tqdm if not settings.disable_tqdm else lambda x: x

    FG = np.zeros_like(AG)
    FG[AG > 0] = 1
    AG_ = AG.values
    FG_prev = FG
    alpha = np.ones(AG.Ex.size)
    reldiffs = np.zeros(settings.iterations)
    maxdiffs = np.zeros(settings.iterations)
    absdiffs = np.zeros(settings.iterations)
    eta = settings.relaxation

    # --- latent component alternating loop (replaces your current W/G/FG update) ---
    # --- censoring-gamma alternating loop ---

    for it in tqdm_(range(settings.iterations)):
        # 1) Build W from current FG (mask your allowed Eg before summing)
        row_sum = FG.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1.0
        W = FG / row_sum

        # 2) Build C once per outer iter
        C = build_C_matrix(mat_to_numba(AG), W, mat_to_numba(N)).astype(np.float32)

        # 3) Solve for gamma (>=1) with current P=FG
        gamma = solve_gamma_pg_njit(
            C=C,
            A=AG_.astype(np.float32),
            P=FG.astype(np.float32),
            lambda_gamma=np.float32(settings.lambda_gamma),
            iters=settings.pg_iters,
            gamma_min=np.float32(settings.gamma_min),
            gamma_max=np.float32(settings.gamma_max),
        )

        # 4) Recycle with uncensored rows: CA_gamma = C @ (Γ A)
        GA = apply_row_gains_to_A(AG_.astype(np.float32), gamma)   # ΓA
        CA_gamma = CA_matmul(C, GA)                                # C (ΓA)

        # 5) Raw primary update, then project and relax
        P_raw = AG_.astype(np.float32) - CA_gamma

        # support mask: Eg <= Ex
        support = (AG.Eg[None, :].astype(np.float32) <= AG.Ex[:, None].astype(np.float32))
        P_raw = np.where(support, P_raw, 0.0)

        if settings.enforce_nonnegativity:
            P_raw = np.maximum(P_raw, 0.0)

        # optional: per-row target area (mean multiplicity) projection
        # T = (AG.sum(axis=1).values / np.maximum(M.values, 1e-12)).astype(np.float32)
        # s = T / np.maximum(P_raw.sum(axis=1), 1e-12)
        # P_raw *= s[:, None]

        # Relaxed update
        FG_next = (1.0 - eta) * FG + eta * P_raw

        # diagnostics & convergence like you already do ...
        diff = FG_next - FG
        abs_diff = np.abs(diff).sum()
        rel_diff = np.sum(np.abs(diff) / np.maximum(np.abs(FG_next), 1e-12))
        max_diff = np.max(np.abs(diff))
        # ... log & stopping tests ...
        reldiffs[it] = rel_diff
        maxdiffs[it] = max_diff
        absdiffs[it] = abs_diff
        LOG.info(
            "Iteration %d:\n\tabs = %g, rel = %g, max = %g",
            it + 1, abs_diff, rel_diff, max_diff,
        )
        if (settings.rel_tol is not None and rel_diff < settings.rel_tol) or \
           (settings.abs_tol is not None and abs_diff < settings.abs_tol):
            break
        FG = FG_next
    # --- end loop ---

    # --- end latent component alternating loop ---

    
    
    FG = AG.clone(values=FG, name="first generation")
    alpha = Matrix(
        Ex=AG.Ex,
        i=np.arange(len(alphas)),
        values=alphas.T,
        ylabel="iteration",
        Y_unit="",
        xlabel="Ex",
        name="alpha",
    )
    reldiffs = Vector(i=np.arange(len(reldiffs)), values=reldiffs, name="relative differences", vlabel='rel. diff.')
    maxdiffs = Vector(i=np.arange(len(maxdiffs)), values=maxdiffs, name="maximum differences", vlabel='max. diff.')
    absdiffs = Vector(i=np.arange(len(absdiffs)), values=absdiffs, name="absolute differences", vlabel='abs. diff.')
    U = Matrix(Ex=AG.Ex, Ef=AG.Ex, values=AG.values, name="latent component")
    res = FirstGenerationResult(
        AG=AG, FG=FG, U=U, alpha=alpha, settings=settings,
        reldiffs=reldiffs, maxdiffs=maxdiffs, absdiffs=absdiffs
    )
    return res