import numpy as np
from .unfolder import Unfolder, UnfoldedResult1DSimple
from .. import Matrix, Vector
from ..stubs import Axes
from ..numbalib import njit, prange, objmode
import time
from iminuit import Minuit
from .loss import loss_factory, print_minuit_convergence
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
from scipy.optimize import OptimizeResult
from scipy.fft import ifft


@dataclass
class FourierResult1D(UnfoldedResult1DSimple):
    nparams: int
    frequencies: np.ndarray
    A0: float
    An: np.ndarray
    Bn: np.ndarray
    res: Minuit 
    minuit_kwargs: dict[str, any] = field(default_factory=dict)

    def plot_fourier(self, ax: Axes | None = None, **kwargs):
        """Plot the Fourier coefficients.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            The axes to plot on, by default None
        """
        if ax is None:
            fig, ax = plt.subplots()
        ax.step(self.frequencies, self.An, label='An', **kwargs)
        ax.step(self.frequencies, self.Bn, label='Bn', **kwargs)
        return ax


class Fourier(Unfolder):
    def __init__(self, R: Matrix, G: Matrix, nparams: int = 100):
        super().__init__(R, G)
        self.nparams = nparams

    def _unfold_vector(self, R: Matrix, data: Vector, initial: Vector, **kwargs) -> FourierResult1D:
        """Unfold a 1D vector using the Fourier method.

        Parameters
        ----------
        R : Matrix
            The response matrix.
        data : Vector
            The data vector.
        initial : Vector
            The initial guess for the unfolded spectrum.

        Returns
        -------
        FourierResult1D
            The result of the unfolding.
        """
        nparams = kwargs.get("nparams", self.nparams)
        res = fourier(R, data, initial, nparams)
        return res



def fourier(R: Matrix, raw: Vector, initial: Vector, nparams: int) -> FourierResult1D:
    """Unfold a 1D vector using the Fourier method.

    Parameters
    ----------
    R : Matrix
        The response matrix.
    raw : Vector
        The data vector.
    initial : Vector
        The initial guess for the unfolded spectrum.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The unfolded spectrum and the cost function.
    """
    assert nparams > 1, "nparams must be greater than 1"
    u, freq, A0, An, Bn, res = fourier_(R.values, raw, raw.X, initial.values, nparams)
    return FourierResult1D(R, raw, initial, u, nparams, freq, A0, An, Bn, res, {})


def fourier_(R: np.ndarray, raw: Vector, E: np.ndarray, initial: np.ndarray, nparams: int,
             verbose: bool = True) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray, Minuit]:
    """Unfold a 1D vector using the Fourier method.

    TODO: Add n peaks.
    TODO: Reparameterize space to fix negative / very small values.
    TODO: Ignore 511

    Parameters
    ----------
    R : np.ndarray
        The response matrix.
    raw : Vector
        The data vector.
    initial : np.ndarray
        The initial guess for the unfolded spectrum.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The unfolded spectrum and the cost function.
    """
    N = 100
    freq = np.arange(nparams) * 2 * np.pi / (E[-1] - E[0])
    A0 = np.mean(raw)
    A = np.zeros(nparams)
    B = np.zeros(nparams)
    #ABbounds = (0, 1e3*np.max(raw))
    #ABbounds = [ABbounds for _ in range(2*nparams+1)]
    #bounds = ABbounds


    loss_ = loss_factory('loglike', R, raw, 'll3', ignore_511=True)
    raw = raw.values
    cossin = wavesum_fast_initial(freq, E)
    #out = np.zeros_like(raw)

    @njit(parallel=True, fastmath=True)
    def lossfn(p):
        #out = wavesum_fast(p, cossin)
        out = p@cossin
        if np.any(out < 0):
            return 1e10
        return loss_(out)


    p0 = np.concatenate((np.array([A0]), A, B))
    #p0 = np.concatenate((np.array([A0]), A))
    m = Minuit(lossfn, p0)
    m.errordef = Minuit.LIKELIHOOD
    #m.limits = bounds
    ret = m.migrad(iterate=100)
    #ret2 = m.hesse()
    print_minuit_convergence(m)
    p = np.array(m.values)
    k = nparams+1
    u = wavesum_fast(p, cossin)
    #u = wavesum_sin(p[0], p[1:], freq, E)
    return u, freq, p[0], p[1:k], p[k:], ret


@njit
def wavesumc(C, freq, x):
    # wrong
    out = np.zeros_like(x)
    for i in range(len(C)):
        out += C[i] * np.exp(1j * freq[i] * x)
    return out

@njit
def wavesum_sin(A0, A, freq, x):
    n = len(A)
    out = np.zeros_like(x) + A0
    for i in range(n):
        out += A[i] * np.sin(freq[i] * x)
    return out

#@njit
def wavesum_fast_initial(freq: np.ndarray, x: np.ndarray) -> np.ndarray:
    cos = np.cos(np.outer(freq, x))
    sin = np.sin(np.outer(freq, x))
    cosin = np.concatenate((cos, sin))
    mean = np.zeros(len(x)) + 1.0
    return np.insert(cosin, 0, mean, axis=0)

@njit(parallel=True, fastmath=True)
def wavesum_fast(p: np.ndarray, cossin: np.ndarray) -> np.ndarray:
    return p@cossin


@njit
def wavesum(A0: float, A: np.ndarray, B: np.ndarray, freq: np.ndarray, x: np.ndarray) -> np.ndarray:
    n = len(A)
    out = np.zeros_like(x) + A0
    for i in range(n):
        out += A[i] * np.cos(freq[i] * x) + B[i] * np.sin(freq[i] * x)
    return out
