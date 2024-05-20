from __future__ import annotations
from .responsedata import ResponseData
from .. import Vector, Matrix, Index
from .numbalib import NVector, index, index_mid, lerp
import numpy as np
from collections import OrderedDict
import warnings
from .comptonmatrixprotocol import ComptonMatrix, is_compton_matrix

"""
TODO: Find a good weighting of the backscattering and compton interpolations
    - The final lerp makes sagging curves. Why? Fix.
       + Nevermind, it seems to be a visual artifact when the compton edge changes a lot between two energies.
    - The first bin is always 0. Why?
      + Because of indexing error. Not fixed.
    - The fan-method is slightly incorrect. A fan will pass through more bins in the higher compton
      than the lower, which is not accounted for.
"""

try:
    #raise ImportError
    from numba import njit, int32, float32, float64, prange
    from numba.experimental import jitclass
    from numba.typed import List as NList
    from numba.types import ListType
    NUMPY = True
except ImportError as e:
    NUMPY = False
    warnings.warn("Numba could not be imported. Falling back to non-jiting which will be much slower")
    int32 = np.int32
    float32 = np.float32
    float64 = np.float64
    prange = range

    def nop_decorator(func, *aargs, **kkwargs):
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper

    def nop_nop(*aargs, **kkwargs):
        def decorator(func):
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            return wrapper
        return decorator

    njit = nop_nop #nop_decorator
    jitclass = nop_nop
    NList = list
    ListType = list


spec2 = OrderedDict()
spec2["E"] = float64[::1] if NUMPY else float64
spec2["compton"] = ListType(NVector.class_type.instance_type) if NUMPY else list[NVector]

@jitclass(spec2)
class ComptonList:
    def __init__(self, E, compton):
        assert len(E) == len(compton)
        self.E = E
        self.compton = compton

    def __getitem__(self, i: int) -> tuple[float64, NVector]:
        return self.E[i], self.compton[i]

    def index(self, e: float64) -> int:
        return index_mid(self.E, e)

    def index_closest(self, e: float64) -> tuple[int, int]:
        """ Indices closest to e """
        return index(self.E, e)

    def at(self, i: int) -> ComptonVector:
        return ComptonVector(self.E[i], self.compton[i])

    def closest(self, e: float64) -> tuple[ComptonVector, ComptonVector]:
        # E are mid-binned, but we want the closest two edges. Treating the indices
        # as left-binned is the easiest way to do this.
        i = self.index_closest(e)
        if i == 0:
            return self.at(0), self.at(1)
        elif i == len(self.E) - 1:
            return self.at(-2), self.at(-1)
        else:
            return self.at(i), self.at(i+1)

    def closest_e(self, e: float64) -> tuple[float64, float64]:
        i = self.index_closest(e)
        if i == 0:
            return self.E[0], self.E[1]
        elif i == len(self.E) - 1:
            return self.E[-2], self.E[-1]
        else:
            return self.E[i], self.E[i+1]

    @property
    def compton_E(self) -> np.ndarray:
        return self.compton[0].E_true


spec3 = OrderedDict()
spec3["e"] = float64
spec3["vector"] = NVector.class_type.instance_type if NUMPY else NVector


@jitclass(spec3)
class ComptonVector:
    def __init__(self, e, vector):
        self.e = e
        self.vector = vector

    def __getitem__(self, i):
        return self.vector.__getitem__(i)

    def __setitem__(self, key, value):
        self.vector[key] = value

    def at(self, e):
        return self.vector.at(e)

    def edge(self):
        """ The Compton edge energy """
        return compton_edge(self.e)

    def np(self) -> np.ndarray:
        return self.vector.values

    @property
    def E(self) -> np.ndarray:
        return self.vector.E_true

    def index(self, e: float64) -> int:
        return self.vector.index(e)

    def __len__(self) -> int:
        return len(self.vector)


def interpolate_compton(p: ResponseData, E: Index, sigma , nsigma: float = 6) -> ComptonMatrix:
    """Interpolate Compton probabilities.

    Args:
        p (ResponseData): ResponseData normalized to probabilities
        E (np.ndarray): Energies to interpolate at
        sigma (np.ndarray): The detector resolution at each energy
        nsigma (float, optional): Number of sigmas to interpolate. Defaults to 6.

    Returns:
        Matrix: Interpolated Compton probabilities
    """
    if not p.is_normalized:
        raise ValueError("ResponseData must be normalized")

    if p.E_true[0] > E[0] or p.E_true[-1] < E[-1]:
        raise ValueError("Compton interpolation range out of bounds")
    compton: ComptonList = make_compton_list(p)
    E_observed = p.E_observed
    sigma_ = NVector(E=E_observed, values=sigma(E_observed))
    assert np.allclose(compton.compton_E, sigma.E), "Compton and sigma must be parameterised at the same energies"
    R = _interpolate_compton(compton, E.bins, sigma_, nsigma)
    Eg = np.asarray(compton.compton_E)
    m = Matrix(true=Eg, observed=E, values=R, xlabel=r"Observed $\gamma$", ylabel=r"True $\gamma$")
    assert is_compton_matrix(m)
    return m


# Parallel fails by some reason
@njit(fastmath=True)
def _interpolate_compton(compton: ComptonList, E: np.ndarray, sigma: NVector, nsigma: float) -> np.ndarray:
    """Interpolate Compton probabilities.

    The squares visible at low E are due to numerical precision of the mama format. TODO: Fix?

    Args:
        compton (ComptonList): ComptonList object
        E (np.ndarray): Energies to interpolate at
        sigma (np.ndarray): The detector resolution in sigma at E

    Returns:
        Matrix: Interpolated Compton probabilities
    """
    N = len(E)
    CE = compton.compton_E
    M = len(CE)
    R = np.zeros((N, M))
    # Logistical weight from 0 to 1 and 0.5 at 800 keV
    weight = 1 / (1 + np.exp(-1/100 * (E - 800)))
    for i in prange(N):
        e = E[i]
        # Faster to allocate new vectors in order to use parallel loop
        backscatter = ComptonVector(e, NVector(CE, np.zeros(M)))  # FIXME When index is fixed, use empty
        fan = ComptonVector(e, NVector(CE, np.zeros(M)))
        # The two closest compton spectra at e
        low, high = compton.closest(e)

        compton_lerp_mut(backscatter, low, high)
        fan_mut(fan, low, high)
        lerp_from_edge_mut(fan, low, high, sigma, nsigma)
        for j in prange(M):
            R[i, j] = weight[i] * fan.vector.values[j] + (1 - weight[i]) * backscatter.vector.values[j]
    return R


@njit
def compton_lerp(low: ComptonVector, high: ComptonVector, e: float) -> np.ndarray:
    intp = np.empty_like(low.np())

    t = (e - low.e) / (high.e - low.e)
    for i in range(len(intp)):
        intp[i] = (1.0 - t) * low[i] + t * high[i]
    return intp


@njit(parallel=True, fastmath=True)
def compton_lerp_mut(mid: ComptonVector, low: ComptonVector, high: ComptonVector):
    t = (mid.e - low.e) / (high.e - low.e)
    stop = len(mid)
    for i in prange(stop):
        # These at() are a massive bottleneck. Can't remove them unless all vectors are equally binned
        # No, wait, we *can* remove them because the bad vectors are only longer, but have the same
        # calibration.
        mid[i] = (1.0 - t) * low[i] + t * high[i]


@njit
def backscatter_mut(mid: ComptonVector, low: ComptonVector, high: ComptonVector) -> int:
    """ Interpolates backscatter probabilities for a given energy. Mutates X in place. """
    i = mid.index(backscatter_energy(e))
    if i > 0:
        partial_compton_lerp_mut(mid, low, high)
    return i


def make_compton_list(p: ResponseData) -> ComptonList:
    """ Make a ComptonList from a ResponseData object.

    Args:
        p (ResponseData): ResponseData object

    Returns:
        ComptonList: ComptonList object
    """
    E = p.E_true
    compton = NList()
    for vec in p.compton:
        compton.append(NVector(vec.E_true, vec.values))
    return ComptonList(E, compton)


@njit(parallel=True, fastmath=True)
def fan_mut(mid: ComptonVector, low: ComptonVector, high: ComptonVector):
    """ Interpolate the probabilities for a given energy at equal angle. Mutates R in place. """
    stop = mid.index(mid.edge())

    e = mid.e
    for i in prange(stop):
        ei = mid.E[i]
        if ei < 0.1:
            continue
        # Map the energy at the bin ei to the corresponding angle
        # Inversion of the compton energy equation to find the angle
        r = 1e-14 if e-ei < 1e-14 else e-ei
        z = ei / (e / 511 * r)
        theta = np.arccos(1-z)
        if not (0 < theta < np.pi):
            continue

        # Map the angle to the energy of the corresponding bin in the low and high
        # compton spectra
        plow = low.at(electron_energy(low.e, theta))
        phigh = high.at(electron_energy(high.e, theta))

        # Linearly interpolate the probability. The interpolation is 2D,
        # so is weighted by the taylor expansion. The dE/dscatter is the same,
        # while the dE/dtheta is different.
        plow *= dEdtheta(low.e, theta)
        phigh *= dEdtheta(high.e, theta)
        intp = lerp(e, low.e, high.e, plow, phigh)

        mid[i] = intp / dEdtheta(e, theta)


@njit(parallel=True, fastmath=True)
def lerp_from_edge_mut(mid: ComptonVector, low: ComptonVector, high: ComptonVector,
                       sigma: NVector, nsigma: float) -> None:
    """ Interpolate the probabilities for a given energy at equal angle. Mutates R in place. """
    edge_low = low.edge()
    edge_mid = mid.edge()
    edge_high = high.edge()

    # FIXME Bug? Should it be edge_low + ...
    # stop_low = low.e + nsigma*sigma.at(edge_low)
    # stop_mid = mid.e + nsigma*sigma.at(edge_mid)
    # stop_high = high.e + nsigma*sigma.at(edge_high)

    stop_low = edge_low + nsigma*sigma.at(edge_low)
    stop_mid = edge_mid + nsigma*sigma.at(edge_mid)
    stop_high = edge_high + nsigma*sigma.at(edge_high)
    # Stop at the end or 10% higher than the high compton. High compton is of course always higher, than mid,
    # but to be on the safe side we go a bit higher in case of higher laying structures.
    #stop = low.index(min(low.E[-1], 1.1*stop_high))
    #print(low.e, stop)
    #stop = min(len(mid), mid.index(stop_mid))  # FIXME. Just a test
    start = mid.index(edge_mid)
    stop = len(mid)

    # Partially precompute the weighting lerp
    t = (mid.e - low.e) / (high.e - low.e)
    wlerp = lambda low, high: (1.0 - t) * low + t * high

    for i in prange(start, stop):
        eg = mid.E[i]
        # TODO Lerps share common t
        elow = lerp(eg, edge_mid, stop_mid, edge_low, stop_low)
        ehigh = lerp(eg, edge_mid, stop_mid, edge_high, stop_high)
        mid[i] = wlerp(low.at(elow), high.at(ehigh))


def test_compton_lerp(p: ResponseData, e: float) -> tuple[Vector, Vector, Vector]:
    compton = make_compton_list(p)
    low, high = compton.closest(e)
    intp = compton_lerp(low, high, e)

    lowv = Vector(E=low.E, values=low.np())
    highv = Vector(E=low.E, values=high.np())
    intpv = Vector(E=low.E, values=intp)
    return intpv, lowv, highv

def test_compton_fan(p: ResponseData, e: float) -> tuple[Vector, Vector, Vector]:
    compton = make_compton_list(p)
    low, high = compton.closest(e)
    R = np.zeros_like(low.np())
    fan_mut(low, high, e, R, 0, len(R))

    lowv = Vector(E=low.E, values=low.np())
    highv = Vector(E=low.E, values=high.np())
    intpv = Vector(E=low.E, values=R)
    return intpv, lowv, highv


def test_lerp_end(p: ResponseData, intp: 'DiscreteInterpolation', e: float, nsigma=6) -> tuple[Vector, Vector, Vector]:
    compton = make_compton_list(p)
    low, high = compton.closest(e)
    R = np.zeros_like(low.np())
    sigmas = intp.sigma(compton.compton_E)
    lerp_from_edge_mut(low, high, sigmas, 6, e, R)

    lowv = Vector(E=low.E, values=low.np())
    highv = Vector(E=low.E, values=high.np())
    intpv = Vector(E=low.E, values=R)
    return intpv, lowv, highv

def test_whole(p: ResponseData, intp: 'DiscreteInterpolation', e: float, nsigma=6) -> tuple[Vector, Vector, Vector]:
    compton = make_compton_list(p)
    low, high = compton.closest(e)
    R = np.zeros_like(low.np())
    sigmas = intp.sigma(compton.compton_E)

    emax = e + 6*sigmas[low.index(e)]
    imax = low.index(emax)

    A = np.zeros_like(R)
    B = np.zeros_like(R)
    weight = 1 / (1 + np.exp(-1/50 * (low.E - 200)))
    partial_compton_lerp_mut(low, high, e, A)
    backscatter_e = backscatter_energy(e)
    fan_mut(low, high, e, B, 0, imax)
    lerp_from_edge_mut(low, high, sigmas, nsigma, e, B)
    R = weight * B + (1-weight) * A
    # CUTOFF = 800
    #
    # if e <= CUTOFF:
    #     #backscatter_mut(low, high, e, R[i, :])
    #     partial_compton_lerp_mut(low, high, e, R)
    # else:
    #     backscatter_end = backscatter_mut(low, high, e, R)
    #     # Interpolate from the backscatter energy to compton edge using the fan method
    #     fan_mut(low, high, e, R, backscatter_end, imax)
    #     # The fan method relies on interpolating along equal angles. This is only
    #     # possible below the compton edge, as the angle is undefined above the edge.
    #     # The rest is interpolated by a weighted average of the two closest spectra.
    #     lerp_from_edge_mut(low, high, sigmas, nsigma, e, R)

    lowv = Vector(E=low.E, values=low.np())
    highv = Vector(E=low.E, values=high.np())
    intpv = Vector(E=low.E, values=R)
    return intpv, lowv, highv


@njit(fastmath=True)
def electron_energy(Eg: float, theta: float) -> float:
    """
    Calculates the energy of an electron that is scattered an angle
    theta by a gamma-ray of energy Eg.

    Note:
        For `Eg <= 0.1` it returns `Eg`. (workaround)

    Args:
        Eg: Energy of incident gamma-ray in keV
        theta: Angle of scatter in radians

    Returns:
        Energy Ee of scattered electron
    """
    Eg_scattered = Eg / (1 + Eg / 511 * (1 - np.cos(theta)))
    electron = Eg - Eg_scattered
    return np.where(Eg > 0.1, electron, Eg)


@njit(fastmath=True)
def compton_edge(e: float) -> float:
    """ The Compton edge energy. Same as electron_energy(e, π) """
    if e < 0.1:
        return e
    #scattered = e / (1 + e / 511 * 2)
    #return e - scattered
    return 2*e**2 / (2*e + 511)


@njit(fastmath=True)
def backscatter_energy(e: float) -> float:
    """ The backscatter energy """
    return e - compton_edge(e)


@njit(fastmath=True)
def dEdtheta(Eg: float, theta: float) -> float:
    """
    Derivative of electron energy with respect to theta.

    Args:
        Eg: Energy of gamma-ray in keV
        theta: Angle of scatter in radians

    Returns:
        TYPE: dEdtheta
    """
    a = (Eg * Eg / 511 * np.sin(theta))
    b = (1 + Eg / 511 * (1 - np.cos(theta))) ** 2
    return a / b

@njit(parallel=True)
def compton_angle(e, ei):
    r = 1e-14 if e-ei < 1e-14 else e-ei
    z = ei / (e / 511 * r)
    theta = np.arccos(1-z)
    if not (0 < theta < np.pi):
        return np.nan
    return theta


