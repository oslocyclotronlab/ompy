from __future__ import annotations

from ompy.response.compton import dEdtheta
from . import ResponseData, ResponseInterpolation
from .. import Vector, Matrix
import warnings
from numba import cuda
from numba import njit, prange
import numba as nb
import numpy as np
import math
from nptyping import NDArray, Shape
from typing import TypeAlias, Callable

# There are a lot of shapes to get right. These are an attempt
# to make it harder to get lost in the code
DTYPE: TypeAlias = np.float32
# 'Observed' is the dimension of the observed Eg in the raw compton spectra
VO: TypeAlias = NDArray[Shape['Observed'], DTYPE]
# 'True' is the dimension of the true Eg in the raw compton spectra
VT: TypeAlias = NDArray[Shape['True'], DTYPE]
# 'E' is the dimension of the user requested energies
VE: TypeAlias = NDArray[Shape['E'], DTYPE]
MEO: TypeAlias = NDArray[Shape['E, Observed'], DTYPE]
MTO: TypeAlias = NDArray[Shape['True, Observed'], DTYPE]
MEO2: TypeAlias = NDArray[Shape['E, Observed, 2'], DTYPE]
vector: TypeAlias = NDArray[Shape['*'], DTYPE]
matrix: TypeAlias = NDArray[Shape['*, *'], DTYPE]
MAT: TypeAlias = NDArray
VEC: TypeAlias = NDArray
D: TypeAlias = DTYPE
S: TypeAlias = Shape

Elementwise = Callable[[float], float] | Callable[[vector], vector]


"""
TODO:
- The weird wavy thing is already in the backscatter lerp(?)
- [ ] Add Docstrings

Note for readers: This is identical to the code in `compton.py` but translated to
the GPU. It is significantly harder to read as the GPU does not like any abstractions, and
indices and energies are precomputed all over the place.

Algorithm:
    - For GPU precompute [N = len(E true), M = len(E measured)]:
      - The compton edge as 1xN
      - Backscattering 1xN
      - theta matrix N x M upper diagonal
      - scattered matrix from theta NxM
      - Fanlow NxM
      - Fanhigh NxM
      - de/dtheta matrices NxM
      - fan lerped NxM
"""

def interpolate_gpu(p: ResponseData, E: np.ndarray,
                    sigma: Elementwise, nsigma: int = 6) -> Matrix:
    """Interpolate Compton probabilities.

    Args:
        p (ResponseData): ResponseData normalized to probabilities
        E (np.ndarray): Energies to interpolate at
        sigma (np.ndarray): The detector resolution at each energy
        nsigma (int, optional): Number of sigmas to interpolate. Defaults to 6.

    Returns:
        Matrix: Interpolated Compton probabilities
    """
    if p.E[0] > E[0] or p.E[-1] < E[-1]:
        raise ValueError("Compton interpolation range out of bounds")
    compton: Matrix = p.compton_matrix()
    E_observed: VO = compton.Eg
    E_true: VT = compton.Ex
    sigma: VO = sigma(E_observed)

    assert len(E) < len(E_observed)
    #assert E[0] < E_observed[0]
    #assert E[-1] > E_observed[-1]

    R = _interpolate(compton.values, E, E_observed, E_true, sigma, nsigma)
    if R.shape[0] == len(E_true):
        A = E_true
    else:
        A = E

    if R.ndim == 1:
        return Vector(E=A, values=R)
    elif R.ndim == 2:
        return Matrix(Eg=E_observed, Ex=A, values=R, xlabel=r"Observed $\gamma$", ylabel=r"True $\gamma$")
    else:
        raise ValueError("Invalid number of dimensions")



def _interpolate(compton: MTO, E: VE, E_observed: VO, E_true: VT,
                 sigma: VO, nsigma: int) -> np.ndarray:
    N = len(E)
    M = len(E_observed)
    Dtype = np.float64
    NPtype = np.float64
    Inttype = np.int32
    # TODO Use cuda stream
    # TODO Use nvprof
    stream = cuda.stream()
    E_to_T: VE = find_closest(E, E_true)
    d_E_to_T = cuda.to_device(E_to_T.astype(Inttype), stream=stream)
    edges_E = edge(E)
    stop_E = edge_thickness(E, edges_E, E_observed, sigma, nsigma)
    edges_T = edge(E_true)
    stop_T = edge_thickness(E_true, edges_T, E_observed, sigma, nsigma)

    d_edges_E = cuda.to_device(edges_E.astype(NPtype), stream=stream)
    d_stop_E = cuda.to_device(stop_E.astype(NPtype), stream=stream)
    d_edges_T = cuda.to_device(edges_T.astype(NPtype), stream=stream)
    d_stop_T = cuda.to_device(stop_T.astype(NPtype), stream=stream)

    c_weight = 1 / (1 + np.exp(-1/100 * (E - 800)))
    d_weight = cuda.to_device(c_weight.astype(NPtype), stream=stream)

    d_E = cuda.to_device(E.astype(NPtype), stream=stream)
    d_Eo = cuda.to_device(E_observed.astype(NPtype), stream=stream)
    d_Et = cuda.to_device(E_true.astype(NPtype), stream=stream)
    d_angle_E = cuda.device_array((N, M), dtype=Dtype, stream=stream)
    d_lerp = cuda.device_array((N, M), dtype=Dtype, stream=stream)
    d_compton = cuda.to_device(compton.astype(NPtype), stream=stream)
    d_dEdtheta_E = cuda.device_array((N, M), dtype=Dtype, stream=stream)
    d_dEdtheta_EO2 = cuda.device_array((N, M, 2), dtype=Dtype, stream=stream)
    d_unscattered_map = cuda.device_array((N, M, 2), dtype=Inttype, stream=stream)
    d_fan = cuda.device_array((N, M), dtype=Dtype, stream=stream)

    threads_per_block = (16, 16)
    blks_p_grid = distribute(threads_per_block, (N, M))
    DEO = (blks_p_grid, threads_per_block, stream)
    thds_p_blk_3D = (8, 8, 2)
    blks_p_grid_3D = distribute(thds_p_blk_3D, (N, M, 2))
    DEO2 = (blks_p_grid_3D, thds_p_blk_3D, stream)

    # Precompute intermediate matrices
    # Angles corresponding to (E incident, E observed)
    # fast
    with cuda.defer_cleanup():
        angle[DEO](d_angle_E, d_E, d_Eo)
        # The derivative of the energy loss with respect to the angle for E incident
        # fast
        dEdtheta[DEO](d_dEdtheta_E, d_E, d_angle_E)
        # The derivative of the energy loss with respect to the angle for neighboring E incident
        dEdtheta_into[DEO2](d_dEdtheta_EO2, d_Et, d_angle_E, E_to_T)
        unscattered_into[DEO2](d_unscattered_map, d_Et, d_Eo, d_angle_E, d_E_to_T)

        lerp_2[DEO](d_lerp, d_E, d_Et, d_E_to_T, d_compton)
        fan[DEO](d_fan, d_E, d_Et, d_unscattered_map,
                 d_E_to_T, d_dEdtheta_E, d_dEdtheta_EO2,
                 d_compton)

        lerp_from_edge[DEO](d_fan, d_E, d_Eo, d_Et,
                            d_E_to_T, d_edges_E, d_edges_T,
                            d_stop_E, d_stop_T, d_compton)
        #total_fan[DEO](d_fan, d_E, d_Et, d_Eo,
        #               d_edges_E, d_edges_T, d_stop_E, d_stop_T,
        #               d_unscattered_map, d_E_to_T, d_dEdtheta_E, d_dEdtheta_EO2, d_compton)

        weight[DEO](d_fan, d_fan, d_lerp, d_weight)
        h_fan = d_fan.copy_to_host()

    return h_fan


@njit(parallel=True)
def edge(E: vector) -> vector:
    X = np.empty_like(E)
    for i in prange(len(X)):
        e = E[i]
        if e < 0.1:
            X[i] = e
        else:
            X[i] = 2*e**2 / (2*e + 511)
    return X


@njit(parallel=True)
def edge_thickness(E: vector, edges: vector, E_sigma, sigma: vector, nsigma: float) -> vector:
    X = np.empty_like(E)
    for i in prange(len(X)):
        edge = edges[i]
        j = index_cpu(E_sigma, edge)  # TODO Can be improved from O(n^2) to O(n), but not in prange
        stop = edge + nsigma*sigma[j]
        X[i] = stop
    return X


@njit
def index_cpu(E: vector, e: float, start=0) -> int:
    i = start
    while i < len(E) and E[i] < e:
        i += 1
    return i - 1


@cuda.jit
def backscatter(out: VE, E: VE, edge: VE) -> None:
    i = cuda.grid(1)
    if i < out.size:
        out[i] = E[i] - edge[i]


@cuda.jit
def angle(out: MEO, E: VE, Eo: VO) -> None:
    """ Angle between the incident and scattered photon

    For incident photon energy E and observed photon energy Eo, the angle between
    them is given by
         θ = arccos(1 - out / (in / 511keV * (in - out)))

    """
    i, j = cuda.grid(2)
    n, m = out.shape
    if i < n and j < m:
        e = E[i]
        eo = Eo[j]
        d = e - eo
        r = 1e-14 if d < 1e-14 else d
        z = eo / (e / 511 * r)
        theta = math.acos(1 - z)
        if 0 < theta < np.pi:
            out[i, j] = theta
        else:
            out[i, j] = 0.0


@cuda.jit
def lerp(out: MEO, E: VE, Et: VT, compton: MTO) -> None:
    i, j = cuda.grid(2)
    n, m = out.shape
    if i < n and j < m:
        e = E[i]
        # Find the closest energy in the compton energy
        for k in range(len(Et)):
            if Et[k] > e:
                low = k - 1
                high = k
                break
        t = (e - Et[low]) / (Et[high] - Et[low])
        out[i, j] = (1 - t) * compton[low, j] + t * compton[high, j]

@cuda.jit
def lerp_2(out, E, Ec, neighbours, compton):
    i, j = cuda.grid(2)
    n, m = out.shape
    if i < n and j < m:
        # Find the closest energy in the compton energy
        #for k in range(len(Ec)):
        #    if Ec[k] > et:
        #        low = k - 1
        #        high = k
        #        break
        low = neighbours[i]
        high = low + 1
        t = (E[i] - Ec[low]) / (Ec[high] - Ec[low])
        out[i, j] = (1 - t) * compton[low, j] + t * compton[high, j]
@njit
def find_closest(X: VEC[S['M'], DTYPE], Y: VEC[S['N'], DTYPE]) -> VEC[S['M'], int]:
    """ Find the closest value in Y for each value in X

    Assumes that X and Y are sorted in ascending order and
    that Y contains point values, i.e. "mid binned" while X is
    left binned.
     """
    x = np.empty_like(X, dtype=np.int32)
    i = 0
    j = 0
    while i < len(x):
        e = X[i]
        while j < len(Y):
            if Y[j] > e:
                j -= 1
                x[i] = j
                break
            j += 1
        i += 1
    return x


@njit
def bidirectional_maps(X: VEC[S['M'], DTYPE], Y: VEC[S['N']]) -> tuple[VEC[S['M'], int], VEC[S['N'], int]]:
    """ Find the closest value in Y for each value in X """
    # len(X) > len(Y) (?)

    assert len(X) > len(Y)
    assert X[0] > Y[0]
    assert X[-1] < Y[-1]
    A = np.empty_like(X, dtype=np.int32)
    B = np.empty_like(Y, dtype=np.int32)
    i = 0
    j = 0
    k = -1  # To keep track of filled elements in B
    # TODO CHECK AND FIX ME
    while i < len(X):
        x = X[i]
        while j < len(Y):
            if Y[j] > x:
                j = max(0, j-1)
                A[i] = j
                if j > k:
                    B[j] = i
                    k += 1
                break
            j += 1
        i += 1
    while j < len(Y):
        B[j] = -1 # Sentinel value for unmapped values
        j += 1
    return A, B


@cuda.jit
def dEdtheta(out: MEO, E: VE, angle: MEO):
    """ The derivative of the Compton energy with respect to the angle """
    i, j = cuda.grid(2)
    n, m = out.shape
    if i < n and j < m:
        e = E[i]
        theta = angle[i, j]
        if theta > 0:
            a = (e*e / 511 *math.sin(theta))
            b = (1 + e/511 * (1 - math.cos(theta)))**2
            out[i, j] = a / b
        else:
            out[i, j] = 0.0

@cuda.jit
def dEdtheta_into(out: MEO2, Et: VT, angle: MEO, E_to_T: VE):
    """ The derivative of the Compton energy with respect to the angle """
    i, j, k = cuda.grid(3)
    n, m, _ = out.shape
    if i < n and j < m and k < 2:
        # i_t is low if k = 0 and high if k = 1
        i_t = E_to_T[i] + k
        et = Et[i_t]
        theta = angle[i, j]
        if theta > 0:
            a = (et*et / 511 *math.sin(theta))
            b = (1 + et/511 * (1 - math.cos(theta)))**2
            out[i, j, k] = a / b
        else:
            out[i, j, k] = 0.0
@cuda.jit
def unscattered(out: MAT[S['N, M'], D], E: VEC[S['N'], D], angle: MAT[S['N, M'], D]):
    i, j = cuda.grid(2)
    n, m = out.shape
    if i < n and j < m:
        e = E[i]
        theta = angle[i, j]
        if theta > 0:
            scattered = e / (1 + e/511 * (1 - math.cos(theta)))
            out[i, j] = e - scattered
        else:
            out[i, j] = 0.0

@cuda.jit
def unscattered_into(out: MEO2, Et: VT, Eo: VO, angle: MEO, E_to_T: VE) -> None:
    i, j, k = cuda.grid(3)
    n, m, _ = out.shape
    if i < n and j < m and k < 2:
        # Lower neighbor if k=0, upper neighbor if k=1
        i_t = E_to_T[i] + k
        # Incident energy
        et = Et[i_t]
        # Scattering angle is set by the mid energy
        # j is implicitly the energy of the scattered photon
        theta = angle[i, j]
        if theta > 0:
            scattered = et / (1 + et/511 * (1 - math.cos(theta)))
            out[i, j, k] = index(Eo, et - scattered)
        else:
            out[i, j, k] = -1  # Sentinel value


@cuda.jit(device=True)
def _index(index: vector, x) -> int:
    """ Find the position of `x` in `index` """
    i = 0
    while i < len(index):
        if index[i] > x:
            return i - 1
        i += 1
    if x >= index[-1]:
        return -1  # Sentinel value
    return len(index) - 1

@cuda.jit(device=True)
def index(X: vector, x) -> int:
    """ Uses binary search. Muuuch faster! O(log n) """
    if x < X[0] or x > X[-1]:
        return -1
    # Perform binary search
    left = 0
    right = len(X) - 1
    while left <= right:
        mid = (left + right) // 2
        if x < X[mid]:
            #if x > X[mid + 1]:
            #    return mid
            right = mid - 1
        elif x > X[mid]:
            if x < X[mid+1]:
                return mid
            left = mid + 1
        else:
            return mid

    # x is not in X
    return -1


@cuda.jit(device=True)
def ___index(X, x):
    # O(loglog(n)) in worst case, O(n) in average case.
    # about equal as binary search in practice
    # Check if x is in X
    if x < X[0] or x > X[-1]:
        return -1

    # Perform interpolation search
    low = 0
    high = len(X) - 1
    while low <= high and x >= X[low] and x <= X[high]:
        # Calculate the interpolation index
        index = low + int(((float(high - low) / (X[high] - X[low])) * (x - X[low])))

        # Check if x is at the interpolation index
        if x == X[index]:
            return index
        # If x is greater, search the right half
        elif x > X[index]:
            low = index + 1
        # If x is smaller, search the left half
        else:
            high = index - 1

    # x is not in X
    return -1

@cuda.jit(device=True)
def index_from(index: vector, x, start) -> int:
    """ Find the position of `x` in `index` """
    i = start
    while i < len(index):
        if index[i] > x:
            return i - 1
        i += 1
    if x >= index[-1]:
        return -1  # Sentinel value
    return len(index) - 1

@cuda.jit
def to_indices(out: MAT[S['N, M'], int], X: MAT[S['N, M'], D], Y: VEC[S['M'], D]):
    i, j = cuda.grid(2)
    n, m = out.shape
    if i < n and j < m:
        x = X[i, j]
        out[i, j] = index(Y, x)

    # T x O, value index_O

@cuda.jit
def fan(out: MEO, E: VE, E_true: VT,
        unscattered_map: MEO2, E_to_T: VE,
        dEdtheta_E: MEO, dEdtheta_EO2: MEO2, compton: MTO) -> None:
    i, j = cuda.grid(2)
    n, m = out.shape
    if i < n and j < m:
        low = E_to_T[i]
        high = low + 1

        j_low = unscattered_map[i, j, 0]
        j_high = unscattered_map[i, j, 1]

        p_low = compton[low, j_low] * dEdtheta_EO2[i, j, 0]
        p_high = compton[high, j_high] * dEdtheta_EO2[i, j, 1]

        e = E[i]
        et_low = E_true[low]
        et_high = E_true[high]
        # lerp
        t = (e - et_low) / (et_high - et_low)
        p = (1 - t) * p_low + t * p_high
        out[i, j] = p / dEdtheta_E[i, j]


@cuda.jit
def lerp_from_edge(out: MEO, E: VE, E_observed: VO, E_true: VT,
                   E_to_T: VE,
                   edges_E: VE, edges_T: VT, stop_E: VE, stop_T: VT, compton: MTO) -> None:
    i, j = cuda.grid(2)
    n, m = out.shape
    if i < n and j < m:
        eo = E_observed[j]
        edge_mid = edges_E[i]
        if eo < edge_mid:
            return

        low = E_to_T[i]
        high = low + 1

        edge_low = edges_T[low]
        edge_high = edges_T[high]

        stop_mid = stop_E[i]
        stop_low = stop_T[low]
        stop_high = stop_T[high]

        t = (eo - edge_mid) / (stop_mid - edge_mid)
        e_intp_low = (1 - t) * edge_low + t * stop_low
        e_intp_high = (1 - t) * edge_high + t * stop_high
        j_intp_low = index(E_observed, e_intp_low)
        #if j_intp_low < 0:
        #    return
        j_intp_high = index_from(E_observed, e_intp_high, j_intp_low)
        #if j_intp_high < 0:
        #    return
        counts_low = compton[low, j_intp_low]
        counts_high = compton[high, j_intp_high]

        e_mid = E[i]
        e_low = E_true[low]
        e_high = E_true[high]
        t = (e_mid - e_low) / (e_high - e_low)
        out[i, j] = (1 - t) * counts_low + t * counts_high



# @cuda.jit
# def total_fan(out: MEO, E: VE, E_true: VT, E_observed: VO,
#               edges_E: VE, edges_T: VT, stop_E: VE, stop_T: VT,
#               unscattered_map: MEO2, E_to_T: VE,
#               dEdtheta_E: MEO, dEdtheta_EO2: MEO2, compton: MTO) -> None:
#     """ Combines fan and lerp_to_edge, but speed is identical ""'
#     i, j = cuda.grid(2)
#     n, m = out.shape
#     if i < n and j < m:
#         low = E_to_T[i]
#         high = low + 1
#
#         e_mid = E[i]
#         e_low = E_true[low]
#         e_high = E_true[high]
#         t = (e_mid - e_low) / (e_high - e_low)
#
#         eo = E_observed[j]
#         edge_mid = edges_E[i]
#
#         if eo < edge_mid:
#             j_low = unscattered_map[i, j, 0]
#             j_high = unscattered_map[i, j, 1]
#
#             p_low = compton[low, j_low] * dEdtheta_EO2[i, j, 0]
#             p_high = compton[high, j_high] * dEdtheta_EO2[i, j, 1]
#
#             # lerp
#             p = (1 - t) * p_low + t * p_high
#             out[i, j] = p / dEdtheta_E[i, j]
#         else:
#             edge_low = edges_T[low]
#             edge_high = edges_T[high]
#
#             stop_mid = stop_E[i]
#             stop_low = stop_T[low]
#             stop_high = stop_T[high]
#
#             t = (eo - edge_mid) / (stop_mid - edge_mid)
#             e_intp_low = (1 - t) * edge_low + t * stop_low
#             e_intp_high = (1 - t) * edge_high + t * stop_high
#             j_intp_low = index(E_observed, e_intp_low)
#             j_intp_high = index_from(E_observed, e_intp_high, j_intp_low)
#             counts_low = compton[low, j_intp_low]
#             counts_high = compton[high, j_intp_high]
#
#             out[i, j] = (1 - t) * counts_low + t * counts_high

def distribute(threads_per_block, dim) -> tuple[int,...]:
    """ Distribute threads over the given dimension """
    assert len(dim) == len(threads_per_block)
    A = np.asarray(threads_per_block)
    B = np.asarray(dim)
    return tuple(np.ceil(B/A).astype(np.int32))

@cuda.jit
def weight(out: MEO, fan: MEO, lerp: MEO, weight: VE) -> None:
    i, j = cuda.grid(2)
    n, m = fan.shape
    if i < n and j < m:
        out[i, j] = weight[i]*fan[i, j] + (1 - weight[i])*lerp[i, j]