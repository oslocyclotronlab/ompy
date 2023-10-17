from .. import njit, prange
from .index_fn import _index_left, is_monotone_uniform, are_congruent, is_close, is_monotone
import numpy as np
from warnings import warn
from typing import TypeAlias, Literal

"""
TODO:
-[ ] Add tests
-[ ] Add docstrings
-[ ] Severe bug in how 2D rebinning handles incongruent arrays, results in a shift.
"""

@njit()
def overlap(Astart, Aend, Bstart, Bend):
    start = max(Astart, Bstart)
    stop = min(Aend, Bend)
    return max(0.0, stop - start)


@njit
def fit_into(old: np.ndarray, new: np.ndarray, values: np.ndarray) -> np.ndarray:
    # Assumes congruent, overlapping, left bins
    overlp = overlap(old[0], old[-1], new[0], new[-1])
    if overlap == 0:
        raise ValueError("Arrays do not overlap.")
    rebinned = np.zeros_like(new)
    # Find overlap indices
    start_o = 0 if new[0] <= old[0] else _index_left(old, new[0])
    stop_o = -1 if new[-1] >= old[-1] else _index_left(old, new[-1])
    start_n = 0 if new[0] > old[0] else _index_left(new, old[0])
    stop_n = -1 if new[-1] < old[-1] else _index_left(new, old[-1])
    print(start_o, stop_o)
    print(start_n, stop_n)
    assert stop_o - start_o == stop_n - start_n
    rebinned[start_n:stop_n] = values[start_o:stop_o]
    return rebinned


Preserve: TypeAlias = Literal['counts', 'area']

def rebin_uniform_left_left(old: np.ndarray, new: np.ndarray, values: np.ndarray, preserve: Preserve = 'counts'):
    if not is_monotone_uniform(old):
        raise ValueError("Old bins are not monotone uniform.")
    if not is_monotone_uniform(new):
        raise ValueError("New bins are not monotone uniform.")
    return _rebin_uniform_left_left(old, new, values, preserve)


def _rebin_uniform_left_left(old: np.ndarray, new: np.ndarray, values: np.ndarray, preserve: Preserve = 'counts'):
    if len(old) == len(new) and np.allclose(old, new):
        # No rebinning
        return values
    # Case for when the rebinning is a simple shift
    if are_congruent(old, new) and False:
        return fit_into(old, new, values)

    dOld = old[1] - old[0]
    dNew = new[1] - new[0]
    if dNew < dOld:
        raise ValueError(f"Rebinning to smaller binwidth is ill defined and not supported: {dNew} < {dOld}")
    if not is_close(round(dNew / dOld), dNew / dOld):
        warn("The new step size is not an integral multiple of the old. Induces numerical inaccuracies.")

    rebinned = np.zeros_like(new)
    start, stop_new, stop_old = __rebin_uniform_left_left(rebinned, old, new, values, dOld, dNew)

    match preserve:
        case 'counts':
            rebinned /= dOld
        case 'area':
            rebinned /= dNew
            # Fix edges
            #new_last = new[stop_new+1] + dNew
            #old_last = old[stop_old] + dOld
            #if new_last > old_last:
            #    pass
                #rebinned[stop_new+1] *= dNew / dOld * (new_last - old_last)
        case _:
            raise ValueError(f"{preserve} is not a valid option. Options are {Preserve}.")
    return rebinned


#@njit
def __rebin_uniform_left_left(rebinned, old, new, values, dOld, dNew) -> tuple[int, int, int]:
    start = 1
    while start < len(old):
        if old[start] > new[0]:
            break
        start += 1
    # Prevent overshooting
    start -= 1
    j = start
    atol = 1e-10
    i = 0
    while i < len(rebinned):
        new_next = new[i] + dNew
        old_next = old[j] + dOld
        c = overlap(old[j], old_next,
                    new[i], new_next)
        rebinned[i] += c * values[j]

        if new_next > old_next + atol:
            # If the current new bins extends beyond the current old bin,
            # go to the next old bin
            j += 1
            if j > len(old) - 1:
                break
        else:
            # If not, go to the next new bin
            i += 1
    return start, i-1, j-1


def __rebin_nonuniform_left_left(rebinned, old, new, values, dOld, dNew) -> None:
    start = 1
    while start < len(old):
        if old[start] > new[0]:
            break
        start += 1
    # Prevent overshooting
    start -= 1
    j = start
    atol = 1e-10
    i = 0
    while i < len(rebinned):
        c = overlap(old[j], old[j] + dOld[j] ,
                    new[i], new[i] + dNew[j])
        rebinned[i] += c * values[j]
        # If the current new bins extends beyond the current old bin,
        # go to the next old bin
        if (new[i] + dNew[i]) > (old[j] + dOld[j]) + atol:
            j += 1
            if j > len(old) - 1:
                break
        else:
            i += 1

@njit
def __rebin_nonuniform_left_left_encode(old, new, dOld, dNew, flag: bool) -> tuple[np.ndarray, np.ndarray]:
    N = 2*len(old)
    do_move_old_ptr = np.zeros(N, dtype=np.bool8)
    C = np.zeros(N, dtype=np.float64)
    start = 1
    k = 1
    while start < len(old):
        if old[start] > new[0]:
            break
        start += 1
        k += 1
    # Prevent overshooting
    start -= 1
    k -= 1
    do_move_old_ptr[:k] = False
    j = start
    atol = 1e-10
    i = 0
    while True:
        new_next = new[i] + dNew[i]
        old_next = old[j] + dOld[j]
        C[k] = overlap(old[j], old_next,
                       new[i], new_next)
        if flag:
            C[k] /= dOld[j]
        else:
            C[k] /= dNew[i]
        # If the current new bins extends beyond the current old bin,
        # go to the next old bin
        if new_next > old_next + atol:
            do_move_old_ptr[k] = True
            j += 1
            if j > len(old) - 1:
                break
        else:
            do_move_old_ptr[k] = False
            i += 1
            if i > len(new) - 1:
                break
        k += 1
    return do_move_old_ptr[:k+1], C[:k+1]

@njit
def __rebin_nonuniform_left_left_decode(rebinned, do_move_old_ptr, C, values) -> np.ndarray:
    # About twice as fast as "normal" rebinning
    k = 0
    while not do_move_old_ptr[k]:
        k += 1
    j = k
    i = 0
    while k < len(C):
        rebinned[i] += C[k] * values[j]
        if do_move_old_ptr[k]:
            j += 1
        else:
            i += 1
        k += 1
    return rebinned



def rebin_2D_uniform_left_left(old: np.ndarray, new: np.ndarray, values: np.ndarray, axis: int, preserve: Preserve = 'counts'):
    if not is_monotone_uniform(old):
        raise ValueError("X is not monotone uniform.")
    if not is_monotone_uniform(new):
        raise ValueError("Y is not monotone uniform.")
    if not (axis == 0 or axis == 1):
        raise ValueError("Axis must be 0 or 1.")
    return _rebin_2D_uniform_left_left(old, new, values, axis, preserve)

def rebin_2D_nonuniform_left_left(old: np.ndarray, new: np.ndarray, values: np.ndarray, axis: int, preserve: Preserve = 'counts'):
    if not is_monotone(old):
        raise ValueError("X is not monotone.")
    if not is_monotone(new):
        raise ValueError("Y is not monotone uniform.")
    if not (axis == 0 or axis == 1):
        raise ValueError("Axis must be 0 or 1.")
    return _rebin_2D_nonuniform_left_left(old, new, values, axis, preserve)

def fit_into_2d(old, new, values, axis) -> np.ndarray:
    # Assumes old and new are monotone and has the same step
    start = np.searchsorted(old, new[0])
    end = np.searchsorted(old, new[-1], side='right')
    if axis == 0:
        return values[start:end, :]
    else:
        return values[:, start:end]


def _rebin_2D_uniform_left_left(old: np.ndarray, new: np.ndarray,
                                values: np.ndarray, axis: int, preserve: Preserve = 'counts'):
    if len(old) == len(new) and np.allclose(old, new):
        return values
    other_axis = (axis + 1) % 2
    N = values.shape[other_axis]
    shape = [0, 0]
    shape[axis] = len(new)
    shape[other_axis] = N
    dOld = old[1] - old[0]
    dNew = new[1] - new[0]
    if is_close(dNew, dOld):
        return fit_into_2d(old, new, values, axis)
    if dNew < dOld:
        raise ValueError(f"Rebinning to smaller binwidth is ill defined and not supported: {dNew} < {dOld}")
    if not is_close(round(dNew / dOld), dNew / dOld):
        warn("The new step size is not an integral multiple of the old. Induces numerical inaccuracies and/or makes the initial and final bins look wierd.")
    rebinned = np.zeros(shape, dtype=values.dtype)
    dOld_ = np.repeat(dOld, len(old)+1)
    dNew_ = np.repeat(dNew, len(new)+1)
    if preserve == 'counts':
        preserve_counts = True
    elif preserve == 'area':
        preserve_counts = False
    else:
        raise ValueError(f"{preserve} is not a valid option. Options are {Preserve}.")
    __rebin_2D_left_left(rebinned, old, new, values, dOld_, dNew_, axis, N, preserve_counts)
    return rebinned

def _rebin_2D_nonuniform_left_left(old: np.ndarray, new: np.ndarray, values: np.ndarray, dOld: np.ndarray, dNew: np.ndarray, axis: int, preserve: Preserve = 'counts'):
    if len(old) == len(new) and np.allclose(old, new):
        return values
    if not (len(old) + 1 == len(dOld) and len(new) + 1 == len(dNew)):
        raise ValueError("The length of dOld and dNew must be one larger than the length of old and new, respectively.")
    other_axis = (axis + 1) % 2
    N = values.shape[other_axis]
    shape = [0, 0]
    shape[axis] = len(new)
    shape[other_axis] = N
    rebinned = np.zeros(shape, dtype=values.dtype)
    min_dNew = np.min(dNew)
    smaller = min_dNew < dOld
    if np.any(smaller):
        raise ValueError(f"Rebinning to smaller binwidth is ill defined and not supported."
                         f" The following bins are smaller than {min_dNew:G}: {np.where(smaller)}")
    if preserve == 'counts':
        preserve_counts = True
    elif preserve == 'area':
        preserve_counts = False
    else:
        raise ValueError(f"{preserve} is not a valid option. Options are {Preserve}.")
    __rebin_2D_left_left(rebinned, old, new, values, dOld, dNew, axis, N, preserve_counts)
    return rebinned

@njit(parallel=True)
def __rebin_2D_left_left(rebinned: np.ndarray, old: np.ndarray, new: np.ndarray, values: np.ndarray, dOld: np.ndarray, dNew: np.ndarray, axis: int, N: int, preserve_counts: bool) -> None:
    """

    Rebin 2D along one axis. To speed up the rebinning, the rebinning process is "encoded" and
    performed in parallel across the other axis.
    """
    ptr, C = __rebin_nonuniform_left_left_encode(old, new, dOld, dNew, preserve_counts)
    if axis == 0:
        for i in prange(N):
            __rebin_nonuniform_left_left_decode(rebinned[:, i], ptr, C, values[:, i])
    else:
        for i in prange(N):
            __rebin_nonuniform_left_left_decode(rebinned[i, :], ptr, C, values[i, :])


# BUG The index edge is not taken into account. FIX!
def rebin_2D(index, bins: np.ndarray, values: np.ndarray, axis: int, preserve: Preserve = 'counts'):
    #print("=====================")
    #print(index)
    #print(bins)
    #print("=====================")
    if not isinstance(bins, np.ndarray):
        bins = bins.bins

    if index.is_uniform():
        return rebin_2D_uniform_left_left(index.bins, bins, values, axis, preserve)
    else:
        return rebin_2D_nonuniform_left_left(index.bins, bins, values, axis, preserve)

def rebin_1D():
    raise NotImplementedError()
