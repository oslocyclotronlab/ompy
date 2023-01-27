from .. import njit, prange
from .index_fn import _index_left, is_monotone_uniform, are_congruent, is_close
import numpy as np
from warnings import warn
from typing import TypeAlias, Literal

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
        raise ValueError("X is not monotone uniform.")
    if not is_monotone_uniform(new):
        raise ValueError("Y is not monotone uniform.")
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
    __rebin_uniform_left_left(rebinned, old, new, values, dOld, dNew)

    match preserve:
        case 'counts':
            rebinned /= dOld
        case 'area':
            rebinned /= dNew
        case _:
            raise ValueError(f"{preserve} is not a valid option. Options are {Preserve}.")
    return rebinned


# TODO Can be parallelized
#@njit
def __rebin_uniform_left_left(rebinned, old, new, values, dOld, dNew) -> None:
    # Find first
    start = -1
    for i in range(1, len(old)):
        if old[i] > new[0]:
            start = i - 1
            break
    if start < 0:
        raise ValueError("No overlap between new and old bins")

    j = start
    atol = 1e-10
    for i in range(len(rebinned)):
        while j < len(old):
            if atol > old[j] + dOld - new[i]:
                j += 1
                continue
            c = overlap(old[j], old[j] + dOld,
                        new[i], new[i] + dNew)
            print(f"i = {i}, j = {j}, c = {c:.2f}")
            print(f"\t[{old[j]:.2f}, {old[j] + dOld:.2f}),  [{new[i]:.2f} {new[i] + dNew:.2f})")
            print(f"\t\t{c / dOld:.2f}")
            rebinned[i] += c * values[j]
            j += 1
            if j >= len(old) or atol > new[i] + dNew - old[j]:
                break
        j -= 1


def rebin_2D_uniform_left_left(old: np.ndarray, new: np.ndarray, values: np.ndarray, axis: int, preserve: Preserve = 'counts'):
    if not is_monotone_uniform(old):
        raise ValueError("X is not monotone uniform.")
    if not is_monotone_uniform(new):
        raise ValueError("Y is not monotone uniform.")
    if not (axis == 0 or axis == 1):
        raise ValueError("Axis must be 0 or 1.")
    return _rebin_2D_uniform_left_left(old, new, values, axis, preserve)


def _rebin_2D_uniform_left_left(old: np.ndarray, new: np.ndarray, values: np.ndarray, axis: int, preserve: Preserve = 'counts'):
    if len(old) == len(new) and np.allclose(old, new):
        return values
    other_axis = (axis + 1) % 2
    N = values.shape[other_axis]
    shape = [0, 0]
    shape[axis] = len(new)
    shape[other_axis] = N
    rebinned = np.zeros(shape, dtype=values.dtype)
    dOld = old[1] - old[0]
    dNew = new[1] - new[0]
    print(dOld, dNew)
    print(old, new)
    if dNew < dOld:
        raise ValueError(f"Rebinning to smaller binwidth is ill defined and not supported: {dNew} < {dOld}")
    if not is_close(round(dNew / dOld), dNew / dOld):
        warn("The new step size is not an integral multiple of the old. Induces numerical inaccuracies.")
    __rebin_2D_uniform_left_left(rebinned, old, new, values, dOld, dNew, axis, N)
    if preserve == 'counts':
        rebinned /= dOld
    elif preserve == 'area':
        rebinned /= dNew
    else:
        raise ValueError(f"{preserve} is not a valid option. Options are {Preserve}.")
    return rebinned


@njit(parallel=True)
def __rebin_2D_uniform_left_left(rebinned: np.ndarray, old: np.ndarray, new: np.ndarray, values: np.ndarray, dOld: float, dNew: float, axis: int, N: int) -> None:
    if axis == 0:
        for i in prange(N):
            __rebin_uniform_left_left(rebinned[:, i], old, new, values[:, i], dOld, dNew)
    else:
        for i in prange(N):
            __rebin_uniform_left_left(rebinned[i, :], old, new, values[i, :], dOld, dNew)




def rebin_2D():
    raise NotImplementedError()


def rebin_1D():
    raise NotImplementedError()