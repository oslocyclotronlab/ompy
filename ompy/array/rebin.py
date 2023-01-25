from .. import njit, prange
from .index_fn import _index_left, overlap, is_monotone, is_monotone_uniform, are_congruent, is_close
import numpy as np
from warnings import warn
from typing import TypeAlias, Literal


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
@njit
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
    for i in range(len(rebinned)):
        while j < len(old):
            if new[i] > old[j] + dOld:
                j += 1
                continue
            c = overlap(old[j], old[j] + dOld,
                        new[i], new[i] + dNew)
            rebinned[i] += c * values[j]
            j += 1
            if j >= len(old) or old[j] > new[i] + dNew:
                break
        j -= 1


def rebin_2D():
    raise NotImplementedError()


def rebin_1D():
    raise NotImplementedError()