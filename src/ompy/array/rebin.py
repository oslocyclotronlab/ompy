"""Histogram rebinning helpers used throughout OMpy.

All routines assume *left-aligned* bin definitions: every entry describes the
left edge of a half-open interval that extends to the next entry.  Rebinning is
implemented for uniform and non-uniform grids, and callers can choose to
preserve either raw counts (suitable when values already represent integrated
counts) or area/integral (suitable when values represent densities).

The doctests use small arrays and explicitly downgrade to the pure-Python
implementations when Numba has compiled the tight loops.  This keeps the
documentation executable in environments where the JIT layer is unavailable.
"""

from ..accel import njit, prange
from .index_fn import _index_left, is_monotone_uniform, is_close, is_monotone
import numpy as np
from warnings import warn
from typing import TypeAlias, Literal

"""
TODO:
-[ ] Add tests
-[ ] Severe bug in how 2D rebinning handles incongruent arrays, results in a shift.
"""


class RebinningError(ValueError):
    """Base error raised when rebinning preconditions are violated."""

    pass


class RebinningBinWidthError(RebinningError):
    """Error used when a requested target grid has narrower bins than the source."""

    pass


@njit()
def overlap(Astart, Aend, Bstart, Bend):
    """Return the length of the overlap between two half-open intervals.

    Examples
    --------
    >>> overlap(0.0, 1.0, 0.5, 2.0)
    0.5
    >>> overlap(0.0, 1.0, 1.0, 2.0)
    0.0
    """
    start = max(Astart, Bstart)
    stop = min(Aend, Bend)
    return max(0.0, stop - start)


@njit
def fit_into(old: np.ndarray, new: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Copy a congruent window of ``values`` into the new grid.

    The function assumes both grids are uniform, left aligned, and congruent
    (same step size) so the result is simply a slice of the original array.
    """
    # Assumes congruent, overlapping, left bins
    overlap_ = overlap(old[0], old[-1], new[0], new[-1])
    if overlap_ == 0:
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


Preserve: TypeAlias = Literal["counts", "area"]


def rebin_uniform_left_left(
    old: np.ndarray, new: np.ndarray, values: np.ndarray, preserve: Preserve = "counts"
):
    """Rebin counts or densities on uniform, left-aligned grids.

    Parameters
    ----------
    old, new :
        Left edges of the source/target bins (uniform and strictly increasing).
    values :
        Source data aligned with ``old``.  Interpreted as counts when
        ``preserve='counts'`` and as densities when ``preserve='area'``.
    preserve :
        Whether to conserve total counts or the integral (area) during rebins.

    Examples
    --------
    >>> import numpy as np
    >>> old = np.array([0.0, 1.0, 2.0, 3.0])
    >>> new = np.array([0.0, 2.0, 4.0])
    >>> values = np.array([2.0, 4.0, 6.0, 8.0])
    >>> rebin_uniform_left_left(old, new, values)
    array([ 6., 14.,  0.])
    >>> rebin_uniform_left_left(old, np.array([0.0, 2.0]), values, preserve="area")
    array([3., 7.])
    """
    if not is_monotone_uniform(old):
        raise ValueError("Old bins are not monotone uniform.")
    if not is_monotone_uniform(new):
        raise ValueError("New bins are not monotone uniform.")
    return _rebin_uniform_left_left(old, new, values, preserve)


def _rebin_uniform_left_left(
    old: np.ndarray, new: np.ndarray, values: np.ndarray, preserve: Preserve = "counts"
):
    if len(old) == len(new) and np.allclose(old, new):
        # No rebinning
        return values
    # Case for when the rebinning is a simple shift
    # if are_congruent(old, new) and False:
    #    return fit_into(old, new, values)

    dOld = old[1] - old[0]
    dNew = new[1] - new[0]
    if dNew < dOld:
        raise RebinningBinWidthError(
            f"Rebinning to smaller binwidth is ill defined and not supported: {dNew} < {dOld}"
        )
    if not is_close(round(dNew / dOld), dNew / dOld):
        warn(
            "The new step size is not an integral multiple of the old. Induces numerical inaccuracies."
        )

    rebinned = np.zeros_like(new)
    start, stop_new, stop_old = __rebin_uniform_left_left(
        rebinned, old, new, values, dOld, dNew
    )

    match preserve:
        case "counts":
            rebinned /= dOld
        case "area":
            rebinned /= dNew
            # Fix edges
            # new_last = new[stop_new+1] + dNew
            # old_last = old[stop_old] + dOld
            # if new_last > old_last:
            #    pass
            # rebinned[stop_new+1] *= dNew / dOld * (new_last - old_last)
        case _:
            raise ValueError(
                f"{preserve} is not a valid option. Options are {Preserve}."
            )
    return rebinned


# @njit
def __rebin_uniform_left_left(
    rebinned, old, new, values, dOld, dNew
) -> tuple[int, int, int]:
    """Populate ``rebinned`` using overlap fractions between uniform grids."""
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
        c = overlap(old[j], old_next, new[i], new_next)
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
    return start, i - 1, j - 1


def __rebin_nonuniform_left_left(rebinned, old, new, values, dOld, dNew) -> None:
    """Populate ``rebinned`` using explicit widths for non-uniform grids."""
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
        c = overlap(old[j], old[j] + dOld[j], new[i], new[i] + dNew[j])
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
def __rebin_nonuniform_left_left_encode(
    old, new, dOld, dNew, flag: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Encode pointer movement/coefficient stream for non-uniform rebinning."""
    N = 2 * len(old)
    do_move_old_ptr = np.zeros(N, dtype=np.bool_)
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
        C[k] = overlap(old[j], old_next, new[i], new_next)
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
    return do_move_old_ptr[: k + 1], C[: k + 1]


@njit
def __rebin_nonuniform_left_left_decode(
    rebinned, do_move_old_ptr, C, values
) -> np.ndarray:
    # About twice as fast as "normal" rebinning
    """Decode the pointer/coefficient stream and fill ``rebinned``."""
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


def rebin_2D_uniform_left_left(
    old: np.ndarray,
    new: np.ndarray,
    values: np.ndarray,
    axis: int,
    preserve: Preserve = "counts",
):
    """Rebin a 2D array along a uniform axis while leaving the other untouched.

    Parameters mirror :func:`rebin_uniform_left_left`, but the work is applied
    column-wise (``axis=0``) or row-wise (``axis=1``).

    Examples
    --------
    >>> import numpy as np
    >>> from ompy.array import rebin as _mod
    >>> old = np.array([0.0, 1.0, 2.0, 3.0])
    >>> new = np.array([0.0, 2.0])
    >>> matrix = np.array([[1.0, 10.0],
    ...                    [2.0, 20.0],
    ...                    [3.0, 30.0],
    ...                    [4.0, 40.0]])
    >>> original = _mod.__rebin_2D_left_left
    >>> helper = getattr(original, "py_func", original)
    >>> _mod.__rebin_2D_left_left = helper
    >>> try:
    ...     rebin_2D_uniform_left_left(old, new, matrix, axis=0)
    ... finally:
    ...     _mod.__rebin_2D_left_left = original
    array([[ 3., 30.],
           [ 7., 70.]])
    """
    if not is_monotone_uniform(old):
        raise ValueError("X is not monotone uniform.")
    if not is_monotone_uniform(new):
        raise ValueError("Y is not monotone uniform.")
    if not (axis == 0 or axis == 1):
        raise ValueError("Axis must be 0 or 1.")
    return _rebin_2D_uniform_left_left(old, new, values, axis, preserve)


def rebin_2D_nonuniform_left_left(
    old: np.ndarray,
    new: np.ndarray,
    values: np.ndarray,
    axis: int,
    preserve: Preserve = "counts",
):
    """Rebin a 2D array along a non-uniform axis while leaving the other untouched."""
    if not is_monotone(old):
        raise ValueError("X is not monotone.")
    if not is_monotone(new):
        raise ValueError("Y is not monotone uniform.")
    if not (axis == 0 or axis == 1):
        raise ValueError("Axis must be 0 or 1.")
    return _rebin_2D_nonuniform_left_left(old, new, values, axis, preserve)


def fit_into_2d(old, new, values, axis) -> np.ndarray:
    """Return the overlapping window when ``new`` is a contiguous subset of ``old``."""
    if axis not in (0, 1):
        raise ValueError("Axis must be 0 or 1.")
    if len(old) < 2 or len(new) == 0:
        raise ValueError("Need at least two old edges and one new edge.")
    if values.shape[axis] != len(old):
        raise ValueError("Values along the rebin axis must match the old grid length.")

    step = old[1] - old[0]
    if not is_close(step, new[1] - new[0] if len(new) > 1 else step):
        raise RebinningError("Grids do not share the same spacing.")

    offset_float = (new[0] - old[0]) / step
    offset = int(round(offset_float))
    aligned_start = is_close(old[0] + offset * step, new[0])
    end = offset + len(new)

    if not aligned_start or offset < 0 or end > len(old):
        raise RebinningError("Target grid is not a window of the source grid.")

    if not np.allclose(old[offset:end], new, rtol=1e-8, atol=1e-12):
        raise RebinningError("Target grid differs from the source grid beyond tolerance.")

    slicer = [slice(None)] * values.ndim
    slicer[axis] = slice(offset, end)
    return values[tuple(slicer)]

def _left_bin_widths(edges: np.ndarray) -> np.ndarray:
    """Estimate bin widths from a left-aligned grid."""
    arr = np.asarray(edges)
    if arr.size < 2:
        raise ValueError("Need at least two edges to determine widths.")
    diffs = np.diff(arr)
    widths = np.empty_like(arr)
    widths[:-1] = diffs
    widths[-1] = diffs[-1]
    return widths


def _rebin_2D_uniform_left_left(
    old: np.ndarray,
    new: np.ndarray,
    values: np.ndarray,
    axis: int,
    preserve: Preserve = "counts",
):
    """Implementation detail for uniform bins; see :func:`rebin_2D_uniform_left_left`."""
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
        try:
            return fit_into_2d(old, new, values, axis)
        except RebinningError:
            pass
    if dNew < dOld:
        raise RebinningBinWidthError(
            f"Rebinning to smaller binwidth is ill defined and not supported: {dNew} < {dOld}"
        )
    if not is_close(round(dNew / dOld), dNew / dOld):
        warn(
            "The new step size is not an integral multiple of the old. Induces numerical inaccuracies and/or makes the initial and final bins look wierd."
        )
    rebinned = np.zeros(shape, dtype=values.dtype)
    dOld_ = _left_bin_widths(old)
    dNew_ = _left_bin_widths(new)
    if preserve == "counts":
        preserve_counts = True
    elif preserve == "area":
        preserve_counts = False
    else:
        raise ValueError(f"{preserve} is not a valid option. Options are {Preserve}.")
    __rebin_2D_left_left(
        rebinned, old, new, values, dOld_, dNew_, axis, N, bool(preserve_counts)
    )
    return rebinned


def _rebin_2D_nonuniform_left_left(
    old: np.ndarray,
    new: np.ndarray,
    values: np.ndarray,
    axis: int,
    preserve: Preserve = "counts",
    dOld: np.ndarray | None = None,
    dNew: np.ndarray | None = None,
):
    """Implementation detail for non-uniform bins; see :func:`rebin_2D_nonuniform_left_left`."""
    axis = int(axis)
    if len(old) == len(new) and np.allclose(old, new):
        return values
    if dOld is None:
        dOld = _left_bin_widths(old)
    if dNew is None:
        dNew = _left_bin_widths(new)
    other_axis = (axis + 1) % 2
    N = values.shape[other_axis]
    shape = [0, 0]
    shape[axis] = len(new)
    shape[other_axis] = N
    rebinned = np.zeros(shape, dtype=values.dtype)
    min_dNew = np.min(dNew)
    smaller = min_dNew < dOld
    if np.any(smaller):
        raise ValueError(
            f"Rebinning to smaller binwidth is ill defined and not supported."
            f" The following bins are smaller than {min_dNew:G}: {np.where(smaller)}"
        )
    if preserve == "counts":
        preserve_counts = True
    elif preserve == "area":
        preserve_counts = False
    else:
        raise ValueError(f"{preserve} is not a valid option. Options are {Preserve}.")
    __rebin_2D_left_left(
        rebinned, old, new, values, dOld, dNew, axis, N, preserve_counts
    )
    return rebinned


@njit(parallel=True)
def __rebin_2D_left_left(
    rebinned: np.ndarray,
    old: np.ndarray,
    new: np.ndarray,
    values: np.ndarray,
    dOld: np.ndarray,
    dNew: np.ndarray,
    axis: int,
    N: int,
    preserve_counts: bool,
) -> None:
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
def rebin_2D(
    index,
    bins: np.ndarray,
    values: np.ndarray,
    axis: int,
    preserve: Preserve = "counts",
):
    """Dispatch to the appropriate 2D rebinning path based on grid uniformity.

    Examples
    --------
    >>> import numpy as np
    >>> from ompy.array import rebin as _mod
    >>> class DummyIndex:
    ...     def __init__(self, bins):
    ...         self.bins = bins
    ...     def is_uniform(self):
    ...         return True
    >>> edges = np.array([0.0, 1.0, 2.0])
    >>> idx = DummyIndex(edges)
    >>> data = np.eye(2, dtype=float)
    >>> original = _mod.__rebin_2D_left_left
    >>> helper = getattr(original, "py_func", original)
    >>> _mod.__rebin_2D_left_left = helper
    >>> try:
    ...     rebin_2D(idx, edges, data, axis=0)
    ... finally:
    ...     _mod.__rebin_2D_left_left = original
    array([[1., 0.],
           [0., 1.]])
    """
    # print("=====================")
    # print(index)
    # print(bins)
    # print("=====================")
    index = index.to_left()
    if not isinstance(bins, np.ndarray):
        bins = bins.bins

    if index.is_uniform():
        x = rebin_2D_uniform_left_left(index.bins, bins, values, axis, preserve)
        return x
    else:
        return rebin_2D_nonuniform_left_left(index.bins, bins, values, axis, preserve)


def rebin_1D():
    """Placeholder for a future 1D wrapper; not yet implemented."""
    raise NotImplementedError()
