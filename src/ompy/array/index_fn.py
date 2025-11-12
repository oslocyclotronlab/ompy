import numpy as np

from ..accel import njit

@njit
def _is_monotone_impl(x: np.ndarray) -> bool:
    """Numba implementation used for float32/float64 arrays."""
    for i in range(len(x) - 1):
        if x[i] >= x[i + 1]:
            return False
    return True


def is_monotone(x: np.ndarray) -> bool:
    """Return True when `x` is strictly increasing."""
    x_arr = np.asarray(x)
    if x_arr.size < 2:
        return True
    if x_arr.dtype == np.float16:
        return bool(np.all(x_arr[:-1] < x_arr[1:]))
    return bool(_is_monotone_impl(x_arr))


#@njit
def is_uniform(X: np.ndarray, rtol=1e-3, atol=None) -> bool:
    """
    Check whether the coordinates in `X` are spaced uniformly.

    Parameters
    ----------
    X : numpy.ndarray
        Sample locations ordered from left to right.
    rtol : float, optional
        Relative tolerance used by `numpy.allclose`.
    atol : float, optional
        Absolute tolerance override. When omitted, the tolerance is derived
        from the dtype of `X`.

    >>> import numpy as np
    >>> is_uniform(np.array([0.0, 0.5, 1.0, 1.5]))
    True
    >>> is_uniform(np.array([0.0, 0.5, 1.1]))
    False
    """
    dX = X[1] - X[0]
    if atol is None:
        if np.issubdtype(X.dtype, np.floating):
            _atol = np.finfo(X.dtype).eps * np.abs(X).max()
        else:
            _atol = 0.5
    else:
        _atol = atol
    return np.allclose(X[1:] - X[:-1], dX, rtol=rtol, atol=_atol, equal_nan=False)


@njit()
def _is_length_congruent_impl(X: np.ndarray, Y: np.ndarray) -> bool:
    if len(X) < 1 or len(Y) < 1:
        return False
    if len(X) == 1 and len(Y) == 1:
        return True
    if len(X) < 2 or len(Y) < 2:
        return False
    return True


def is_length_congruent(X: np.ndarray, Y: np.ndarray) -> bool:
    """Determine whether two grids may be compared for congruency."""
    if np.asarray(X).dtype == np.float16 or np.asarray(Y).dtype == np.float16:
        if len(X) < 1 or len(Y) < 1:
            return False
        if len(X) == len(Y) == 1:
            return True
        if len(X) < 2 or len(Y) < 2:
            return False
        return True
    return bool(_is_length_congruent_impl(np.asarray(X), np.asarray(Y)))


#@njit()
def is_monotone_uniform(X: np.ndarray) -> bool:
    """
    Return True when `X` is strictly increasing and evenly spaced.

    >>> import numpy as np
    >>> is_monotone_uniform(np.array([0.0, 1.0, 2.0]))
    True
    >>> is_monotone_uniform(np.array([0.0, 1.0, 1.5]))
    False
    """
    return is_monotone(X) and is_uniform(X)


@njit
def is_close(x, y, rtol=1e-5, atol=1e-8) -> bool:
    """
    Lightweight `numpy.isclose` alternative compatible with Numba.

    >>> is_close(1.0, 1.0 + 1e-7)
    True
    >>> is_close(1.0, 1.01)
    False
    """
    return abs(x - y) <= (atol + rtol * abs(y))


#@njit()
def are_congruent(X: np.ndarray, Y: np.ndarray) -> bool:
    """
    Check whether two coordinate arrays have identical spacing and phase.

    Two arrays are congruent when they are both monotone and uniform, share the
    same spacing, and align modulo that spacing.

    >>> import numpy as np
    >>> X = np.array([0.0, 1.0, 2.0])
    >>> Y = np.array([2.0, 3.0, 4.0])
    >>> are_congruent(X, Y)
    True
    >>> are_congruent(X, np.array([2.0, 3.1, 4.2]))
    False
    """
    if not is_length_congruent(X, Y):
        return False
    if not is_monotone_uniform(X) and is_monotone_uniform(Y):
        return False

    dx = X[1] - X[0]
    dy = Y[1] - Y[0]
    if is_close(dx, dy) and is_close(X[0] % dx, Y[0] % dy):
        return True
    return False


def index_left(X: np.ndarray, x: float) -> int:
    """
    Locate the index of the element directly to the left of `x`.

    Raises a `ValueError` when `x` lies outside the open interval covered by
    `X`. Requires strictly increasing coordinates.

    >>> import numpy as np
    >>> index_left(np.array([0.0, 1.0, 2.0]), 1.4)
    1
    >>> index_left(np.array([0.0, 1.0, 2.0]), -0.5)
    Traceback (most recent call last):
    ...
    ValueError: -0.5 out of range of left edge 0.0
    """
    if X[0] > x:
        raise ValueError(f"{x} out of range of left edge {X[0]}")
    if X[-1] < x:
        raise ValueError(f"{x} out of range of right edge {X[-1]}")
    return _index_left(np.asarray(X), x)


@njit
def _index_left_impl(X: np.ndarray, x: float) -> int:
    i = 0
    while i < len(X):
        if X[i] > x:
            return i - 1
        i += 1
    return i - 1


def _index_left(X: np.ndarray, x: float) -> int:
    if X.dtype == np.float16:
        i = 0
        while i < len(X):
            if X[i] > x:
                return i - 1
            i += 1
        return i - 1
    return int(_index_left_impl(X, x))


def index_mid(X: np.ndarray, x: float) -> int:
    """
    Return the index of the grid point whose cell midpoint is nearest to `x`.

    The first and last cells are treated using half-widths inferred from their
    neighboring points. Values outside this extended range raise `ValueError`.

    >>> import numpy as np
    >>> index_mid(np.array([0.0, 1.0, 2.0]), 0.75)
    1
    >>> index_mid(np.array([0.0, 1.0, 2.0]), 2.6)
    Traceback (most recent call last):
    ...
    ValueError: 2.6 out of range of right edge 2.5
    """
    dX0 = (X[1] - X[0]) / 2
    dXend = (X[-1] - X[-2]) / 2
    if X[0] - dX0 > x:
        raise ValueError(f"{x} out of range of left edge {X[0] - dX0}")
    if X[-1] + dXend < x:
        raise ValueError(f"{x} out of range of right edge {X[-1] + dXend}")
    return _index_mid(X, x)


@njit
def _index_mid_impl(X: np.ndarray, x: float) -> int:
    i = 1
    d0 = abs(X[0] - x)
    while i < len(X):
        d1 = X[i] - x
        if d1 > 0:
            if d1 > d0:
                return i - 1
            else:
                return i
        d0 = abs(d1)
        i += 1
    return i - 1


def _index_mid(X: np.ndarray, x: float) -> int:
    if X.dtype == np.float16:
        i = 1
        d0 = abs(float(X[0]) - float(x))
        while i < len(X):
            d1 = float(X[i]) - float(x)
            if d1 > 0:
                if d1 > d0:
                    return i - 1
                else:
                    return i
            d0 = abs(d1)
            i += 1
        return i - 1
    return int(_index_mid_impl(X, x))


def index_mid_uniform(X: np.ndarray, x: float) -> int:
    """
    Specialized midpoint indexer for uniformly spaced grids.

    The grid is assumed uniform; no extra checks are performed before
    delegating to `_index_mid`.

    >>> import numpy as np
    >>> index_mid_uniform(np.array([0.0, 1.0, 2.0]), 1.6)
    2
    """
    dX = (X[1] - X[0]) / 2
    if X[0] - dX > x:
        raise ValueError(f"{x} out of range of left edge {X[0] - dX}")
    if X[-1] + dX < x:
        raise ValueError(f"{x} out of range of right edge {X[-1] + dX}")
    return _index_mid_uniform(X, x)


def _index_mid_uniform(X: np.ndarray, x: float) -> int:
    """
    Internal helper that simply proxies to `_index_mid`.

    >>> import numpy as np
    >>> _index_mid_uniform(np.array([0.0, 1.0, 2.0]), 0.2)
    0
    """
    return _index_mid(X, x)


def index_mid_nonuniform(X: np.ndarray, dX: np.ndarray, x: float) -> int:
    """
    Midpoint indexer for non-uniform grids with per-point cell widths.

    Parameters
    ----------
    X : numpy.ndarray
        Strictly increasing grid coordinates.
    dX : numpy.ndarray
        Positive cell widths associated with each entry in `X`.
    x : float
        Location to index.

    >>> import numpy as np
    >>> X = np.array([0.0, 1.0, 2.5])
    >>> dX = np.array([1.0, 1.5, 1.5])
    >>> index_mid_nonuniform(X, dX, 1.4)
    1
    """
    if X[0] - dX[0] / 2 > x:
        raise ValueError(f"{x} out of range of left edge {X[0] - dX[0] / 2}")
    if X[-1] + dX[-1] / 2 < x:
        raise ValueError(f"{x} out of range of right edge {X[-1] + dX[-1] / 2}")
    return _index_mid_nonuniform(X, dX, x)

def _index_mid_nonuniform(X, dX, x):
    if np.asarray(X).dtype == np.float16 or np.asarray(dX).dtype == np.float16:
        i = 0
        while i < len(X):
            d = abs(float(X[i]) - float(x))
            if d < float(dX[i]) / 2:
                return i
            i += 1
        return i - 1
    return int(_index_mid_nonuniform_impl(X, dX, x))

@njit
def _index_mid_nonuniform_impl(X, dX, x):
    """
    Numba implementation for non-uniform midpoint indexing.

    The function walks the grid until the absolute distance to the candidate
    point is less than half the local cell width.

    >>> import numpy as np
    >>> X = np.array([0.0, 1.0, 2.5])
    >>> dX = np.array([1.0, 1.5, 1.5])
    >>> __index_mid_nonuniform(X, dX, 0.2)
    0
    """
    i = 0
    #d0 = abs(X[0] - x)
    while i < len(X):
        d = abs(X[i] - x)
        if d < dX[i]/2:
            return i
        #if d1 > 0:
        #    if d1 > d0:
        #        return i - 1
        #    else:
        #        return i
        #d0 = abs(d1)
        i += 1
    return i - 1
