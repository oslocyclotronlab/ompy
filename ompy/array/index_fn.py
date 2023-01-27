from .. import njit, prange
import numpy as np

@njit
def is_monotone(x: np.ndarray) -> bool:
    """ Check if x is strictly monotone increasing """
    for i in range(len(x) - 1):
        if x[i] >= x[i + 1]:
            return False
    return True


@njit
def is_uniform(X: np.ndarray, rtol=1e-5, atol=1e-8) -> bool:
    """ Check if X is equidistant """
    dX = X[1] - X[0]
    return np.allclose(X[1:] - X[:-1], dX, rtol=rtol, atol=atol, equal_nan=False)


@njit()
def is_length_congruent(X: np.ndarray, Y: np.ndarray) -> bool:
    if len(X) < 1 or len(Y) < 1:
        return False
    if len(X) == 1 and len(Y) == 1:
        return True
    if len(X) < 2 or len(Y) < 2:
        return False
    return True

@njit()
def is_monotone_uniform(X: np.ndarray) -> bool:
    return is_monotone(X) and is_uniform(X)

@njit
def is_close(x, y, rtol=1e-5, atol=1e-8) -> bool:
    """ Reimplementation of numpy.isclose because numba """
    return abs(x-y) <= (atol + rtol*abs(y))


@njit()
def are_congruent(X: np.ndarray, Y: np.ndarray) -> bool:
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
    if X[0] > x:
        raise ValueError(f"{x} out of range of left edge {X[0]}")
    if X[-1] < x:
        raise ValueError(f"{x} out of range of right edge {X[-1]}")
    return _index_left(X, x)

@njit
def _index_left(X: np.ndarray, x: float) -> int:
    i = 0
    while i < len(X):
        if X[i] > x:
            return i - 1
        i += 1
    return i - 1

def index_mid(X: np.ndarray, x: float) -> int:
    dX0 = (X[1] - X[0])/2
    dXend = (X[-1] - X[-2])/2
    if X[0] - dX0 > x:
        raise ValueError(f"{x} out of range of left edge {X[0] - dX0}")
    if X[-1] + dXend < x:
        raise ValueError(f"{x} out of range of right edge {X[-1] + dXend}")
    return _index_mid(X, x)

@njit
def _index_mid(X: np.ndarray, x: float) -> int:
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

def index_mid_uniform(X: np.ndarray, x: float) -> int:
    dX = (X[1] - X[0])/2
    if X[0] - dX > x:
        raise ValueError(f"{x} out of range of left edge {X[0] - dX}")
    if X[-1] + dX < x:
        raise ValueError(f"{x} out of range of right edge {X[-1] + dX}")
    return _index_mid_uniform(X, x)

def _index_mid_uniform(X: np.ndarray, x: float) -> int:
    return _index_mid(X, x)
