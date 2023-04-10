from numba import njit, types, prange
from typing import Callable, Union, Literal
import numpy as np
# from headers import arraylike, array
from scipy.signal import savgol_filter
from typing import TypeAlias
from .. import Vector, Matrix
arraylike: TypeAlias = np.ndarray
array: TypeAlias = np.ndarray

"""
TODO:
 - [ ] Add a cost for count difference (is this a good idea?)
 - [ ] Generalize L_n_p
 - [ ] Add KL divergence
"""


readonly = types.Array(types.float64, 1, 'C', readonly=True)


@njit((readonly, readonly))
def neglog(nu: arraylike, n: arraylike) -> array:
    prod = np.where(n == 0, 0, n*np.log(nu / n))
    vec = prod - nu + n
    return -vec


@njit((readonly, readonly))
def neglog2(nu: arraylike, n: arraylike) -> array:
    V = np.diag(n)
    return (n - nu).T @ np.linalg.pinv(V) @ (n - nu)


@njit((readonly, readonly))
def loglike(nu: arraylike, n: arraylike):
    err = np.maximum(3.0**2, n)
    return (n - nu)**2 / err  # *-0.5?


@njit((readonly, readonly))
def mse(nu: arraylike, n: arraylike) -> array:
    return np.sqrt((nu - n)**2)


@njit((readonly, readonly, readonly))
def loglike_bg(nu: array, raw: array, bg: array):
    err_fg = np.maximum(3.0**2, raw)
    err_bg = np.maximum(3.0**2, bg)
    return (raw - nu - bg)**2 / (err_fg + err_bg)


LogLikeStr = Literal['ll', 'll2', 'll3', 'mse']
LogLikeFn = Callable[[array, array], array]
LogLike = Union[LogLikeStr, LogLikeFn]

LogLikeBgStr = Literal['ll']
LogLikeBgFn = Callable[[array, array, array], array]
LogLikeBg = LogLikeBgStr | LogLikeBgFn


def get_loglike(name: LogLike) -> LogLikeFn:
    if not isinstance(name, str):
        return name
    match name:
        case 'll':
            return loglike
        case 'll2':
            return neglog
        case 'll3':
            return neglog2
        case 'mse':
            return mse
        case _:
            raise ValueError(f"Loglikelihood {name} not supported")


def get_loglike_bg(name: LogLikeBg) -> LogLikeBgFn:
    if not isinstance(name, str):
        return name
    match name:
        case 'll':
            return loglike_bg
        case _:
            raise ValueError(f"Loglikelihood {name} not supported")


LossStr = Literal['loglike', 'smooth', 'derivative',
                  'derivative2', 'discrete']
LossFn = Callable[[array], float]
Loss = Union[LossStr, LossFn]
MapFn: TypeAlias = Callable[[array], array]


@njit
def idmap(x):
    return x


def loss_factory(name: Loss, R: Matrix, n: Vector,
                 loglike: LogLike,
                 mask: array | None = None,
                 mapfn: MapFn = idmap,
                 imapfn: MapFn = idmap,
                 G: Matrix | None = None,
                 **kwargs) -> LossFn:

    ll = get_loglike(loglike)
    if not isinstance(name, str):
        return name(ll)

    X = n.X
    n = n.values
    R = R.values

    if mask is None:
        mask_ = np.ones_like(n, dtype=bool)
    else:
        mask_ = mask

    match name:
        case 'loglike':
            if mask is not None:
                @njit
                def lossfn(mu: arraylike) -> float:
                    mu = imapfn(mu)
                    nu = R @ mu
                    loglikelihood: array = ll(nu, n)[mask_]
                    return np.sum(loglikelihood)
                return lossfn
            else:
                @njit
                def lossfn(mu: arraylike) -> float:
                    mu = imapfn(mu)
                    nu = R @ mu
                    loglikelihood: array = ll(nu, n)
                    return np.sum(loglikelihood)
                return lossfn
        case 'smooth':
            alpha = kwargs.get('alpha', 2.0)

            def lossfn_reg(mu: arraylike) -> float:
                mu = imapfn(mu)
                nu = R @ mu
                loglikelihood: array = ll(nu, n)[mask_]
                smooth: array = savgol_filter(nu, 9, 3, mode='nearest')
                # logprior: array = ll(smooth, n)[mask_]
                return np.sum(loglikelihood) + alpha*np.sum((smooth - nu)**2)
            return lossfn_reg
        case 'dx':
            alpha = kwargs.get('alpha', 0.1)
            dx = X[1] - X[0]

            @njit
            def lossfn(mu: arraylike) -> float:
                mu = imapfn(mu)
                nu = R @ mu
                loglikelihood: array = ll(nu, n)[mask_]
                # derivative: array = np.sum(np.abs(diff(mu, dx)[mask_]))
                derivative: array = np.sum((diff(mu, dx)[mask_])**2)
                return np.sum(loglikelihood) + alpha*derivative
            return lossfn
        case 'd2x':
            alpha = kwargs.get('alpha', 0.1)
            dx = X[1] - X[0]

            @njit
            def lossfn(mu: arraylike) -> float:
                mu = imapfn(mu)
                nu = R @ mu
                loglikelihood: array = ll(nu, n)[mask_]
                # derivative: array = np.sum(np.abs(diff(mu, dx)[mask_]))
                derivative: array = np.sum((diff(mu, dx)[mask_])**2)
                return np.sum(loglikelihood) + alpha*derivative
            return lossfn
        case 'derivative2':
            alpha = kwargs.get('alpha', 2.0)

            def lossfn(mu: arraylike) -> float:
                nu = R @ mu
                loglikelihood: array = ll(nu, n)[mask_]
                derivative2: float = second_diff_sum(n, mu)[mask_]
                return np.sum(loglikelihood) + alpha*derivative2
            return lossfn
        case 'discrete':
            alpha = kwargs.get('alpha', 2.0)

            @njit
            def lossfn(mu: arraylike) -> float:
                mu = imapfn(mu)
                nu = R @ mu
                loglikelihood: array = ll(nu, n)[mask_]
                cost: float = discrete_bin_cost(mu)
                return np.sum(loglikelihood) + alpha*cost
            return lossfn
        case 'TV':
            alpha = kwargs.get('alpha', 1.0)

            @njit
            def lossfn(mu: arraylike) -> float:
                mu = imapfn(mu)
                nu = R @ mu
                loglikelihood: array = ll(nu, n)[mask_]
                cost: float = TV(mu)
                return np.sum(loglikelihood) + alpha * cost
            return lossfn
        case 'zero':
            alpha = kwargs.get('alpha', 1.0)

            @njit
            def lossfn(mu: arraylike) -> float:
                mu = imapfn(mu)
                nu = R @ mu
                loglikelihood: array = ll(nu, n)[mask_]
                zerocost = np.where(mu < 0, alpha*np.abs(mu), 0)
                return np.sum(loglikelihood) + np.sum(zerocost)
            return lossfn
        case 'I':
            alpha = kwargs.get('alpha', 1.0)

            @njit
            def lossfn(mu: arraylike) -> float:
                mu = imapfn(mu)
                nu = R @ mu
                loglikelihood: array = ll(nu, n)[mask_]
                Icost = (np.sum(nu) - np.sum(n))**2
                return np.sum(loglikelihood) + alpha*Icost
            return lossfn

        case _:
            raise ValueError(f"Loss function {name} is not supported.")


def get_transform(name: str | tuple[MapFn, MapFn]) -> tuple[MapFn, MapFn]:
    if not isinstance(name, str):
        return name[0], name[1]
    match name:
        case 'log':
            @njit
            def exp(x):
                return np.exp(x/1e2)

            @njit
            def log(x):
                return np.log(x)*1e2
            return log, exp
        case 'id':
            return idmap, idmap
        case 'sqrt':
            @njit
            def sqrt(x):
                return np.sqrt(x)

            @njit
            def isqrt(x):
                return x**2
            return sqrt, isqrt
        case _:
            raise ValueError(f"Transform {name} is not supported.")


def loss_factory_bg(name: Loss, R: Matrix, n: Vector, bg: Vector,
                    loglike: LogLikeBg,
                    mask: array | None = None,
                    mapfn: MapFn = idmap,
                    imapfn: MapFn = idmap,
                    **kwargs) -> LossFn:

    ll = get_loglike_bg(loglike)
    if not isinstance(name, str):
        return name(ll)

    if mask is None:
        mask = np.ones_like(n, dtype=bool)

    X = n.X
    R: np.ndarray = R.values
    n: np.ndarray = n.values
    bg: np.ndarray = bg.values

    match name:
        case 'loglike':
            @njit
            def lossfn(mu: arraylike) -> float:
                mu = imapfn(mu)
                nu = R @ mu
                loglikelihood: array = ll(nu, n, bg)[mask]
                return np.sum(loglikelihood)
            return lossfn
        case 'dx':
            alpha = kwargs.get('alpha', 2.0)

            def lossfn(mu: arraylike) -> float:
                mu = imapfn(mu)
                nu = R @ mu
                loglikelihood: array = ll(nu, n, bg)[mask]
                derivative: float = diff_sum(X, mu)[mask]
                return np.sum(loglikelihood) + alpha*derivative
            return lossfn
        case _:
            raise ValueError(f"Loss function {name} is not supported.")


@njit(inline='always')
def TV(x: array) -> float:
    """ Total variation """
    return np.sum(np.abs(np.diff(x)))


@njit
def diff(x: array, dx: float) -> array:
    D = np.empty_like(x)
    D[-1] = 0
    for i in range(len(x)-1):
        D[i] = (x[i+1] - x[i])
    D /= dx
    return D


@njit(parallel=True)
def ddiff(x: array, dx: float) -> array:
    D = np.empty_like(x)
    D[0] = 0
    D[-1] = 0
    for i in prange(1, len(x)-1):
        D[i] = (x[i+1] - 2*x[i] + x[i-1])
    D /= dx**2
    return D


@njit
def diff_sum(x, y):
    """ Forward difference """
    d = x[1] - x[0]
    s = 0.0
    for i in range(1, len(y)-1):
        s += y[i+1] - y[i]
    return s / d


@njit
def second_diff_sum(x, y):
    d = x[1] - x[0]
    s = 0.0
    for i in range(1, len(y)-1):
        s += y[i+1] - 2*y[i] + y[i-1]
    return s / d**2


@njit
def discrete_bin_cost(y):
    return np.sum(y > 0.0)


def strc(s: str, color: str | int):
    if isinstance(color, str):
        if color in {'g', 'green'}:
            c = 32
        elif color in {'r', 'red'}:
            c = 31
        else:
            raise ValueError(f"Color {color} not supported")
    elif not isinstance(color, int):
        raise ValueError(f"Color {color} not supported")
    return f'\033[{c}m{s}\x1b[0m'


def print_minuit_convergence(m):
    def printres(what, expected: bool):
        value = getattr(m.fmin, what)
        if value == expected:
            s = f'{what:<25} is ' + \
                strc(f'{value} (Expected {expected})', 'green')
        else:
            s = f'{what:<25} is ' + \
                strc(str(value) + f" (Expected {expected})", 'red')
        print(s)
    if m.valid:
        print('Valid: ' + strc('True', 'green'))
    else:
        print('Valid: ' + strc('False', 'red'))
    print(f"{m.valid=}")
    printres('has_covariance', True)
    printres('has_accurate_covar', True)
    printres('has_posdef_covar', True)
    printres('has_made_posdef_covar', False)
    printres('hesse_failed', False)
