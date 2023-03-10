from numba import njit, types
from typing import Callable, Union, Literal
import numpy as np
#from headers import arraylike, array
from scipy.signal import savgol_filter
from typing import TypeAlias
arraylike: TypeAlias = np.ndarray
array: TypeAlias = np.ndarray


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
    return (n - nu)**2 / err # *-0.5?

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
            return neglog
        case 'll2':
            return neglog2
        case 'll3':
            return loglike
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


def loss_factory(name: Loss, R: array, n: array,
                 loglike: LogLike,
                 mask: array | None = None,
                 **kwargs) -> LossFn:

    ll = get_loglike(loglike)
    if not isinstance(name, str):
        return name(ll)

    if mask is None:
        mask = np.ones_like(n, dtype=bool)

    if name == 'loglike':
        @njit
        def lossfn(mu: arraylike) -> float:
            nu = R @ mu
            loglikelihood: array = ll(nu, n)[mask]
            return np.sum(loglikelihood)
        return lossfn
    elif name == 'smooth':
        alpha = kwargs.get('alpha', 2.0)
        def lossfn_reg(mu: arraylike) -> float:
            nu = R @ mu
            loglikelihood: array = ll(nu, n)[mask]
            smooth: array = savgol_filter(mu, 5, 3, mode='nearest')
            logprior: array = ll(mu, smooth)[mask]
            return np.sum(loglikelihood) + alpha*np.sum(logprior)
        return lossfn_reg
    elif name == 'derivative':
        alpha = kwargs.get('alpha', 2.0)
        def lossfn(mu: arraylike) -> float:
            nu = R @ mu
            loglikelihood: array = ll(nu, n)[mask]
            derivative: float = diff_sum(n, mu)[mask]
            return np.sum(loglikelihood) + alpha*derivative
        return lossfn
    elif name == 'derivative2':
        alpha = kwargs.get('alpha', 2.0)
        def lossfn(mu: arraylike) -> float:
            nu = R @ mu
            loglikelihood: array = ll(nu, n)[mask]
            derivative2: float = second_diff_sum(n, mu)[mask]
            return np.sum(loglikelihood) + alpha*derivative2
        return lossfn
    elif name == 'discrete':
        alpha = kwargs.get('alpha', 2.0)
        def lossfn(mu: arraylike) -> float:
            nu = R @ mu
            loglikelihood: array = ll(nu, n)[mask]
            cost: float = discrete_bin_cost(mu)[mask]
            return np.sum(loglikelihood) + alpha*cost
        return lossfn
    else:
        raise ValueError(f"Loss function {name} is not supported.")


def loss_factory_bg(name: Loss, R: array, n: array, bg: array,
                    loglike: LogLikeBg,
                    mask: array | None = None,
                    **kwargs) -> LossFn:

    ll = get_loglike_bg(loglike)
    if not isinstance(name, str):
        return name(ll)

    if mask is None:
        mask = np.ones_like(n, dtype=bool)

    match name:
        case 'loglike':
            def lossfn(mu: arraylike) -> float:
                nu = R @ mu
                loglikelihood: array = ll(nu, n, bg)[mask]
                return np.sum(loglikelihood)
            return lossfn
        case _:
            raise ValueError(f"Loss function {name} is not supported.")

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
