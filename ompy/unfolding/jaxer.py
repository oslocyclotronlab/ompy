from __future__ import annotations
from .unfolder import Unfolder
from .result1d import UnfoldedResult1DSimple, Cost1D, Parameters1D, ResultMeta1D
from .result2d import UnfoldedResult2DSimple, Cost2D, Parameters2D, ResultMeta2D
from .stubs import Space
from .. import Matrix, Vector
from ..stubs import Plot1D, Axes, array1D
import numpy as np
import time
from tqdm.autonotebook import tqdm
from dataclasses import dataclass, fields
import matplotlib.pyplot as plt
from typing import Any, TypedDict, Iterable
from functools import partial
from pathlib import Path
from itertools import product
from typing_extensions import override


import jax
from jax import numpy as jnp
from jax import Array

"""
TODO
-[ ] Optimize vector
-[x] Hyperparameter search for NAG++
     Parameter transform, logistic or inverse hyperbolic tangent
-[x] GPU. Remember to set correct environment variables
-[ ] Test more optimizers
-[ ] Initial optimizer hyperparameter search
-[ ] KL + ME
-[x] Background
-[x] Why does taking SiRi into account worsen the result?
     Because I am stupid.
"""

# JAX Jit hates this function
def slog(x):
    return jnp.where(x <= 1e-5, 0.0, jnp.log(x))

def kl(nu, n):
    #return nu - n + n * jnp.log(n / (nu+1e-10) + 1e-10)
    #return (nu - n) + n * (slog(n) - slog(nu))
    eps = 1e-5
    #mask = (nu <= eps) | (n <= eps)
    return nu - n + n * jnp.log(n / (nu+1e-10) + 1e-10)
    #return jnp.where(mask, 0.0, (nu - n) + n * (jnp.log(n) - jnp.log(nu)))
    #return (nu - n) + n * (jnp.log(n) - jnp.log(nu))

def entropy(mu):
    #mask = mu <= 1e-5
    #return jnp.where(mask, 0.0, mu * jnp.log(mu))
    return mu * jnp.log(mu + 1.0)
    #return -jnp.sum(mu * slog(mu))
    #return mu * jnp.log(mu)

def split_entropy(mu, lower: float, upper: float, midpoint: float):
    entropy_ = entropy(mu)
    return logistic_interpolation(entropy_, lower, upper, midpoint)


def difference_cost(n, nu):
    return (jnp.sum(n) - jnp.sum(nu))**2

def sigmoid(x):
    return 1 / (1 + jnp.exp(-x*1e-2))

def logistic_interpolation(t, lower, upper, midpoint):
    #d = find_d(C, A, B, k)
    return lower + (upper - lower) * sigmoid(t - midpoint)

def onecost(mu): # Make smooth by sigmoid
    return jnp.sum(jnp.arctan(mu))

def cost(mu, R, G_ex, n, bg, n_err, bg_err,
         alpha=1.0, beta=1e-3, lower=1e-3, upper=-1e-3, midpoint=1e4):#, alpha=0.3e-1):
    mu = mu**2
    nu = G_ex@mu@R
    #return jnp.sum((n - bg - nu)**2/(n_err + bg_err))# + onecost(mu)
    return jnp.sum(kl(nu, n)) + alpha*onecost(mu) #beta*jnp.sum(entropy(mu)) + alpha*onecost(mu) # + 1e-6*difference_cost(n, nu)
    #return jnp.sum(kl(nu, n)) - jnp.sum(split_entropy(mu, lower, upper, midpoint))# + alpha*onecost(mu) + 1e-5*difference_cost(n, nu)
    #return jnp.sum((nu - n)**2/e
    #return jnp.sum(kl(nu, n))
    #jnp.sum(kl(nu, n)) #+ alpha*entropy(mu)

def _cost_1d(mu, R, G_ex, n, bg, n_err, bg_err,
            alpha=1.0, beta=1e-3):#, alpha=0.3e-1):
    mu = mu**2
    nu = R@mu
    # Jax will jit and remove this branch
    #if bg is not None:
    #    return jnp.sum((n - bg - nu)**2/(n_err + bg_err))# + onecost(mu)
    #else:
    #    return jnp.sum(kl(nu, n)) + 1e3*onecost(nu) # 2000*entropy(mu)
    #return jnp.sum(kl(nu, n)) + jnp.sum(1e-4*entropy(mu)) + difference_cost(n, nu)
    # The entropy must be taken in eta space
    return jnp.sum(kl(nu, n)) + alpha*onecost(mu)**2 - beta*jnp.sum(entropy(mu))# + difference_cost(n, nu)
    #total = jnp.sum(kl(nu, n)) #- alpha*jnp.sum(entropy(mu)) + beta*difference_cost(n, nu)


def cost_1d(mu, R, G_ex, n, bg, n_err, bg_err,
            alpha=1.0, beta=1e-3):#, alpha=0.3e-1):
    mu = mu**2
    nu = R@mu
    nu = jnp.log(nu + 1e-1)
    n = jnp.log(n + 1e-1)
    return jnp.sum(jnp.abs(nu - n)) + alpha*onecost(mu)**2 - beta*jnp.sum(entropy(mu))# + difference_cost(n, nu)
    #total = jnp.sum(kl(nu, n)) #- alpha*jnp.sum(entropy(mu)) + beta*difference_cost(n, nu)


def cost_1d_v2(mu, R, G_eg, G_ex, n, bg, n_err, bg_err,
            alpha=1.0, beta=1e-3):#, alpha=0.3e-1):
    mu = mu**2
    nu = R@mu
    eta = G_eg@mu
    # The entropy must be taken in eta space
    return jnp.sum(kl(nu, n)) + alpha*onecost(mu)**2 - beta*jnp.sum(entropy(eta))# + difference_cost(n, nu)
    #total = jnp.sum(kl(nu, n)) #- alpha*jnp.sum(entropy(mu)) + beta*difference_cost(n, nu)


def cost_components(n, mu, R, eta=None, G_eg=None, G_ex=None, alpha=0, beta=0):
    nu = R@mu
    loss = jnp.sum(kl(nu, n))
    regularization = onecost(mu)**2# - beta*jnp.sum(entropy(eta))
    if eta is not None:
        eta_ = G_eg@mu
        validation = jnp.sum(kl(eta_, eta))
        return loss, regularization, validation
    return loss, regularization


@dataclass(kw_only=True)
class JaxResult2D(Cost1D, UnfoldedResult2DSimple):
    def _save(self, path: Path, exist_ok: bool = False):
        Cost2D._save(self, path, exist_ok)
        UnfoldedResult2DSimple._save(self, path, exist_ok)

    @classmethod
    def _load(cls, path: Path) -> dict[str, np.ndarray | Matrix]:
        a = Cost2D._load(path)
        b = UnfoldedResult2DSimple._load(path)
        return a | b

@dataclass(kw_only=True)
class JaxResult1D(Cost1D, UnfoldedResult1DSimple):
    def _save(self, path: Path, exist_ok: bool = False):
        UnfoldedResult1DSimple._save(self, path, exist_ok)
        Cost1D._save(self, path, exist_ok)

    @classmethod
    def _load(cls, path: Path) -> dict[str, np.ndarray | Vector]:
        a = Cost1D._load(path)
        b = UnfoldedResult1DSimple._load(path)
        return a | b



class Jaxer(Unfolder):
    @override
    @staticmethod
    def supports_background():
        return True

    @override
    def _unfold_vector(self, R: Matrix, data: Vector, background: Vector | None,
                       initial: Vector, space: Space,
                       G: Matrix | None = None, **kwargs) -> JaxResult1D:
        # Check if cost_1d parameters are iterable
        # At most one can be iterable
        # Construct matrix NxM where N is vector and M is parameters
        # How can the cost function be applied correctly?
        # Probably need to use jax.vmap? Or linear algebra trick
        # How does this work with Bootstrap?
        # Perhaps instead have a _unfold_vectors()?

        loss = jax.jit(cost_1d)
        grad = jax.jit(jax.grad(cost_1d))
        value_and_grad = jax.jit(jax.value_and_grad(cost_1d))
        u = jnp.asarray(np.sqrt(initial.values))
        R_ = jnp.asarray(R.values)
        raw = jnp.asarray(data.values)
        if background is None:
            bg = 0
        else:
            bg = jnp.asarray(background.values)
        start = time.time()
        u, total_loss = unfold_adam_1d(u, raw=raw, bg=bg, R=R_, G_ex=None,
                                       loss=loss, grad=grad, value_and_grad=value_and_grad,
                                       **kwargs)
        u = data.clone(values=u)
        elapsed = time.time() - start
        parameters = Parameters1D(R=R, raw=data, background=background, initial=initial,
                                  kwargs=kwargs, G=G)
        meta = ResultMeta1D(time=elapsed, space=space, parameters=parameters,
                            method=self.__class__)
        return JaxResult1D(meta=meta, cost=total_loss, u=u)


    def _unfold_matrix(self, R: Matrix, data: Matrix, background: Matrix | None, initial: Matrix,
                       use_previous: bool, space: Space, G: Matrix, G_ex: Matrix, **kwargs) -> UnfoldedResult2DSimple:

        mask = np.zeros_like(data, dtype=bool)
        bins = np.zeros(data.shape[0])
        for i in range(data.shape[0]):
            j = data.iloc[i, :].last_nonzero()
            mask[i, :j] = True
            bins[i] = j
        u = jnp.asarray(np.sqrt(initial.values))
        R_ = jnp.asarray(R.values.T)
        G_ex_ = jnp.asarray(G_ex.values)
        n = jnp.asarray(data.values)
        if background is None:
            bg = 0
        else:
            bg = jnp.asarray(background.values)
        mask = jnp.asarray(mask)
        loss = jax.jit(cost, static_argnames=('alpha', 'beta'))
        grad = jax.grad(cost)
        grad = jax.jit(grad, static_argnames=('alpha', 'beta'))
        method = kwargs.pop('method', 'adam')
        match method:
            case 'NAG':
                unfold = unfold_NAG
            case 'GD':
                unfold = unfold_GD
            case 'adam':
                unfold = unfold_adam
            case _:
                raise ValueError(f"Unknown method {method}")
        value_and_grad = jax.jit(jax.value_and_grad(cost), static_argnames=('alpha', 'beta'))
        start = time.time()
        u, total_cost = unfold(u, raw=n, bg=bg, R=R_, G_ex=G_ex_,
                               loss=loss, grad=grad, value_and_grad=value_and_grad,
                               mask=mask, **kwargs)
        elapsed = time.time() - start
        # TODO Add Response coefficients as optimisation parameter
        # TODO Loop over Ex and make error
        #print("Approximating variance")
        #hessian = jax.jit(jax.jacfwd(jax.jacrev(cost)))
        #hessian = hessian(u[160], R_, n[160])
        parameters = Parameters2D(R=R, raw=data, background=background, initial=initial,
                                  G=G, kwargs=kwargs | {'method': method},
                                  G_ex=G_ex)
        meta = ResultMeta2D(time=elapsed, space=space, parameters=parameters,
                            method=self.__class__)
        u = data.clone(values=u)
        return JaxResult2D(meta=meta, cost=total_cost, u=u)

    def grid_search(self, eta: Matrix,
                    *args,
                    unfkwargs: dict[str, Any] | None = None,
                    **kwargs) -> GridSearchResult:
        if unfkwargs is None:
            raise ValueError("unfkwarg must be provided")
        if len(args) == 0:
            raise ValueError("At least one hyperparameter must be provided")
        if len(args) > 2:
            raise ValueError("Up to two hyperparameters supported")
        if len(args) == 1:
            param, values = args[0]
            return self.grid_search_1D(eta, param, values, unfkwargs)
        if len(args) == 2:
            raise NotImplementedError

    def grid_search_1D(self, eta: Matrix, param: str, values: np.ndarray,
                       mask: np.ndarray,
                       unfkwargs: dict[str, Any]) -> GridSearchResult1D:
        kw = unfkwargs.copy()
        results: list[JaxResult1D] = []
        for i, value in tqdm(enumerate(values), total=len(values)):
            kw[param] = value
            res = self.unfold(**kw)
            results.append(res)
        return GridSearchResult1D(hyperparameter=param, grid=values, results=results)

    def grid_search_2D(self, eta: Matrix, param1: str, values1: np.ndarray,
                       param2: str, values2: np.ndarray,
                       mask: np.ndarray,
                       unfkwargs: dict[str, Any]) -> GridSearchResult2D:
        kw = unfkwargs.copy()
        results: list[JaxResult2D] = []
        values = list(product(values1, values2))
        for i, (value1, value2) in tqdm(enumerate(values), total=len(values)):
            print(value1, value2)
            kw[param1] = value1
            kw[param2] = value2
            res = self.unfold(**kw)
            results.append(res)
        return GridSearchResult2D(param1=param1, grid1=values1,
                                  param2=param2, grid2=values2, results=results)


@dataclass(kw_only=True)
class GridSearchResult:
    pass

@dataclass(kw_only=True)
class GridSearchResult1D(GridSearchResult):
    hyperparameter: str
    grid: np.ndarray
    results: list[JaxResult1D]

@dataclass(kw_only=True)
class GridSearchResult2D(GridSearchResult):
    param1: str
    grid1: np.ndarray
    param2: str
    grid2: np.ndarray
    results: list[JaxResult2D]



def unfold_GD(u, raw, R, loss, grad, value_and_grad, mask, max_iter=10,
              lr=1.0, abs_tol=1e-3, rel_tol=1e-3,
           use_abs_tol: bool = False, use_rel_tol: bool = False, **kwargs):
    max_iter = int(max_iter)
    total_cost = np.zeros(max_iter)
    #print(loss(u, R, raw))
    mask = ~mask
    alpha = 1e-3
    @jax.jit
    def body(u):
        tloss, g = value_and_grad(u, R, raw, alpha=alpha)
        u = u - lr*g
        u = u.at[mask].set(0)
        #tloss = loss(u, R, raw, alpha=alpha)
        return u, tloss
    j = -1
    for i in tqdm(range(max_iter)):
        u, total_cost[i] = body(u)#, alpha=kwargs['alpha'])

        if i > 0:
            if use_abs_tol and np.abs(total_cost[i] - total_cost[i-1]) < abs_tol:
                j = i
                break
            if use_rel_tol and np.abs(total_cost[i] - total_cost[i-1])/total_cost[i-1] < rel_tol:
                j = i
                break
        #g = grad(u, R, raw, **kwargs)
        #print(g)
        #u = u - lr*g
        #u = u.at[mask].set(0)
        #u = u.at[mask | jnp.isnan(g)].set(0)
        #total_cost[i] = loss(u, R, raw, **kwargs)
        #print(total_cost[i])
    return u**2, total_cost[:j+1]


def unfold_NAG(u, raw, R, loss, grad, value_and_grad, mask, max_iter=10,
               lr=1.0, momentum=0.9,
               abs_tol=1e-3, rel_tol=1e-3,
           use_abs_tol: bool = False, use_rel_tol: bool = False, **kwargs):
    max_iter = int(max_iter)
    total_cost = np.zeros(max_iter)
    #print(loss(u, R, raw))
    mask = ~mask
    alpha = 1e-3
    @jax.jit
    def body(u, v):
        u_first = u - momentum*v
        tloss, g = value_and_grad(u_first, R, raw, alpha=alpha)
        v = momentum*v + lr*g
        u = u - v
        u = u.at[mask].set(0)
        #tloss = loss(u, R, raw, alpha=alpha)
        return u, v, tloss
    j = -1
    v = jnp.zeros_like(u) #grad(u, R, raw, alpha=alpha)
    for i in tqdm(range(max_iter)):
        u, v, total_cost[i] = body(u, v)#, alpha=kwargs['alpha'])

        if i > 0:
            if use_abs_tol and np.abs(total_cost[i] - total_cost[i-1]) < abs_tol:
                j = i
                break
            if use_rel_tol and np.abs(total_cost[i] - total_cost[i-1])/total_cost[i-1] < rel_tol:
                j = i
                break
    return u**2, total_cost[:j]


def unfold_adam(u: Array, *, raw: Array, bg: Array, R: Array,
                G_ex: Array, loss, grad, value_and_grad, mask, max_iter=10,
               lr=0.001, beta1=0.9, beta2=0.999,
               abs_tol=1e-3, rel_tol=1e-3,
           use_abs_tol: bool = False, use_rel_tol: bool = False, **kwargs):
    max_iter = int(max_iter)
    total_cost = np.zeros(max_iter)
    #print(loss(u, R, raw))
    mask = ~mask
    alpha = kwargs.pop('alpha', 1.0)
    beta = kwargs.pop('beta', 1.0e-3)

    #@jax.jit
    eps = 1e-8
    n_err = jnp.where(raw <= eps, 3.0**2 ,raw)
    if jnp.sum(abs(bg)) == 0:
        bg_err = 0
    else:
        bg_err = jnp.where(bg <= eps, 3.0**2, bg)
    @jax.jit
    def body(u, mean, var, i):
        tloss, g = value_and_grad(u, R, G_ex, raw, bg, n_err, bg_err,
                                  alpha=alpha, beta=beta)
        mean = beta1*mean + (1-beta1)*g
        var = beta2*var + (1-beta2)*jnp.multiply(g, g)
        mean_cor = mean/(1-beta1**i)
        var_cor = var/(1-beta2**i)
        v = jnp.multiply(lr/(jnp.sqrt(var_cor) + eps), mean_cor)
        u = u - v
        u = u.at[mask].set(0)
        #tloss = loss(u, R, raw, alpha=alpha)
        return u, mean, var, tloss
    j = -1
    mean = jnp.zeros_like(u) #grad(u, R, raw, alpha=alpha)
    var = jnp.zeros_like(u) #grad(u, R, raw, alpha=alpha)
    disable_tqdm = kwargs.get('disable_tqdm', False)
    for i in tqdm(range(max_iter), disable=disable_tqdm):
        u, mean, var, total_cost[i] = body(u, mean, var, i+1)#, alpha=kwargs['alpha'])
        if i > 0:
            if use_abs_tol and np.abs(total_cost[i] - total_cost[i-1]) < abs_tol:
                j = i
                break
            if use_rel_tol and np.abs(total_cost[i] - total_cost[i-1])/total_cost[i-1] < rel_tol:
                j = i
                break
    return u**2, total_cost[:j]

@dataclass
class AdamParams:
    lr: float | Iterable[float] = 0.001     # learning rate
    beta1: float | Iterable[float] = 0.9    # decay rate for first moment estimate
    beta2: float | Iterable[float] = 0.999  # decay rate for second moment estimate
    max_iter: int | Iterable[int] = 10      # maximum number of iterations

    def get_iterables(self) -> list[str]:
        iterables = []
        for field in fields(self):
            if is_iterable(getattr(self, field.name)):
                iterables.append(field.name)
        return list(iterables)

def is_iterable(x) -> bool:
    try:
        iter(x)
        return True
    except TypeError:
        return False

def unfold_adam_1d(u, raw, bg, R, G_ex, loss, grad, value_and_grad, mask=None, max_iter=10,
               lr=0.001, beta1=0.9, beta2=0.999,
               abs_tol=1e-3, rel_tol=1e-3,
           use_abs_tol: bool = False, use_rel_tol: bool = False, **kwargs):
    max_iter = int(max_iter)
    total_cost = np.zeros(max_iter)
    if mask is not None:
        mask = ~mask
    eps = 1e-8
    n_err = jnp.where(raw <= eps, 3.0**2 ,raw)
    if jnp.sum(abs(bg)) == 0:
        bg_err = 0
    else:
        bg_err = jnp.where(bg <= eps, 3.0**2, bg)
    alpha = kwargs.pop('alpha', 1.0)
    beta = kwargs.pop('beta', 1.0e-3)

    @jax.jit
    def body(u, mean, var, i):
        tloss, g = value_and_grad(u, R, G_ex, raw, bg, n_err, bg_err, alpha=alpha, beta=beta)
        mean = beta1*mean + (1-beta1)*g
        var = beta2*var + (1-beta2)*jnp.multiply(g, g)
        mean_cor = mean/(1-beta1**i)
        var_cor = var/(1-beta2**i)
        v = jnp.multiply(lr/(jnp.sqrt(var_cor) + eps), mean_cor)
        u = u - v
        if mask is not None:
            u = u.at[mask].set(0)
        return u, mean, var, tloss
    j = -1
    mean = jnp.zeros_like(u) #grad(u, R, raw, alpha=alpha)
    var = jnp.zeros_like(u) #grad(u, R, raw, alpha=alpha)
    disable_tqdm = kwargs.get('disable_tqdm', False)
    for i in tqdm(range(max_iter), disable=disable_tqdm):
        u, mean, var, total_cost[i] = body(u, mean, var, i+1)#, alpha=kwargs['alpha'])
        if i > 0:
            if use_abs_tol and np.abs(total_cost[i] - total_cost[i-1]) < abs_tol:
                j = i
                break
            if use_rel_tol and np.abs(total_cost[i] - total_cost[i-1])/total_cost[i-1] < rel_tol:
                j = i
                break
    return u**2, total_cost[:j]


def unfold_1d(u, raw, R, loss, grad, max_iter=10, lr=1.0, abs_tol=1e-3, rel_tol=1e-3,
              use_abs_tol: bool = False, use_rel_tol: bool = False, **kwargs):
    max_iter = int(max_iter)
    i = 0
    total_cost = np.zeros(max_iter)
    while i < max_iter:
        #print(loss(u, R, raw))

        #nu = R@(u**2)
        #print(loss(u, R, raw))
        g = grad(u, R, raw, **kwargs)
        g = g.at[jnp.isnan(g)].set(0)
        #print(g)
        #if jnp.any(jnp.isnan(g)):
        #    raise RuntimeError("NaN gradient")
        #print(g)
        #print(jnp.log(nu))
        #print(jnp.log(raw))
        #print(jnp.sum(kl(R@(u**2), raw)))
        u = u - lr*g
        total_cost[i] = loss(u, R, raw, **kwargs)
        if i > 0:
            if use_abs_tol and np.abs(total_cost[i] - total_cost[i-1]) < abs_tol:
                break
            if use_rel_tol and np.abs(total_cost[i] - total_cost[i-1])/total_cost[i-1] < rel_tol:
                break
        i += 1
    return u, total_cost

