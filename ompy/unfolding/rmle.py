from __future__ import annotations
from .unfolder import Unfolder
from .result1d import UnfoldedResult1DSimple, Cost1D, Parameters1D, ResultMeta1D, Result
from .result2d import UnfoldedResult2DSimple, Cost2D, Parameters2D, ResultMeta2D
from .stubs import Space
from .. import Matrix, Vector, OPTAX_AVAILABLE
from ..stubs import Plot1D, Axes, array1D
import numpy as np
import time
from tqdm.autonotebook import tqdm
from dataclasses import dataclass, fields
import matplotlib.pyplot as plt
from typing import Any, TypedDict, Iterable, Callable, TypeAlias, Literal
from functools import partial
from pathlib import Path
from itertools import product
from typing_extensions import override
import os
from abc import ABC, abstractmethod

def is_jupyter_notebook():
    return 'JPY_PARENT_PID' in os.environ

if OPTAX_AVAILABLE:
    import optax


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

def onecost(mu, C):
    # Smooth approximation of the Heaviside step function using a logistic function
    #return jnp.sum(0.5 * (1 + jnp.tanh((mu - C) / (1e-6 + C / 10))))
    return jnp.sum(0.5*(1 + 2/3.141592*jnp.arctan((mu - C)/(C/100))))

def onecost_2(mu, C):
    # Smooth approximation of the Heaviside step function using a logistic function
    #return jnp.sum(0.5 * (1 + jnp.tanh((mu - C) / (1e-6 + C / 10))))
    return jnp.sum(0.5*(1 + 2/3.141592*jnp.arctan((mu - C)/(C/10))))

def to_tau(mu):
    return jnp.sqrt(mu)
    #return jnp.sqrt(mu)
    #return jnp.log(mu + 1e-10)

def from_tau(tau):
    #return jnp.where(tau < 0, tau**2, tau)
    return tau**2 + 1e-3*tau
    #return tau**2
    #return jnp.exp(tau)

def cost(tau, R, G_ex, y, bg=None,
         alpha=0.0, alpha_c=1.0):
    mu_ = from_tau(tau)
    if bg is not None:
        mu, beta = jnp.vsplit(mu_, 2)
        nu = G_ex@mu@R
        nu = nu + beta
        loglike = jnp.sum(kl(nu, y)) + jnp.sum(kl(beta, bg))
        cost = loglike
    else:
        mu = mu_
        nu = G_ex@mu@R
        loglike = jnp.sum(kl(nu, y))

        penalty = alpha*onecost(mu, alpha_c)**2
        # Normalize row wise
        #prop = mu / jnp.sum(mu, axis=0)[None, 1]
        #entropy = -jnp.sum(prop * jnp.log(prop+1e-10))
        #penalty = alpha * entropy
        cost = loglike + penalty

    return cost
    

def cost_1d(tau, R, y, bg=None,
            alpha=0.0, alpha_c=1.0, alpha_bg=0.0, 
            alpha_xi_alpha=0.0, alpha_xi_beta=0.0, alpha_xi_c=10.0):#, alpha=0.3e-1):
    mu_ = from_tau(tau)
    # "mu" might be [prompt..., background...]
    if bg is not None:
        mu = mu_[:-len(bg)]
        beta = mu_[-len(bg):]
        nu = R@mu
        nu = nu + beta
        loglike = jnp.sum(kl(nu, y)) + jnp.sum(kl(beta, bg))
        beta_diff = jnp.diff(beta)
        beta_cost = alpha_bg * jnp.sum(beta_diff**2)
        loglike = loglike + beta_cost
    else:
        mu = mu_
        nu = R@mu
        loglike = jnp.sum(kl(nu, y))
    # Rescaling seems to make the optimization much slower
    penalty = alpha*onecost(mu, alpha_c)**2
    #penalty = alpha*onecost(mu / (1+jnp.abs(mu)), 0.01)**2
    # try a entropy penalty
    #prop = jax.nn.softmax(mu)
    #prop = mu / jnp.sum(mu)
    #penalty = alpha * -jnp.sum(prop * jnp.log(prop+1e-10))
    cost = loglike + penalty

    aux = {'loglike': loglike, 'penalty': penalty}

    return cost, aux
    #nu = jnp.log(nu + 1e-1)
    #n = jnp.log(n + 1e-1)
    #return jnp.sum(jnp.abs(nu - n)) + alpha*onecost(mu)**2 #- beta*jnp.sum(entropy(mu))# + difference_cost(n, nu)
    #total = jnp.sum(kl(nu, n)) #- alpha*jnp.sum(entropy(mu)) + beta*difference_cost(n, nu)

    
def cost_1d_v3(tau, R, y, bg=None, R_xi=None, 
               xi_constraints = None,
            alpha=0.0, alpha_c=1.0, alpha_bg=0.0,
            alpha_xi_alpha=1.0, alpha_xi_beta=1.0, alpha_xi_c=10.0):
    """
    Unified cost function for unfolding with optional background and contaminants.
    
    Parameters
    ----------
    tau : array
        Parameters to optimize (includes all components)
    R : array
        Main response matrix
    y : array
        Observed spectrum
    bg : array, optional
        Background spectrum data
    R_xsi : list of arrays, optional
        Response matrices for contaminants
    alpha : float
        Regularization parameter for main spectrum
    alpha_c : float
        Regularization parameter for contaminants
    zeta : float
        Smoothness parameter for background
    """
    # Split parameters based on what components are present
    n_bg = len(bg) if bg is not None else 0
    n_cont = len(R_xi) if R_xi is not None else 0
    # n_bg is the length of the vector, and all vectors must be the same length
    n_cont *= n_bg
    
    # Handle main spectrum
    main_tau = tau[:-n_bg-n_cont] if (n_bg + n_cont) > 0 else tau
    mu = main_tau**2  # Non-negativity via square
    nu = R@mu

    # Add regularization for main spectrum if needed
    if alpha > 0:
        cost += alpha * onecost(mu, alpha_c)**2
    
    # Initialize total folded spectrum
    nu_total = nu.copy()
    
    # Initialize cost
    cost = 0.0 
    
    # Add background component if present
    if bg is not None:
        bg_tau = tau[-n_bg-n_cont:-n_cont] if n_cont > 0 else tau[-n_bg:]
        beta = bg_tau**2  # Non-negativity
        nu_total = nu_total + beta
        # Likelihood of background
        cost += jnp.sum(kl(beta, bg))
        
        # Background smoothness
        if alpha_bg > 0:
            beta_diff = jnp.diff(beta)
            cost += alpha_bg * jnp.sum(beta_diff**2)

    # Handle contaminants
    for i in range(n_cont):
        # Unpack the xi vector
        mu_xi_i = tau[-n_bg*i-n_bg:-n_bg*i]
        nu_xi_i = R_xi[i]@mu_xi_i
        nu_total = nu_total + nu_xi_i

        # Add penalty
        probs = jnp.softmax(mu_xi_i)
        # entropy
        penalty = -jnp.sum(probs * jnp.log(probs))
        cost += alpha_xi_alpha * penalty
        
        # We can't allow the peak to be outside of the constraints
        idx = jnp.arange(len(mu))
        left, right = xi_constraints[i]
        lower_mask = jax.nn.sigmoid((idx - left) / alpha_xi_c)
        upper_mask = jax.nn.sigmoid((right - idx) / alpha_xi_c)
        valid_region = lower_mask * upper_mask
        # penalize values outside boundaries
        penalty = jnp.sum(mu_xi_i**2 * (1 - valid_region))
        cost += alpha_xi_beta * penalty

    # Final likelihood using total folded spectrum
    cost += jnp.sum(kl(nu_total, y))
    
    
    return cost


def cost_1d_v2(mu, R, G_eg, G_ex, n, bg, n_err, bg_err,
            alpha=1.0, beta=1e-3):#, alpha=0.3e-1):
    mu = mu**2
    nu = R@mu
    eta = G_eg@mu
    # The entropy must be taken in eta space
    return jnp.sum(kl(nu, n)) + alpha*onecost(mu)**2 #- beta*jnp.sum(entropy(eta))# + difference_cost(n, nu)
    #total = jnp.sum(kl(nu, n)) #- alpha*jnp.sum(entropy(mu)) + beta*difference_cost(n, nu)

def cost_components_from_result(result, eta, alpha=0, beta=0):
    return cost_components(result.raw.values, result.unfolded().values,
                           result.R.values, eta=eta, G_eg=result.G.values, G_ex=result.G_ex,
                           alpha=result.meta.kwargs['alpha'])


def cost_components(n, mu, R, eta=None, G_eg=None, G_ex=None, alpha=0, beta=0):
    """ Return the loss, regularization and validation components of the cost function
    """
    nu = R@mu
    loss = jnp.sum(kl(nu, n))
    regularization = onecost(mu)**2# - beta*jnp.sum(entropy(eta))
    if eta is not None:
        eta_ = G_eg@mu
        validation = jnp.sum(kl(eta_, eta))
        return loss, regularization, validation
    return loss, regularization


@dataclass(kw_only=True)
class RMLEResult2D(Cost1D, UnfoldedResult2DSimple):
    beta: Matrix | None = None

    def _save(self, path: Path, exist_ok: bool = False):
        Cost2D._save(self, path, exist_ok)
        UnfoldedResult2DSimple._save(self, path, exist_ok)

    @classmethod
    def _load(cls, path: Path) -> dict[str, np.ndarray | Matrix]:
        a = Cost2D._load(path)
        b = UnfoldedResult2DSimple._load(path)
        return a | b

@dataclass(kw_only=True)
class RMLEResult1D(Cost1D, UnfoldedResult1DSimple):
    beta: Vector | None = None
    
    def _save(self, path: Path, exist_ok: bool = False):
        UnfoldedResult1DSimple._save(self, path, exist_ok)
        Cost1D._save(self, path, exist_ok)

    @classmethod
    def _load(cls, path: Path) -> dict[str, np.ndarray | Vector]:
        a = Cost1D._load(path)
        b = UnfoldedResult1DSimple._load(path)
        return a | b

class RMLE(Unfolder):
    @staticmethod
    @override
    def supports_background():
        return True

    def richardson_rate(self, tol: float | None = None) -> float:
        #kappa = np.linalg.cond(self.R.values, tol)
        # get the largest and smallest singular values
        s = np.linalg.svd(self.R.values, compute_uv=False)
        s_max = s.max()
        s_min = s.min()
        #return 1 - 2 / (kappa + 1)
        return 2 / (s_max + s_min)


    @override
    def _unfold_vector(self, R: Matrix, data: Vector, background: Vector | None,
                       initial: Vector, space: Space,
                       G: Matrix | None = None, **kwargs) -> RMLEResult1D:
        # Check if cost_1d parameters are iterable
        # At most one can be iterable
        # Construct matrix NxM where N is vector and M is parameters
        # How can the cost function be applied correctly?
        # Probably need to use jax.vmap? Or linear algebra trick
        # How does this work with Bootstrap?
        # Perhaps instead have a _unfold_vectors()?
        # Add mask for trilu
        # Add mask to Result

        loss = jax.jit(cost_1d)
        grad = jax.jit(jax.grad(cost_1d))
        value_and_grad = jax.jit(jax.value_and_grad(cost_1d, has_aux=True))
        u = jnp.asarray(initial.values)
        R_ = jnp.asarray(R.values)
        raw = jnp.asarray(data.values)

        
        if 'mask' not in kwargs:
            mask = np.ones_like(data, dtype=bool)
        else:
            mask = kwargs.pop('mask')
            if isinstance(mask, (str, int, float)):
                idx = data.index(mask)
                mask = np.ones_like(data, dtype=bool)
                mask[idx:] = False
        mask = jnp.asarray(mask)
        
        if background is None:
            bg = None
        else:
            bg = jnp.asarray(background.values)

        start = time.time()
        if 'lr' not in kwargs or kwargs['lr'] == 'auto':
            kwargs['lr'] = self.richardson_rate()
        u, total_loss, aux = unfold_adam_1d(u, raw=raw, bg=bg, R=R_, G_ex=None,
                                       loss=loss, grad=grad, value_and_grad=value_and_grad,
                                       mask=mask,
                                       **kwargs)
        u = np.asarray(u)
        # If we have a background, we need to unpack u
        if bg is not None:
            mu = u[:-len(bg)]
            beta = u[-len(bg):]
            beta = background.clone(values=beta)
        else:
            mu = u
            beta = None

        u = data.clone(values=mu)
        elapsed = time.time() - start
        parameters = Parameters1D(R=R, raw=data, background=background, initial=initial,
                                  kwargs=kwargs, G=G, mask=np.asarray(mask))
        meta = ResultMeta1D(time=elapsed, space=space, parameters=parameters,
                            method=self.__class__)
        return RMLEResult1D(meta=meta, cost=total_loss, u=u, beta=beta, aux=aux)


    def _unfold_matrix(self, R: Matrix, data: Matrix, background: Matrix | None, initial: Matrix,
                       use_previous: bool, space: Space, G: Matrix, G_ex: Matrix,
                       mask: np.ndarray | Literal['tril', 'last nonzero'] = 'last nonzero', **kwargs) -> UnfoldedResult2DSimple:

        bins = np.zeros(data.shape[0])
        if isinstance(mask, str):
            if mask == 'tril':
                # this only works for square matrices
                mask = np.tril(np.ones_like(data, dtype=bool))
            elif mask == 'last nonzero':
                mask = np.zeros_like(data, dtype=bool)
                for i in range(data.shape[0]):
                    j = data.iloc[i, :].last_nonzero()
                    mask[i, :j] = True
                    bins[i] = j
            else:
                raise ValueError(f"Unknown mask option {mask}. Must be 'tril' or 'last nonzero', or array")
        mask = jnp.asarray(mask)

        u = jnp.asarray(np.sqrt(initial.values))
        R_ = jnp.asarray(R.values.T)
        G_ex_ = jnp.asarray(G_ex.values)
        n = jnp.asarray(data.values)
        if background is None:
            bg = None
        else:
            bg = jnp.asarray(background.values)
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
            case 'optax':
                unfold = unfold_optax
            case _:
                raise ValueError(f"Unknown method {method}")
        value_and_grad = jax.jit(jax.value_and_grad(cost), static_argnames=('alpha', 'beta'))

        if 'lr' not in kwargs or kwargs['lr'] == 'auto':
            kwargs['lr'] = self.richardson_rate()
        start = time.time()
        u, total_cost = unfold(u, raw=n, bg=bg, R=R_, G_ex=G_ex_,
                               loss=loss, grad=grad, value_and_grad=value_and_grad,
                               mask=mask, **kwargs)
        elapsed = time.time() - start
        # If we have a background, we need to unpack u
        if bg is not None:
            mu, beta = jnp.vsplit(u, 2)
            beta = background.clone(values=np.asarray(beta))
        else:
            mu = u
            beta = None
        mu = data.clone(values=np.asarray(mu))

        # TODO Add Response coefficients as optimisation parameter
        # TODO Loop over Ex and make error
        #print("Approximating variance")
        #hessian = jax.jit(jax.jacfwd(jax.jacrev(cost)))
        #hessian = hessian(u[160], R_, n[160])
        parameters = Parameters2D(R=R, raw=data, background=background, initial=initial,
                                  G=G, kwargs=kwargs | {'method': method},
                                  G_ex=G_ex, mask=mask)
        meta = ResultMeta2D(time=elapsed, space=space, parameters=parameters,
                            method=self.__class__)
        return RMLEResult2D(meta=meta, cost=total_cost, u=mu, beta=beta)

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
                       unfkwargs: dict[str, Any]) -> GridSearchResult1D:
        kw = unfkwargs.copy()
        results: list[RMLEResult1D] = []
        if 'leave_tqdm' not in kw:
            kw['leave_tqdm'] = False
        bar = tqdm(enumerate(values), total=len(values))
        for i, value in bar: 
            bar.set_postfix({param: value})
            kw[param] = value
            res = self.unfold(**kw)
            results.append(res)
        return GridSearchResult1D(hyperparameter=param, grid=values, results=results)

    def grid_search_2D(self, eta: Matrix, param1: str, values1: np.ndarray,
                       param2: str, values2: np.ndarray,
                       mask: np.ndarray,
                       unfkwargs: dict[str, Any]) -> GridSearchResult2D:
        kw = unfkwargs.copy()
        results: list[RMLEResult2D] = []
        values = list(product(values1, values2))
        bar = tqdm(enumerate(values), total=len(values))
        for i, (value1, value2) in bar:
            bar.set_postfix()
            kw[param1] = value1
            kw[param2] = value2
            res = self.unfold(**kw)
            results.append(res)
        return GridSearchResult2D(param1=param1, grid1=values1,
                                  param2=param2, grid2=values2, results=results)

    def tune_learning_rate(self, lr: np.ndarray | None = None, unfkwargs: dict[str, Any] | None = None) -> tuple[GridSearchResult1D, float]:
        # Perform a grid search over the learning rates
        if lr is None:
            lr = np.logspace(-3, 1, 10)
        if unfkwargs is None:
            unfkwargs = {}
        if 'iterations' not in unfkwargs:
            unfkwargs['iterations'] = 1000
        result = self.grid_search_1D(None, 'lr', lr, unfkwargs)

        # Find the learning rate that gave the lowest cost
        min_cost = np.inf
        for i, res in enumerate(result.results):
            if res.cost[-1] < min_cost:
                min_cost = res.cost[-1]
                best = i

        return result, result.grid[best]


@dataclass(kw_only=True)
class GridSearchResult:
    pass

@dataclass(kw_only=True)
class GridSearchResult1D(GridSearchResult):
    hyperparameter: str
    grid: np.ndarray
    results: list[RMLEResult1D]

    def plot(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()

        costs = [res.cost[-1] for res in self.results]
        line = ax.plot(self.grid, costs, '-o')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(self.hyperparameter)
        ax.set_ylabel('Cost')
        ax.set_title('Grid Search Results')
        ax.grid(True)

        return ax, line


@dataclass(kw_only=True)
class GridSearchResult2D(GridSearchResult):
    param1: str
    grid1: np.ndarray
    param2: str
    grid2: np.ndarray
    results: list[RMLEResult2D]



def unfold_GD(u, raw, R, loss, grad, value_and_grad, mask, iterations=10,
              lr=1.0, abs_tol=1e-3, rel_tol=1e-3,
           use_abs_tol: bool = False, use_rel_tol: bool = False, **kwargs):
    iterations = int(iterations)
    total_cost = np.zeros(iterations)
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
    for i in tqdm(range(iterations)):
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


def unfold_NAG(u, raw, R, loss, grad, value_and_grad, mask, iterations=10,
               lr=1.0, momentum=0.9,
               abs_tol=1e-3, rel_tol=1e-3,
           use_abs_tol: bool = False, use_rel_tol: bool = False, **kwargs):
    iterations = int(iterations)
    total_cost = np.zeros(iterations)
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
    for i in tqdm(range(iterations)):
        u, v, total_cost[i] = body(u, v)#, alpha=kwargs['alpha'])

        if i > 0:
            if use_abs_tol and np.abs(total_cost[i] - total_cost[i-1]) < abs_tol:
                j = i
                break
            if use_rel_tol and np.abs(total_cost[i] - total_cost[i-1])/total_cost[i-1] < rel_tol:
                j = i
                break
    return u**2, total_cost[:j]


Schedule: TypeAlias = Callable[[int], float]


class Scheduler(ABC):
    @abstractmethod
    def make(self) -> Schedule: ...

    def plot(self, x: np.ndarray | None = None, ax = None):
        if ax is None:
            fig, ax = plt.subplots()
        if x is None:
            x = np.arange(0, 1_000_000, 10_000)
        fn = np.vectorize(self.make())
        ax.plot(x, fn(x))
        return ax

class ConstScheduler(Scheduler):
    def __init__(self, lr):
        self.lr = lr

    def make(self):
        lr = self.lr
        return lambda t: lr


class ExponentialDecayScheduler(Scheduler):
    def __init__(self, initial_lr, decay_rate):
        self.initial_lr = initial_lr
        self.decay_rate = decay_rate

    def make(self):
        # Return a function that computes the learning rate given the epoch
        lr0 = self.initial_lr
        theta = self.decay_rate
        def lr_schedule(epoch):
            return lr0 * (theta ** epoch)
        return lr_schedule

    @classmethod
    def from_point(cls, initial_lr, epoch, lr):
        # Calculate the necessary decay_rate
        decay_rate = (lr / initial_lr) ** (1 / epoch)
        return cls(initial_lr, decay_rate)


class StepScheduler(Scheduler):
    def __init__(self, initial_lr, drop_factor, drop_every):
        self.initial_lr = initial_lr
        self.drop_factor = drop_factor
        self.drop_every = drop_every

    def make(self):
        def lr_schedule(epoch):
            steps = epoch // self.drop_every
            return self.initial_lr - (self.drop_factor * steps)
        return lr_schedule


class StepsScheduler(Scheduler):
    def __init__(self, points):
        # Ensure points are sorted by epoch
        self.points = sorted(points)

    def make(self):
        def lr_schedule(epoch):
            for i in range(len(self.points) - 1):
                if epoch < self.points[i + 1][0]:
                    return self.points[i][1]
            return self.points[-1][1]
        return lr_schedule


class CosineAnnealingScheduler(Scheduler):
    def __init__(self, initial_lr, min_lr, T_max):
        """
        :param initial_lr: The initial learning rate.
        :param min_lr: The minimum learning rate.
        :param T_max: The maximum number of iterations (epochs) for the schedule.
        """
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.T_max = T_max

    def make(self):
        def lr_schedule(epoch):
            # Cosine Annealing formula
            return self.min_lr + (self.initial_lr - self.min_lr) * (1 + np.cos(np.pi * epoch / self.T_max)) / 2
        return lr_schedule


def unfold_adam(u: Array, *, raw: Array, bg: Array, R: Array,
                G_ex: Array, loss, grad, value_and_grad, mask, iterations=10,
                lr=0.001, beta1=0.9, beta2=0.999,
                abs_tol=1e-3, rel_tol=1e-3,
                use_abs_tol: bool = False, use_rel_tol: bool = False,
                lr_scheduler: Scheduler | None = None,
                **kwargs):
    iterations = int(iterations)
    total_cost = np.zeros(iterations)
    #print(loss(u, R, raw))
    mask = ~mask
    
    alpha = kwargs.pop('alpha', 0.0)
    beta = kwargs.pop('beta', 0.0)

    if lr_scheduler is None:
        lr_schedule = ConstScheduler(lr).make()
    else:
        lr_schedule = lr_scheduler.make()
    #print(lr_scheduler, lr_schedule(0))

    # Background and prompt are combined in the input
    u = to_tau(u)
    if bg is not None:
        mask = jnp.vstack((mask, jnp.zeros_like(bg, dtype=bool)))
        u = jnp.vstack((u, 1.0+jnp.zeros_like(bg)))

    #@jax.jit
    eps = 1e-8

    @jax.jit
    def body(u, mean, var, i):
        lr_i = lr_schedule(i)
        tloss, g = value_and_grad(u, R, G_ex, raw, bg,
                                  alpha=alpha)
        mean = beta1*mean + (1-beta1)*g
        var = beta2*var + (1-beta2)*jnp.multiply(g, g)
        mean_cor = mean/(1-beta1**i)
        var_cor = var/(1-beta2**i)
        v = jnp.multiply(lr_i/(jnp.sqrt(var_cor) + eps), mean_cor)
        u = u - v
        u = u.at[mask].set(0)
        #tloss = loss(u, R, raw, alpha=alpha)
        return u, mean, var, tloss
    j = -1
    mean = jnp.zeros_like(u) #grad(u, R, raw, alpha=alpha)
    var = jnp.zeros_like(u) #grad(u, R, raw, alpha=alpha)
    disable_tqdm = kwargs.get('disable_tqdm', False)
    leave_tqdm = kwargs.get('leave_tqdm', True)
    min_interval = kwargs.get('mininterval_tqdm', None)
    # Check if is in jupyter
    if is_jupyter_notebook() and min_interval is None:
        # We need to slow down the output since jupyter can't keep up
        min_interval = 1
    bar = tqdm(range(iterations), disable=disable_tqdm, leave=leave_tqdm, mininterval=min_interval)
    time0 = time.time()
    for i in bar:
        u, mean, var, total_cost[i] = body(u, mean, var, i+1)#, alpha=kwargs['alpha'])
        # Check if cost is NaN
        if jnp.isnan(total_cost[i]):
            raise RuntimeError(f"NaN cost at iteration {i}")    
    
        if i > 0:
            if use_abs_tol and np.abs(total_cost[i] - total_cost[i-1]) < abs_tol:
                j = i
                bar.container.close()
                break
            if use_rel_tol and np.abs(total_cost[i] - total_cost[i-1])/total_cost[i-1] < rel_tol:
                j = i
                bar.container.close()
                break
        if min_interval is not None and time.time() - time0 > min_interval:
            std = rolling_standard_deviation(total_cost[:i], 10)
            cv = rolling_coefficient_of_variation(total_cost[:i], 10)
            ema = exponential_moving_average(total_cost[:i], 0.1)
            bar.set_postfix({'cost': total_cost[i], 'std': std, 'cv': cv, 'ema': ema}, refresh=True)
            time0 = time.time()

    return from_tau(u), total_cost[:j]


def rolling_standard_deviation(cost_array, window_size):
    if len(cost_array) < window_size:
        return np.std(cost_array)  # If not enough data, use the entire array
    return np.std(cost_array[-window_size:])

def rolling_coefficient_of_variation(cost_array, window_size):
    if len(cost_array) < window_size:
        mean = np.mean(cost_array)
        std_dev = np.std(cost_array)
    else:
        recent_values = cost_array[-window_size:]
        mean = np.mean(recent_values)
        std_dev = np.std(recent_values)
    
    return std_dev / mean if mean != 0 else float('inf')

def exponential_moving_average(cost_array, alpha=0.1):
    if len(cost_array) == 0:
        return np.inf
    ema = [cost_array[0]]  # Start with the first cost
    for cost in cost_array[1:]:
        ema.append(alpha * cost + (1 - alpha) * ema[-1])
    return ema[-1]

def unfold_optax(*args, **kwargs):
    raise ImportError("Optax is not available on your system")

def requires_lr(f) -> bool:
    try:
        return 'learning_rate' in f.__code__.co_varnames
    except AttributeError:
        if f == optax.nadam:
            return True
    return False  # Je ne sais pas, let the error be thrown

def requires_max_learning_rate(f) -> bool:
    try:
        return 'max_learning_rate' in f.__code__.co_varnames
    except AttributeError:
        if f == optax.nadam:
            return False
    return False  # Je ne sais pas, let the error be thrown

def requires_values(f) -> bool:
    match f:
        case optax.polyak_sgd:
            return True
        case _:
            return False

if OPTAX_AVAILABLE:
    def unfold_optax(u, raw, bg, R, G_ex, loss, grad, value_and_grad, mask, **kwargs):
        
        # Initialize the Adam optimizer
        rename_key(kwargs, 'lr', 'learning_rate')
        num_iters = int(kwargs.pop('iterations', 1000))
        bar = tqdm(range(num_iters), disable=kwargs.pop('disable_tqdm', False),
                leave=kwargs.pop('leave_tqdm', True))
        break_at_nan = kwargs.pop('break_at_nan', True)

        method = kwargs.pop('optimizer', optax.adam)
        alpha = kwargs.pop('alpha', 0.0)
        beta = kwargs.pop('beta', 0.0)
        optim_kwargs = kwargs.pop('optimizer_kwargs', {})
        rename_key(optim_kwargs, 'lr', 'learning_rate')
        if 'learning_rate' in optim_kwargs and 'learning_rate' in kwargs:
            raise ValueError("Only provide 'learning_rate' in 'optimizer_kwargs' or 'kwargs', not both")
        if 'learning_rate' not in optim_kwargs and 'learning_rate' not in kwargs and requires_lr(method):
            optim_kwargs['learning_rate'] = 0.001
        elif 'learning_rate' in kwargs:
            optim_kwargs['learning_rate'] = kwargs.pop('learning_rate')

        if requires_max_learning_rate(method):
            if 'learning_rate' in optim_kwargs:
                rename_key(optim_kwargs, 'learning_rate', 'max_learning_rate')
            # Let optax throw the error for missing keyword

        
        # All keyword arguments should be handled
        if len(kwargs) > 0:
            raise ValueError(f"Unknown keyword arguments: {kwargs.keys()}")

        optimizer = method(**optim_kwargs)

        # Initialize the optimizer state
        state = optimizer.init(u)

        # Perform the optimization
        total_cost = np.zeros(num_iters)
        for i in bar:
            # Compute the gradient
            value, gradients = value_and_grad(u, R, G_ex, raw, bg, None, None, alpha=alpha)

            # Update the parameters and the optimizer state
            updates, state = optimizer.update(gradients, state, u, value=value)
            u = optax.apply_updates(u, updates)
            total_cost[i] = value ##loss(u, R, G_ex, raw, bg, None, None, alpha=alpha)
            if break_at_nan and not np.isfinite(value):
                i = i+1
                break

            std = rolling_standard_deviation(total_cost[:i], 10)
            cv = rolling_coefficient_of_variation(total_cost[:i], 10)
            ema = exponential_moving_average(total_cost[:i], 0.1)
            bar.set_postfix({'cost': total_cost[i], 'std': std, 'cv': cv, 'ema': ema}, refresh=True)

        return u**2, total_cost[:i]

def rename_key(kw, old_key, new_key, default_value=None):
    if old_key in kw and new_key in kw:
        raise ValueError(f"Only provide '{old_key}' or '{new_key}', not both")
    if old_key in kw:
        kw[new_key] = kw.pop(old_key)
    elif new_key not in kw and default_value is not None:
        kw[new_key] = default_value


@dataclass
class AdamParams:
    lr: float | Iterable[float] = 0.001     # learning rate
    beta1: float | Iterable[float] = 0.9    # decay rate for first moment estimate
    beta2: float | Iterable[float] = 0.999  # decay rate for second moment estimate
    iterations: int | Iterable[int] = 10      # maximum number of iterations

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

def unfold_adam_1d(u, raw, bg, R, G_ex, loss, grad, value_and_grad, mask=None, iterations=10,
               lr=0.001, beta1=0.9, beta2=0.999,
               abs_tol=1e-3, rel_tol=1e-3,
           use_abs_tol: bool = False, use_rel_tol: bool = False, 
           break_in: int = 1000, **kwargs):
    iterations = int(iterations)
    total_cost = np.zeros(iterations)
    loglike = np.zeros(iterations)
    penalty = np.zeros(iterations)
    if mask is not None:
        mask = ~mask
    else:
        mask = jnp.zeros_like(raw, dtype=bool)

    # we combine the prompt and the background
    u = to_tau(u)
    if bg is not None:
        mask = jnp.concatenate([mask, jnp.zeros_like(bg, dtype=bool)])
        u = jnp.concatenate([u, 1.0+jnp.zeros_like(bg)])
    eps = 1e-8
    #n_err = jnp.where(raw <= eps, 3.0**2 ,raw)
    alpha = kwargs.pop('alpha', 0.0)
    alpha_c = kwargs.pop('alpha_c', 1.0)
    # contaminant keywords
    alpha_xi_alpha = kwargs.pop('alpha_xi_alpha', 1.0)
    alpha_xi_beta = kwargs.pop('alpha_xi_beta', 1.0)
    alpha_xi_c = kwargs.pop('alpha_xi_c', 10.0)
    #beta = kwargs.pop('beta', 0.0)

    @jax.jit
    def body(u, mean, var, i):
        (tloss, aux), g = value_and_grad(u, R, raw, bg, alpha=alpha,
                                  alpha_c=alpha_c,
                                  alpha_xi_alpha=alpha_xi_alpha,
                                  alpha_xi_beta=alpha_xi_beta,
                                  alpha_xi_c=alpha_xi_c)
        mean = beta1*mean + (1-beta1)*g
        var = beta2*var + (1-beta2)*jnp.multiply(g, g)
        mean_cor = mean/(1-beta1**i)
        var_cor = var/(1-beta2**i)
        v = jnp.multiply(lr/(jnp.sqrt(var_cor) + eps), mean_cor)
        u = u - v
        u = u.at[mask].set(0)
        return u, mean, var, tloss, aux
    j = -1
    mean = jnp.zeros_like(u) #grad(u, R, raw, alpha=alpha)
    var = jnp.zeros_like(u) #grad(u, R, raw, alpha=alpha)
    disable_tqdm = kwargs.get('disable_tqdm', False)
    leave_tqdm = kwargs.get('leave_tqdm', True)
    min_interval = kwargs.get('min_interval_tqdm', None)
    # Check if is in jupyter
    if is_jupyter_notebook() and min_interval is None:
        # We need to slow down the output since jupyter can't keep up
        min_interval = 0.5
    bar = tqdm(range(iterations), disable=disable_tqdm, leave=leave_tqdm, mininterval=min_interval)
        
    for i in bar:
        u, mean, var, total_cost[i], aux = body(u, mean, var, i+1)#, alpha=kwargs['alpha'])

        loglike[i] = aux['loglike']
        penalty[i] = aux['penalty']
        
        if i > 0:
            if use_abs_tol and np.abs(total_cost[i] - total_cost[i-1]) < abs_tol:
                j = i
                break
            if use_rel_tol and np.abs(total_cost[i] - total_cost[i-1])/total_cost[i-1] < rel_tol:
                j = i
                break

        bar.set_postfix({'cost': total_cost[i], 'loglike': loglike[i], 'penalty': penalty[i]}, refresh=True)
    return from_tau(u), total_cost[:j], {'loglike': loglike[:j], 'penalty': penalty[:j]}


def unfold_1d(u, raw, R, loss, grad, iterations=10, lr=1.0, abs_tol=1e-3, rel_tol=1e-3,
              use_abs_tol: bool = False, use_rel_tol: bool = False, **kwargs):
    iterations = int(iterations)
    i = 0
    total_cost = np.zeros(iterations)
    while i < iterations:
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


def kl_2(nu: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """Compute the KL divergence."""
    # Avoid division by zero and log of zero
    safe_nu = jnp.where(nu == 0, 1e-10, nu)
    safe_y = jnp.where(y == 0, 1e-10, y)
    return safe_nu * jnp.log(safe_nu / safe_y) - safe_nu + safe_y

def loglikelihood(mu: jnp.ndarray, R: jnp.ndarray, G_in: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """Compute the log-likelihood."""
    nu = G_in @ (mu ** 2) @ R
    return jnp.sum(kl_2(nu, y))   


def hessian_vector_product(mu: jnp.ndarray, v: jnp.ndarray, R: jnp.ndarray, G_in: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """Compute the Hessian-vector product H * v."""
    # First derivative (gradient)
    grad = jax.grad(loglikelihood, argnums=0)
    # Corrected line: Pass the function `grad` instead of calling it
    hvp = jax.jvp(grad, (mu,R, G_in, y), (v,))[1]
    return hvp

def estimate_largest_eigenvalue(mu: jnp.ndarray, R: jnp.ndarray, G_in: jnp.ndarray, y: jnp.ndarray, 
                                num_iters: int = 100, tol: float = 1e-6, 
                                key: jax.random.PRNGKey = jax.random.PRNGKey(0)) -> float:
    """Estimate the largest eigenvalue of the Hessian using Power Iteration."""
    # Initialize a random vector v with unit norm
    v = jax.random.normal(key, shape=mu.shape)
    v = v / jnp.linalg.norm(v)
    
    for _ in range(num_iters):
        # Compute H * v
        Hv = hessian_vector_product(mu, v, R, G_in, y)
        
        # Compute the norm of Hv
        Hv_norm = jnp.linalg.norm(Hv)
        if Hv_norm == 0:
            break
        
        # Normalize Hv to get the next iteration's vector
        v_new = Hv / Hv_norm
        
        # Check for convergence (cosine similarity)
        cosine_sim = jnp.dot(v, v_new)
        if jnp.abs(cosine_sim - 1.0) < tol:
            break
        
        v = v_new
    
    # Estimate of the largest eigenvalue
    eigenvalue = jnp.dot(v, hessian_vector_product(mu, v, R, G_in, y))
    return eigenvalue
import jax
import jax.numpy as jnp
from functools import partial

def hessian_vector_product(f, x, v):
    """Compute Hessian-vector product without materializing the full Hessian."""
    def grad_dot_v(x):
        return jnp.vdot(jax.grad(f)(x), v)
    return jax.grad(grad_dot_v)(x)

def power_iteration(hvp_func, x_shape, num_iterations=5, tol=1e-6):
    """
    Compute largest eigenvalue using power iteration.
    
    Args:
        hvp_func: Function that computes Hessian-vector product
        x_shape: Shape of the input vector
        num_iterations: Maximum number of iterations
        tol: Convergence tolerance
    
    Returns:
        largest_eigenvalue: Estimated largest eigenvalue
    """
    # Initialize random vector and normalize it
    key = jax.random.PRNGKey(0)
    v = jax.random.normal(key, x_shape)
    v = v / jnp.linalg.norm(v)
    
    def body_fun(carry):
        i, v, prev_eigenvalue, _ = carry
        # Compute Hv
        Hv = hvp_func(v)
        # Calculate Rayleigh quotient (approximate eigenvalue)
        eigenvalue = jnp.vdot(v, Hv)
        # Normalize the new vector
        v_new = Hv / jnp.linalg.norm(Hv)
        # Check convergence
        converged = jnp.abs(eigenvalue - prev_eigenvalue) < tol
        return (i + 1, v_new, eigenvalue, converged)

    def cond_fun(carry):
        i, _, _, converged = carry
        return jnp.logical_and(i < num_iterations, jnp.logical_not(converged))

    # Initial state
    init_state = (0, v, jnp.inf, False)
    
    # Run power iteration
    final_state = jax.lax.while_loop(cond_fun, body_fun, init_state)
    _, _, eigenvalue, _ = final_state
    
    return eigenvalue

def compute_largest_eigenvalue(loglikelihood, mu, R, G_in, y):
    """
    Compute the largest eigenvalue of the Hessian of the log-likelihood function.
    """
    # Create partial function with fixed parameters except mu
    def f(mu):
        return loglikelihood(mu, R, G_in, y)
    
    # Create Hessian-vector product function
    def hvp(v):
        return hessian_vector_product(f, mu, v)
    
    # Run power iteration
    eigenvalue = power_iteration(hvp, mu.shape)
    return eigenvalue

# Example usage
@jax.jit
def find_largest_eigenvalue(mu, R, G_in, y):
    return compute_largest_eigenvalue(loglikelihood, mu, R, G_in, y)



def loglikelihood_hessian(mu: Array, R: Array, G_in: Array, y: Array) -> Array:
    """Compute the Hessian of the log-likelihood term (KL divergence)."""
    def loglikelihood(mu):
        nu = G_in@(mu**2)@R
        return jnp.sum(kl(nu, y))
    
    return jax.hessian(loglikelihood)(mu)

def regularization_hessian(mu: Array, alpha: float, C: float) -> Array:
    """Compute the Hessian of the regularization term (onecost)."""
    def regularization(mu):
        return alpha * onecost(mu**2, C)**2
    
    return jax.hessian(regularization)(mu)

def get_max_eigenvalue(H: Array) -> float:
    """Compute the largest eigenvalue of a Hessian matrix."""
    # Using power iteration method for efficiency
    v = jnp.ones_like(H[0])
    for _ in range(10):  # Usually converges quickly
        v_new = H @ v
        v = v_new / jnp.linalg.norm(v_new)
    
    return jnp.dot(v, H @ v)


def estimate_lipschitz_constant(samples: Array, R: Array, y: Array) -> float:
    """
    Estimate Lipschitz constant using gradient norm bounding over a specified range.
    
    Args:
        theta_range: Array of theta values to evaluate over
        R: Response matrix
        G_ex: Extended response matrix
        n: Data vector
        alpha: Regularization parameter
    Returns:
        L: Estimated Lipschitz constant
    """
    L_estimates = []
    
    # Define gradient function with inlined cost (no bg case)
    @jax.jit
    def grad_f(mu):
        nu = R@mu
        return jnp.sum(kl(nu, y))
    
    grad_func = jax.grad(grad_f)

    @jax.jit
    def estimate_lipschitz_constant_pair(theta1, theta2):
        grad1 = grad_func(theta1)
        grad2 = grad_func(theta2)
        grad_diff_norm = jnp.linalg.norm(grad1 - grad2)
        param_diff_norm = jnp.linalg.norm(theta1 - theta2)
        return grad_diff_norm / param_diff_norm
    
    # Compare all pairs of points in the range
    for i in tqdm(range(len(samples)),leave=False):
        theta1 = samples[i]
        for j in range(i+1, len(samples)):
            theta2 = samples[j]
            L_estimates.append(estimate_lipschitz_constant_pair(theta1, theta2))
    
    return np.max(np.asarray(L_estimates))


def estimate_lipschitz_constant_reg(samples: Array, alpha: float = 0.0, C: float = 1.0) -> float:
    """
    Estimate Lipschitz constant using gradient norm bounding over a specified range.
    
    Args:
        theta_range: Array of theta values to evaluate over
        R: Response matrix
        G_ex: Extended response matrix
        n: Data vector
        alpha: Regularization parameter
    Returns:
        L: Estimated Lipschitz constant
    """
    L_estimates = []
    
    # Define gradient function with inlined cost (no bg case)
    @jax.jit
    def grad_f(mu):
        return alpha*onecost(mu, C)**2
    
    grad_func = jax.grad(grad_f)

    @jax.jit
    def estimate_lipschitz_constant_pair(theta1, theta2):
        grad1 = grad_func(theta1)
        grad2 = grad_func(theta2)
        grad_diff_norm = jnp.linalg.norm(grad1 - grad2)
        param_diff_norm = jnp.linalg.norm(theta1 - theta2)
        return grad_diff_norm / param_diff_norm

    # Compare all pairs of points in the range
    for i in tqdm(range(len(samples)),leave=False):
        theta1 = samples[i]
        for j in range(i+1, len(samples)):
            theta2 = samples[j]
            L_estimates.append(estimate_lipschitz_constant_pair(theta1, theta2))
    
    return np.max(np.asarray(L_estimates))

def estimate_learning_rate_1d(
    R: Matrix, y: Vector, mu: Vector,
    sample_points: int | list[Array] = 100,
    alpha: float = 0.0,
    C: float = 1.0
) -> float:
    """
    Estimate optimal learning rate using Lipschitz constant over a specified range.
    
    Args:
        R: Response matrix
        y: Data vector
        sample_points: Number of points to sample over
        alpha: Regularization parameter
    Returns:
        lr: Estimated optimal learning rate
    """
    if isinstance(sample_points, int):
        samples = [np.abs(mu.values + mu.values*np.random.normal(0, 1)) + 
                   np.random.normal(0, 1000, size=(y.shape[0])) for _ in range(sample_points)]
    else:
        samples = sample_points
    R = jnp.array(R)
    y = jnp.array(y)
    # Estimate Lipschitz constant with inlined cost function
    L0 = estimate_lipschitz_constant(samples, R, y)
    L1 = estimate_lipschitz_constant_reg(samples, alpha=alpha, C=C)

    print(f"Estimated Lipschitz constant for loglikelihood: {L0}")
    print(f"Estimated Lipschitz constant for regularization: {L1}")
    
    # Compute learning rate as 1/L
    lr_1 = 1.0 / (L0 + L1 + 1e-10)  # Add small constant for numerical stability
    lr_2 = 2.0 / (L0 + L1 + 1e-10)
    return lr_1, lr_2



def model_loglikelihood(res: Result) -> float:
    y = res.raw
    nu = res.best_folded()
    loglike = jnp.sum(kl(jnp.asarray(nu), jnp.asarray(y)))
    return loglike
