import numpy as np
from .unfolder import Unfolder, UnfoldedResult1DSimple, Errors1DCovariance, ResultMeta, Space, UnfoldedResult2DSimple, ResultMeta2D
from .. import Matrix, Vector
import time
from iminuit import Minuit
from .loss import loss_factory_bg, loss_factory, LogLike, LossFn, print_minuit_convergence, get_transform
from dataclasses import dataclass
from typing import Any
from tqdm.autonotebook import tqdm


@dataclass
class MLResult1D(Errors1DCovariance, UnfoldedResult1DSimple):
    loss: LossFn
    res: Minuit
    minuit: Minuit


class ML(Unfolder):
    def __init__(self, R: Matrix, G: Matrix):
        super().__init__(R, G)

    def _unfold_vector(self, R: Matrix, data: Vector, background: Vector | None,
                       initial: Vector,
                       mask: Vector,
                       space: Space,
                       G: Matrix | None = None,
                       limit: bool | Any = False,
                        **kwargs):
        transform = kwargs.pop('transform', 'id')
        tmap, imap, Jacobian = get_transform(transform)
        loss: LossFn = kwargs.pop('loss', 'loglike')
        ll: LogLike = kwargs.pop('loglike', 'll')
        mask_ = mask.values
        if background is None:
            loss = loss_factory(loss, R, data,
                                ll, mapfn=tmap, imapfn=imap, mask=mask_, **kwargs)
        else:
            loss = loss_factory_bg(loss, R, data, background,
                                   ll, mapfn=tmap, imapfn=imap, mask=mask_, **kwargs)
        mu0: np.ndarray = tmap(initial.values)
        m = Minuit(loss, mu0)
        m.tol = kwargs.pop('tol', 1e-1)
        m.errordef = Minuit.LEAST_SQUARES
        if isinstance(limit, bool) and limit:
            m.limits = (0, None)
        elif not isinstance(limit, bool):
            m.limits = limit
        start = time.time()
        ret: Minuit = m.migrad(iterate=100)
        ret2 = m.hesse()
        elapsed = time.time() - start
        u: Vector = data.clone(values=imap(np.asarray(m.values)))
        J = Jacobian(np.asarray(m.values))
        cov = J@np.asarray(m.covariance)@J.T
        return MLResult1D(meta=ResultMeta(time=elapsed, space=space),
                          R=R, raw=data, background=background, initial=initial,
                          G=G,
                          mask=mask,
                          u=u, cov=cov, loss=loss,
                          res=ret, minuit=m)

    @staticmethod
    def supports_background() -> bool:
        return True

from numba import njit

@njit
def nlog(x):
    return np.where(x <= 1e-3, 0.0, np.log(x))

def make_cost_numba(R, n, alpha=0.3e-1, beta=2.0):
    @njit
    def cost(mu):
        mu = mu ** 2
        nu = R @ mu
        entropy = -np.sum(mu * nlog(mu))
        #kl = np.sum(nu - n + n * np.log((n+1e-10) / (nu+1e-10)))
        kl = np.sum(nu - n + n * (nlog(n) - nlog(nu)))
        return kl + alpha * entropy + beta * (np.sum(nu) - np.sum(n))**2

    @njit
    def grad(mu):
        #mu = mu
        #nu = R @ mu
        # Obs! I computed derivative using scalars, not matrices and vectors
        entropy_grad = -np.sum(np.where(mu <= 0, 0.0, 2*mu*(2*np.log(mu) + 1)))
        kl_grad = 2*R@mu-2*n/mu
        #kl_grad = -2*n/mu
        return kl_grad + alpha*entropy_grad + beta * 4 * (np.sum(mu**2) - np.sum(n)) * mu
    return cost, grad


class ML_numba(ML):
    def _unfold_vector(self, R: Matrix, data: Vector, background: Vector | None,
                       initial: Vector,
                       mask: Vector,
                       space: Space,
                       G: Matrix | None = None,
                       limit: bool | Any = False,
                        **kwargs):
        #transform = kwargs.pop('transform', 'id')
        #tmap, imap, Jacobian = get_transform(transform)
        #loss: LossFn = kwargs.pop('loss', 'loglike')
        #ll: LogLike = kwargs.pop('loglike', 'll')
        #mu0: np.ndarray = tmap(initial.values)
        loss, grad = make_cost_numba(R.values, data.values)
        print("running")
        mu0 = np.sqrt(initial.values)
        print(loss(initial.values))
        m = Minuit(loss, mu0)#, grad=grad)
        m.strategy = 1
        m.tol = kwargs.pop('tol', 1e-1)
        m.errordef = Minuit.LEAST_SQUARES
        if isinstance(limit, bool) and limit:
            m.limits = (0, 8000.0)
        elif not isinstance(limit, bool):
            m.limits = limit
        start = time.time()
        ret: Minuit = m.migrad()#ncall=10000)#iterate=10, ncall=int(1e6))
        #ret2 = m.hesse()
        elapsed = time.time() - start
        #u: Vector = data.clone(values=imap(np.asarray(m.values)))
        u = data.clone(values=np.asarray(m.values)) ** 2
        print(loss(u.values))
        cov = np.asarray(m.covariance)
        #J = Jacobian(np.asarray(m.values))
        #cov = J@np.asarray(m.covariance)@J.T
        return MLResult1D(meta=ResultMeta(time=elapsed, space=space),
                          R=R, raw=data, background=background, initial=initial,
                          G=G,
                          mask=mask,
                          u=u, cov=cov, loss=loss,
                          res=ret, minuit=m)


import jax
from jax import numpy as jnp

def slog(x):
    return jnp.where(x <= 1e-5, 0.0, jnp.log(x))

def kl(nu, n) -> jax.Array:
    #return nu - n + n * jnp.log(n / (nu+1e-10) + 1e-10)
    return (nu - n) + n * (slog(n) - slog(nu))
    #return n * (slog(n) - slog(nu))

def entropy(mu):
    #return -jnp.sum(jnp.where(mu <= 0, 0.0, mu * jnp.log(mu)))
    return -jnp.sum(mu * slog(mu))

def difference_cost(n, nu):
    return (jnp.sum(n) - jnp.sum(nu))**2

def make_cost(R, n, alpha=10.0, beta=2.0):
    def cost(mu):
        mu = mu**2
        nu = R@mu
        zerocost = jnp.sum(mu > 0)
        return jnp.sum(kl(nu, n)) - alpha*entropy(mu) + beta*zerocost#difference_cost(n, nu)
    return cost

from concurrent.futures import ProcessPoolExecutor

class ML_JAX(ML):
    def __init__(self, R: Matrix, G: Matrix):
        super().__init__(R, G)
    def _unfold_vector(self, R: Matrix, data: Vector, background: Vector | None,
                       initial: Vector,
                       mask: Vector,
                       space: Space,
                       G: Matrix | None = None,
                       limit: bool | Any = False,
                        **kwargs):
        #transform = kwargs.pop('transform', 'id')
        tmap, imap, Jacobian = get_transform('sqrt')
        #loss: LossFn = kwargs.pop('loss', 'loglike')
        #ll: LogLike = kwargs.pop('loglike', 'll')
        #mu0: np.ndarray = tmap(initial.values)
        loss = make_cost(R.values, data.values, **kwargs)
        #print("Grading")
        grad = jax.grad(loss)
        #print("jiting")
        loss = jax.jit(loss, static_argnames=('alpha', 'beta'))
        grad = jax.jit(grad, static_argnames=('alpha', 'beta'))
        #print("running")
        mu0 = initial.values
        mu0 = np.sqrt(mu0)
        m = Minuit(loss, mu0, grad=grad)
        m.strategy = 0
        m.tol = kwargs.pop('tol', 1e-1)
        m.errordef = Minuit.LEAST_SQUARES
        if isinstance(limit, bool) and limit:
            m.limits = (0, 8000.0)
        elif not isinstance(limit, bool):
            m.limits = limit
        start = time.time()
        ret: Minuit = m.migrad(iterate=100)#ncall=int(5e4))
        #ret2 = m.hesse()
        elapsed = time.time() - start
        #u: Vector = data.clone(values=imap(np.asarray(m.values)))
        u = data.clone(values=np.asarray(m.values))
        u = u**2

        cov = None if m.covariance is None else m.covariance
        J = Jacobian(np.asarray(m.values))
        cov = J@np.asarray(m.covariance)@J.T
        return MLResult1D(meta=ResultMeta(time=elapsed, space=space),
                          R=R, raw=data, background=background, initial=initial,
                          G=G,
                          mask=mask,
                          u=u, cov=cov, loss=loss,
                          res=ret, minuit=m)

    def __unfold_matrix(self, R: Matrix, data: Matrix, background: Matrix | None, initial: Matrix,
                       use_previous: bool, space: Space, G: Matrix | None = None, **kwargs) -> UnfoldedResult2DSimple:
        """ A default, simple implementation of unfolding a matrix

         """
        best = np.zeros((data.shape[0], R.shape[1]))
        N = data.shape[0]
        time = np.zeros(N)
        bins = np.zeros(N)
        def process_row(i):
            vec: Vector = data.iloc[i, :]
            j = vec.last_nonzero()
            vec: Vector = vec.iloc[:j]
            if background is not None:
                bvec: Vector | None = background.iloc[i, :j]
            else:
                bvec = None
            if use_previous and i > 0:
                init = best[i-1, :j]
            else:
                init = initial.iloc[i, :j]
            R_: Matrix = R.iloc[:j, :j]
            if G is not None:
                G_ = G.iloc[:j, :j]
            else:
                G_ = None
            res = self._unfold_vector(R_, vec, bvec, init, None, space, G=G_, **kwargs)

        with ProcessPoolExecutor() as executor:
            args = list(range(N))
            results = list(tqdm(executor.map(lambda args: process_row(*args), args), total=N))

        for i, res in enumerate(results):
            u = res.best()
            j = len(u)
            best[i, :j] = u.values
            time[i] = res.time
            bins[i] = j

        return UnfoldedResult2DSimple(meta=ResultMeta2D(time=time, bins=bins, space=space), R=R,
                                      raw=data, u=best, background=background,
                                      initial=initial, mask=None)
