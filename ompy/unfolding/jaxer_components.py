from __future__ import annotations
from .unfolder import Unfolder
from .result2d import (UnfoldedResult2DSimple, Cost2D,
                       Parameters2D, ResultMeta2D)
from .result1d import ComponentsRes
from .stubs import Space
from .. import Matrix, Vector
from ..stubs import Plot1D, Axes
import numpy as np
import time
from tqdm.autonotebook import tqdm
from dataclasses import dataclass
import matplotlib.pyplot as plt
from typing import Any
from functools import partial
from pathlib import Path
from ..response import Components, Response
from .jaxer import JaxResult2D


import jax
from jax import numpy as jnp

"""
TODO
-[ ] Optimize vector
-[ ] Hyperparameter search for NAG++
     Parameter transform, logistic or inverse hyperbolic tangent
-[x] GPU. Remember to set correct environment variables
-[ ] Test more optimizers
-[ ] Initial optimizer hyperparameter search
-[ ] KL + ME
-[ ] Background
-[ ] Why does taking SiRi into account worsen the result?
"""

# JAX Jit hates this function
def slog(x):
    return jnp.where(x <= 1e-5, 0.0, jnp.log(x))

def kl(nu, n):
    #return nu - n + n * jnp.log(n / (nu+1e-10) + 1e-10)
    #return (nu - n) + n * (slog(n) - slog(nu))
    eps = 1e-5
    mask = (nu <= eps) | (n <= eps)
    return jnp.where(mask, 0.0, (nu - n) + n * (jnp.log(n) - jnp.log(nu)))
    #return (nu - n) + n * (jnp.log(n) - jnp.log(nu))

def entropy(mu):
    mask = mu <= 1e-5
    return jnp.where(mask, 0.0, mu * jnp.log(mu))
    #return -jnp.sum(mu * slog(mu))
    #return mu * jnp.log(mu)

def difference_cost(n, nu):
    return (jnp.sum(n) - jnp.sum(nu))**2

def onecost(mu):
    return jnp.sum(mu > 0)

def cost(mu, w, FE, SE, DE, AP, Compton, G, n):
    mu = mu**2
    w = hyperbolic_tangent_map(w)
    R = w[0] * FE + w[1] * SE + w[2] * DE + w[3] * AP + w[4] * Compton
    #R = FE + SE + DE + Compton
    R = R / jnp.sum(R, axis=1)[:, jnp.newaxis]
    R = (G@R).T
    nu = mu@R
    e = jnp.where(n <= 1e-5, 3.0**2, n)
    return jnp.sum((nu - n)**2/e) # + onecost(mu)

def hyperbolic_tangent_map(x):
    a = 0.9
    b = 1.1
    return a + ((jnp.tanh(x) + 1)/2) * (b - a)

def inverse_hyperbolic_tangent_map(x):
    a = 0.9
    b = 1.1
    return jnp.arctanh(2*(x - a)/(b - a) - 1)

@dataclass(kw_only=True)
class JaxCResult2D(ComponentsRes, JaxResult2D):
    ws: np.ndarray
    def _save(self, path: Path, exist_ok: bool = False):
        Cost2D._save(self, path, exist_ok)
        UnfoldedResult2DSimple._save(self, path, exist_ok)

    @classmethod
    def _load(cls, path: Path) -> dict[str, np.ndarray | Matrix]:
        a = Cost2D._load(path)
        b = UnfoldedResult2DSimple._load(path)
        return a | b


class JaxerComponents(Unfolder):
    def __init__(self, R: Matrix, G: Matrix, response: Response):
        self.response = response
        super().__init__(R, G)

    @classmethod
    def from_response(cls, response: Response, data: Matrix | Vector,
                      **kwargs) -> JaxerComponents:
        R = response.specialize_like(data)
        G = response.gaussian_like(data)
        return cls(R, G, response)

    @staticmethod
    def supports_background():
        return False

    def _unfold_vector(self, R: Matrix, data: Vector, background: Vector | None,
                       initial: Vector,
                       mask: Vector,
                       space: Space,
                       G: Matrix | None = None,
                        **kwargs):
        raise NotImplementedError()


    def _unfold_matrix(self, R: Matrix, data: Matrix, background: Matrix | None, initial: Matrix,
                       use_previous: bool, space: Space, G: Matrix, **kwargs) -> UnfoldedResult2DSimple:
        mask = np.zeros_like(data, dtype=bool)
        bins = np.zeros(data.shape[0])
        for i in range(data.shape[0]):
            j = data.iloc[i, :].last_nonzero()
            mask[i, :j] = True
            bins[i] = j
        u = jnp.asarray(np.sqrt(initial.values))
        #R_ = jnp.asarray(R.values.T)
        G_ = jnp.asarray(G.values)
        n = jnp.asarray(data.values)
        mask = jnp.asarray(mask)
        loss = jax.jit(cost, )
        grad = jax.grad(cost, argnums=(0, 1))
        grad = jax.jit(grad)
        method = kwargs.pop('method', 'adam')
        components = kwargs.pop('components', Components())
        w = jnp.asarray([components.FE, components.SE, components.DE, components.AP, components.compton])
        matrices: dict[str, Matrix] = self.response.component_matrices_like(R)
        FE, SE, DE, AP, compton = [jnp.asarray(matrices[name].values) for name in
                                   ['FE', 'SE', 'DE', 'AP', 'compton']]
        match method:
            case 'adam':
                unfold = unfold_adam
            case _:
                raise ValueError(f"Unknown method {method}")
        value_and_grad = jax.jit(jax.value_and_grad(cost), static_argnames=('alpha'))
        start = time.time()
        u, w, total_cost, ws = unfold(u, n, w, FE, SE, DE, AP, compton, G_, loss, grad, value_and_grad, mask, **kwargs)
        elapsed = time.time() - start
        # TODO Add Response coefficients as optimisation parameter
        # TODO Loop over Ex and make error
        #print("Approximating variance")
        #hessian = jax.jit(jax.jacfwd(jax.jacrev(cost)))
        #hessian = hessian(u[160], R_, n[160])

        R_ = w[0] * FE + w[1] * SE + w[2] * DE + w[3] * AP + w[4] * compton
        R_ = np.asarray(R_)
        T = R_.sum(axis=1)
        T[T == 0] = 1
        R_ = R_ / T[:, np.newaxis]
        R = R.clone(values=(G@R_).T)
        parameters = Parameters2D(R=R, raw=data, background=background, initial=initial,
                                  G=G, kwargs=kwargs | {'method': method})
        meta = ResultMeta2D(time=elapsed, space='GR', parameters=parameters,
                            method=self.__class__)
        u = data.clone(values=u)
        c = dict(FE=w[0], SE=w[1], DE=w[2], AP=w[3], compton=w[4])
        return JaxCResult2D(meta=meta, cost=total_cost, u=u, components=c,ws=ws)



def unfold_adam(u, raw, w, FE, SE, DE, AP, Compton, G, loss, grad, value_and_grad, mask, max_iter=10,
               lr=0.001, beta1=0.9, beta2=0.999,
               abs_tol=1e-3, rel_tol=1e-3,
           use_abs_tol: bool = False, use_rel_tol: bool = False, **kwargs):
    max_iter = int(max_iter)
    total_cost = np.zeros(max_iter)
    #print(loss(u, R, raw))
    mask = ~mask
    alpha = 1e-3
    #@jax.jit
    w = inverse_hyperbolic_tangent_map(w)
    eps = 1e-8
    @jax.jit
    def body(u, mean_u, var_u,
             w, mean_w, var_w, i):
        ug, wg = grad(u, w, FE, SE, DE, AP, Compton, G, raw)
        mean_w = beta1*mean_w + (1-beta1)*wg
        var_w = beta2*var_w + (1-beta2)*jnp.multiply(wg, wg)
        mean_w_cor = mean_w/(1-beta1**i)
        var_w_cor = var_w/(1-beta2**i)
        v = jnp.multiply(lr/(jnp.sqrt(var_w_cor) + eps), mean_w_cor)
        w = w - v

        mean_u = beta1*mean_u + (1-beta1)*ug
        var_u = beta2*var_u + (1-beta2)*jnp.multiply(ug, ug)
        mean_u_cor = mean_u/(1-beta1**i)
        var_u_cor = var_u/(1-beta2**i)
        v = jnp.multiply(lr/(jnp.sqrt(var_u_cor) + eps), mean_u_cor)
        u = u - v
        u = u.at[mask].set(0)

        tloss = loss(u, w, FE, SE, DE, AP, Compton, G, raw)
        return u, mean_u, var_u, w, mean_w, var_w, tloss
    j = -1
    mean_u = jnp.zeros_like(u) #grad(u, R, raw, alpha=alpha)
    var_u = jnp.zeros_like(u) #grad(u, R, raw, alpha=alpha)
    mean_w = jnp.zeros_like(w) #grad(u, R, raw, alpha=alpha)
    var_w = jnp.zeros_like(w) #grad(u, R, raw, alpha=alpha)
    disable_tqdm = kwargs.get('disable_tqdm', False)
    ws = np.zeros((max_iter, len(w)))

    for i in tqdm(range(max_iter), disable=disable_tqdm):
        u, mean_u, var_u, w, mean_w, var_w, total_cost[i] = body(u, mean_u, var_u,
                                                                 w, mean_w, var_w, i+1)#, alpha=kwargs['alpha'])
        ws[i, :] = hyperbolic_tangent_map(w)
        if i > 0:
            if use_abs_tol and np.abs(total_cost[i] - total_cost[i-1]) < abs_tol:
                j = i
                break
            if use_rel_tol and np.abs(total_cost[i] - total_cost[i-1])/total_cost[i-1] < rel_tol:
                j = i
                break
    return u**2, hyperbolic_tangent_map(w), total_cost[:j], ws[:j, :]
