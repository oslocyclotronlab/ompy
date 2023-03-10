from __future__ import annotations
import numpy as np
from ..numbalib import njit, prange, NumpyArray, objmode
from .. import Vector, Matrix, Response, zeros_like
from .. import USE_GPU
from .unfolder import Unfolder, UnfoldedResult1D, UnfoldedResult2DCost
from dataclasses import dataclass
from .loss import loss_factory
from typing import TypeAlias, Literal
import time

if USE_GPU:
    from numba import cuda
    from numpy import float32

@dataclass#(frozen=True, slots=True)
class EMResult1D(UnfoldedResult1D):
    def best(self) -> Vector:
        return self.unfolded(len(self)-1)

# TODO
# [ ] Add convergence criteria
# [ ] Add GPU support
# [ ] Loss functions incompatible with matrix unfolding because of

class EM(Unfolder):
    """ Implementation of the Expectation maximization algorithm

    Scales very poorly as O(iterations * N^3) where N is matrix size.

    """
    def __init__(self, R: Matrix, G: Matrix, iterations: int, use_gpu: bool = USE_GPU):
        super().__init__(R, G)
        self.iterations = iterations
        self.use_gpu = False #use_gpu

    @classmethod
    def from_response(cls, response: Response, data: Matrix | Vector, iterations: int = 10) -> Unfolder:
        R = response.specialize_like(data)
        G = response.gaussian_like(data)
        return cls(R, G, iterations=iterations)

    def _unfold_vector(self, R: Matrix, data: Vector, initial: Vector, **kwargs) -> EMResult1D:
        it = kwargs.get('iterations', self.iterations)
        gpu = kwargs.get('use_gpu', self.use_gpu)
        convergence_criterion = kwargs.get('convergence_criterion', 'iterations')
        convergence_fn = criterion_factory(convergence_criterion, R, data, **kwargs)
        if gpu:
            if not USE_GPU:
                raise ValueError("GPU support not supported.")
            print("using gpu")
            u, cost = EM_cuda(R, data, initial, it)
        else:
            u, cost = EM_(R, data, initial, it, convergence_fn)
        return EMResult1D(R, data, initial, u, cost)

    def _unfold_matrix(self, R: Matrix, data: Matrix, initial: Matrix,
                       use_previous: bool, **kwargs) -> EMResult1D:
        it = kwargs.get('iterations', self.iterations)
        gpu = kwargs.get('use_gpu', self.use_gpu)
        convergence_criterion = kwargs.get('convergence_criterion', 'iterations')
        convergence_fn = criterion_factory(convergence_criterion, R, data, **kwargs)
        # TODO: Implement GPU support
        u_cube, cost_cube = EM_matrix(R, data, initial, it, convergence_fn, use_previous)
        return UnfoldedResult2DCost(R, data, initial, u_cube, cost_cube)


Criterion: TypeAlias = Literal['iterations', 'relative', 'absolute', 'likelihood']

def criterion_factory(criterion: Criterion, R: Matrix, raw: Vector, **kwargs):
    R = R.values
    raw = raw.values
    match criterion:
        case 'iterations':
            @njit
            def fn(u, k):
                return 0, False
            return fn
        case 'relative':
            rtol = kwargs.get('rtol', 1e-3)
            @njit
            def fn(u, k):
                if k == 0:
                    return np.NaN, False
                reldiff = np.abs((u[k] - u[k-1])/u[k-1])
                return np.sum(reldiff), np.all(reldiff < rtol)
            return fn
        case 'absolute':
            atol = kwargs.get('atol', 1)
            @njit
            def fn(u, k):
                if k == 0:
                    return np.NaN, False
                diff = np.abs(u[k] - u[k-1])
                return np.sum(diff), np.all(diff < atol)
            return fn
        case 'likelihood':
            kwargs = kwargs.copy()
            kwargs.setdefault('name', 'loglike')
            kwargs.setdefault('loglike', 'll3')
            loss = loss_factory(R=R, n=raw, **kwargs)
            rtol = kwargs.get('rtol', 1e-3)
            @njit
            def fn(u, k):
                uloss = loss(u[k, :])
                if k == 0:
                    return uloss, False
                ploss = loss(u[k-1, :])
                if ploss < 0:
                    return uloss, False
                return uloss, abs((uloss - ploss)/uloss) < rtol
            return fn
        case _:
            raise ValueError(f"Unknown convergence criterion: {criterion}.\nAvailable: {str(Criterion)}.")

def EM_matrix(R: Matrix, raw: Matrix, u0: Matrix, iterations: int, convergence_fn,
              use_previous: bool) -> tuple[np.ndarray, np.ndarray]:
    values, cost = _EM_matrix(R.values, raw.values, u0.values, iterations,
                              convergence_fn, use_previous)
    return values, cost

@njit
def _EM_matrix(R: np.ndarray, raw: np.ndarray, u0: np.ndarray,
               iterations: int, convergence_fn, use_previous: bool) -> tuple[np.ndarray, np.ndarray]:
    nrows, ncols = u0.shape
    cube = np.zeros((nrows, iterations, ncols))
    cost = np.zeros((nrows, iterations))
    cube[:, 0, :] = u0
    start_time = 0
    with objmode(start_time='int64'):
        start_time = time.time()
    last_best = 0
    for i in range(nrows):
        # Reshape the arrays to account for the "diagonal"
        raw_i = raw[i, :]
        j = last_nonzero(raw_i)
        if j < 2:
            continue
        raw_i = raw_i[:j]
        if use_previous and i > 0:
            u0_i = cube[i-1, last_best, :j]
        else:
            u0_i = u0[i, :j]
        last_best = _EM_mut(R[:j, :j], raw_i, u0_i, iterations, convergence_fn, cube[i, :, :], cost[i, :])

        with objmode():
            progress = (i / nrows) * 100
            elapsed_time = time.time() - start_time
            iterations_per_sec = i / elapsed_time
            bar = "[" + "=" * int(progress / 2) + ">" + " " * (50 - int(progress / 2)) + "]"
            #print(f"{bar}  {i}/{nrows}  {iterations_per_sec:.1f} iterations/sec")
            print(str(bar) + "  " + str(i) + "/" + str(nrows) + "  " + str(iterations_per_sec)[:4] + " its/sec")
    # swamp indices
    cube = np.swapaxes(cube, 0, 1)
    cost = np.swapaxes(cost, 0, 1)
    return cube, cost

@njit
def last_nonzero(arr: np.ndarray) -> int:
    j = len(arr) - 1
    while j >= 0:
        if arr[j] != 0:
            break
        j -= 1
    return j


def EM_(R: Matrix, raw: Vector, u0: Vector, iterations: int, convergence_fn) -> tuple[np.ndarray, np.ndarray]:
    values, cost = _EM(R.values, raw.values, u0.values, iterations, convergence_fn)
    return values, cost


@njit(parallel=True)
def _EM(R: np.ndarray, raw: np.ndarray, u0: np.ndarray, iterations: int, convergence_fn):
    cost = np.empty(iterations)
    u = np.empty((iterations, len(u0)))
    u[0] = u0
    k = _EM_mut(R, raw, u0, iterations, convergence_fn, u, cost)
    return u[:k], cost[:k]

@njit
def _EM_mut(R: np.ndarray, raw: np.ndarray, u0: np.ndarray, iterations: int, convergence_fn,
            u: np.ndarray, cost: np.ndarray) -> int:
    N = len(u0)
    Kj = R.sum(axis=0)
    Kj[Kj == 0] = 1.0  # Avoid /0
    for k in range(1, iterations):
        for j in prange(N):
            ks = 0.0
            for i in range(N):
                ls = 0.0
                for l in range(N):
                    ls += R[i, l]*u[k-1, l]
                if ls == 0: # Avoid /0
                    ls = 1.0
                ks += R[i,j]*raw[i] / ls

            u[k, j] = u[k-1, j] / Kj[j] * ks
        loss, converged = convergence_fn(u, k)
        cost[k] = loss
        if converged:
            return k
    return iterations

@cuda.jit
def _EM_cuda(R, raw, u, N, Kj, iterations):
    j = cuda.grid(1)
    if j < N:
        for k in range(1, iterations):
            ks = 0.0
            for i in range(N):
                ls = 0.0
                for l in range(N):
                    ls += R[i, l]*u[k-1, l]
                if ls == 0:
                    ls = 1.0
                ks += R[i,j]*raw[i] / ls

            u[k, j] = u[k-1, j] / Kj[j] * ks

def EM_cuda(R: Matrix, raw: Vector, u0: Vector, iterations: int):
    values = EM_cuda_(R.values, raw.values, u0.values, iterations)
    return values

def EM_cuda_(R, raw, u0, iterations):
    u = np.empty((iterations, len(u0)))
    u[0] = u0
    N = len(u0)
    Kj = R.sum(axis=0)
    Kj[Kj == 0] = 1.0
    block_size = 32
    num_blocks = (N + block_size - 1) // block_size
    R_gpu = cuda.to_device(R.astype(float32))
    raw_gpu = cuda.to_device(raw.astype(float32))
    u_gpu = cuda.to_device(u.astype(float32))
    _EM_cuda[num_blocks, block_size](R_gpu, raw_gpu, u_gpu, N, Kj, iterations)
    u = u_gpu.copy_to_host()
    return u