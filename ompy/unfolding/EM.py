from __future__ import annotations
import numpy as np
from ..numbalib import njit, prange, NumpyArray
from .. import Vector, Matrix, Response, zeros_like
from .unfolder import Unfolder, UnfoldedResult1D
from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class EMResult1D(UnfoldedResult1D):
    def best(self) -> Vector:
        return self.unfolded(len(self)-1)


class EM(Unfolder):
    def __init__(self, R: Matrix, G: Matrix, iterations: int):
        super().__init__(R, G)
        self.iterations = iterations

    @classmethod
    def from_response(cls, response: Response, data: Matrix | Vector, iterations: int = 10) -> Unfolder:
        R = response.specialize_like(data)
        G = response.gaussian_like(data)
        return cls(R, G, iterations=iterations)

    def _unfold_vector(self, R: Matrix, data: Vector, initial: Vector, **kwargs) -> EMResult1D:
        it = kwargs.get('iterations', self.iterations)
        u = EM_(R, data, initial, it)
        return EMResult1D(R, data, initial, u, [])

    def _unfold_matrix(self, data: Matrix, **kwargs) -> Matrix:
        raise NotImplementedError("EM unfolding for matrices not implemented")


def EM_(R: Matrix, raw: Vector, u0: Vector, iterations: int):
    values = _EM(R.values, raw.values, u0.values, iterations)
    return values


@njit(parallel=True)
def _EM(R: np.ndarray, raw: np.ndarray, u0: np.ndarray, iterations: int):
    u = np.empty((iterations, len(u0)))
    u[0] = u0
    N = len(u0)
    Kj = R.sum(axis=0)
    Kj[Kj == 0] = 1.0  # Avoid /0
    for k in range(1, iterations):
        for j in prange(N):
            ks = 0.0
            for i in range(N):
                ls = 0.0
                for l in range(N):
                    ls += R[i,l]*u[k-1, l]
                if ls == 0: # Avoid /0
                    ls = 1.0
                ks += R[i,j]*raw[i] / ls

            u[k, j] = u[k-1, j] / Kj[j] * ks
    return u
