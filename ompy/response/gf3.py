from __future__ import annotations
from collections import OrderedDict

import numpy as np
from scipy.optimize import curve_fit
from sympy import diff, exp, log, lambdify

from . import Interpolator, Interpolation
from .numbalib import prange, njit, jitclass, float64
from .. import Vector
from ..stubs import Pathlike
from pathlib import Path

spec = OrderedDict()
spec["a"] = float64
spec["b"] = float64
spec["c"] = float64
spec["d"] = float64
spec["e"] = float64
spec["f"] = float64
spec["g"] = float64
spec["C"] = float64


@jitclass(spec)
class GSF3:

    def __init__(self, a, b, c, d, e, f, g, C):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.e = e
        self.f = f
        self.g = g
        self.C = C

    def call(self, E: np.ndarray) -> np.ndarray:
        e1 = 1e2
        e2 = 1e6
        a, b, c, d, e, f, g, C = self.a, self.b, self.c, self.d, self.e, self.f, self.g, self.C
        x = np.log(E / e1)
        y = np.log(E / e2)
        A = a + b * x + c * x ** 2
        B = d + e * y + f * y ** 2
        out = np.empty_like(E)
        for i in prange(len(out)):
            if A[i] < 0 or B[i] < 0:
                out[i] = np.inf
            else:
                out[i] = np.exp(C * (A[i] ** (-g) + B[i] ** (-g)) ** (-1 / g))
        return out


class GF3Interpolation(Interpolation):
    def __init__(self, points: Vector, a: float, b: float, c: float, d: float, e: float, f: float, g: float):
        super().__init__(points)
        self.gsf3 = GSF3(a, b, c, d, e, f, g, 1.0)

    def eval(self, E: np.ndarray) -> np.ndarray:
        return self.gsf3.call(E)

    def _metadata(self) -> dict[str, any]:
        return {"a": self.gsf3.a,
                "b": self.gsf3.b,
                "c": self.gsf3.c,
                "d": self.gsf3.d,
                "e": self.gsf3.e,
                "f": self.gsf3.f,
                "g": self.gsf3.g}

    @classmethod
    def from_path(cls, path: Pathlike) -> GF3Interpolation:
        path = Path(path)
        points, meta = Interpolation._load(path)
        a, b, c, d, e, f, g = meta["a"], meta["b"], meta["c"], meta["d"], meta["e"], meta["f"], meta["g"]
        return GF3Interpolation(points, a,b,c,d,e,f,g)


    def __str__(self) -> str:
        s = ("GF3Interpolation with parameters:\n"
             f"a = {self.gsf3.a:.3f}\n"
             f"b = {self.gsf3.b:.3f}\n"
             f"c = {self.gsf3.c:.3f}\n"
             f"d = {self.gsf3.d:.3f}\n"
             f"e = {self.gsf3.e:.3f}\n"
             f"f = {self.gsf3.f:.3f}\n"
             f"g = {self.gsf3.g:.3f}\n"
             f"C = {self.gsf3.C:.3f}")
        return s


class GF3Interpolator(Interpolator):
    def interpolate(self) -> GF3Interpolation:
        x, y = self.x, self.y
        p0 = [1.5, -0.5, 0.05, 0.5, 0.3, 0.08, 1.8]
        popt, pcov = curve_fit(gf3, x, y, p0=p0, jac=gf3_jac, maxfev=int(1e6))
        self.cov = pcov
        return GF3Interpolation(self.points, *popt)


def jacobian(fn, x, a, b, c, d, e, f, g):
    # Numba.njit complains a lot about lambdify and sympy, so
    # the code is not nice nor general.
    df = [diff(fn, s) for s in (a, b, c, d, e, f, g)]
    df_l = [lambdify((x, a, b, c, d, e, f, g), dfi) for dfi in df]
    fnc = [njit(dfi) for dfi in df_l]
    f1 = njit(df_l[0])
    f2 = njit(df_l[1])
    f3 = njit(df_l[2])
    f4 = njit(df_l[3])
    f5 = njit(df_l[4])
    f6 = njit(df_l[5])
    f7 = njit(df_l[6])

    @njit
    def foo(x, a, b, c, d, e, f, g):
        return np.vstack((f1(x, a, b, c, d, e, f, g), f2(x, a, b, c, d, e, f, g), f3(x, a, b, c, d, e, f, g),
                          f4(x, a, b, c, d, e, f, g), f5(x, a, b, c, d, e, f, g), f6(x, a, b, c, d, e, f, g),
                          f7(x, a, b, c, d, e, f, g))).T

    return foo


# The GF3 model is a pain to fit, as it is very unstable.
# However, it is analytical, so we can compute the Jacobian and use
# that to find the best fit. The derivatives are a pain to compute, so
# we use sympy to do it for us and then jit the result.
# Yields very fast code:).
def define_gf3():
    from sympy.abc import x, a, b, c, d, e, f, g
    x1 = log(x / 1e2)
    x2 = log(x / 1e6)

    f1 = a + b * x1 + c * x1 * x1
    f2 = d + e * x2 + f * x2 * x2
    fg = exp((f1 ** (-g) + f2 ** (-g)) ** (-1 / g))
    return fg, (x, a, b, c, d, e, f, g)


def define_gf3_jac():
    fg, (x, a, b, c, d, e, f, g) = define_gf3()
    jac = jacobian(fg, x, a, b, c, d, e, f, g, )
    func = njit(lambdify((x, a, b, c, d, e, f, g), fg))
    return func, jac


gf3, gf3_jac = define_gf3_jac()

# old code for differential evolution
#     X, Y = self.x, self.y
#
#     out = np.empty_like(X)
#     #x = np.empty_like(X)
#     #y = np.empty_like(Y)
#     @njit(parallel=True)
#     def loss_m(p, X, Y, x, y, A, B, out) -> float:
#         a, b, c, d, e, f, g, C = p
#         e1 = 1e2
#         e2 = 1e6
#         x[:] = np.log(X/e1)
#         y[:] = np.log(X/e2)
#         A[:] = a + b*x + c*x**2
#         B[:] = d + e*y + f*y**2
#         for i in prange(len(out)):
#             if A[i] < 0 or B[i] < 0:
#                 out[i] = np.inf
#             else:
#                 out[i] = np.exp(C*(A[i]**(-g) + B[i]**(-g))**(-1/g))
#         return np.sum((out - Y)**2)
#
#     @njit(parallel=True)
#     def loss(p) -> float:
#         out = np.empty_like(X)
#         a, b, c, d, e, f, g, C = p
#         e1 = 1e2
#         e2 = 1e6
#         x = np.log(X/e1)
#         y = np.log(X/e2)
#         A = a + b*x + c*x**2
#         B = d + e*y + f*y**2
#         for i in prange(len(out)):
#             if A[i] < 0 or B[i] < 0:
#                 out[i] = np.inf
#             else:
#                 out[i] = np.exp(C*(A[i]**(-g) + B[i]**(-g))**(-1/g))
#         return np.sum((out - Y)**2)
#
#     bounds = [(-10, 10), (-10, 10), (-1e-1, 1e-1),
#               (-10, 10), (-10, 10), (-1e-1, 1e-1),
#               (0.1, 20.0), (-1e5, 1e5)]
#     # SE
#     #  x: array([  1.40853485,  -0.55571893,   0.06829715,   0.71091904,
#     # 0.30544212,   0.08453222,   1.81302555, -17.72508124])
#     p0 = [1.5, -0.5, 0.05, 0.5, 0.3, 0.08, 1.8, -17]
#     #res = minimize(loss, p0, bounds=bounds, method="Nelder-Mead")
#     #res = differential_evolution(loss, bounds=bounds, disp=True, workers=-1,
#     #                             popsize=1000)
#     out = np.empty_like(X)
#     t1 = np.empty_like(X)
#     t2 = np.empty_like(Y)
#     t3 = np.empty_like(Y)
#     t4 = np.empty_like(Y)
#     res = differential_evolution(loss_m, args=(X, Y, t1, t2, t3, t4, out), bounds=bounds, disp=True, workers=-1,
#                                  popsize=1000, polish=True, maxiter=20)
#
#     intp = GF3Interpolation(self.points, *res.x)
#
#     return res, intp
