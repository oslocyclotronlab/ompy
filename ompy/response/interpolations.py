from .. import Vector
from .interpolation import Interpolator, Interpolation, LinearInterpolator, LinearInterpolation, PoissonInterpolation
from .numbalib import prange, njit, jitclass, float64
from collections import OrderedDict
import numpy as np
from scipy.optimize import curve_fit
from .gf3 import GF3Interpolation, GF3Interpolator

ESCAPE_LIMIT = 511*2
ESCAPE_LINEAR_TO= 2.5e3
ANNIHILATION_LINEAR_TO = 4e3
assert(ESCAPE_LINEAR_TO > ESCAPE_LIMIT)

@njit
def zero(x):
    return np.zeros_like(x)

@njit
def loglerp(X, a, b, Y1, Y2):
    w = 1 / (1 + np.exp(-1/b * (X - a)))
    return (1-w)*Y1 + w*Y2

class EscapeInterpolation(PoissonInterpolation):
    def __init__(self, points: Vector, linear: LinearInterpolation, gf3: GF3Interpolation,
                 linear_to: float):
        super().__init__(points)
        self.gf3 = gf3
        self.linear = linear
        self.linear_to = linear_to

    def eval(self, points: np.ndarray) -> np.ndarray:
        # 0 to 1022keV, then linear to 2500keV, then GF3 model
        # For a smooth transition
        linear = np.maximum(zero(points), self.linear.eval(points))
        gf3 = self.gf3.eval(points)
        gf3[~np.isfinite(gf3)] = 0
        # Logistic lerp to remove the discontinuity
        return loglerp(points, self.linear_to, 100, linear, gf3)

class EscapeInterpolator(Interpolator):
    linear_to = ESCAPE_LINEAR_TO
    def interpolate(self, linear_to: float | None = None) -> EscapeInterpolation:
        X, Y = self.x, self.y
        # Linear
        linear_to = linear_to or self.linear_to
        mask_lin = (ESCAPE_LIMIT < X) & (X < linear_to)
        lin_vec = self.points.iloc[mask_lin]
        lin_intp: LinearInterpolation = LinearInterpolator(lin_vec).interpolate()
        # GSF3
        mask_gf = X > ESCAPE_LINEAR_TO
        gf_vec = self.points.iloc[mask_gf]
        gf = GF3Interpolator(gf_vec)
        gf_intp = gf.interpolate()
        # HACK: Wrong, but for convenience
        self.cov = gf.cov
        return EscapeInterpolation(self.points, lin_intp, gf_intp, linear_to=linear_to)

class AnnihilationInterpolation(EscapeInterpolation): ...
class AnnihilationInterpolator(Interpolator):
    linear_to = ANNIHILATION_LINEAR_TO
    def interpolate(self, linear_to: float | None = None) -> AnnihilationInterpolation:
        X, Y = self.x, self.y
        # Linear
        linear_to = linear_to or self.linear_to
        mask_lin = (ESCAPE_LIMIT < X) & (X < linear_to)
        lin_vec = self.points.iloc[mask_lin]
        lin_intp: LinearInterpolation = LinearInterpolator(lin_vec).interpolate()
        # GSF3
        mask_gf = X > ESCAPE_LINEAR_TO
        gf_vec = self.points.iloc[mask_gf]
        gf = GF3Interpolator(gf_vec)
        gf_intp = gf.interpolate()
        return AnnihilationInterpolation(self.points, lin_intp, gf_intp, linear_to=linear_to)

@njit
def fwhm(E: np.ndarray, a0: float, a1: float, a2: float) -> np.ndarray:
    return np.sqrt(a0 + a1*E + a2*E**2)

@njit
def fwhm_jac(E: np.ndarray, a0: float, a1: float, a2: float) -> np.ndarray:
    val = fwhm(E, a0, a1, a2)
    return np.stack((1/val, E/val, E**2/val)).T

class FWHMInterpolation(Interpolation):
    def __init__(self, points: Vector, a0: float, a1: float, a2: float, cov: np.ndarray | None = None):
        super().__init__(points)
        self.a0 = a0
        self.a1 = a1
        self.a2 = a2
        self.cov = cov

    def eval(self, points: np.ndarray) -> np.ndarray:
        return fwhm(points, self.a0, self.a1, self.a2)

    def __repr__(self) -> str:
        return f"FWHMInterpolation({self.a0}, {self.a1}, {self.a2})"

    def __str__(self) -> str:
        err = np.sqrt(np.diag(self.cov))
        a0 = f"{self.a0: .2e} ± {err[0]: .1e}"
        a1 = f"{self.a1: .2e} ± {err[1]: .1e}"
        a2 = f"{self.a2: .2e} ± {err[2]: .1e}"
        return f"FWHMInterpolation:\n a0: {a0}\n a1: {a1}\n a2: {a2}"

class FWHMInterpolator(Interpolator):
    def interpolate(self) -> FWHMInterpolation:
        X, Y = self.x, self.y
        p0 = [1.0, 0.1, 1e-5]
        p, pcov = curve_fit(fwhm, X, Y, p0=p0, jac=fwhm_jac)
        self.cov = pcov
        return FWHMInterpolation(self.points, *p, cov=pcov)


@njit
def polylog(x: np.ndarray, *p: np.ndarray) -> np.ndarray:
    y = np.zeros_like(x)
    xlog = np.log(x)
    xlog[~np.isfinite(xlog)] = 0
    for i in range(len(p)):
        y += p[i] * xlog**i
    return np.exp(y)

@njit
def polylog_jac(x: np.ndarray, *p: np.ndarray) -> np.ndarray:
    jac = np.empty((len(x), len(p)))
    xlog = np.log(x)
    value = polylog(x, *p)
    for i in range(len(p)):
        jac[:, i] = value * xlog**i
    return jac

class FEInterpolation(Interpolation):
    def __init__(self, points: Vector, lerp: LinearInterpolation, p: np.ndarray, cov: np.ndarray | None = None):
        super().__init__(points)
        self.lerp = lerp
        self.p = p
        self.cov = cov

    def eval(self, points: np.ndarray) -> np.ndarray:
        limit = self.lerp.points.E[-1]
        pl = polylog(points, *self.p)
        lin = self.lerp.eval(points)
        return np.where(points < limit, lin, pl)

    def __str__(self) -> str:
        err = np.sqrt(np.diag(self.cov))
        s = f"FE logarithmic polynomial of order {len(self.p)}:\n"
        for i, (p, e) in enumerate(zip(self.p, err)):
            s += f" p{i}: {p: .2e} ± {e: .1e}\n"
        return s

class FEInterpolator(Interpolator):
    def interpolate(self, order: int = 7, linear_num_points: int = 5) -> FEInterpolation:
        X, Y = self.x, self.y
        # Linear
        vec_linear = self.points.iloc[:linear_num_points]
        lin_intp: LinearInterpolation = LinearInterpolator(vec_linear).interpolate()
        p0 = [10**(-i) for i in range(order)]
        p, cov = curve_fit(polylog, X, Y, p0=p0, jac=polylog_jac)
        self.cov = cov
        return FEInterpolation(self.points, lin_intp, p, cov=cov)
