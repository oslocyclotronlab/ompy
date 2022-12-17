from __future__ import annotations
from . import ResponseData, ResponseInterpolation
from .. import Vector, Matrix
import warnings
import numpy as np
from collections import OrderedDict


try:
    from numba import njit, int32, float32, float64
    from numba.experimental import jitclass
except ImportError:
    warnings.warn("Numba could not be imported. Falling back to non-jiting which will be much slower")
    int32 = np.int32
    float32 = np.float32
    float64 = np.float64

    def nop_decorator(func, *aargs, **kkwargs):
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    njit = nop_decorator
    jitclass = nop_decorator


class Response:
    def __init__(self, probabilities: ResponseData,
                 interpolation: ResponseInterpolation | None = None):
        self.probabilities: ResponseData = probabilities
        self.interpolation: ResponseInterpolation = interpolation or ResponseInterpolation.from_data(probabilities)

    def __call__(self, E: Vector) -> Matrix:
        return self.evaluate(E)

    def evaluate(self, E: Vector) -> Matrix:
        if self.interpolation is None:
            raise ValueError("No interpolation set. Use interpolate() first.")
        pass

    def interpolate_compton(self, E: Vector) -> Matrix:
        return interpolate_compton(self.probabilities, E)

    def clone(self, probabilities: ResponseData = None,
              interpolation: ResponseInterpolation = None) -> Response:
        return Response(probabilities=probabilities or self.probabilities,
                        interpolation=interpolation or self.interpolation)
