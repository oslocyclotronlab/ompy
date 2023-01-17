from __future__ import annotations
from . import ResponseData, ResponseInterpolation, interpolate_compton
from .. import Vector, Matrix, USE_GPU
import warnings
import numpy as np
if USE_GPU:
    from . import interpolate_gpu
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
                 interpolation: ResponseInterpolation | None = None,
                 compton: Matrix | None = None):
        if not probabilities.is_normalized:
            raise ValueError("Probabilities must be normalized.")
        if not probabilities.is_fwhm_normalized:
            raise ValueError("FWHM must be normalized to experiment.")
        self.probabilities: ResponseData = probabilities
        self.interpolation: ResponseInterpolation = interpolation or ResponseInterpolation.from_data(probabilities)
        self.compton: Matrix = compton

    def __call__(self, E: Vector) -> Matrix:
        return self.evaluate(E)

    def evaluate(self, E: Vector) -> Matrix:
        if self.interpolation is None:
            raise ValueError("No interpolation set. Use interpolate() first.")
        pass

    def interpolate_compton(self, E: np.ndarray, GPU: bool = True,
                            sigma: float = 6) -> Matrix:
        if self.interpolation is None:
            raise RuntimeError("Peak interpolations must be done before compton interpolation.")
        sigmafn = self.interpolation.sigma
        if USE_GPU and GPU:
            compton: Matrix = interpolate_gpu(self.probabilities, E, sigmafn, sigma)
        else:
            compton: Matrix = interpolate_compton(self.probabilities, E, sigmafn, sigma)
        self.compton = compton

    def clone(self, probabilities: ResponseData = None,
              interpolation: ResponseInterpolation = None,
              compton: Matrix = None) -> Response:
        return Response(probabilities=probabilities or self.probabilities,
                        interpolation=interpolation or self.interpolation,
                        compton=compton or self.compton)
