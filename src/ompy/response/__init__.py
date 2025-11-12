"""
This module is a model for the OSCAR response functions.
It was not written with the intent of being used as a library, and hence
is not well-suited for modelling other response functions.
The necessary features the input must have are points defining the
single escape (SE), double escape (DE), the full energy peak (FE),
the annihilation peak (AP), the efficiency (eps), and the "Compton"
background at different Eg and Ex energies.

The Compton class handles the interpolation of the Compton background,
which is by far the most complex part of the response and the most computationally
intensive. It has a numba_array and a numba_array cuda implementation.

ResponseData is an abstraction over the input data points of the discrete features,
while interpolations are the actual interpolations of the response functions. Most
are modelled as logarithmic polynomials, while SE and DE use the model found in GF3.
Together they form the Response class, which is the main interface for the response functions. It
also handles the shitty calibration of the FWHM. Loading and interpolating FWHM is an
vestigeal part as copied from older OMpy and MAMA. It is highly recommended to instead
directly input the FWHM function/polynomial.

perturb.py and calibrator.py are for development, they test the inherent uncertainty of the response
(turns out to be unrealisticlly low), and the calibration of the FWHM. The calibration should take
the entire Eg into account, but I never found a good way to do this. 
"""
from .numbalib import *
from .calibrator import *
from .io import save, load
from .responsedata import *
from .interpolation import *
from .gf3 import GF3Interpolator, GF3Interpolation
from .interpolations import *
from .discreteinterpolation import *
from .compton import *
from ..accel import numba_cuda_available

if numba_cuda_available():
    from .comptongpu import *


from .response import Response as Response
from .response import ResponseMatrices
from .perturb import *
