# Version control taken from numpy
# We first need to detect if we're being called as part of the ompy setup
# procedure itself in a reliable manner.
try:
    __OMPY_SETUP__
except NameError:
    __OMPY_SETUP__ = False

if __OMPY_SETUP__:
    import sys
    sys.stderr.write('Running from ompy source directory.\n')
else:

    import os
    from pint import UnitRegistry
    from pint.errors import DimensionalityError
    ureg = UnitRegistry()
    ureg.setup_matplotlib()
    u = ureg
    Q_ = ureg.Quantity
    Quantity = ureg.Quantity
    Unit = ureg.Unit

    try:
        from ompy.decomposition import *
        # if importing one of the cython modules fails, it may hint on it
        # not beeing installed correctly
    except ImportError as e:
        msg = f"""Error importing ompy: you should not try to import ompy from
        its source directory unless it is a submodule; please exit the ompy
        source tree, and relaunch your python interpreter from there.
        If it is a submodule, you probably forgor to run
        `python setup.py build_ext --inplace` in the folder.
        Original exception was: {e}"""
        raise ImportError(msg)
    from .version import GIT_REVISION as __git_revision__
    from .version import VERSION as __version__
    from .version import FULLVERSION as __full_version__

    import warnings
    # warnings.simplefilter('always', DeprecationWarning)

    import pkgutil

    def is_available(pkg, load=True, suppress_warnings=True):
        try:
            exists = pkgutil.find_loader(pkg) is not None
        except ImportError: 
            # As usual, ROOT is a special case. It can refuse to
            # import for no particular reason, in which case it
            # throws an exception
            exists = False
        if exists and load:
            if suppress_warnings:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    __import__(pkg)
            else:
                __import__(pkg)
        return exists

    NUMBA_AVAILABLE = is_available("numba")
    GPU_AVAILABLE = False
    NUMBA_CUDA_AVAILABLE = False
    if NUMBA_AVAILABLE:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                from numba import cuda
            NUMBA_CUDA_AVAILABLE = True
        except ImportError:
            warnings.warn(
                "Numba.CUDA could not be imported. GPU acceleration will not be available")
    else:
        warnings.warn(
            "Numba could not be imported. Falling back to non-jiting which will be much slower")
    # Whether Numba cuda is actually working can only be determined by trying to compile code.
    # That is done in the numbalib module
    NUMBA_CUDA_WORKING = [False]
    # If numba is not available, these imports are dummies
    from .numbalib import jit, njit, nop_nop, int32, float32, float64, prange

    # ROOT messes with the version of system libraries, so it is not imported
    # unless the user explicitly requests it.
    # TODO: This is not working yet. ROOT is currently imported
    ROOT_AVAILABLE = is_available("ROOT", load=True)
    ROOT_IMPORTED = True
    ROOT_CALLBACKS = []

    def import_ROOT() -> None:
        global ROOT_IMPORTED, ROOT_CALLBACKS
        if ROOT_IMPORTED:
            return
        if not ROOT_AVAILABLE:
            raise ImportError("ROOT is not available")
        import ROOT
        ROOT_IMPORTED = True
        # Call all callbacks that were registered before ROOT was imported
        # Will redefine relevant functions
        for callback in ROOT_CALLBACKS:
            callback()

    MINUIT_AVAILABLE = is_available("iminuit")
    JAX_AVAILABLE = is_available("jax")
    JAX_WORKING = False
    if JAX_AVAILABLE:
        import jax
        devices = jax.devices()
        JAX_WORKING = any("gpu" in device.platform.lower()
                          for device in devices)
    GPU_AVAILABLE = NUMBA_CUDA_AVAILABLE or JAX_WORKING

    H5PY_AVAILABLE = is_available("h5py")

    MULTINEST_AVAILABLE = False
    if is_available("pymultinest", load=False):
        # Check if MultiNest's LD_LIBRARY_PATH is set
        if "LD_LIBRARY_PATH" in os.environ:
            MULTINEST_AVAILABLE = is_available("pymultinest")
        else:
            warnings.warn("pymultinest is installed, but LD_LIBRARY_PATH is not set."
                          "\nSee http://johannesbuchner.github.io/PyMultiNest/install.html#installing-the-python-module"
                          )
    XARRAY_AVAILABLE = is_available("xarray")

    from .status import print_status

    from .stubs import Axes
    from .helpers import make_axes, make_combined_legend
    # Simply import all functions and classes from all files to make them
    # available at the package level
    from .validator import Unitful, Bounded, Choice, Toggle
    # from .spinfunctions import (SpinFunction, Const, EB05, EB09CT, EB09Emp,
    #                            DiscAndEB05, SpinModel)
    from .geometry import Geometry, Line
    # Abstracts away hierachical saving to and from different formats
    #from .fileio import JSONWriter
    from .array import AbstractArrayProtocol, MatrixProtocol
    from .array import Vector, Matrix, zeros_like, empty_like, empty, transition_matrix
    from .array import to_index, Index, fmap, umap, omap, linspace, unpack_to_vectors
    from .array import ErrorVector, SymmetricVector, AsymmetricVector, CorrelationMatrix, PoissonVector, ArrayList
    # from .database import Nucleus, get_nucleus, get_nucleus_df
    from .unfolder import Unfolder
    # from .examples import example_raw, list_examples
    # from .ensemble import Ensemble
    from . import response
    from .response import Response, Calibrator, ResponseData, DiscreteInterpolation
    from .unfolding import Guttormsen
    # from .gauss_smoothing import *
    from .action import Action
    from . import firstgeneration
    from .firstgeneration import (FirstGenerationParameters, first_generation, FGP)
    from .extractor import Extractor
    # from .decomposition import nld_T_product, index, index_2
    from .normalization import *
    from .shape import Shape, normalize_to_shape
    from .detector import Detector, OSCAR
    from .peakfit import fit
    from .peakselect import fit_gauss
    from .peakselect import gaussian as pgaussian
    from .introspection import logging, hooks
    from .clicker import Clicker
