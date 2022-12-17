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

    from pint import UnitRegistry
    from pint.errors import DimensionalityError
    ureg = UnitRegistry()
    ureg.setup_matplotlib()
    u = ureg
    Q_ = ureg.Quantity
    Quantity = ureg.Quantity

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
    warnings.simplefilter('always', DeprecationWarning)

    # Simply import all functions and classes from all files to make them
    # available at the package level
    from .validator import Unitful, Bounded, Choice, Toggle
    #from .spinfunctions import (SpinFunction, Const, EB05, EB09CT, EB09Emp,
    #                            DiscAndEB05, SpinModel)
    from .spinfunctions import SpinFunctions
    from .geometry import Geometry, Line
    from .array import Vector, Matrix, zeros_like, empty_like, empty
    from .database import Nucleus, get_nucleus, get_nucleus_df
    from .models import Model, NormalizationParameters, ResultsNormalized
    from .unfolder import Unfolder
    from .examples import example_raw, list_examples
    from .ensemble import Ensemble
    from . import response
    from .response import Response, Calibrator, ResponseData, ResponseInterpolation
    from .gauss_smoothing import *
    from .firstgeneration import FirstGeneration, normalize_rows
    from .extractor import Extractor
    from .action import Action
    #from .decomposition import nld_T_product, index, index_2
    from .normalizer_nld import (NormalizerNLD, load_levels_discrete,
                                 load_levels_smooth)
    from .normalizer_gsf import NormalizerGSF
    from .normalizer_simultan import NormalizerSimultan
    from .ensembleNormalizer import EnsembleNormalizer
    from .models import NormalizationParameters, ResultsNormalized
    from .shape import Shape, normalize_to_shape
    from .library import (div0, fill_negative_gauss, fill_negative_max,
                          plot_trapezoid, contains_zeroes_patches,
                          ascii_plot, plot_projection_rectangle)
    from .detector import Detector, OSCAR
    from .peakselect import fit_gauss
    from .introspection import logging, hooks
