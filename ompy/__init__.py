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
    try:
        from ompy.rebin import *
        # if importing one of the cython modules fails, it may hint on it
        # not beeing installed correctly
    except ImportError:
        msg = """Error importing ompy: you should not try to import ompy from
        its source directory unless it is a submodule; please exit the ompy
        source tree, and relaunch your python interpreter from there.
        If it is a submodule, you probably forgor to run
        `python setup.py build_ext --inplace` in the folder."""
        raise ImportError(msg)
    from .version import GIT_REVISION as __git_revision__
    from .version import VERSION as __version__
    from .version import FULLVERSION as __full_version__

    # Simply import all functions and classes from all files to make them
    # available at the package level
    from .library import div0, fill_negative_gauss, fill_negative_max
    from .spinfunctions import SpinFunctions
    from .abstractarray import AbstractArray
    from .matrix import Matrix, ZerosMatrix
    from .models import Model, NormalizationParameters, ResultsNormalized
    from .vector import Vector
    from .unfolder import Unfolder
    from .examples import example_raw, list_examples
    from .ensemble import Ensemble
    from .response import Response
    from .gauss_smoothing import *
    from .firstgeneration import FirstGeneration, normalize_rows
    from .extractor import Extractor
    from .action import Action
    from .decomposition import nld_T_product, index
    from .normalizer_nld import (NormalizerNLD, load_levels_discrete,
                                 load_levels_smooth)
    from .normalizer_gsf import NormalizerGSF
    from .normalizer_simultan import NormalizerSimultan
    from .ensembleNormalizer import EnsembleNormalizer
    from .models import NormalizationParameters, ResultsNormalized
    from .introspection import logging, hooks
