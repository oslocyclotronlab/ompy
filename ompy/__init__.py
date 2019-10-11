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
    except ImportError:
        msg = """Error importing ompy: you should not try to import ompy from
        its source directory; please exit the ompy source tree, and relaunch
        your python interpreter from there."""
        raise ImportError(msg)
    from .version import git_revision as __git_revision__
    from .version import version as __version__
    from .version import full_version as __full_version__


# Simply import all functions and classes from all files to make them available
# at the package level
from .library import *
from .rebin import *
from .spinfunctions import *
from .fit_rho_T import *
from .first_generation_method import *
from .matrix import Matrix
from .vector import Vector
from .rhosig import *
from .unfolder import *
from .examples import *
from .ensemble import Ensemble
from .norm_nld import *
from .norm_gsf import *
from .multinest_setup import *
from .response import *
from .response_class import *
from .compton_subtraction_method import *
from .gauss_smoothing import *
from .firstgeneration import FirstGeneration
from .extractor import Extractor
from .setable import Setable
from .filehandling import *
from .action import Action
from .decomposition import *
from .normalizer import Normalizer, load_levels_discrete, load_levels_smooth, Eshift_from_T, Sn_from_D0, errfn
from .introspection import logging, hooks
