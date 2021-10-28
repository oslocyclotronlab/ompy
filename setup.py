# -*- coding: utf-8 -*-
from setuptools import setup, Extension, find_packages
from pkg_resources import get_build_platform
import numpy
import subprocess
import os
import builtins
import platform
from ctypes.util import find_library
from pybind11.setup_helpers import Pybind11Extension

try:
    from Cython.Build import cythonize
    from Cython.Distutils import build_ext
except ImportError:
    raise ImportError("Need to have installed Cython module for compilation")


# build me (i.e. compile Cython modules) for testing in this directory using
# python setup.py build_ext --inplace

# Version rutine taken from numpy
MAJOR = 1
MINOR = 1
MICRO = 0
VERSION = '%d.%d.%d' % (MAJOR, MINOR, MICRO)


def check_if_clang_compiler():
    """Check if the compiler is clang or gcc"""

    # Find the CC variable
    cc = os.getenv("CC")
    cc = 'gcc' if cc is None else cc  # Will be gcc if none.
    std_err = subprocess.run(cc, capture_output=True, text=True).stderr
    if "clang" in std_err:
        return True


# Return the git revision as a string
# See also ompy/version.py
def git_version():
    def _minimal_ext_cmd(cmd):
        # construct minimal environment
        env = {}
        for k in ['SYSTEMROOT', 'PATH', 'HOME']:
            v = os.environ.get(k)
            if v is not None:
                env[k] = v
        # LANGUAGE is used on win32
        env['LANGUAGE'] = 'C'
        env['LANG'] = 'C'
        env['LC_ALL'] = 'C'
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, env=env)
        return out

    try:
        out = _minimal_ext_cmd(['git', 'rev-parse', 'HEAD'])
        GIT_REVISION = out.strip().decode('ascii')
    except (subprocess.SubprocessError, OSError):
        GIT_REVISION = "Unknown"

    if not GIT_REVISION:
        # this shouldn't happen but apparently can (see gh-8512)
        GIT_REVISION = "Unknown"

    return GIT_REVISION

# BEFORE importing setuptools, remove MANIFEST. Otherwise it may not be
# properly updated when the contents of directories change (true for distutils,
# not sure about setuptools).
if os.path.exists('MANIFEST'):
    os.remove('MANIFEST')

# This is a bit hackish: we are setting a global variable so that the main
# ompy __init__ can detect if it is being loaded by the setup routine, to
# avoid attempting to load components that aren't built yet.  While ugly, it's
# a lot more robust than what was previously being used.
builtins.__OMPY_SETUP__ = True


def get_version_info():
    # Adding the git rev number needs to be done inside write_version_py(),
    # otherwise the import of ompy.version messes up the build under Python 3.
    FULLVERSION = VERSION
    if os.path.exists('.git'):
        GIT_REVISION = git_version()
    elif os.path.exists('ompy/version_setup.py'):
        # must be a source distribution, use existing version file
        try:
            from ompy.version import git_revision as GIT_REVISION
        except ImportError:
            raise ImportError("Unable to import git_revision. Try removing "
                              "ompy/version_setup.py and the build directory "
                              "before building.")
    else:
        GIT_REVISION = "Unknown"

    FULLVERSION += '.dev0+' + GIT_REVISION[:7]

    return FULLVERSION, GIT_REVISION


def write_version_py(filename='ompy/version_setup.py'):
    cnt = """
# THIS FILE IS GENERATED FROM OMPY SETUP.PY
# State of last built is:
short_version = '%(version)s'
version = '%(version)s'
full_version = '%(full_version)s'
git_revision = '%(git_revision)s'
"""
    FULLVERSION, GIT_REVISION = get_version_info()

    a = open(filename, 'w')
    try:
        a.write(cnt % {'version': VERSION,
                       'full_version': FULLVERSION,
                       'git_revision': GIT_REVISION})
    finally:
        a.close()
write_version_py()

# If macOS, use ctypes.util.find_library to determine if OpenMP is avalible.
openmp = os.getenv("ompy_OpenMP")
if openmp is None and platform.system() == 'Darwin':  # Check if macOS
    if find_library("omp") != None:
        openmp = True
        print("libOMP found, building with OpenMP")
    else:
        print("libOMP not found, building without OpenMP")
elif openmp in (None, True, "True", "true"):
    openmp = True
elif openmp in (False, "False", "false"):
    openmp = False
    print("Building without OpenMP")
else:
    raise ValueError("Env var ompy_OpenMP must be either True or False "
                     "(or not set); use eg. 'export ompy_OpenMP=False'"
                     f"Now it is: {openmp}")
fname = "ompy/decomposition.c"  # otherwise it may not recompile
if os.path.exists(fname):
    os.remove(fname)

extra_compile_args_cython = ["-O3", "-ffast-math"]
extra_compile_args_cpp = ["-std=c++11", "-O3"]

extra_link_args = []
if openmp and platform.system() == 'Darwin' and check_if_clang_compiler():
    extra_compile_args_cython.insert(-1, "-Xpreprocessor -fopenmp")
    extra_link_args.insert(-1, "-lomp")
elif openmp:
    extra_compile_args_cython.insert(-1, "-fopenmp")
    extra_link_args.insert(-1, "-fopenmp")

ext_modules = [
        Extension("ompy.decomposition",
                  ["ompy/decomposition.pyx"],
                  # on MacOS the clang compiler pretends not to support OpenMP, but in fact it does so
                  extra_compile_args=extra_compile_args_cython,
                  extra_link_args=extra_link_args,
                  include_dirs=[numpy.get_include()]
                  ),
        Extension("ompy.rebin", ["ompy/rebin.pyx"], include_dirs=[numpy.get_include()]),
        Extension("ompy.gauss_smoothing", ["ompy/gauss_smoothing.pyx"], include_dirs=[numpy.get_include()]),
        ]

ext_modules_pybind11 = [
        Pybind11Extension("ompy.stats",
                          ["src/stats.cpp"],
                          extra_compile_args=extra_compile_args_cpp)
]

# Superhacky solution to get pybind11 to play nicely with GCC...
if platform.system() == 'Darwin' and not check_if_clang_compiler():
    try:
        idx = ext_modules_pybind11[0].extra_compile_args.index(
            '-stdlib=libc++')
        del ext_modules_pybind11[0].extra_compile_args[idx]
    except ValueError:
        pass
    try:
        idx = ext_modules_pybind11[0].extra_link_args.index('-stdlib=libc++')
        del ext_modules_pybind11[0].extra_link_args[idx]
    except ValueError:
        pass

install_requires = [
 "cython",
 "numpy>=1.20.0",
 "pandas",
 "matplotlib",
 "termtables",
 "pymultinest",  # needed only for multinest-runs
 "scipy",
 "uncertainties>=3.0.3",
 "tqdm",
 "pathos",
 "pybind11>=2.6.0"
]

setup(name='OMpy',
      version=get_version_info()[0],
      author="Jørgen Eriksson Midtbø, Fabio Zeiser, Erlend Lima",
      author_email=("jorgenem@gmail.com, "
                    "fabio.zeiser@fys.uio.no, "
                    "erlenlim@fys.uio.no"),
      url="https://github.com/oslocyclotronlab/ompy",
      packages=find_packages(),
      ext_modules=cythonize(ext_modules,
                            compiler_directives={'language_level': "3",
                                                 'embedsignature': True},
                            compile_time_env={"OPENMP": openmp}
                            )+ext_modules_pybind11,
      zip_safe=False,
      install_requires=install_requires
      )
