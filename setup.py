# -*- coding: utf-8 -*-
from setuptools import setup, Extension, find_packages
from pkg_resources import get_build_platform
import numpy
import subprocess
import os
import platform
import builtins

try:
    from Cython.Build import cythonize
    from Cython.Distutils import build_ext
except ImportError:
    raise ImportError("Need to have installed Cython module for compilation")


# build me (i.e. compile Cython modules) for testing in this directory using
# python setup.py build_ext --inplace

# Version rutine taken from numpy
MAJOR = 0
MINOR = 4
MICRO = 0
VERSION = '%d.%d.%d' % (MAJOR, MINOR, MICRO)


# Return the git revision as a string
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
    elif os.path.exists('ompy/version.py'):
        # must be a source distribution, use existing version file
        try:
            from ompy.version import git_revision as GIT_REVISION
        except ImportError:
            raise ImportError("Unable to import git_revision. Try removing "
                              "ompy/version.py and the build directory "
                              "before building.")
    else:
        GIT_REVISION = "Unknown"

    FULLVERSION += '.dev0+' + GIT_REVISION[:7]

    return FULLVERSION, GIT_REVISION


def write_version_py(filename='ompy/version.py'):
    cnt = """
# THIS FILE IS GENERATED FROM OMPY SETUP.PY
#
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


# some machines have difficulties with OpenMP
openmp = os.getenv("ompy_OpenMP")
if openmp is None and platform.system() == 'Darwin':
    openmp = False
    print("MacOS detected: Building without OpenMP")
elif openmp in (True, "True", "true"):
    openmp = True
elif openmp in (False, "False", "false"):
    openmp = False
    print("Building without OpenMP")
else:
    raise ValueError("Env var ompy_OpenMP must be either True or False "
                     "(or not set); use eg. 'export ompy_OpenMP=False'")
fname = "ompy/decomposition.c"  # otherwise it may not recompile
if os.path.exists(fname):
    os.remove(fname)

extra_compile_args = ["-O3", "-ffast-math", "-march=native"]
extra_link_args = []
if openmp:
    extra_compile_args.insert(-1, "-fopenmp")
    extra_link_args.insert(-1, "-fopenmp")

ext_modules = [
        Extension("ompy.decomposition",
                  ["ompy/decomposition.pyx"],
                  # on MacOS the clang compiler pretends not to support OpenMP, but in fact it does so
                  extra_compile_args=extra_compile_args,
                  extra_link_args=extra_link_args,
                  include_dirs=[numpy.get_include()]
                  ),
        Extension("ompy.rebin", ["ompy/rebin.pyx"], include_dirs=[numpy.get_include()]),
        Extension("ompy.gauss_smoothing", ["ompy/gauss_smoothing.pyx"], include_dirs=[numpy.get_include()]),
        Extension("ompy.rhosig", ["ompy/rhosig.pyx"], include_dirs=[numpy.get_include()])
        ]

install_requires = numpy.loadtxt("requirements.txt", dtype="str").tolist()
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
                            ),
      zip_safe=False,
      install_requires=install_requires
      )

