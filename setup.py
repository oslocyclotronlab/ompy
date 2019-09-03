# -*- coding: utf-8 -*-
from setuptools import setup, Extension
from pkg_resources import get_build_platform
import numpy
import os

try:
    from Cython.Build import cythonize
    from Cython.Distutils import build_ext
except ImportError:
    raise ImportError("Need to have installed Cython module for compilation")


# build me (i.e. compile Cython modules) for testing in this directory using
# python setup.py build_ext --inplace

# some machines have difficulties with OpenMP
openmp = os.getenv("ompy_OpenMP")
if openmp in (None, True, "True", "true"):
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

setup(name='OMpy',
      version='0.3',
      author="Jørgen Eriksson Midtbø, Fabio Zeiser, Erlend Lima",
      author_email=("jorgenem@gmail.com, "
                    "fabio.zeiser@fys.uio.no, "
                    "erlenlim@fys.uio.no"),
      url="https://github.com/oslocyclotronlab/ompy",
      py_modules=['ompy'],
      ext_modules=cythonize(ext_modules,
                            compiler_directives={'language_level': "3"},
                            compile_time_env={"OPENMP": openmp}
                            ),
      zip_safe=False,
      install_requires=[
          'cython',
          'numpy',
          'matplotlib',
          'termtables',
          'pymultinest',
          'scipy',
          'uncertainties',
          'tqdm'
      ]
      )

