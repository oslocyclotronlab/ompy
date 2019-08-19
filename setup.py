# -*- coding: utf-8 -*-
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import numpy

# build me (i.e. compile Cython modules) for testing in this directory using
# python setup.py build_ext --inplace

ext_modules = [
        Extension("decomposition",
                  ["ompy/decomposition.pyx"],
                  extra_compile_args=["-O3", "-ffast-math", "-march=native",
                                      "-fopenmp"],
                  extra_link_args=['-fopenmp'],
                  include_dirs=[numpy.get_include()]
                  ),
        Extension("rebin", ["ompy/rebin.pyx"], include_dirs=[numpy.get_include()]),
        Extension("gauss_smoothing", ["ompy/gauss_smoothing.pyx"], include_dirs=[numpy.get_include()]),
        Extension("rhosig", ["ompy/rhosig.pyx"], include_dirs=[numpy.get_include()])
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
                            ),
      zip_safe=False,
      install_requires=[
          'cython',
          'numpy',
          'matplotlib',
          'termtables',
          'pymultinest',
          'scipy',
          'uncertainties'
      ]
      )

