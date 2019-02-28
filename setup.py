# -*- coding: utf-8 -*-
from distutils.core import setup
from Cython.Build import cythonize
import numpy

# build me (i.e. compile Cython modules) for testing in this directory using
# python setup.py build_ext --inplace

setup(name='Oslo Method',
      version='0.1',
      author="Jørgen Eriksson Midtbø, Erlend Lima",
      author_email="jorgenem@gmail.com, erlendlima@outlook.com",
      url="https://github.com/jorgenem/oslo_method_python",
      py_modules=['ompy'],
      include_dirs=[numpy.get_include()],
      ext_modules=cythonize(
                            [
                            "ompy/rebin.pyx",
                            "ompy/rhosig.pyx"
                            ])
      )
