# -*- coding: utf-8 -*-
from distutils.core import setup
from Cython.Build import cythonize

# build me (i.e. compile Cython modules) for testing in this directory using
# python setup.py build_ext --inplace

setup(name='oslo_method_python',
      version='0.1',
      author="Jørgen Eriksson Midtbø",
      author_email="jorgenem@gmail.com",
      url="https://github.com/jorgenem/oslo_method_python",
      py_modules=['oslo_method_python'],
      ext_modules=cythonize(
                            [
                            "oslo_method_python/rebin.pyx",
                            "oslo_method_python/rhosig.pyx"
                            ])
      )
