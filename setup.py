# -*- coding: utf-8 -*-
from distutils.core import setup
from Cython.Build import cythonize

# build me (i.e. compile Cython modules) for testing in this directory using
# python setup.py build_ext --inplace

setup(name='OMpy',
      version='0.2',
      author="Jørgen Eriksson Midtbø, Fabio Zeiser, Erlend Lima",
      author_email="jorgenem@gmail.com, fabio.zeiser@fys.uio.no,\
                    erlendlima@outlook.com",
      url="https://github.com/oslocyclotronlab/ompy",
      py_modules=['ompy'],
      ext_modules=cythonize(
                            [
                             "ompy/rebin.pyx",
                             "ompy/rhosig.pyx",
                             "ompy/response.pyx"
                            ])
      )
