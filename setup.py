from distutils.core import setup
from Cython.Build import cythonize

setup(name='Rebin function',
      ext_modules=cythonize("oslo_method_python/rebin.pyx"))