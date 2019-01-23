from distutils.core import setup
from Cython.Build import cythonize

# build me using
# python setup.py build_ext --inplace

setup(name='rebin',
      ext_modules=cythonize("oslo_method_python/rebin.pyx"))

setup(name='fit_rho_T',
      ext_modules=cythonize("oslo_method_python/fit_rho_T.pyx"))
