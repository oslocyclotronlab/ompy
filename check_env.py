import numpy
import cython
import pandas
import matplotlib
import pymultinest
import scipy
import uncertainties
import tqdm
import pathos
import pybind11
import watermark


from IPython import get_ipython
ipython = get_ipython()
ipython.magic("load_ext watermark")
ipython.magic("watermark -m -u -d -v -iv -w")
