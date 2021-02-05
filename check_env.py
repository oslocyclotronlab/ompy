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

from IPython import get_ipython
ipython = get_ipython()

import watermark


ipython.magic("load_ext watermark")
ipython.magic("watermark -m -u -d -v -iv -w")