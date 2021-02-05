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

try:
    from IPython import get_ipython
    ipython = get_ipython()
    ipython.magic("load_ext watermark")
    ipython.magic("watermark -m -u -d -v -iv -w")
except:
    try:
        import IPython.core.ipapi  
        ipython = IPython.core.ipapi.get()
        ipython.magic("load_ext watermark")
        ipython.magic("watermark -m -u -d -v -iv -w")
    except:
        try:
            import IPython.ipapi  
            ipython = IPython.ipapi.get()
            ipython.magic("load_ext watermark")
            ipython.magic("watermark -m -u -d -v -iv -w")
        except:
            print("Unable to show versions")