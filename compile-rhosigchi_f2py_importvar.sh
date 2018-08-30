#!/bin/bash

f2py -h --overwrite-signature rhosigchi_f2py_importvar.pyf -m rhosigchi_f2py_importvar rhosigchi_f2py-import_variance_matrix-20161114.f

f2py -c rhosigchi_f2py_importvar.pyf rhosigchi_f2py-import_variance_matrix-20161114.f