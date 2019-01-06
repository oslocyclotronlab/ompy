#!/bin/bash

f2py -h --overwrite-signature rhosigchi_f2py_origvar.pyf -m rhosigchi_f2py_origvar rhosigchi_f2py-original_variance_matrix.f

f2py -c rhosigchi_f2py_origvar.pyf rhosigchi_f2py-original_variance_matrix.f