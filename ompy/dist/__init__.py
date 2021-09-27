import pymc3 as pm
from packaging import version

if version.parse(pm.__version__) < version.parse("4.0.0"):
    from .fermi_dirac import FermiDirac
else:
    from .fermi_dirac_pymc4 import FermiDirac
