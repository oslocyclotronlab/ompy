from . import (ROOT_AVAILABLE, MINUIT_AVAILABLE,
                JAX_AVAILABLE, H5PY_AVAILABLE, NUMBA_AVAILABLE,
                GPU_AVAILABLE, NUMBA_CUDA_AVAILABLE,
                NUMBA_CUDA_WORKING, JAX_WORKING, ROOT_IMPORTED,
                XARRAY_AVAILABLE)
from .version import get_version_info
import os
import platform

def color_status(status: bool) -> str:
    """Returns a string with the status in color."""
    if status:
        return "\033[92mOK\033[0m"
    else:
        return "\033[91mNO\033[0m"

def get_cpu() -> str:
    """Returns the CPU name."""
    cpu = platform.processor()
    if cpu:
        return cpu
    # Branch on the platform
    try:
        if platform.system() == "Linux":
            cpu = os.popen("cat /proc/cpuinfo | grep 'model name' | uniq").read()
            if cpu:
                return cpu.split(":")[1].strip()
        elif platform.system() == "Darwin":
            cpu = os.popen("sysctl -n machdep.cpu.brand_string").read()
            if cpu:
                return cpu.strip()
        elif platform.system() == "Windows":
            cpu = os.popen("wmic cpu get name").read()
            if cpu:
                return cpu.split("\n")[2].strip()
    except Exception:
        # A lot can go wrong, so just give up at any resistance
        pass
    return "Unknown"

def print_status():
    """Prints a report of the status of the dependencies."""
    full_version, git_version = get_version_info()
    availabe_cpus = len(os.sched_getaffinity(0))
    msg = f"""
                OMpy Status
==============================================
Version:              {full_version}
Git version:          {git_version}
GPU available:        {color_status(GPU_AVAILABLE)}
NUMBA available:      {color_status(NUMBA_AVAILABLE)}
NUMBA CUDA available: {color_status(NUMBA_CUDA_AVAILABLE)}
NUMBA CUDA working:   {color_status(NUMBA_CUDA_WORKING[0])}
ROOT available:       {color_status(ROOT_AVAILABLE)}
ROOT imported:        {color_status(ROOT_IMPORTED)}
MINUIT available:     {color_status(MINUIT_AVAILABLE)}
JAX available:        {color_status(JAX_AVAILABLE)}
JAX working:          {color_status(JAX_WORKING)}
H5PY available:       {color_status(H5PY_AVAILABLE)}
XARRAY available:     {color_status(XARRAY_AVAILABLE)}

Platform:             {platform.platform()}
CPU:                  {get_cpu()}
  + architecture:     {platform.architecture()[0]}
  + number:           {availabe_cpus}"""
    try:
        import psutil
        virtual_memory = psutil.virtual_memory()
        msg += f"""
  + frequency:        {psutil.cpu_freq().current:.2f} MHz
Total memory:         {virtual_memory.total / 1024**3:.2f} GB
Available memory:     {virtual_memory.available / 1024**3:.2f} GB
        """
    except ImportError:
        pass

    if JAX_AVAILABLE:
        import jax, jaxlib
        gpus = [device.device_kind for device in jax.devices() if "gpu" in device.platform.lower()]
        msg += f"""
JAX version:          {jax.__version__}
JAXlib version:       {jaxlib.__version__}
"""
        if gpus:
            msg += f"""
Available GPUs:       {len(gpus)}
  + kind:             {gpus if len(gpus) > 1 else gpus[0]}
"""
    print(msg)
