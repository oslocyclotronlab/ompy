from __future__ import annotations
from . import (ROOT_AVAILABLE, MINUIT_AVAILABLE,
                JAX_AVAILABLE, H5PY_AVAILABLE, NUMBA_AVAILABLE,
                GPU_AVAILABLE, NUMBA_CUDA_AVAILABLE,
                NUMBA_CUDA_WORKING, JAX_WORKING, ROOT_IMPORTED,
                XARRAY_AVAILABLE, GAMBIT_AVAILABLE, EMCEE_AVAILABLE,
                PYMC_AVAILABLE, PYRO_AVAILABLE, SKLEARN_AVAILABLE)

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
"""
    if NUMBA_AVAILABLE:
        msg += f"""    + CUDA available: {color_status(NUMBA_CUDA_AVAILABLE)}
    + CUDA working:   {color_status(NUMBA_CUDA_WORKING[0])}"""
    msg += f"""
ROOT available:       {color_status(ROOT_AVAILABLE)}"""
    if ROOT_AVAILABLE:
        msg += f"""
    + imported:       {color_status(ROOT_IMPORTED)}
"""
    msg += f"""MINUIT available:     {color_status(MINUIT_AVAILABLE)}
JAX available:        {color_status(JAX_AVAILABLE)}
"""
    if JAX_AVAILABLE:
        msg += f"""    + working:        {color_status(JAX_WORKING)}"""
    msg += f"""
H5PY available:       {color_status(H5PY_AVAILABLE)}
XARRAY available:     {color_status(XARRAY_AVAILABLE)}
GAMBIT available:     {color_status(GAMBIT_AVAILABLE)}
EMCEE available:      {color_status(EMCEE_AVAILABLE)}
PYMC available:       {color_status(PYMC_AVAILABLE)}
PYRO available:       {color_status(PYRO_AVAILABLE)}
SKLEARN available:    {color_status(SKLEARN_AVAILABLE)}

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



from dataclasses import dataclass, field
from abc import ABC, abstractmethod

@dataclass
class Entry(ABC):
    name: str

    @abstractmethod
    def render_parts(self, pad: int = 0) -> tuple[str, str]:
        pass

    def render(self, pad: int = 0) -> str:
        name, info = self.render_parts(pad)
        return f"{name}: {info}"

    def __len__(self) -> int:
        a, b = self.render_parts()
        return len(a) + len(b)


@dataclass
class StatusEntry(Entry):
    status: bool

    def render_parts(self, pad: int = 0) -> tuple[str, str]:
        s = color_status(self.status)
        s = s.ljust(pad)
        return f"{self.name}", s


@dataclass
class InfoEntry(Entry):
    info: str

    def render_parts(self, pad: int = 0) -> tuple[str, str]:
        s = self.info
        s = s.ljust(pad)
        return f"{self.name}", s


class Menu:
    def __init__(self, title: str):
        self.title = title
        self.entries = []
        self.submenus = []

    def append(self, Entry: Entry) -> None:
        self.entries.append(Entry)

    def add_submenu(self, menu: Menu) -> None:
        self.submenus.append(menu)

    def render_text(self, level=0):
        indent = "  " * level
        text = f"{indent}{self.title}\n"
        text += f"{indent}{'=' * len(self.title)}\n"

        pad = self.pad_length()
        for entry in self.entries:
            text += f"{indent}{entry.render(pad)}"
            text += "\n"
        for submenu in self.submenus:
            text += submenu.render_text(level + 1)
        return text

    def render_html(self, level=0):
        html = f"<div><strong>{self.title}</strong><br>"
        for entry in self.entries:
            html += entry.render()
            html += "<br>"

        for submenu in self.submenus:
            submenu_id = f"{self.title.replace(' ', '_')}_{submenu.title.replace(' ', '_')}"
            html += f'<a href="javascript:void(0);" onclick="toggleVisibility(\'{submenu_id}\')">{submenu.title}</a>'
            html += f'<div id="{submenu_id}" style="display:none; margin-left: {20 * (level + 1)}px;">'
            html += submenu.render_html(level + 1)
            html += "</div>"

        html += "</div>"
        return html

    def pad_length(self) -> int:
        i = max(len(entry) for entry in self.entries)
        for submenu in self.submenus:
            i = max(i, submenu.pad_length())
        return i

    def _repr_html_(self):
        script = """
        <script>
        function toggleVisibility(id) {
            var x = document.getElementById(id);
            if (x.style.display === "none") {
                x.style.display = "block";
            } else {
                x.style.display = "none";
            }
        }
        </script>
        """

        html_content = self.render_html()
        return html_content + script


def get_status_menu() -> Menu:
    menu = Menu("OMpy Status")
    full_version, git_version = get_version_info()
    menu.append(InfoEntry("Version", full_version))
    menu.append(InfoEntry("Git version", git_version))
    menu.append(StatusEntry("GPU available", GPU_AVAILABLE))
    menu.append(StatusEntry("NUMBA available", NUMBA_AVAILABLE))
    if NUMBA_AVAILABLE:
        menu.append(StatusEntry("CUDA available", NUMBA_CUDA_AVAILABLE))
        menu.append(StatusEntry("CUDA working", NUMBA_CUDA_WORKING[0]))
    menu.append(StatusEntry("ROOT available", ROOT_AVAILABLE))
    if ROOT_AVAILABLE:
        menu.append(StatusEntry("ROOT imported", ROOT_IMPORTED))
    menu.append(StatusEntry("MINUIT available", MINUIT_AVAILABLE))
    menu.append(StatusEntry("JAX available", JAX_AVAILABLE))
    if JAX_AVAILABLE:
        menu.append(StatusEntry("JAX working", JAX_WORKING))
    menu.append(StatusEntry("H5PY available", H5PY_AVAILABLE))
    menu.append(StatusEntry("XARRAY available", XARRAY_AVAILABLE))
    menu.append(StatusEntry("GAMBIT available", GAMBIT_AVAILABLE))
    menu.append(StatusEntry("EMCEE available", EMCEE_AVAILABLE))
    menu.append(StatusEntry("PYMC available", PYMC_AVAILABLE))
    menu.append(StatusEntry("PYRO available", PYRO_AVAILABLE))
    menu.append(StatusEntry("SKLEARN available", SKLEARN_AVAILABLE))

    menu.add_submenu(get_platform_menu())
    if JAX_AVAILABLE:
        menu.add_submenu(get_jax_menu())
    return menu

def get_jax_menu() -> Menu:
    menu = Menu("JAX")
    import jax, jaxlib
    gpus = [device.device_kind for device in jax.devices() if "gpu" in device.platform.lower()]
    menu.append(InfoEntry("JAX version", jax.__version__))
    menu.append(InfoEntry("JAXlib version", jaxlib.__version__))
    if gpus:
        menu.append(InfoEntry("Available GPUs", str(len(gpus))))
        menu.append(InfoEntry("GPU kind", str(gpus if len(gpus) > 1 else gpus[0])))
    return menu

def get_platform_menu() -> Menu:
    menu = Menu("Platform")
    menu.append(InfoEntry("Platform", platform.platform()))
    menu.append(InfoEntry("CPU", get_cpu()))
    menu.append(InfoEntry("Architecture", platform.architecture()[0]))
    menu.append(InfoEntry("Number of CPUs", str(len(os.sched_getaffinity(0)))))
    try:
        import psutil
        virtual_memory = psutil.virtual_memory()
        menu.append(InfoEntry("CPU frequency", f"{psutil.cpu_freq().current:.2f} MHz"))
        menu.append(InfoEntry("Total memory", f"{virtual_memory.total / 1024**3:.2f} GB"))
        menu.append(InfoEntry("Available memory", f"{virtual_memory.available / 1024**3:.2f} GB"))
    except ImportError:
        pass
    return menu
